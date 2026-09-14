#!/usr/bin/env python3
"""Standalone SM90 DeepGEMM MegaMoE correctness and performance test.

The default dimensions reproduce one tensor-parallel Qwen4Exp expert shard:
512 experts, hidden size 2560, intermediate size 256, and top-k 10.
Only DeepGEMM/PyTorch are used; GGML and GGUF are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
import shutil
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

# DeepGEMM captures CUDA_HOME when imported. Some CUDA runtime installations
# leave /usr/local/cuda without nvcc even though a compiler is on PATH.
if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        nvcc = next(
            (str(path) for path in sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True)),
            None,
        )
    if nvcc is not None:
        os.environ["CUDA_HOME"] = str(Path(nvcc).resolve().parent.parent)

import deep_gemm
from deep_gemm.utils import ceil_div, per_block_cast_to_fp8, per_token_cast_to_fp8


def symmetric_relative_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double().reshape(-1)
    expected = expected.double().reshape(-1)
    denominator = (actual.square() + expected.square()).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (actual * expected).sum() / denominator)


def init_single_rank(port: int) -> dist.ProcessGroup:
    params: dict[str, object] = {
        "backend": "nccl",
        "init_method": f"tcp://127.0.0.1:{port}",
        "rank": 0,
        "world_size": 1,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device("cuda:0")
    dist.init_process_group(**params)
    return dist.new_group([0])


def make_routing(
    tokens: int,
    experts: int,
    top_k: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ids = torch.stack([
        torch.randperm(experts, generator=generator)[:top_k]
        for _ in range(tokens)
    ])
    logits = torch.randn((tokens, top_k), generator=generator, dtype=torch.float32)
    weights = torch.softmax(logits, dim=-1)
    return ids.to(device=device, dtype=torch.int32), weights.to(device=device)


def quantize_grouped_weights(
    shape: tuple[int, int, int],
    std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, cols = shape
    fp8 = torch.empty(shape, device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (experts, ceil_div(rows, 128), ceil_div(cols, 128)),
        device=device,
        dtype=torch.float32,
    )

    # Quantize one expert at a time to keep peak memory well below one H100.
    for expert in range(experts):
        weight = torch.randn((rows, cols), device=device, dtype=torch.bfloat16)
        weight.mul_(std)
        fp8[expert], scales[expert] = per_block_cast_to_fp8(
            weight, use_ue8m0=False
        )
    return fp8, scales


def dequantize_weight(fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, cols = fp8.shape
    return (
        fp8.float().reshape(rows // 128, 128, cols // 128, 128)
        * scales.float().reshape(rows // 128, 1, cols // 128, 1)
    ).reshape(rows, cols)


def dequantize_rows(fp8: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    rows, cols = fp8.shape
    return (
        fp8.float().reshape(rows, cols // group, group)
        * scales.float().reshape(rows, cols // group, 1)
    ).reshape(rows, cols)


def quantize_dequantize_rows(x: torch.Tensor, group: int) -> torch.Tensor:
    rows, cols = x.shape
    blocks = x.float().reshape(rows, cols // group, group)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    quantized = (blocks / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return (quantized.float() * scales.unsqueeze(-1)).reshape(rows, cols)


def make_reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    expert_size: int,
) -> torch.Tensor:
    tokens, top_k = ids.shape
    hidden = x.shape[1]

    # The SM90 pre-dispatch quantizes input activations per 128 channels.
    x_fp8, x_scales = per_token_cast_to_fp8(x, use_ue8m0=False)
    x_ref = dequantize_rows(x_fp8.cpu(), x_scales.cpu(), 128)
    ids_cpu = ids.cpu()
    routing_cpu = routing_weights.cpu()
    result = torch.zeros((tokens, hidden), dtype=torch.float32)

    # Cache only selected experts.  This keeps the CPU reference small even
    # though Qwen4Exp has 512 resident routed experts.
    l1_cache: dict[int, torch.Tensor] = {}
    l2_cache: dict[int, torch.Tensor] = {}
    for expert in torch.unique(ids_cpu).tolist():
        l1_cache[expert] = dequantize_weight(
            l1_weights[0][expert].cpu(), l1_weights[1][expert].cpu()
        )
        l2_cache[expert] = dequantize_weight(
            l2_weights[0][expert].cpu(), l2_weights[1][expert].cpu()
        )

    for token in range(tokens):
        for slot in range(top_k):
            expert = int(ids_cpu[token, slot])
            gate_up = x_ref[token].float() @ l1_cache[expert].float().t()
            gate, up = gate_up.split(expert_size)
            activated = F.silu(gate) * up

            # SM90 MegaMoE emits one activation scale per 64 intermediate
            # channels before the down projection.
            activated = quantize_dequantize_rows(
                activated.reshape(1, expert_size), 64
            ).reshape(expert_size)
            down = activated.float() @ l2_cache[expert].float().t()
            result[token] += down * routing_cpu[token, slot]

    return result.to(torch.bfloat16)


def run_mega_moe(
    output: torch.Tensor,
    x: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
) -> None:
    deep_gemm.mega_moe_pre_dispatch_sm90(
        x,
        ids,
        routing_weights,
        sym_buffer.x,
        sym_buffer.x_sf,
        sym_buffer.topk_idx,
        sym_buffer.topk_weights,
        num_tokens=x.shape[0],
        group_size=128,
        routed_scaling_factor=1.0,
    )
    deep_gemm.fp8_mega_moe(
        output,
        l1_weights,
        l2_weights,
        sym_buffer,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=None,
        fast_math=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--expert-size", type=int, default=256)
    parser.add_argument("--logical-expert-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--master-port", type=int, default=29651)
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    assert torch.cuda.get_device_capability(0) == (9, 0), "SM90/Hopper is required"
    assert args.tokens <= args.max_tokens
    assert args.top_k <= args.experts
    assert args.hidden_size % 128 == 0
    assert args.expert_size % 128 == 0
    logical_expert_size = args.logical_expert_size or args.expert_size
    assert 0 < logical_expert_size <= args.expert_size

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    group = init_single_rank(args.master_port)
    sym_buffer = None

    try:
        print(f"DeepGEMM {getattr(deep_gemm, '__version__', 'unknown')}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"shape: tokens={args.tokens}, experts={args.experts}, top_k={args.top_k}, "
            f"hidden={args.hidden_size}, intermediate={logical_expert_size}, "
            f"physical_intermediate={args.expert_size}"
        )

        l1 = quantize_grouped_weights(
            (args.experts, 2 * args.expert_size, args.hidden_size),
            1 / math.sqrt(args.hidden_size),
            device,
        )
        l2 = quantize_grouped_weights(
            (args.experts, args.hidden_size, args.expert_size),
            1 / math.sqrt(args.expert_size),
            device,
        )

        # The SM90 workspace requires its per-64 activation-scale row to be
        # 16-byte aligned. A 384-wide shard therefore uses a 512-wide physical
        # representation with zero-padded gate, up, and down channels.
        if logical_expert_size < args.expert_size:
            l1[0][:, logical_expert_size:args.expert_size].zero_()
            l1[0][:, args.expert_size + logical_expert_size:].zero_()
            l2[0][:, :, logical_expert_size:].zero_()

        mega_l1, mega_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(l1, l2)

        x = torch.randn(
            (args.tokens, args.hidden_size), device=device, dtype=torch.bfloat16
        )
        ids, routing_weights = make_routing(
            args.tokens, args.experts, args.top_k, args.seed + 1, device
        )
        output = torch.empty_like(x)
        sym_buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
            group,
            args.experts,
            args.max_tokens,
            args.top_k,
            args.hidden_size,
            args.expert_size,
            use_fp8_dispatch=True,
            activation="swiglu",
        )

        run_mega_moe(output, x, ids, routing_weights, mega_l1, mega_l2, sym_buffer)
        torch.cuda.synchronize()

        if not args.skip_reference:
            reference = make_reference(
                x, ids, routing_weights, l1, l2, args.expert_size
            )
            actual_cpu = output.cpu()
            diff = symmetric_relative_diff(actual_cpu, reference)
            max_abs = float((actual_cpu.float() - reference.float()).abs().max())
            print(f"correctness: symmetric_relative_diff={diff:.6e}, max_abs={max_abs:.6e}")
            assert torch.isfinite(actual_cpu).all(), "MegaMoE output contains non-finite values"
            assert diff < args.threshold, (
                f"MegaMoE diff {diff:.6e} exceeds threshold {args.threshold:.6e}"
            )

        for _ in range(args.warmup):
            run_mega_moe(output, x, ids, routing_weights, mega_l1, mega_l2, sym_buffer)
        torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_mega_moe(output, x, ids, routing_weights, mega_l1, mega_l2, sym_buffer)
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))

        median_ms = statistics.median(times_ms)
        print(
            f"steady_state: median={median_ms:.3f} ms, "
            f"tokens_per_second={args.tokens * 1000 / median_ms:.2f}, "
            f"samples_ms={[round(value, 3) for value in times_ms]}"
        )
        print("PASS: SM90 DeepGEMM MegaMoE matches the quantization-aware reference")
    finally:
        if sym_buffer is not None:
            sym_buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
