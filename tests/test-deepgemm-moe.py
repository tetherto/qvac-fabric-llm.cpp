#!/usr/bin/env python3
"""Standalone DeepGEMM MoE correctness test for NVIDIA Hopper.

This intentionally does not use GGML or GGUF.  It exercises the proposed
MOE_FFN numerical pipeline with synthetic tensors:

    route/pack -> FP8 gate+up -> SwiGLU -> FP8 down -> weighted reduce

DeepGEMM performs the two grouped FP8 GEMMs on the GPU.  The test dequantizes
the exact FP8 operands and performs the corresponding matrix multiplications
in FP32 on the CPU.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.utils import (
    align,
    ceil_div,
    get_mk_alignment_for_contiguous_layout,
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)


@dataclass
class PackedAssignments:
    activations: torch.Tensor
    grouped_layout: torch.Tensor
    token_for_row: torch.Tensor
    expert_for_row: torch.Tensor
    weight_for_row: torch.Tensor
    expected_m: int
    layout: str


@dataclass
class MoeResult:
    output: torch.Tensor
    gate_up: torch.Tensor
    activated: torch.Tensor
    activated_fp8: tuple[torch.Tensor, torch.Tensor]
    down: torch.Tensor


def symmetric_relative_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """DeepGEMM's scale-independent correctness metric."""
    actual = actual.double().reshape(-1)
    expected = expected.double().reshape(-1)
    denominator = (actual.square() + expected.square()).sum()
    if denominator == 0:
        return 0.0
    similarity = 2 * (actual * expected).sum() / denominator
    return float(1 - similarity)


def make_routing(tokens: int, experts: int, top_k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    ids = torch.stack([torch.randperm(experts, generator=generator)[:top_k] for _ in range(tokens)])
    logits = torch.randn((tokens, top_k), generator=generator, dtype=torch.float32)
    weights = torch.softmax(logits, dim=-1)
    return ids.to(torch.int32), weights


def pack_contiguous(
    x: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    experts: int,
) -> PackedAssignments:
    tokens, top_k = ids.shape
    alignment = get_mk_alignment_for_contiguous_layout()
    flat_experts = ids.reshape(-1).to(torch.int64)
    flat_tokens = torch.arange(tokens).repeat_interleave(top_k)
    flat_weights = routing_weights.reshape(-1)

    counts = torch.bincount(flat_experts, minlength=experts)
    padded_counts = [align(int(count), alignment) for count in counts]
    total_rows = sum(padded_counts)

    packed_x = torch.zeros((total_rows, x.shape[1]), device=x.device, dtype=x.dtype)
    m_indices = torch.full((total_rows,), -1, device=x.device, dtype=torch.int32)
    token_for_row = torch.full((total_rows,), -1, device=x.device, dtype=torch.int64)
    expert_for_row = torch.full((total_rows,), -1, device=x.device, dtype=torch.int64)
    weight_for_row = torch.zeros((total_rows,), device=x.device, dtype=torch.float32)

    start = 0
    for expert, padded_count in enumerate(padded_counts):
        assignment_indices = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        count = assignment_indices.numel()
        if count:
            end = start + count
            token_indices = flat_tokens[assignment_indices]
            packed_x[start:end] = x[token_indices.to(x.device)]
            m_indices[start:end] = expert
            token_for_row[start:end] = token_indices.to(x.device)
            expert_for_row[start:end] = expert
            weight_for_row[start:end] = flat_weights[assignment_indices].to(x.device)
        start += padded_count

    return PackedAssignments(
        activations=packed_x,
        grouped_layout=m_indices,
        token_for_row=token_for_row,
        expert_for_row=expert_for_row,
        weight_for_row=weight_for_row,
        expected_m=max(1, math.ceil(tokens * top_k / experts)),
        layout="contiguous",
    )


def pack_masked(
    x: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    experts: int,
) -> PackedAssignments:
    tokens, top_k = ids.shape
    alignment = get_mk_alignment_for_contiguous_layout()
    flat_experts = ids.reshape(-1).to(torch.int64)
    flat_tokens = torch.arange(tokens).repeat_interleave(top_k)
    flat_weights = routing_weights.reshape(-1)
    counts = torch.bincount(flat_experts, minlength=experts)

    # The physical M extent remains fixed for CUDA graph capture.  masked_m
    # tells DeepGEMM how many rows in each expert are actually valid.
    max_m = align(max(1, int(counts.max())), alignment)
    packed_x = torch.zeros((experts, max_m, x.shape[1]), device=x.device, dtype=x.dtype)
    token_for_row = torch.full((experts, max_m), -1, device=x.device, dtype=torch.int64)
    expert_for_row = torch.full((experts, max_m), -1, device=x.device, dtype=torch.int64)
    weight_for_row = torch.zeros((experts, max_m), device=x.device, dtype=torch.float32)

    for expert in range(experts):
        assignment_indices = torch.nonzero(flat_experts == expert, as_tuple=False).flatten()
        count = assignment_indices.numel()
        if not count:
            continue
        token_indices = flat_tokens[assignment_indices]
        packed_x[expert, :count] = x[token_indices.to(x.device)]
        token_for_row[expert, :count] = token_indices.to(x.device)
        expert_for_row[expert, :count] = expert
        weight_for_row[expert, :count] = flat_weights[assignment_indices].to(x.device)

    return PackedAssignments(
        activations=packed_x,
        grouped_layout=counts.to(device=x.device, dtype=torch.int32),
        token_for_row=token_for_row,
        expert_for_row=expert_for_row,
        weight_for_row=weight_for_row,
        expected_m=max(1, math.ceil(tokens * top_k / experts)),
        layout="masked",
    )


def quantize_grouped_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, cols = weights.shape
    fp8 = torch.empty_like(weights, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (experts, ceil_div(rows, 128), ceil_div(cols, 128)),
        device=weights.device,
        dtype=torch.float32,
    )
    for expert in range(experts):
        fp8[expert], scales[expert] = per_block_cast_to_fp8(weights[expert], use_ue8m0=False)
    return fp8, scales


def quantize_grouped_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 2:
        return per_token_cast_to_fp8(x, use_ue8m0=False)

    experts, rows, cols = x.shape
    fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (experts, rows, ceil_div(cols, 128)),
        device=x.device,
        dtype=torch.float32,
    )
    for expert in range(experts):
        fp8[expert], scales[expert] = per_token_cast_to_fp8(x[expert], use_ue8m0=False)
    return fp8, scales


def grouped_gemm(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    packed: PackedAssignments,
    output_rows: int,
) -> torch.Tensor:
    output_shape = (*a[0].shape[:-1], output_rows)
    output = torch.empty(output_shape, device=a[0].device, dtype=torch.bfloat16)

    if packed.layout == "contiguous":
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a,
            b,
            output,
            packed.grouped_layout,
            disable_ue8m0_cast=True,
        )
    else:
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a,
            b,
            output,
            packed.grouped_layout,
            packed.expected_m,
            disable_ue8m0_cast=True,
        )
    return output


def run_deepgemm_moe(
    packed: PackedAssignments,
    gate_up: tuple[torch.Tensor, torch.Tensor],
    down: tuple[torch.Tensor, torch.Tensor],
    expert_size: int,
    tokens: int,
    hidden_size: int,
) -> MoeResult:
    x_fp8 = quantize_grouped_activations(packed.activations)
    gate_up_out = grouped_gemm(x_fp8, gate_up, packed, 2 * expert_size)

    gate, up = gate_up_out.split(expert_size, dim=-1)
    activated = F.silu(gate.float()) * up.float()

    # Masked output rows beyond masked_m are unspecified.  They must not
    # influence scale calculation for the second GEMM.
    valid_mask = packed.token_for_row >= 0
    activated = torch.where(valid_mask.unsqueeze(-1), activated, torch.zeros_like(activated))
    activated_fp8 = quantize_grouped_activations(activated)
    down_out = grouped_gemm(activated_fp8, down, packed, hidden_size)

    flat_valid = valid_mask.reshape(-1)
    flat_tokens = packed.token_for_row.reshape(-1)[flat_valid]
    flat_weights = packed.weight_for_row.reshape(-1)[flat_valid]
    flat_down = down_out.reshape(-1, hidden_size)[flat_valid]
    output = torch.zeros((tokens, hidden_size), device=flat_down.device, dtype=torch.float32)
    output.index_add_(0, flat_tokens, flat_down.float() * flat_weights.unsqueeze(-1))

    return MoeResult(
        output=output,
        gate_up=gate_up_out,
        activated=activated,
        activated_fp8=activated_fp8,
        down=down_out,
    )


def dequantize_rows(fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, cols = fp8.shape
    assert cols % 128 == 0
    return (
        fp8.float().reshape(rows, cols // 128, 128)
        * scales.float().reshape(rows, cols // 128, 1)
    ).reshape(rows, cols)


def dequantize_weight(fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, cols = fp8.shape
    assert rows % 128 == 0 and cols % 128 == 0
    return (
        fp8.float().reshape(rows // 128, 128, cols // 128, 128)
        * scales.float().reshape(rows // 128, 1, cols // 128, 1)
    ).reshape(rows, cols)


def cpu_quantize_dequantize_rows(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert cols % 128 == 0
    blocks = x.float().reshape(rows, cols // 128, 128)
    scales = blocks.abs().amax(dim=-1).clamp(1e-4) / 448.0
    fp8 = (blocks / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return (fp8.float() * scales.unsqueeze(-1)).reshape(rows, cols)


def make_cpu_reference(
    packed: PackedAssignments,
    result: MoeResult,
    gate_up_weights: tuple[torch.Tensor, torch.Tensor],
    down_weights: tuple[torch.Tensor, torch.Tensor],
    tokens: int,
    hidden_size: int,
    expert_size: int,
) -> dict[str, torch.Tensor]:
    valid_mask = (packed.token_for_row >= 0).reshape(-1)
    valid_rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_experts = packed.expert_for_row.reshape(-1)[valid_mask].cpu()
    valid_tokens = packed.token_for_row.reshape(-1)[valid_mask].cpu()
    valid_weights = packed.weight_for_row.reshape(-1)[valid_mask].cpu()

    x_fp8 = quantize_grouped_activations(packed.activations)
    x_values = x_fp8[0].reshape(-1, hidden_size)[valid_rows].cpu()
    x_scales = x_fp8[1].reshape(-1, hidden_size // 128)[valid_rows].cpu()
    x_dequant = dequantize_rows(x_values, x_scales)

    activated_values = result.activated_fp8[0].reshape(-1, expert_size)[valid_rows].cpu()
    activated_scales = result.activated_fp8[1].reshape(-1, expert_size // 128)[valid_rows].cpu()
    activated_dequant = dequantize_rows(activated_values, activated_scales)

    assignment_count = valid_rows.numel()
    gate_up_ref = torch.empty((assignment_count, 2 * expert_size), dtype=torch.bfloat16)
    down_ref = torch.empty((assignment_count, hidden_size), dtype=torch.bfloat16)
    down_e2e_ref = torch.empty_like(down_ref)

    active_experts = torch.unique(valid_experts).tolist()
    for expert in active_experts:
        assignment_indices = torch.nonzero(valid_experts == expert, as_tuple=False).flatten()

        gate_up_weight = dequantize_weight(
            gate_up_weights[0][expert].cpu(),
            gate_up_weights[1][expert].cpu(),
        )
        gate_up_ref[assignment_indices] = (
            x_dequant[assignment_indices].float() @ gate_up_weight.float().t()
        ).to(torch.bfloat16)

        down_weight = dequantize_weight(
            down_weights[0][expert].cpu(),
            down_weights[1][expert].cpu(),
        )
        down_ref[assignment_indices] = (
            activated_dequant[assignment_indices].float() @ down_weight.float().t()
        ).to(torch.bfloat16)

        gate, up = gate_up_ref[assignment_indices].split(expert_size, dim=-1)
        activated_ref = F.silu(gate.float()) * up.float()
        activated_ref = cpu_quantize_dequantize_rows(activated_ref)
        down_e2e_ref[assignment_indices] = (
            activated_ref.float() @ down_weight.float().t()
        ).to(torch.bfloat16)

    output_ref = torch.zeros((tokens, hidden_size), dtype=torch.float32)
    output_ref.index_add_(0, valid_tokens, down_ref.float() * valid_weights.unsqueeze(-1))

    output_e2e_ref = torch.zeros_like(output_ref)
    output_e2e_ref.index_add_(0, valid_tokens, down_e2e_ref.float() * valid_weights.unsqueeze(-1))

    return {
        "gate_up": gate_up_ref,
        "down": down_ref,
        "output": output_ref,
        "output_e2e": output_e2e_ref,
    }


def test_case(
    name: str,
    layout: str,
    tokens: int,
    x: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weights: tuple[torch.Tensor, torch.Tensor],
    down_weights: tuple[torch.Tensor, torch.Tensor],
    experts: int,
    hidden_size: int,
    expert_size: int,
    kernel_threshold: float,
    end_to_end_threshold: float,
) -> None:
    pack = pack_contiguous if layout == "contiguous" else pack_masked
    packed = pack(x, ids, routing_weights, experts)

    started = time.monotonic()
    result = run_deepgemm_moe(
        packed,
        gate_up_weights,
        down_weights,
        expert_size,
        tokens,
        hidden_size,
    )
    torch.cuda.synchronize()
    gpu_seconds = time.monotonic() - started

    reference = make_cpu_reference(
        packed,
        result,
        gate_up_weights,
        down_weights,
        tokens,
        hidden_size,
        expert_size,
    )

    valid_mask = (packed.token_for_row >= 0).reshape(-1)
    gpu_gate_up = result.gate_up.reshape(-1, 2 * expert_size)[valid_mask].cpu()
    gpu_down = result.down.reshape(-1, hidden_size)[valid_mask].cpu()
    gpu_output = result.output.cpu()

    diffs = {
        "gate_up": symmetric_relative_diff(gpu_gate_up, reference["gate_up"]),
        "down": symmetric_relative_diff(gpu_down, reference["down"]),
        "weighted_output": symmetric_relative_diff(gpu_output, reference["output"]),
        "end_to_end": symmetric_relative_diff(gpu_output, reference["output_e2e"]),
    }

    print(
        f"{name}: layout={layout}, tokens={tokens}, assignments={tokens * ids.shape[1]}, "
        f"physical_rows={packed.activations.numel() // hidden_size}, first_run={gpu_seconds:.3f}s"
    )
    for metric, value in diffs.items():
        print(f"  {metric:16s} symmetric_relative_diff={value:.6e}")

    for metric in ("gate_up", "down", "weighted_output"):
        assert diffs[metric] < kernel_threshold, (
            f"{name} {metric} diff {diffs[metric]:.6e} exceeds {kernel_threshold:.6e}"
        )
    assert diffs["end_to_end"] < end_to_end_threshold, (
        f"{name} end-to-end diff {diffs['end_to_end']:.6e} exceeds {end_to_end_threshold:.6e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("both", "contiguous", "masked"), default="both")
    parser.add_argument("--prefill-tokens", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--expert-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kernel-threshold", type=float, default=1e-3)
    parser.add_argument("--end-to-end-threshold", type=float, default=5e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    assert torch.cuda.get_device_capability()[0] == 9, "this test currently targets SM90/Hopper"
    assert args.top_k <= args.experts
    assert args.hidden_size % 128 == 0
    assert args.expert_size % 128 == 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda", 0)

    print(f"DeepGEMM {getattr(deep_gemm, '__version__', 'unknown')}")
    print(f"GPU: {torch.cuda.get_device_name(device)} (visible device 0)")
    print(
        f"shape: experts={args.experts}, top_k={args.top_k}, "
        f"hidden={args.hidden_size}, expert={args.expert_size}"
    )

    gate_up_bf16 = torch.randn(
        (args.experts, 2 * args.expert_size, args.hidden_size),
        device=device,
        dtype=torch.bfloat16,
    ).mul_(1 / math.sqrt(args.hidden_size))
    down_bf16 = torch.randn(
        (args.experts, args.hidden_size, args.expert_size),
        device=device,
        dtype=torch.bfloat16,
    ).mul_(1 / math.sqrt(args.expert_size))
    gate_up_weights = quantize_grouped_weights(gate_up_bf16)
    down_weights = quantize_grouped_weights(down_bf16)
    del gate_up_bf16, down_bf16

    cases: list[tuple[str, str, int]] = []
    if args.layout in ("both", "masked"):
        cases.append(("decode", "masked", args.decode_tokens))
    if args.layout in ("both", "contiguous"):
        cases.append(("prefill", "contiguous", args.prefill_tokens))

    for case_index, (name, layout, tokens) in enumerate(cases):
        case_seed = args.seed + 100 + case_index
        x = torch.randn((tokens, args.hidden_size), device=device, dtype=torch.bfloat16)
        ids, routing_weights = make_routing(tokens, args.experts, args.top_k, case_seed)
        test_case(
            name,
            layout,
            tokens,
            x,
            ids,
            routing_weights,
            gate_up_weights,
            down_weights,
            args.experts,
            args.hidden_size,
            args.expert_size,
            args.kernel_threshold,
            args.end_to_end_threshold,
        )

    print("PASS: DeepGEMM MoE matches the CPU FP32 reference")


if __name__ == "__main__":
    main()
