#!/usr/bin/env python3
"""Convert the CosyVoice3 flow model (flow.pt) to GGUF.

Second converter of the CosyVoice3 bring-up (stage 4). The flow model is
`CausalMaskedDiffWithDiT`:

    speech tokens --(embed 6561x80)--> PreLookahead(2 convs + residual)
        --> repeat_interleave x2 --> CausalConditionalCFM (10-step Euler,
        cosine schedule, CFG rate 0.7) wrapping a 22-layer DiT estimator
        --> mel [80, T]

All weights are plain Linear/Conv1d/LayerNorm/Embedding (no weight_norm).
Tensors are written under `flow/<slash-separated key>` -- the same naming
convention the HiFT converter uses -- with each block's to_q/to_k/to_v fused
into one to_qkv tensor, and the architecture hparams recorded as metadata so
the C++ graph cannot silently disagree with the weights.  In q8_0/q4_0 mode
only the 2-D matmul weights quantize; conv kernels, norms, biases, the token
embedding and the baked rand_noise stay float (the C++ reads rand_noise raw
as f32).  bf16 mode stores those same 2-D matmul weights as bf16 (the fastest
CPU tier on AVX512-BF16 hardware) and everything else like f16 mode, so the
conv path keeps its kernel-typed f16 im2col.  f16 mode keeps 1-D tensors
(biases, norms) f32: the engine adds biases to f32 activations and the CPU
backend has no F32+F16 add.

    python3 convert-cosyvoice3-flow-to-gguf.py --flow flow.pt \\
        --config cosyvoice3.yaml --outfile cosyvoice3-flow-f32.gguf --dtype f32

Requires: pip install gguf numpy torch
"""
import argparse

import numpy as np


def load_state_dict(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "generator"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k: v for k, v in obj.items() if hasattr(v, "detach")}


QUANT_MIN_ELEMENTS = 1024
QUANT_BLOCK = 32


def to_numpy(t, dtype):
    import torch
    a = t.detach().to(torch.float32).cpu().numpy()
    if dtype == "f16" and a.ndim >= 2 and a.size >= 32:
        a = a.astype(np.float16)
    else:
        a = a.astype(np.float32)
    return np.ascontiguousarray(a)


def should_quant(a):
    """Only 2-D matmul weights quantize: ggml mul_mat handles the block-quant
    types natively, while conv kernels, norms, biases and the token embedding
    must stay float."""
    return a.ndim == 2 and a.shape[-1] % QUANT_BLOCK == 0 and a.size >= QUANT_MIN_ELEMENTS


def add_weight(w, name, t, dtype):
    import gguf
    if dtype in ("q8_0", "q4_0"):
        qt = gguf.GGMLQuantizationType.Q8_0 if dtype == "q8_0" else gguf.GGMLQuantizationType.Q4_0
        a = to_numpy(t, "f32")
        if should_quant(a):
            q = gguf.quants.quantize(a, qt)
            w.add_tensor(name, q, raw_dtype=qt)
        else:
            w.add_tensor(name, a)
        return
    if dtype == "bf16":
        a = to_numpy(t, "f32")
        if should_quant(a):
            qt = gguf.GGMLQuantizationType.BF16
            w.add_tensor(name, gguf.quants.quantize(a, qt), raw_dtype=qt)
        else:
            w.add_tensor(name, to_numpy(t, "f16"))
        return
    w.add_tensor(name, to_numpy(t, dtype))


def fuse_attention_qkv(sd, depth):
    """Concatenate each DiT block's to_q/to_k/to_v into one to_qkv tensor so
    the C++ graph runs one wide matmul per block instead of three.  Row
    concatenation preserves every output row bit-exactly."""
    import torch
    for i in range(depth):
        base = f"decoder.estimator.transformer_blocks.{i}.attn"
        weights = [sd.pop(f"{base}.to_{x}.weight") for x in ("q", "k", "v")]
        biases = [sd.pop(f"{base}.to_{x}.bias") for x in ("q", "k", "v")]
        sd[f"{base}.to_qkv.weight"] = torch.cat(weights, dim=0)
        sd[f"{base}.to_qkv.bias"] = torch.cat(biases, dim=0)


def infer_hparams(sd):
    """Derive architecture hparams from tensor shapes / key counts, so the
    metadata is a fact about the weights rather than a guessed config."""
    import re
    depth = 1 + max(
        int(m.group(1))
        for k in sd
        if (m := re.search(r"transformer_blocks\.(\d+)\.", k))
    )
    dim = sd["decoder.estimator.input_embed.proj.bias"].shape[0]          # 1024
    proj_in = sd["decoder.estimator.input_embed.proj.weight"].shape[1]    # 320 = 80*3 + spk80
    ff_inner = sd["decoder.estimator.transformer_blocks.0.ff.ff.0.0.bias"].shape[0]  # 2048
    inv_freq = sd["decoder.estimator.rotary_embed.inv_freq"].shape[0]     # 32 -> dim_head 64
    vocab, in_size = sd["input_embedding.weight"].shape                   # 6561, 80
    spk_in = sd["spk_embed_affine_layer.weight"].shape[1]                 # 192
    mel_dim = sd["decoder.estimator.proj_out.weight"].shape[0]            # 80
    return {
        "depth": depth,
        "dim": dim,
        "dim_head": inv_freq * 2,
        "heads": dim // (inv_freq * 2),
        "ff_inner": ff_inner,
        "proj_in": proj_in,
        "vocab_size": vocab,
        "in_size": in_size,
        "spk_embed_dim": spk_in,
        "mel_dim": mel_dim,
        "token_mel_ratio": 2,
        "pre_lookahead_len": 3,
        "n_timesteps": 10,
        "conv_pos_kernel": sd["decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight"].shape[2],
        "conv_pos_groups": 16,
    }


def main():
    import gguf
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, help="path to flow.pt")
    ap.add_argument("--config", default=None, help="cosyvoice3.yaml (embedded as metadata)")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16", "bf16", "q8_0", "q4_0"], default="f32")
    args = ap.parse_args()

    sd = load_state_dict(args.flow)
    hp = infer_hparams(sd)
    print(f"hparams (inferred from weights): {hp}")
    fuse_attention_qkv(sd, hp["depth"])

    w = gguf.GGUFWriter(args.outfile, "cosyvoice3-flow")
    w.add_float32("cosyvoice3.flow.inference_cfg_rate", 0.7)
    w.add_float32("cosyvoice3.flow.sigma_min", 1e-06)
    w.add_string("cosyvoice3.flow.t_scheduler", "cosine")
    w.add_string("cosyvoice3.flow.solver", "euler")
    for k, v in hp.items():
        w.add_uint32(f"cosyvoice3.flow.{k}", int(v))
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as fh:
                w.add_string("cosyvoice3.flow.config_yaml", fh.read())
        except OSError:
            pass

    # flow/<slash-separated key>, matching the HiFT converter (hift/<...>), but
    # abbreviated so names stay under ggml's GGML_MAX_NAME (64) limit -- the full
    # "decoder.estimator.transformer_blocks.21..." path overflows it and the C++
    # gguf reader then fails ("failed to read tensor info"). The C++ side
    # (cosyvoice_flow.cpp) uses the same short names.
    def gguf_name(k):
        k = k.replace("decoder.estimator.", "")     # DiT lives at flow/<...>
        k = k.replace("transformer_blocks.", "blk.")  # flow/blk/<i>/<...>
        return "flow/" + k.replace(".", "/")

    n = 0
    for name, tensor in sorted(sd.items()):
        add_weight(w, gguf_name(name), tensor, args.dtype)
        n += 1

    # CausalConditionalCFM.__init__ does set_all_random_seed(0); rand_noise =
    # torch.randn([1,80,50*300]). This fixed noise is the Euler start point (not a
    # state_dict param), so bake it in for a self-contained C++ flow.
    import torch
    torch.manual_seed(0)
    rand_noise = torch.randn([1, 80, 50 * 300])[0]   # [80, 15000]
    w.add_tensor("flow/rand_noise", to_numpy(rand_noise, "f32"))
    n += 1
    print(f"writing {n} tensors -> {args.outfile} ({args.dtype})")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("done.")


if __name__ == "__main__":
    main()
