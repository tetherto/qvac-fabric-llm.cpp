#!/usr/bin/env python3
"""Widen a Qwen3.5 GGUF's gated-delta-net layers to twice the value heads.

The 4B and 9B have 32 value heads on 16 q/k groups; the 0.8B and 2B have 16
on 16. This writes a model of the smaller width with the bigger head count,
so the 32-head path of the backend can be tested where everything else of
the model already runs on the NPU (tests/test-xdna-gdn.cpp).

Value heads h and 16 + h both read q/k group h (ggml's GATED_DELTA_NET:
head % groups). Head h takes the v, z, alpha, beta, dt, A and conv weights of
old head (h + --shift-lo) % 16 and head 16 + h those of (h + --shift-hi) % 16;
the defaults (0, 5) keep the old heads and make no new head a copy of another,
so a swapped head shows. ssm_out reads the two halves at --out-lo and
--out-hi (0.5 each keeps the layer's output at the scale the rest of the
model expects); a zero half takes those heads out of the output, which tests
one half on its own.

    python gdn_widen_model.py Qwen_Qwen3.5-2B-bf16.gguf out-gdn32.gguf

then quantize the result as the other test models are (Q4_K_M,
--tensor-type ssm_out=q5_K).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "gguf-py"))
import gguf  # noqa: E402

ARCH = "qwen35"
SV = 128     # value head dim


def to_f32(t: gguf.ReaderTensor) -> np.ndarray:
    """A tensor's values as float32, numpy shape (ne1, ne0)."""
    shape = tuple(reversed([int(d) for d in t.shape]))
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.array(t.data, dtype=np.float32).reshape(shape)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        u = np.array(t.data).view(np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    raise SystemExit(f"{t.name}: {t.tensor_type.name} is not F32 or BF16")


def head_rows(n_heads: int, shift_lo: int, shift_hi: int) -> np.ndarray:
    """For each new head, the old head it takes its weights from."""
    return np.array([(h + shift_lo) % n_heads for h in range(n_heads)]
                    + [(h + shift_hi) % n_heads for h in range(n_heads)])


def widen(name: str, w: np.ndarray, n_vh: int, n_qk: int, args: argparse.Namespace) -> np.ndarray:
    """One GDN tensor at twice the value heads, numpy shape (ne1, ne0)."""
    src = head_rows(n_vh, args.shift_lo, args.shift_hi)
    if name.endswith("attn_qkv.weight"):
        qk = w[: 2 * n_qk * SV]
        v = w[2 * n_qk * SV:].reshape(n_vh, SV, -1)
        return np.concatenate([qk, v[src].reshape(-1, w.shape[1])])
    if name.endswith("ssm_conv1d.weight"):
        qk = w[: 2 * n_qk * SV]
        v = w[2 * n_qk * SV:].reshape(n_vh, SV, -1)
        return np.concatenate([qk, v[src].reshape(-1, w.shape[1])])
    if name.endswith("attn_gate.weight"):
        return w.reshape(n_vh, SV, -1)[src].reshape(-1, w.shape[1])
    if name.endswith(("ssm_alpha.weight", "ssm_beta.weight")):
        return w[src]
    if name.endswith(("ssm_dt.bias", "ssm_a")):
        return w.reshape(-1)[src]
    if name.endswith("ssm_out.weight"):
        cols = w.reshape(w.shape[0], n_vh, SV)[:, src]
        cols[:, :n_vh] *= args.out_lo
        cols[:, n_vh:] *= args.out_hi
        return cols.reshape(w.shape[0], -1)
    return w


WIDENED = ("attn_qkv.weight", "ssm_conv1d.weight", "attn_gate.weight", "ssm_alpha.weight",
           "ssm_beta.weight", "ssm_dt.bias", "ssm_a", "ssm_out.weight")


def field(reader: gguf.GGUFReader, key: str):
    f = reader.get_field(key)
    return f.contents() if f else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--shift-lo", type=int, default=0, help="head h takes old head (h + this) %% 16")
    ap.add_argument("--shift-hi", type=int, default=5, help="head 16 + h takes old head (h + this) %% 16")
    ap.add_argument("--out-lo", type=float, default=0.5, help="ssm_out weight of heads 0-15")
    ap.add_argument("--out-hi", type=float, default=0.5, help="ssm_out weight of heads 16-31")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.input)
    n_vh = int(field(reader, f"{ARCH}.ssm.time_step_rank"))
    n_qk = int(field(reader, f"{ARCH}.ssm.group_count"))
    inner = int(field(reader, f"{ARCH}.ssm.inner_size"))
    if n_vh != n_qk or inner != n_vh * SV:
        raise SystemExit(f"expected as many value heads as q/k groups, got {n_vh} and {n_qk}")
    new_kv = {f"{ARCH}.ssm.time_step_rank": 2 * n_vh, f"{ARCH}.ssm.inner_size": 2 * inner}

    writer = gguf.GGUFWriter(args.output, arch=ARCH, endianess=reader.endianess)
    for f in reader.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(f.name, new_kv.get(f.name, f.contents()), vtype, sub_type=sub)

    out = []
    for t in reader.tensors:
        if t.name.endswith(WIDENED) and t.name.startswith("blk."):
            a = np.ascontiguousarray(widen(t.name, to_f32(t), n_vh, n_qk, args), dtype=np.float32)
            writer.add_tensor_info(t.name, a.shape, a.dtype, a.nbytes)
            out.append(a)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            out.append(t.data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for a in out:
        writer.write_tensor_data(a, tensor_endianess=reader.endianess)
    writer.close()


if __name__ == "__main__":
    main()
