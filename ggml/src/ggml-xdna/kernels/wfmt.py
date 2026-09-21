"""Packed NPU weight formats shared by the GEMM design and its harness.

The format itself is defined in xdna-quant.h (host side) and expanded by
dequant_b.cc (device side); this module is the single place the Python side
states the geometry, so the design and the standalone harness cannot drift.

One B tile is (K_TILE x N_TILE) and is stored sub-tile major in the mmul's own
order - (K_TILE/S) x (N_TILE/T) sub-tiles of S*T values, (si, ti) row-major -
followed by the group-parameter planes: per group, one bf16 row of rounded
values and one of residuals, which sum to the parameter. That makes the
expansion a linear read and a linear write, and leaves the layout decision with
the host packer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import ml_dtypes
import numpy as np

from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import _include_dirs

# Sub-tile shape of the mmul micro-kernel, i.e. (s, t) of its mac_dims. It is
# a property of the compiled kernel and the target, so callers pass the value
# they read off the kernel rather than trusting a constant here; these are the
# defaults for bf16 on AIE2P.
MAC_S = 8
MAC_T = 8

Q4_GROUP = 32
Q8_GROUP = 16

FORMATS = ("q4g32", "q8g16")


def tile_bytes(fmt: str, k: int, n: int) -> int:
    """Packed size of one (k x n) B tile."""
    if fmt == "q4g32":
        return k * n // 2 + 2 * (k // Q4_GROUP) * n * 4
    if fmt == "q8g16":
        return k * n + 2 * (k // Q8_GROUP) * n * 4
    raise ValueError(f"unknown weight format {fmt!r}")


def group_size(fmt: str) -> int:
    return Q4_GROUP if fmt == "q4g32" else Q8_GROUP


def dequant_fn(fmt: str, k: int, n: int, s: int = MAC_S, t: int = MAC_T) -> ExternalFunction:
    """The core function that expands one packed tile into a bf16 (k, n) tile."""
    src = (Path(__file__).resolve().parent / "gemm-dequant.cc").read_text()
    flags = [
        f"-DK_TILE={k}", f"-DN_TILE={n}",
        f"-DMAC_S={s}", f"-DMAC_T={t}", f"-DQ4_GROUP={Q4_GROUP}",
    ]
    digest = hashlib.sha256((src + "\0".join(flags)).encode()).hexdigest()[:8]
    in_ty = np.ndarray[(tile_bytes(fmt, k, n),), np.dtype[np.uint8]]
    # Flat, matching the mmul micro-kernel's B operand type.
    out_ty = np.ndarray[(k * n,), np.dtype[ml_dtypes.bfloat16]]
    name = "dequant_q4g32_bf16" if fmt == "q4g32" else "dequant_q8g16_bf16"
    return ExternalFunction(
        name,
        object_file_name=f"dequant_b_{fmt}_{k}x{n}_s{s}t{t}_{digest}.o",
        source_string=src,
        arg_types=[in_ty, out_ty],
        include_dirs=_include_dirs(),
        compile_flags=flags,
    )


def subtile_order(k: int, n: int, s: int = MAC_S, t: int = MAC_T) -> np.ndarray:
    """Index map from (k, n) row-major to the position in the mmul sub-tile
    order, (kb, nb, si, ti)."""
    idx = np.arange(k * n).reshape(k, n)
    return (idx.reshape(k // s, s, n // t, t)
               .transpose(0, 2, 1, 3)
               .reshape(-1))


def code_order(k: int, n: int, s: int = MAC_S, t: int = MAC_T) -> np.ndarray:
    """Index map from (k, n) row-major to the order the codes are stored in,
    (nb, kb, si, ti) - column block first, so the expansion holds the group
    parameters constant across a run."""
    idx = np.arange(k * n).reshape(k, n)
    return (idx.reshape(k // s, s, n // t, t)
               .transpose(2, 0, 1, 3)
               .reshape(-1))


def split_pairs(x: np.ndarray) -> np.ndarray:
    """[ng, n] f32 -> [2*ng, n] bf16: per group a rounded row then a residual
    row, which sum back to the parameter."""
    hi = x.astype(ml_dtypes.bfloat16)
    lo = (x - hi.astype(np.float32)).astype(ml_dtypes.bfloat16)
    out = np.empty((x.shape[0] * 2, x.shape[1]), dtype=ml_dtypes.bfloat16)
    out[0::2] = hi
    out[1::2] = lo
    return out


def join_pairs(p: np.ndarray) -> np.ndarray:
    """Inverse of split_pairs, as f32."""
    return p[0::2].astype(np.float32) + p[1::2].astype(np.float32)


def pack_tile(fmt: str, codes: np.ndarray, d: np.ndarray, m: np.ndarray | None,
              s: int = MAC_S, t: int = MAC_T) -> np.ndarray:
    """codes [k, n] ints, d/m [k//group, n] f32 -> the packed tile bytes."""
    k, n = codes.shape
    flat = codes.reshape(-1)[code_order(k, n, s, t)]
    if fmt == "q4g32":
        body = ((flat[0::2].astype(np.uint8) & 0xF)
                | ((flat[1::2].astype(np.uint8) & 0xF) << 4))
        params = np.concatenate([split_pairs(d).reshape(-1),
                                 split_pairs(m).reshape(-1)])
    else:
        body = flat.astype(np.int8).view(np.uint8)
        params = np.concatenate([split_pairs(d).reshape(-1),
                                 split_pairs(m).reshape(-1)])
    return np.concatenate([body, params.view(np.uint8)])


def unpack_reference(fmt: str, codes: np.ndarray, d: np.ndarray, m: np.ndarray | None,
                     s: int = MAC_S, t: int = MAC_T) -> np.ndarray:
    """The bf16 tile the expansion must produce, in sub-tile order."""
    k, n = codes.shape
    g = np.arange(k) // group_size(fmt)
    # Mirror the device: the parameters are the bf16 pairs, not the f32 input.
    de = join_pairs(split_pairs(d))
    w = codes.astype(np.float32) * de[g]
    if m is not None:
        w = w + join_pairs(split_pairs(m))[g]
    return w.reshape(-1)[subtile_order(k, n, s, t)]
