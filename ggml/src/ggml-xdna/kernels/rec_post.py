# The transition between a layer's ssm_out projection and its FFN, on the
# array instead of the host: add the residual, normalise, scale by the layer's
# gamma and quantize into the activation tiles the next projection reads.
#
# It exists to remove the host from the middle of a layer. As long as the host
# computes this, the projection before it and the projection after it cannot
# be in one dispatch, and the array's fixed per-phase cost is paid twice.
#
# One object in, one object out, so the tile needs a single channel each way:
#   in   [so_out D f32][residual D f32][gamma D f32][flags i32][pad]
#   out  [hattn D f32][header tile][NT tiles]
# where a tile is ACT_TILE bytes of K_TILE int8 codes, a f32 code sum and a
# f32 scale per group of 32, and the two trailing flag words.

from pathlib import Path

D        = 1024
ACT_TILE = 2112
GROUP    = 32

POST_SRC = Path(__file__).resolve().parent / "rec-post.cc"


def _kernel_src(d: int = D, k_tile: int = 256, act_tile: int = ACT_TILE) -> str:
    src = POST_SRC.read_text()
    for k, v in (("PD", d), ("PNT", d // k_tile), ("PKT", k_tile),
                 ("PGPT", k_tile // GROUP), ("PGRP", GROUP), ("PACT", act_tile)):
        src = src.replace(f"@{k}@", str(v))
    return src
