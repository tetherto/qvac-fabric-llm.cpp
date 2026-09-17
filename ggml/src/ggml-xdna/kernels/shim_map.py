#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Read the shim DMA endpoints out of a built instruction stream.
#
# A design's ObjectFifos are placed by the compiler, and which column and
# channel each one ends up on is not something the source says - in the merged
# layer (fused_layer.py) pinning them is not even possible, because the other
# half of the design has already taken the obvious choices. The backend builds
# its own per-token streams and has to push each transfer to the same endpoint
# the artifact expects, so the mapping has to come from the artifact itself.
#
# The stream IRON compiles alongside the xclbin contains exactly that: every
# push-queue write names a column, a direction, a channel and a descriptor, in
# the order the design's runtime sequence issues them. This prints them in that
# order, which is the order the backend's builder walks its own transfers in.
#
#   python3 shim_map.py fused_layer.insts.bin [--tail N]

from __future__ import annotations

import argparse
import subprocess
import sys

# Shim DMA push-queue registers: S2MM ch0 is the base, +8 selects channel 1 and
# +0x10 selects MM2S (xdna-seq.h has the same constants).
PUSHQ_BASE = 0x1D204
PUSHQ_LAST = 0x1D21C
COL_SHIFT = 25
REG_MASK = 0xFFFFF


def endpoints(insts: str) -> list[tuple[int, str, int, int]]:
    """(column, direction, channel, descriptor id) in issue order."""
    import re

    out = []
    pat = re.compile(r"\s*XAIE_IO_WRITE\s+@0x([0-9a-f]+),\s*(0x[0-9a-f]+)")
    for line in insts.splitlines():
        m = pat.match(line)
        if not m:
            continue
        addr, val = int(m.group(1), 16), int(m.group(2), 16)
        reg = addr & REG_MASK
        if not PUSHQ_BASE <= reg <= PUSHQ_LAST:
            continue
        off = reg - PUSHQ_BASE
        out.append(((addr >> COL_SHIFT) & 0x7F,
                    "MM2S" if off & 0x10 else "S2MM",
                    1 if off & 8 else 0,
                    val & 0xF))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="shim_map")
    ap.add_argument("insts", help="the .insts.bin the design was built with")
    ap.add_argument("--tail", type=int, default=0,
                    help="print only the last N endpoints (the last phase of "
                         "a merged design)")
    opts = ap.parse_args()

    try:
        text = subprocess.run(["aiebu-dump", "-d", opts.insts],
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"shim_map: cannot disassemble {opts.insts}: {e}")

    eps = endpoints(text)
    if opts.tail:
        eps = eps[-opts.tail:]
    print(f"# {len(eps)} shim transfers, in the order the design issues them")
    print("# index  column  direction  channel  descriptor")
    for i, (col, d, ch, bd) in enumerate(eps):
        print(f"{i:6d}  {col:6d}  {d:>9s}  {ch:7d}  {bd:10d}")


if __name__ == "__main__":
    main()
