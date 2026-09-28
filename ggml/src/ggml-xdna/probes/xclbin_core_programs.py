#!/usr/bin/env python3
# Group an xclbin's AIE cores by identical program image, read out of the CDO in
# its AIE_PARTITION PDI.  No XRT or xclbinutil needed, so it runs on the Mac.
#
#   python3 xclbin_core_programs.py a.xclbin [b.xclbin ...]
#
# A design whose cores mostly share one program is a homogeneous worker pool; a
# design where every core has its own program is a pipeline of roles.  That is
# the difference between FastFlowLM's Qwen3.5-0.8B layer (16 of 29 cores on one
# GEMV program, cols 0/1/6/7) and ours (32 cores, 24 programs, largest group 4).
#
# CDO framing: a command header is (len << 16) | cmd; len 0xff means the real
# length is in the next word.  0x105 is a block write: addr_hi, addr_lo, payload.
# Program memory is 0x20000-0x23fff of a core tile (AIE2: col << 25 | row << 20).
import struct
import hashlib
import collections
import sys


def load(fn):
    b=open(fn, "rb").read()
    nsec=struct.unpack_from("<I", b, 448)[0]
    sh=456
    for i in range(nsec):
        kind, =struct.unpack_from("<I", b, sh)
        soff, ssz=struct.unpack_from("<QQ", b, sh +24)
        if kind==32:
            s=b[soff:soff +ssz]
            break
        sh+=40
    colw, =struct.unpack_from("<H", s, 32)
    npdi, pdio=struct.unpack_from("<II", s, 120)
    imgn, imgo=struct.unpack_from("<II", s, pdio +16)
    return colw, s[imgo:imgo +imgn]


def programs(img):
    W=struct.unpack_from("<%dI" %(len(img) //4), img)
    chunks=collections.defaultdict(dict)
    for i in range(len(W) -4):
        h=W[i]
        if h &0xffff!=0x105: continue
        L=h >>16
        a=i +1
        if L==0xff:
            L=W[i +1]
            a=i +2
        if L<3 or L>100000 or W[a]!=0: continue
        lo=W[a +1]
        col=(lo >>25) &0x7f
        row=(lo >>20) &0x1f
        off=lo &0xfffff
        if col<8 and 2<=row<=5 and 0x20000<=off<0x24000:
            chunks[(col, row)][off]=bytes(img[4 *(a +2):4 *(a +L)])
    return {t: b"".join(v[o] for o in sorted(v)) for t, v in chunks.items()}


for fn in sys.argv[1:]:
    colw, img=load(fn)
    progs=programs(img)
    groups=collections.defaultdict(list)
    for t, pg in progs.items(): groups[hashlib.sha1(pg).hexdigest()[:10]].append((t, len(pg)))
    tot=sum(len(p) for p in progs.values())
    print(f"== {fn.split('/')[-1]}: cols={colw} cores={len(progs)} distinct_programs={len(groups)} total_prog={tot} B")
    for h, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"   x{len(ts):2d} {ts[0][1]:6d} B  {sorted(t for t,_ in ts)}")
