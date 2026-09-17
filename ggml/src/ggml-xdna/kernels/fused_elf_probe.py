import sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fused_layer

spec = fused_layer.fused_layer.specialize(full_elf=True, dev_name="npu2")
elf_path, _ = spec.compile(elf_path="/tmp/fused_full.elf")
shutil.copy(elf_path, "/tmp/fused_full.elf")
cname = spec.compilable._full_elf_kernel_name
print("kernel:", cname, flush=True)

from aie.utils.npukernel import NPUKernel
from aie.utils import DefaultNPURuntime
kern = NPUKernel(elf_path="/tmp/fused_full.elf", kernel_name=cname,
                 xclbin_path=None, insts_path=None)
handle = DefaultNPURuntime.load(kern)
print("loaded", flush=True)
# how many args does the full-ELF kernel declare?
k = handle.kernel if hasattr(handle, "kernel") else None
if k is not None:
    try:
        print("declared args:", k.get_num_args() if hasattr(k, "get_num_args") else "n/a", flush=True)
    except Exception as e:
        print("args err:", e, flush=True)

T = DefaultNPURuntime._tensor_class
def mk(n, dt=np.float32):
    return T(np.zeros(n, dtype=dt).view(np.uint32))

# core buffers + the GEMV's (compile-time K1024 N7168)
feed  = mk(2 * 24 * 4096)
x     = mk(16 * 1024)
pkvb  = mk(16 * 132 * 4)
state = mk(8 * 8 * 128 * 2, np.uint16)
azg   = mk(16 * 385 * 4)
out   = mk(10496 + (1 + 2048 // 128) * 2112, np.uint8)
w     = mk((1024 * 7168 * 9) // 16, np.uint8)
# the activation: a header tile then 7 chunks of 8 tiles (compile-time shape)
actn  = (1 + 7 * 8) * 2112
abuf  = np.zeros(actn, dtype=np.uint32)
hdr   = abuf[:2112//4]
hdr[0] = 8          # n tiles
hdr[1] = 7          # n out
hdr[2112//4 - 1] = 0
hdr[2112//4 - 2] = 0
for t in range(1, 1 + 7 * 8):
    tl = abuf[t*2112//4:(t+1)*2112//4]
    tl[2112//4 - 1] = 0
    tl[2112//4 - 2] = (t == 7 * 8) and 1 or 0
a     = T(abuf)
o     = mk(7168)

r = DefaultNPURuntime.run(handle, [feed, x, pkvb, state, azg, out, w, a, o])
print("run1:", r, flush=True)
r = DefaultNPURuntime.run(handle, [feed, x, pkvb, state, azg, out, w, a, o])
print("run2:", r, flush=True)
