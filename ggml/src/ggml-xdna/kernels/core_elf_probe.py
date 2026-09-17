import sys, os, shutil
sys.path.insert(0, "/home/npu-bench/qvac-clean/ggml/src/ggml-xdna/kernels")
import numpy as np
import attn_gdn_gated

spec = attn_gdn_gated.attn_gdn_gated.specialize(full_elf=True, dev_name="npu2")
elf_path, _ = spec.compile(elf_path="/tmp/core_full.elf")
shutil.copy(elf_path, "/tmp/core_full.elf")
cname = spec.compilable._full_elf_kernel_name
print("kernel:", cname, flush=True)

from aie.utils.npukernel import NPUKernel
from aie.utils import DefaultNPURuntime
kern = NPUKernel(elf_path="/tmp/core_full.elf", kernel_name=cname,
                 xclbin_path=None, insts_path=None)
handle = DefaultNPURuntime.load(kern)
print("loaded", flush=True)
T = DefaultNPURuntime._tensor_class
def mk(n, dt=np.float32):
    return T(np.zeros(n, dtype=dt).view(np.uint32))
feed  = mk(49152)
x     = mk(6192)
pkvb  = mk(49536)
state = mk(262144, np.uint16)
azg   = mk(6400)
out   = mk(46400, np.uint8)
r = DefaultNPURuntime.run(handle, [feed, x, pkvb, state, azg, out])
print("run1:", r, flush=True)
