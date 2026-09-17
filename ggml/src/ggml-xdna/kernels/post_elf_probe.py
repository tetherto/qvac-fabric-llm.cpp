import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import post_norm

spec = post_norm.post_norm.specialize(full_elf=True, dev_name="npu2")
elf_path, _ = spec.compile(elf_path="/tmp/post_norm_full.elf")
cname = spec.compilable._full_elf_kernel_name
print("kernel name:", cname, flush=True)
import shutil
shutil.copy(elf_path, "/tmp/post_norm_full.elf")
print("copied", elf_path, flush=True)

from aie.utils.npukernel import NPUKernel
from aie.utils import DefaultNPURuntime
kern = NPUKernel(elf_path="/tmp/post_norm_full.elf",
                 kernel_name=cname,
                 xclbin_path=None, insts_path=None)
handle = DefaultNPURuntime.load(kern)
print("loaded", flush=True)

def run_once(tag):
    inb = np.zeros((3*1024+4,), dtype=np.float32)
    rng = np.random.default_rng(7)
    inb[0:1024] = rng.standard_normal(1024).astype(np.float32)
    inb[1024:2048] = rng.standard_normal(1024).astype(np.float32)
    inb[2048:3072] = 0.5 + rng.standard_normal(1024).astype(np.float32)
    inb[3072] = np.float32(5.0).view(np.int32)
    outb = np.zeros(1024*4 + (1+1024//256)*2112, dtype=np.uint8)
    T = DefaultNPURuntime._tensor_class
    r = DefaultNPURuntime.run(handle, [T(inb.view(np.uint32)), T(outb.view(np.uint32))])
    print(tag, "state:", r, flush=True)

run_once("run1")
run_once("run2")
run_once("run3")
