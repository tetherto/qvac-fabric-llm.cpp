# AMD XDNA backend

The XDNA backend offloads weight `MUL_MAT` operations to the AMD XDNA2 NPU.
All other operations use the standard GGML CPU backend. The production build
contains two BF16 GEMM artifacts and no model-specific fused kernels.

## Validated environment

The backend targets the XDNA2 (`aie2p`) NPU on Linux. Everything in this
document was built and measured on one machine:

| | Value |
| --- | --- |
| Host | ASUS ProArt PX13 HN7306EA, AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (Strix Halo), 27 GiB RAM |
| NPU | `NPU Strix Halo`, `aie2p`, 6x8 topology, BDF `0000:c5:00.1` |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic, Python 3.12 |
| XRT | 2.25.37 under `/opt/xilinx/xrt` |
| `amdxdna` driver | 2.25.260102.56.release_20260630 (DKMS) |
| NPU firmware | 1.1.2.65 |
| IRON | mlir-aie **1.3.4** + llvm-aie 21 (`~/ironenv`) |
| Models | Qwen3.5-0.8B Q4_K_M and BF16, Qwen3.5-9B Q4_K_M |

`xrt-smi examine` reports the XRT, driver, firmware and device rows. The NPU
has to appear there before `--device XDNA` can do anything.

mlir-aie **1.4.1 is not compatible** with this branch. The production GEMM
design constructs `Runtime()` with no arguments; 1.4.1 requires `seq_fn` and
fails at compile time with `TypeError: Runtime.__init__() missing 1 required
positional argument: 'seq_fn'`. Use 1.3.4.

## Dependencies

Install these **before** running CMake. Kernel compilation needs all three;
the C++ binary needs XRT at runtime even if the `.xclbin` files are copied in.

### 1. XRT and the `amdxdna` driver

XRT 2.25.x must live under `/opt/xilinx/xrt` (or `$XILINX_XRT`) and the
`amdxdna` DKMS module must match that XRT. After install:

```sh
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

The table must list an `aie2p` device (on the validated box: `NPU Strix Halo`).
`xclbinutil` has to be on `PATH` (it comes with XRT) or kernel compilation
fails. `pyxrt` is imported from `/opt/xilinx/xrt/python`, which `setup.sh`
puts on `PYTHONPATH`.

Optional, before a timed run:

```sh
xrt-smi configure --pmode performance
```

### 2. IRON toolchain (mlir-aie 1.3.4)

Wheels are not on PyPI. Install them from the Xilinx GitHub release indexes
into a dedicated venv. Do **not** point CMake at a 1.4.1 environment.

```sh
python3 -m venv "$HOME/ironenv"
"$HOME/ironenv/bin/pip" install -U pip
"$HOME/ironenv/bin/pip" install mlir-aie==1.3.4 \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.4
"$HOME/ironenv/bin/pip" install 'llvm-aie==21.0.0.2026073101+cfd8aae2' \
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/latest-wheels

source /opt/xilinx/xrt/setup.sh
"$HOME/ironenv/bin/python3" -c "import mlir_aie, aie.iron, pyxrt; print('ok')"
```

The llvm-aie pin is the version used with mlir-aie 1.3.4 on the validated
box. A later `llvm-aie` 22 wheel (the current `latest-wheels` default) belongs
with mlir-aie 1.4.1 and is not the toolchain for this branch.

CMake looks for `$HOME/ironenv/bin/python3` by default
(`-DGGML_XDNA_IRON_PYTHON`). Override that if the venv lives elsewhere.

For interactive kernel work, `source ggml/src/ggml-xdna/iron/env_setup.sh`
after activating the venv; CMake already does the equivalent through
`build-kernels.sh`.

## Build

XRT must be sourced in the CMake shell so the backend can link
`libxrt_coreutil.so`. `-DGGML_OPENMP=ON` is the default and is required for
the host pack/scatter path used at runtime.

```sh
source /opt/xilinx/xrt/setup.sh
cmake -B build -DGGML_XDNA=ON -DGGML_OPENMP=ON \
    -DGGML_XDNA_IRON_PYTHON="$HOME/ironenv/bin/python3"
cmake --build build -j --target llama-completion llama-server
```

`GGML_XDNA_BUILD_KERNELS` defaults to `ON`. The production build generates
two GEMM artifact pairs next to the binaries:

```text
build/bin/ggml-xdna-gemm-npu2-bf16-16-16.{xclbin,insts.bin}   # MB=16, M block 2048
build/bin/ggml-xdna-gemm-npu2-bf16-16-32.{xclbin,insts.bin}   # MB=32, M block 4096
```

Kernel compilation takes several minutes and needs `xclbinutil`. Set
`GGML_XDNA_BUILD_KERNELS=OFF` only when those four files are already in
`build/bin/` (or `$GGML_XDNA_KERNELS_DIR`). The runtime searches
`GGML_XDNA_KERNELS_DIR`, `GGML_BACKEND_DIR`, the executable directory, and
the current directory.

The artifacts are build products. They are not stored in git.

Without XRT headers the C++ backend still compiles, but
`ggml_backend_xdna_reg_get_device_count` returns 0 and `--list-devices`
will not show `XDNA`.

### Dynamic backend library

`ggml-xdna` uses the same `ggml_add_backend_library` path as other ggml
backends, including `GGML_BACKEND_DL_IMPL`. To build it as a loadable module:

```sh
cmake -B build-dl -DGGML_XDNA=ON -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
    -DGGML_XDNA_IRON_PYTHON="$HOME/ironenv/bin/python3"
cmake --build build-dl -j --target llama-completion ggml-xdna
```

That writes `build-dl/bin/libqvac-ggml-xdna.so` next to `llama-completion`.
Keep the GEMM artifacts in the same directory, or set `GGML_XDNA_KERNELS_DIR`.

## `--device` / `-dev`

This backend registers a ggml device whose **name is always `XDNA`**. That is
the string `--device` / `-dev` must be given. XRT's PCI name (`NPU Strix Halo`
on the validated box) is only the description; it is not accepted by
`--device`.

`--device` is parsed by `parse_device_list`: a comma-separated list of ggml
device names, looked up with `ggml_backend_dev_by_name`. The special value
`none` disables offload. The same list is accepted via `LLAMA_ARG_DEVICE`.
`XDNA` has no spaces, so it does not need quoting. `--list-devices` prints
every non-CPU device and exits.

| Flag | Effect on this backend |
| --- | --- |
| `--device XDNA` | Use the XDNA ACCEL device for offload. Required for the NPU GEMM path. |
| `-dev XDNA` | Same as `--device XDNA`. |
| `--device none` | Do not put XDNA on the llama.cpp offload device list. Weight repack can still happen: `XDNA_REPACK` is an extra ACCEL buffer type on the CPU list. For a CPU-only run also set `GGML_XDNA_NPU=0`. |
| (omit `--device`) | llama.cpp may still pick XDNA as an ACCEL device. Pass `--device XDNA` explicitly. |
| `-ngl 99` | Standard llama.cpp offload count. Use this with `--device XDNA`. |
| `-ngl 0` | Layers stay on the CPU device, but `XDNA_REPACK` extra bufts are still considered unless `GGML_XDNA_NPU=0`. |

`GGML_XDNA_NPU` is independent of `--device`:

- `GGML_XDNA_NPU=1` (default): if the NPU is present and both GEMM artifacts
  are found, `XDNA_REPACK` is offered and prefill `MUL_MAT` with `M >= 128`
  runs on the NPU.
- `GGML_XDNA_NPU=0`: the device can still appear in `--list-devices` (XRT
  probe succeeded), but extra bufts are not offered and nothing is submitted
  to the NPU.

`--list-devices` shows `XDNA` only when XRT actually found an NPU
(`npu_present`). It still shows `XDNA` when artifacts are missing or
`GGML_XDNA_NPU=0`; the description then says `(no NPU kernels)` or the
init log lists `kernels=none`. If `XDNA` is absent entirely: XRT is not
sourced, `libxrt_coreutil` was not linked, or no NPU is visible to
`xrt-smi examine`.

On a healthy production build the line looks like:

```text
  XDNA: AMD XDNA NPU Strix Halo (NPU GEMM_MB16, GEMM_MB32) [own-graph] (0 MiB, 0 MiB free)
```

The memory fields are always 0; this backend does not report device DRAM.

## E2E CLI

XRT must be sourced in the **same shell** that runs the binary. The
implementation check is `llama-completion`; `llama-server` is the same
backend with an HTTP front-end. `llama-cli` accepts the same `--device`
flag.

### 1. Confirm the device

```sh
source /opt/xilinx/xrt/setup.sh
./build/bin/llama-completion --list-devices
```

### 2. One-shot completion (implementation check)

```sh
GGML_XDNA_NPU=1 OMP_WAIT_POLICY=PASSIVE \
    ./build/bin/llama-completion \
    -m Qwen3.5-0.8B-Q4_K_M.gguf \
    --device XDNA -ngl 99 \
    -p "The capital of France is" -n 32 \
    --top-k 1 --temp 0 -s 42 \
    -t 16 --poll 0 --flash-attn off -c 2048
```

The init log must contain:

```text
ggml_backend_xdna_init: XDNA backend init: device=NPU Strix Halo, kernels=GEMM_MB16, GEMM_MB32, own_graph=on
```

`device=` is the XRT name of this machine's NPU. `kernels=` must list
`GEMM_MB16, GEMM_MB32` and must not list ADD, MUL, RMS_NORM, or GDN.

Greedy CPU reference (same binary, NPU disabled):

```sh
GGML_XDNA_NPU=0 OMP_WAIT_POLICY=PASSIVE \
    ./build/bin/llama-completion \
    -m Qwen3.5-0.8B-Q4_K_M.gguf \
    --device none -ngl 0 \
    -p "The capital of France is" -n 32 \
    --top-k 1 --temp 0 -s 42 \
    -t 16 --poll 0 --flash-attn off -c 2048
```

The first greedy answer should match. Token equality after that is not a
numerical correctness test; use `test-xdna-production micro` for NMSE against
a CPU reference.

`GGML_XDNA_DEBUG=1` reports which ops the backend accepted, per-artifact NPU
run counts and submit time, host pack/scatter time, and nested CPU totals.
Output text alone cannot tell the two paths apart.

### 3. Server (the Qwen3.5-0.8B measurement configuration)

```sh
GGML_XDNA_NPU=1 OMP_WAIT_POLICY=PASSIVE \
    ./build/bin/llama-server -m Qwen3.5-0.8B-Q4_K_M.gguf \
    --host 127.0.0.1 --port 8080 \
    -c 147456 -np 4 -b 4096 -ub 4096 --flash-attn off \
    -t 16 --poll 0 --device XDNA -ngl 99
```

CPU baseline for a fair comparison. Keep `OMP_WAIT_POLICY` identical: the NPU
path blocks on XRT completions, and an active OpenMP spin loop burns CPU that
the baseline never spends. `GGML_XDNA_NPU=0` is required; `--device none`
alone is not enough (see `--device` above).

```sh
GGML_XDNA_NPU=0 OMP_WAIT_POLICY=PASSIVE \
    ./build/bin/llama-server -m Qwen3.5-0.8B-Q4_K_M.gguf \
    --host 127.0.0.1 --port 8080 \
    -c 147456 -np 4 -b 4096 -ub 4096 --flash-attn off \
    -t 16 --poll 0 --device none -ngl 0
```

### 4. 9B memlock

Qwen3.5-9B packs about 13 GiB into `XDNA_REPACK` (`MAP_LOCKED`). If
`RLIMIT_MEMLOCK` is the usual 1/8 of RAM, raise it for that session:

```sh
sudo prlimit --pid $$ --memlock=unlimited:unlimited
```

## Execution contract

The backend's default `own_graph` mode claims the graph so model weights can be
placed in `XDNA_REPACK`. Within that graph:

- supported weight `MUL_MAT` operations with sufficiently large M run on the
  NPU;
- decode-sized weight `MUL_MAT` operations use the original quantized weights
  through a nested instance of the standard GGML CPU backend;
- every other operation uses that same standard CPU backend unchanged.

The XDNA backend does not contain private CPU implementations of model
operations. Host work in the NPU path is limited to weight conversion at model
load and activation packing/result scattering for each GEMM.

`GGML_XDNA_NPU=0`, a missing NPU, or failure to initialize both GEMM artifacts
prevents XDNA weight placement and leaves execution on the regular CPU path.
A failure after an NPU submission has started is reported as a compute error;
it is not silently recomputed on the CPU.

## Supported MUL_MAT

For `dst = src0 * src1`:

- `src0` is a two-dimensional weight tensor stored in `XDNA_REPACK`;
- `src1` is a contiguous F32 activation matrix;
- `dst` is contiguous F32;
- K is a positive multiple of 64;
- N is a positive multiple of 512 and no greater than 16384;
- M is unrestricted for placement, but only M greater than or equal to
  `GGML_XDNA_GEMM_M_MIN` runs on the NPU by default;
- batched dimensions `ne[2]` and `ne[3]` must both be one.

GGUF weights with a `to_float` trait (quantized, BF16, F16) are converted row
by row. The NPU therefore reads a swizzled BF16 copy rather than the source
layout. F32 activations are not placed in `XDNA_REPACK`.
With `GGML_XDNA_KEEP_SRC=1` (the default), the original GGUF bytes live in a
regular host allocation so nested CPU decode uses the same kernels and the
same DRAM as `-ngl 0`. The XRT buffer holds only the swizzled BF16 copy the
NPU reads.

## GEMM layout

The production geometry is:

```text
TILE_M = 32
TILE_K = 64
TILE_N = 64
grid   = 4 x 8
KT     = 16
K per submit = 1024
N per submit = 512
```

Two artifacts differ only in the number of M blocks:

```text
MB=16: M block = 16 * 4 * 32 = 2048
MB=32: M block = 32 * 4 * 32 = 4096
```

M greater than `GGML_XDNA_GEMM_MB32_MIN` selects MB=32. K values larger than
1024 are split and accumulated. M and partial K blocks are zero-padded by the
host. N is not padded in the production weight path.

Weights are laid out as:

```text
[n_block][k_block][grid_column][k_tile][8x8-swizzled tile]
```

Activations and results use the corresponding 8x8 MMUL micro-tile order.
Weights and activations are converted to BF16; each NPU K-slice produces F32,
and the host accumulates K-slices into the final F32 tensor.

## Buffer and kernel lifetime

`XDNA_REPACK` allocations are XRT host-only buffer objects mapped into the
process and visible to the NPU. They hold the swizzled BF16 copy. With
`KEEP_SRC=1`, original GGUF bytes are a separate host allocation used by nested
CPU decode. Weights are packed and synchronized once when the model is loaded.

Each GEMM artifact owns one `xrt::hw_context` and one persistent kernel handle.
Contexts are initialized lazily, shared by backend instances, and remain alive
until process exit. Per-submit A and C buffers are also persistent. Two banks
allow the host to pack the next M block while the previous run completes.
Multiple K-slices are submitted through one `xrt::runlist`.

The production build therefore uses at most two of the 16 hardware contexts
available on the validated XDNA2 device.

## Runtime settings

A first run needs only `--device XDNA`, `-ngl 99`, and a sourced XRT
environment. Defaults already enable the NPU (`GGML_XDNA_NPU=1`,
`GGML_XDNA_GEMM=1`, `GGML_XDNA_OWN_GRAPH=1`). For any timed comparison against
CPU, also set `OMP_WAIT_POLICY=PASSIVE` on **both** sides, and set
`GGML_XDNA_NPU=0` on the CPU side.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `GGML_XDNA_NPU` | `1` | Enable XDNA device execution |
| `GGML_XDNA_GEMM` | `1` | Enable production weight GEMM |
| `GGML_XDNA_GEMM_MB` | `16` | MB used by the first artifact |
| `GGML_XDNA_GEMM_MB16` | `1` | Enable the MB=16 slot |
| `GGML_XDNA_GEMM_MB32` | `1` | Enable the MB=32 slot |
| `GGML_XDNA_GEMM_MB32_MIN` | `2048` | Select MB=32 when M is greater than this |
| `GGML_XDNA_GEMM_M_MIN` | `128` | Minimum M submitted to the NPU |
| `GGML_XDNA_GEMM_KB_MAX` | `16` | Maximum K-slices in one runlist |
| `GGML_XDNA_GEMM_BANKS` | `2` | Number of A/C ping-pong banks |
| `GGML_XDNA_KEEP_SRC` | `1` | Keep original GGUF bytes in host DRAM |
| `GGML_XDNA_OWN_GRAPH` | `1` | Use nested standard CPU fallback |
| `GGML_XDNA_HOST_THREADS` | `8` | Threads used for GEMM pack/scatter |
| `GGML_XDNA_KERNELS_DIR` | unset | Additional artifact directory |
| `GGML_XDNA_WARMUP` | `4` | Warm-up submissions after context creation |
| `GGML_XDNA_CLFLUSH` | `1` | Invalidate mapped result cache lines |
| `GGML_XDNA_DEBUG` | `0` | Enable backend diagnostics |
| `GGML_XDNA_HOST_PROFILE` | `0` | Profile nested CPU operations; diagnostic only |

## Experimental kernels

Default Phase 1 (`-DGGML_XDNA=ON`, `GGML_XDNA_EXPERIMENTAL` off) compiles and
runs only the two weight-GEMM artifacts. GDN, ADD, MUL, RMS_NORM, fused
pre-norm, and activation-GEMM do **not** run.

To compile those research kernels:

```sh
cmake -B build-exp -DGGML_XDNA=ON -DGGML_XDNA_EXPERIMENTAL=ON \
    -DGGML_XDNA_IRON_PYTHON="$HOME/ironenv/bin/python3"
cmake --build build-exp -j --target llama-completion
```

That is not enough at runtime. The exact value `GGML_XDNA_ENABLE_EXPERIMENTAL=1`
is also required; any other value leaves the experimental slots off.

```sh
GGML_XDNA_ENABLE_EXPERIMENTAL=1 ./build-exp/bin/llama-completion \
    --device XDNA -ngl 99 -m MODEL -p "Hello" -n 32
```

With both gates set, experimental kernels still have their own switches:

| Kernel | Artifact | Runtime enable | Default after both gates |
| --- | --- | --- | --- |
| ADD | `ggml-xdna-add-npu2-f32-<TILE>` | `GGML_XDNA_ADD` | on (`1`) |
| MUL | `ggml-xdna-mul-npu2-f32-<TILE>` | `GGML_XDNA_MUL` | on (`1`) |
| RMS_NORM | `ggml-xdna-rms-norm-npu2-f32-<RMS_ROW>` | `GGML_XDNA_RMS_NORM` | on (`1`) |
| fused ADD+RMS_NORM+MUL | `ggml-xdna-add-rms-mul-npu2-f32-<RMS_ROW>` | off with `GGML_XDNA_DISABLE_FUSION=1` | on |
| GATED_DELTA_NET | `ggml-xdna-gdn-npu2-s128-r32-cs64[-w4]` | `GGML_XDNA_GDN=1` | **off** |
| GDN pre-L2 absorb | (host packing, no extra xclbin) | `GGML_XDNA_GDN_FUSE_PRE=1` | **off** |
| activation GEMM | production GEMM artifact | `GGML_XDNA_ACT_GEMM=1` | **off** |

GDN example:

```sh
GGML_XDNA_ENABLE_EXPERIMENTAL=1 GGML_XDNA_GDN=1 \
    ./build-exp/bin/llama-completion --device XDNA -ngl 99 -m MODEL ...
```

None of this is part of the Phase 1 execution or performance contract.
Phase 1 is single-op weight GEMM dispatch with nested CPU for everything else,
including GDN.

`GGML_XDNA_DEBUG` and `GGML_XDNA_HOST_PROFILE` are diagnostics on the
production path. They do not enable experimental kernels.

## Phase 1 coverage

Present in this tree:

- Weight `MUL_MAT` offload to the NPU for Qwen3.5 (0.8B, and 9B FFN `N=12288`
  now that the compile-time `GGML_XDNA_GEMM_N_MAX` is 16384).
- `llama-completion` E2E with `--device XDNA` / `-dev XDNA` and `-ngl 99`.
- Two BF16 GEMM `.xclbin` artifacts, built by CMake when IRON is available.
- Backend as a dynamic library via `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON`.
- Nested standard CPU fallback (`GGML_XDNA_OWN_GRAPH=1`); no fusion on the
  default path.
- Microtest target `test-xdna-production` (`micro`, `fallback`, `no-npu`).

Not in this tree, or not Phase 1:

- `.xclbin` files are not committed; they must be built or copied into
  `build/bin/`.
- This branch is `xdna/selective-hybrid` on top of `exp-gemm`. It is not
  automatically rebased onto a later Qvac drop; do that when the team provides
  the new base.
- Decode `MUL_MAT` (`M < GGML_XDNA_GEMM_M_MIN`) stays on nested CPU.
- Attention QK/AV, GDN state update, RMS_NORM, ADD, MUL stay on CPU unless the
  experimental build above is used.
- `test-xdna-production` cannot be linked when `GGML_BACKEND_DL=ON`.
- SoC RAPL energy is not available on the validated Ryzen AI MAX+ 395.

## Validation

```sh
ctest --test-dir build -L xdna --output-on-failure

GGML_XDNA_NPU=1 GGML_XDNA_OWN_GRAPH=1 ./build/bin/test-xdna-production micro
GGML_XDNA_NPU=1 GGML_XDNA_OWN_GRAPH=1 ./build/bin/test-xdna-production fallback
GGML_XDNA_NPU=0 GGML_XDNA_OWN_GRAPH=1 ./build/bin/test-xdna-production no-npu

./build/bin/llama-completion \
    -m Qwen3.5-0.8B-Q4_K_M.gguf \
    --device XDNA -ngl 99 -p "Hello" -n 32
```

A passing `micro` run must increase the NPU GEMM call counter. Greedy token
match alone is not a numerical correctness test.

## Correctness

The IRON GEMM design has a compile-time NumPy check using `rtol=0.2` and an
absolute tolerance scaled by K. This loose elementwise bound reflects BF16
input conversion and a different accumulation order. End-to-end changes must
also pass the CPU-reference microtest and Qwen3.5 perplexity validation; greedy
token equality alone is not a numerical correctness test.
