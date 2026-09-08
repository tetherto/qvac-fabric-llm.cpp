# IRON kernels for ggml-xdna

The NPU kernels used by the `ggml-xdna` backend, written with IRON
(`@iron.jit` / MLIR-AIE).

Phase 1 builds **only** the production GEMM. Everything else is experimental
and is compiled only with `-DGGML_XDNA_EXPERIMENTAL=ON`. Even then the
process must set `GGML_XDNA_ENABLE_EXPERIMENTAL=1`. GDN still needs
`GGML_XDNA_GDN=1`. See [`docs/backend/XDNA.md`](../../../../docs/backend/XDNA.md).

| File | Status | Purpose |
| --- | --- | --- |
| `ggml-xdna-gemm.py` | production | BF16 weight GEMM |
| `ggml-xdna-add.py` | experimental | element-wise ADD |
| `ggml-xdna-mul.py` | experimental | element-wise MUL |
| `ggml-xdna-rms-norm.py` | experimental | per-row RMS_NORM |
| `ggml-xdna-add-rms-mul.py` | experimental | fused pre-norm block |
| `ggml-xdna-gdn.py` | experimental | GATED_DELTA_NET strip (off unless `GGML_XDNA_GDN=1`) |
| `build-kernels.sh` | build | wrapper called by `../CMakeLists.txt` |
| `env_setup.sh` | build | interactive env setup |

ADD and MUL are element-wise, so `transform_binary` covers them with a lambda
and their artifact is sized by the elementwise tile. RMS_NORM has to sum a whole
row before it can emit anything, so it carries an inline `ExternalFunction`
compiled by Peano and its artifact is sized by the model's `ne0`
(`GGML_XDNA_RMS_ROW`, default 1024). `eps` travels in a one-element buffer
rather than being baked in, so one xclbin serves any model.

Each design compiles to its own xclbin. The backend gives every one of them a
separate `xrt::hw_context` over a shared `xrt::device`, so all of them stay
resident on the NPU and dispatching between ops never reloads a design.

The backend never calls Python: `build-kernels.sh` produces
`ggml-xdna-<op>-npu2-f32-<N>.*`, and the C++ side loads them through XRT.
See [`docs/backend/XDNA.md`](../../../../docs/backend/XDNA.md).

## Setup

Production kernels need mlir-aie **1.3.4** (not 1.4.1). Install steps are in
[`docs/backend/XDNA.md`](../../../../docs/backend/XDNA.md).

```bash
source ~/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
source ggml/src/ggml-xdna/iron/env_setup.sh
```

## Run / compile

```bash
python ggml/src/ggml-xdna/iron/ggml-xdna-add.py -n 1024 --tile-size 256 --dtype f32 -v
python ggml/src/ggml-xdna/iron/ggml-xdna-mul.py -n 1024 --tile-size 256 --dtype f32 -v

# compile-only (CMake does this with -DGGML_XDNA_BUILD_KERNELS=ON)
python ggml/src/ggml-xdna/iron/ggml-xdna-add.py -d npu2 -n 1024 --tile-size 256 --dtype f32 \
  --xclbin-path build/bin/ggml-xdna-add-npu2-f32-1024.xclbin \
  --insts-path  build/bin/ggml-xdna-add-npu2-f32-1024.insts.bin
```

## Runtime toggles (experimental build only)

These apply only after `-DGGML_XDNA_EXPERIMENTAL=ON` and
`GGML_XDNA_ENABLE_EXPERIMENTAL=1`. The production GEMM path ignores them.

`GGML_XDNA_ADD=0` / `GGML_XDNA_MUL=0` / `GGML_XDNA_RMS_NORM=0` drop one kernel
from `supports_op`. `GGML_XDNA_NPU=0` disables the backend entirely.
`GGML_XDNA_DISABLE_FUSION=1` keeps every experimental op separate.
`GGML_XDNA_GDN=1` enables the GDN strip kernel (still off by default).

## Stage 5 fusion boundary (GDN surround)

Elementwise ops around GDN (`SSM_CONV`, `CONCAT`, `SIGMOID`, `L2_NORM`,
`RMS_NORM`, `SWIGLU`) must not be ported one-by-one: NPU streaming is ~30 GB/s
vs ~140 GB/s on the host. They only win when absorbed into the GDN kernel so
intermediates stay in MemTile/L1.

Current knobs:

- `GGML_XDNA_GDN=1` enables the strip kernel (default off: not yet faster).
- `GGML_XDNA_GDN_FUSE_PRE=1` absorbs surrounding `L2_NORM(q/k)` into GDN
  packing (reads pre-L2 inputs, applies L2 on the host while packing; L2 nodes
  become no-ops). SSM_CONV / SIGMOID / SWIGLU still need MemTile-full state.


## Residency

`xdna-probe` (built next to the backend) separates three costs that are easy to
confuse: *holding* a design resident, *switching* to it, and *reloading* it. It
reports the one-time load per design, how many `hw_context`s the driver grants,
submit latency while rotating over K contexts, submit latency on a single context
while N others merely exist, and a control run that rebuilds the context before
every submit. It also dumps the driver's own `aie-partitions` view while the
contexts are held.

On Strix Halo the answer is 16 contexts, all inside one partition spanning all 8
columns, and the count does not change with the data buffer size. Holding them
costs nothing measurable; switching costs a constant.
