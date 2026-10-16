# Third-party notices

This directory holds the ggml-xdna backend. Most of it is original work under
the repository's MIT licence (`LICENSE` at the repository root). One file is
not original, and its notice is repeated here so that it travels with whatever
copy of this directory is distributed.

## kernels/gemm.py

Derived from the matrix-multiplication designs under mlir-aie's
`programming_examples`:

    programming_examples/basic/matrix_multiplication/whole_array/whole_array.py
    programming_examples/basic/matrix_multiplication/cascade/cascade.py
    programming_examples/ml/block_datatypes/matrix_multiplication/whole_array_shuffle/whole_array.py
    programming_examples/ml/block_datatypes/matrix_multiplication/whole_array_mixed/whole_array.py

    Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
    SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The array and ObjectFifo plumbing is theirs; the kernel, the tile layouts, the
RTP counts and the runtime sequence are this project's. The file carries the
notice and the SPDX identifier in its own header, and is distributed under
Apache-2.0 WITH LLVM-exception - the stricter of the two licences, which covers
both parts. The licence texts are `LICENSES/Apache-2.0.txt` and
`LICENSES/LLVM-exception.txt` in this directory.

Nothing else under `kernels/` is derived: the other designs (including their
argparse/`run_design_cli` skeletons) use the published IRON API only.

## mlir-aie itself - a build dependency, not redistributed here

The IRON designs are compiled by an out-of-tree mlir-aie installation
(`GGML_XDNA_GEMM_PYTHON`). mlir-aie is Apache-2.0 WITH LLVM-exception, and the
`.xclbin` and `.insts.bin` it produces are covered by the LLVM exception, so
the artifacts this backend ships need no notice added. `kernels/attn-mm.cc`
includes mlir-aie's `aie_kernels/aie2p/mm.cc` at compile time: source consumed,
not redistributed.

## The host-side TXN stream builders

`xdna-seq.cpp`, `xdna-seq.h`, `xdna-seq-attn.cpp`, `xdna-gemv.cpp`,
`xdna-gemv.h` and the stream code in `xdna-pgemm.cpp` are this project's own
implementation of the NPU2 instruction-stream format. They were written from
the BD field semantics documented in mlir-aie (`AIEDmaToNpu.cpp`,
`include/aie/Runtime/TxnEncoding.h`) and from instruction streams read back out
of artifacts this design built. No mlir-aie or IRON source was copied or
translated into them; the one sentence quoted from mlir-aie, in
`xdna-seq.cpp`, is attributed where it appears. They are MIT, like the rest of
the backend.
