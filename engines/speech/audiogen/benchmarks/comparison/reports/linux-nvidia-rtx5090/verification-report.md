# Linux NVIDIA RTX 5090 artifact verification

## Hardware and operating system

The run used host `qvac-dev-linux-x64` with two NVIDIA GeForce RTX 5090 GPUs,
each reporting 32607 MiB. The reported NVIDIA driver was 595.71.05 on
`linux-x64`. Exact CPU, memory, and operating-system versions were not retained
with the committed artifacts.

## Source revisions

The run instructions pinned QVAC to
`57db2bb33f20ed7c75b50a21105853b17a47a741`, qvac-ext-ggml to
`0a76e3ed969781da6de41d6c9a1c3fc471c0978b`, acestep.cpp to
`9761469d95fc204b5468623c68a1a2203e50b1f9`, and its ggml submodule to
`c044c6f03892f9d5e98213b05f8afea1f8b0d3c9`. The generated JSON does not
independently embed those revisions.

## Toolchain and build flags

The host reported CMake 3.31.6, Node.js 20.19.4, CUDA compiler 12.4.131, and a
driver supporting CUDA 13.2. Both engines enabled `GGML_CUDA`, disabled Metal,
and compiled `89-virtual` PTX because CUDA 12.4 cannot emit native `sm_120`
binaries. The driver JIT-compiled the PTX for the RTX 5090.

## Models

The JSON contains the expected Q8_0 text encoder, Q8_0 0.6B LM, Q8_0 turbo DiT,
and BF16 VAE. Their SHA-256 hashes match the other committed platform reports.

## Workload

The run used the committed manifest, four threads, one warm-up, three timed
runs, alternating engine order, and a 5000 ms cooldown. CLAP used
`laion/larger_clap_music_and_speech`, caption-only text, and CPU scoring.

## Backend verification

The retained records identify CUDA for both engines. The harness was configured
to fail on GPU-to-CPU fallback, and no fallback failure appears in the report.

## Result validation

The report contains 30 successful timed rounds per engine and no failed,
malformed, or silent output. Every prompt has one deterministic WAV hash across
its three timed repetitions. CLAP scores are present for timed rounds.

## Limitations and sign-off

This is an implementation-level comparison with the process-shape and ggml-pin
asymmetries documented in `../../architecture.md`. The build used
forward-compatible PTX rather than native Blackwell code, peak RSS is
unavailable, and `environment.json` was not retained. Do not present these
numbers as native `sm_120` performance or use the memory chart for comparison.

- Review basis: committed JSON, Markdown, prompt manifest, and captured session output
- Decision: internally consistent; PTX and provenance limitations disclosed
