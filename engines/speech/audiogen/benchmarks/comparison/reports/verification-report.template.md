# ACE-Step comparison verification

## Hardware and operating system

- Host:
- CPU:
- GPU:
- Memory:
- Operating system:

## Source revisions

- QVAC:
- qvac-ext-ggml:
- acestep.cpp:
- acestep.cpp ggml submodule:

## Toolchain and build flags

- Compiler:
- CMake:
- GPU toolkit:
- QVAC backend flags:
- acestep.cpp backend flags:
- CUDA or GPU architecture:
- Visible GPU selection:

## Models

List each GGUF filename, quantization, and SHA-256 hash.

## Workload

- Prompt manifest:
- Threads:
- Warm-ups:
- Timed runs:
- Order:
- Cooldown:
- CLAP model, revision, text policy, and device:

## Backend verification

Record log evidence showing the requested backend for both engines and confirm
that no GPU-to-CPU fallback occurred.

## Result validation

- Failed rounds:
- Malformed WAVs:
- Silent WAVs:
- Missing or duplicate outputs:
- CLAP coverage:

## Limitations and sign-off

Document implementation asymmetries, non-native GPU compilation, missing
environment data, or other interpretation limits.

- Reviewed by:
- Review date:
- Decision:
