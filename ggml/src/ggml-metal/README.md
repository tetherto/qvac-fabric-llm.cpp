# Embedded precompiled Metal shaders

`GGML_METAL_EMBED_PRECOMPILED=ON` embeds a precompiled macOS library alongside
all embedded shader sources. It is off by default and requires Metal,
`GGML_METAL_EMBED_LIBRARY=ON`, a macOS target and Xcode's `metal`/`metallib`
tools through `xcrun`. Each split kernel source is compiled separately, then
linked into the embedded library. No external shader resources are installed.

The precompiled variant uses Metal 3.1, macOS 14 and BF16, without the native
tensor API. It is attempted only on macOS 14 or newer, with BF16 support and
without native tensor operations. A rejected library falls back to the normal
per-source runtime compilation. Other devices retain source compilation.
The precompiled variant has a fixed language/deployment baseline independent
of `GGML_METAL_STD` and `GGML_METAL_MACOSX_VERSION_MIN`.
`GGML_METAL_SHADER_DEBUG` applies to both precompiled and runtime compilation.

The existing self-hosted Metal workflow tests successful loading, BF16-disabled
source fallback and rejection of a deliberately corrupted embedded library.
Run `bash ci/test-metal-precompiled.sh` on a BF16-capable Mac with the Metal
compiler installed. The script only corrupts its own build output for the last
case; rebuild from a clean build directory before using that artifact elsewhere.

Precompilation removes shader source compilation from eligible startup paths;
pipeline specialization and model loading still occur. Measure initialization
separately from inference before claiming a startup improvement.
