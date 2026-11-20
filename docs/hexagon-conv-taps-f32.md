# Hexagon F32 convolution taps

The optional registry procedure `ggml_backend_hexagon_conv_taps_f32` constructs a
custom convolution node from contiguous F16 weights `[K, C, taps]` and padded F32
input `[K, time]`. The output is contiguous F32 `[C, rows]`. It preserves F32 input
values and accumulates each tap separately before adding the rounded result.

Supported shapes have K at least 64 and divisible by 32, C divisible by four,
2–16 taps, and a positive output row count. Tap steps 1–64 read forward from the
padded input; step -1 reads from `taps - 1` backwards through taps. The input must
contain all referenced rows. Unsupported shapes return null, allowing the caller
to retain its ordinary graph.

This path supports a single physical session with DMA64 disabled. Extended-address
weights are not directly addressable by this kernel, so the constructor returns
null when DMA64 is enabled; the caller retains its ordinary graph. It requires the per-tap working set
to exceed VTCM, while a 64-row output tile per worker must fit VTCM. Inputs and
weights remain in shared memory; workers accumulate separate output tiles in
VTCM. No full-input staging or F16 conversion of activations is required. This is
a standalone optional kernel; it does not change ordinary matmul routing.

Build the Android backend and the appropriate DSP skeleton, then run
`GGML_HEXAGON_DMA64=0 test-hexagon-conv-taps` on HTP0. CTest sets this environment
for that test. It checks positive, dilated and reverse tap steps,
odd output tails and worker/tile boundaries against independent CPU arithmetic.
A missing device exits 77; a missing procedure on an available device fails.
Hardware execution is required to validate numerical behavior and performance;
a successful cross-build alone is insufficient.
