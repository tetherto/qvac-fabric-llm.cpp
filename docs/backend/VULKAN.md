# Vulkan backend

## Experimental batched host uploads

Set `GGML_VK_BATCH_UPLOADS=1` to stage unregistered host uploads in a reusable
64 MiB buffer per backend context. The default is off; `GGML_VK_BATCH_UPLOADS=0`
also disables it. No additional dependency is required.

The compute-queue path copies each host payload into a separate mapped slice,
records its transfer, and defers waiting until the existing backend
synchronization point. When the arena fills, the backend synchronizes before
reusing it. This avoids a staging copy followed by a wait for every small upload.

Safety constraints:

- The arena requires host-visible, host-coherent memory. Host writes occur before
  queue submission; the existing transfer/compute barriers remain in place.
- Slices are not reused until all submitted compute work has completed.
- Already registered host sources retain their direct upload path.
- Separate transfer queues and serialized submissions retain the existing path.
- Uploads exceeding the arena capacity, or not meeting four-byte copy alignment,
  retain the existing synchronous fallback.
- Failure to allocate the arena disables batching for that backend context and
  falls back to synchronous staging. This does not guarantee that the fallback
  allocation itself succeeds under memory pressure.

This is intended for validation on additional Vulkan implementations before
considering a default change. NVIDIA RTX 5090 and RTX 4090 results do not establish safety or
performance on AMD, Intel, integrated GPUs, or other drivers. The extra 64 MiB
allocation can also matter on memory-constrained devices.

### Tests

With `GGML_VULKAN=ON` and `LLAMA_BUILD_TESTS=ON`, build
`test-vulkan-upload` and run `ctest -R test-vulkan-upload --output-on-failure`.
The tests exercise wrap/reuse, immediate reuse of unregistered source memory,
upload followed by GPU computation, strided views with preserved padding, and
oversized fallback. Separate tests enable batching, disable it, and select the
transfer-queue and serialized-submission bypasses.

Use `GGML_VK_VISIBLE_DEVICES` to select the target GPU. These tests use the first
visible Vulkan device and skip if no Vulkan backend can be initialized.

The PR version passed the upload test with batching on/off, transfer-queue mode,
and serialized submissions on an RTX 5090, and with batching on an RTX 4090.
Synchronization validation layers and allocation-failure injection remain
untested.

A focused CPU-reference operator check passed 29 of 30 selected Q4_0, IQ2_XS,
and F32 MUL_MAT_ID cases with batching enabled, disabled, and the unchanged base
library. All three failed the same F32 case with
`n_mats=4,n_used=2,b=0,m=64,n=16,k=3`. This pre-existing failure is not fixed here.

### Prototype measurements

Each condition used two initial warm-ups, a full untimed 24-prompt pass, then
24 measured prompts with 128 generated tokens and automatic expert caching.

| Host/model | Load mode | Original uploads | Batched uploads |
| --- | --- | ---: | ---: |
| EPYC 7742, DDR4, RTX 5090; Qwen3.8-Flash-Next Q4_0 | none | 29.73 tok/s | 32.26 tok/s |
| Same | mmap | 18.28 tok/s | 22.81 tok/s |
| Ryzen host, RTX 5090; Qwen3.8-Flash-Next UD-Q2_K_XL | mmap | 35.13 tok/s | 42.14 / 42.50 tok/s |

All 24 output hashes matched within each batching on/off comparison. The DDR4
runs had zero measured major faults or swap-in; the pinned batched run recorded
17 host-wide swap-out pages. The Ryzen full passes had paging activity; a short
four-prompt off/on/on/off control with nearly no paging also showed about a 21%
gain. Quantization, host memory, and placement differ between hosts, so these
numbers do not isolate hardware effects.

These measurements are from the prototype before the opt-in flag was renamed
and allocation-failure fallback was added. They compare two cache-enabled
configurations, not cache against no cache. Decode throughput is generated
tokens divided by summed server decode time. API-level profiling was enabled;
this is not a statistical or cross-vendor performance guarantee.

The staging synchronization follows the
[Vulkan synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html).
