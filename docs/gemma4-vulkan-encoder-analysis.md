# QVAC-21361 — Gemma 4 E2B Vision Encoder: Vulkan Op-Coverage & Hotspot Analysis

Device-independent analysis of the Gemma 4 E2B vision encoder on the **Vulkan**
backend, produced to guide kernel optimization for Pixel 9 Pro (Arm Mali) ahead of
on-device measurement.

## Method & caveats

- Built `qvac-fabric-llm.cpp` natively with `-DGGML_VULKAN=ON` and ran the
  **encoder-only** pass (`llama-mtmd-debug -p encode -n 768 --image gray`,
  Gemma 4 E2B Q8_0 + mmproj).
- The only Vulkan device available locally is **`llvmpipe`** (software Vulkan on
  CPU), enabled via `GGML_VK_ALLOW_CPU_DEVICES=1`.
- **Therefore absolute timings are NOT used** — llvmpipe latency is meaningless for
  Mali. What *is* transferable and reported here:
  1. **Op coverage** (which ops the Vulkan backend supports / where it falls back).
  2. **Op composition** and **exact kernel shapes** (fixed by the graph, not the GPU).
  3. **Analytical FLOP breakdown** and how it scales with token budget.

Encoder config (from load): `projector=gemma4v`, `n_embd=768`, `n_head=12`
(`head_dim=64`), `n_layer=16`, `n_ff=3072`, `patch_size=16`, `n_merge=3`,
`projection_dim=1536`. For a 768x768 input: `48x48 = 2304` patch tokens through the
ViT, pooled 3x3 to `16x16 = 256` output tokens.

---

## 1. Op coverage — no CPU fallbacks (good news)

The encoder graph runs **entirely on the Vulkan backend**:

```
alloc_compute_meta: graph splits = 1, nodes = 940
```

`graph splits = 1` over 940 nodes means **every encoder op is supported by the
Vulkan backend** — there are **no ops silently falling back to CPU**. So the common
"first easy win" on mobile (eliminating fallbacks) does **not** apply here; the
encoder is already a single Vulkan dispatch chain. Optimization must come from
making the existing kernels faster, not from coverage gaps.

### Op composition (encoder graph)

| Op | Role in the ViT encoder |
|---|---|
| `MUL_MAT` | QKV / output / FFN / projector matmuls — **primary compute** |
| `FLASH_ATTN_EXT` | attention (1 per layer, 16 layers) — **O(N^2) compute** |
| `RMS_NORM` (+ fused `RMS_NORM_MUL`, `RMS_NORM_MUL_ROPE...`) | normalization (Gemma uses RMS) |
| `ROPE` | Gemma 4's 2D rotary on Q/K |
| `GLU` | gated FFN activation |
| `IM2COL` + `MUL_MAT` (f32) | patch embedding (conv lowered to im2col+GEMM) |
| `POOL_2D` | 3x3 average pool (token reduction, once) |
| `CLAMP`, `MUL`, `ADD`, `CONCAT`, `CPY`, `SCALE`, `GET_ROWS`, `SET_ROWS` | Gemma4-specific scaling/bias, residuals, glue |

Note: several ops are **fused** in the Vulkan backend (`RMS_NORM_MUL`,
`RMS_NORM_MUL_ROPE`, `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS`), so norm+mul+rope already
run as single kernels — good for Mali (fewer dispatches/round-trips).

---

## 2. The kernels that matter (exact shapes)

Per-layer matmuls over `N = 2304` tokens (`d = 768`, `ff = 3072`), Q8_0 weights:

| Kernel (ggml) | Shape (m x n x k) | What it is |
|---|---|---|
| `MUL_MAT q8_0` | `768 x 2304 x 768` | attention Q/K/V & output projections |
| `MUL_MAT q8_0` | `3072 x 2304 x 768` | FFN up + gate (d -> ff) |
| `MUL_MAT q8_0` | `768 x 2304 x 3072` | FFN down (ff -> d) |
| `MUL_MAT q8_0` | `1536 x 256 x 768` | final projector (after pooling, 256 tokens) |
| `MUL_MAT f32`  | `2304 x 768 x 768` | patch-embed GEMM (im2col output, **f32**) |
| `FLASH_ATTN_EXT` | `q/k/v = (64, 2304, 12)`, no mask | full bidirectional attention, head_dim 64, 12 heads, 2304 tokens |

Two immediate observations:
- The attention is **maskless full attention over 2304 tokens** at `head_dim=64` —
  exactly the `flash_attn` shader's domain.
- The patch-embed GEMM is **f32** (not quantized/f16) — a candidate for f16 on Mali.

---

## 3. FLOP breakdown (analytical, hardware-independent)

Per transformer layer at `N=2304`, `d=768`, `ff=3072`, 12 heads x 64
(MAC = multiply-add; FLOP = 2 x MAC):

| Component | MAC / layer | Share |
|---|---:|---:|
| FFN (gate + up + down, gated) | ~16.3 G | **54.5%** |
| Attention (QK^T + A·V) | ~8.16 G | **27.3%** |
| QKV + output projections | ~5.44 G | 18.2% |
| **Total / layer** | ~29.9 G | 100% |

Whole encoder (16 layers) at 768px / 256 output tokens ≈ **~0.96 TFLOP**.

So at this token budget the encoder is **matmul-bound**: FFN + projections ≈ **73%**,
attention ≈ **27%**.

### The key insight: priority shifts with token budget

The ticket's budgets (70/140/280/560/1120) are **output** tokens; input patches are
`9x` that (3x3 pool). Matmul FLOPs scale **O(N)** with tokens, attention scales
**O(N^2)**. So the dominant kernel changes with budget:

| Output tokens | Input patches N | Matmul vs Attention (approx) | Optimize first |
|---:|---:|---|---|
| 70 | 630 | matmul ~90% | **mul_mm / FFN** |
| 256 (measured) | 2304 | matmul ~73%, attn ~27% | **mul_mm / FFN** |
| 560 | 5040 | roughly even | both |
| 1120 | 10080 | **attention ~62%**, matmul ~38% | **flash_attn** |

This means a single kernel won't win across the board — the optimization target
should track the token budgets actually used in production.

---

## 4. Kernel optimization priorities for Mali (Pixel 9 Pro)

Derived from the above, to validate/measure once a device is available:

1. **`mul_mm` (Q8_0) for the FFN and projection shapes** — highest payoff at the
   common budgets. On Mali: tune tile sizes / workgroup shape for `k=768` and
   `k=3072`; evaluate `GGML_VK_DISABLE_F16` on/off and whether Mali's
   cooperative-matrix path (if exposed by the driver) helps these shapes.
2. **`flash_attn` for the 2304+ token, head_dim=64, maskless case** — becomes the
   dominant cost at high budgets; confirm the flash path is selected (`-fa on`) and
   tuned for Mali subgroup width rather than falling to the generic softmax matmul.
3. **Patch-embed GEMM is f32** — try f16 to halve bandwidth for that op.
4. **Already-fused norm/rope kernels** (`RMS_NORM_MUL_ROPE*`) are good — keep; no
   action unless the profile says otherwise.
5. **`CLAMP`/`SCALE`/`MUL` glue ops** are numerous but cheap; only worth fusing if
   on-device profiling shows dispatch overhead dominating at small budgets.

## 5. What still requires the device (DoD #1/#2)

- Real per-op latency on Mali (`GGML_VK_PERF_LOGGER=1` + `LLAMA_LOG_VERBOSITY=4`,
  analyzed with `vulkan_profiling_analyzer.py`).
- Confirming which of the above levers actually move Mali latency, and by how much.
- The agreed speedup target, measured before/after across the token budgets.

Plan to run this on a cloud device farm (Pixel 9 Pro): push the Android Vulkan
binaries (`build-android-vulkan/bin` + libs) and the GGUFs, run
`llama-mtmd-cli ... -fa on` with `MTMD_BACKEND_DEVICE=Vulkan0`, pull the perf log,
and feed it to `vulkan_profiling_analyzer.py`.
