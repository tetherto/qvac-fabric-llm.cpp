# Direct MMA/WMMA Gated Delta Net

The persistent prefill kernel uses the internal `mma.cuh` BF16 primitives, an FP32 triangular solve, and FP32 recurrent state. It replaces the GDN CuTe and FlashInfer helpers. CUTLASS remains an independent optional dependency for FP4 matrix multiplication.

Each block owns one sequence/value head. Eight warps keep the state across 64-token CUDA chunks or 32-token HIP chunks. The inverse application and state update retain BF16 residual terms; a single BF16 product did not meet the precision tests during development.

## Validation

Hardware: NVIDIA GB10 (SM121, CUDA 13.0) and AMD Radeon 8060S (gfx1151, ROCm HIP 7.15). Both builds used `GGML_CUDA_CUTLASS=OFF`.

- All 121 GDN/precision cases and all nine continuation/replay cases passed on each backend.
- The continuation harness confirmed the accelerated allocation (`mma=1`) on both backends and the default native fallback (`mma=0`) on ROCm.
- CUDA memcheck reported zero errors for 129-token, two-sequence, permuted and nonpermuted inputs. Racecheck reported zero hazards for 129-token continuation.
- SM75 and SM120 compilation passed; only SM121 and gfx1151 received runtime testing.
- At 2048 tokens, CUDA output/state NMSE was 5.443e-5 / 1.116e-6. ROCm output/state NMSE was 3.892e-5 / 7.882e-7. The continuation tolerance is 3e-4.

These are operator and throughput checks, not a representative model-quality evaluation. A Qwen3.6 GGUF was not present on Strix, so its measurements use synthetic Qwen-shaped operator inputs.

## Strix performance

One sequence, width 128, 16 query/key heads, 48 value heads. Repeated measurements after removing the CUDA shared-memory swizzle from the HIP path:

| Tokens | Native ROCm | BF16 WMMA | Native / WMMA |
| --- | ---: | ---: | ---: |
| 512 | 1.892 ms | 1.988 ms | 0.95x |
| 2048 | 15.790 ms | 8.597 ms | 1.84x |

The initial swizzled layout took 23.507 ms at 2048 tokens. The plain HIP layout passed the complete correctness suite after the change. ROCm remains opt-in because the shorter prompt regressed and the performance coverage is narrow.

## GB10 performance

With FP4 CUTLASS matrix multiplication enabled, Qwen3.6-27B NVFP4 PP4096 (batch 4096, microbatch 2048, flash attention enabled, full GPU offload) remains near 1.2k tokens/s. Five-repetition confirmation runs:

| GDN path | Mean prompt tokens/s | Sample standard deviation |
| --- | ---: | ---: |
| Previous CuTe TF32 | 1232.11 | 1.23 |
| Direct BF16 MMA | 1205.46 | 2.98 |

The direct kernel is 2.2% slower in this matched configuration. Initial five-repetition runs were more variable: TF32 measured 1045.01 +/- 13.16 and MMA 1165.61 +/- 73.69 tokens/s. A reverse-order TF32 run and another MMA run produced the stable confirmation pair above; the initial measurements must not be used to claim a speedup over TF32.

The GDN kernel itself does not need CUTLASS. The independent FP4 matrix multiplication path still benefits from it. Disabling CUTLASS globally therefore changes more than GDN, as the following separate comparison illustrates.

Qwen3.6-27B NVFP4, PP4096, batch 4096, microbatch 2048, flash attention enabled, all layers on GPU, three repetitions per path using the same CUTLASS-off build:

| GDN path | Mean prompt tokens/s | Sample standard deviation |
| --- | ---: | ---: |
| Native | 834.38 | 6.79 |
| Direct BF16 MMA | 872.93 | 4.95 |

This pair measured a 4.6% model throughput improvement over native GDN. It is not a comparison with the PR's former CuTe TF32 kernel or with CUTLASS FP4 matrix multiplication enabled. The direct MMA operator measured 2.48-2.87 ms at 2048 tokens across runs; earlier CuTe BF16 and TF32 experiments measured about 2.14 and 1.79 ms, respectively. Removing the dependency did not establish a speedup over those implementations.

## Reproduce

Configure CUDA with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=121`, `GGML_CUDA_CUTLASS=OFF`, and `LLAMA_BUILD_TESTS=ON`. For the FP4-enabled model comparison, set `GGML_CUDA_CUTLASS=ON` (this does not change the GDN implementation). Configure HIP with `GGML_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx1151`, and `LLAMA_BUILD_TESTS=ON`.

```sh
# CUDA: enabled by default for supported prefill shapes
build/bin/test-backend-ops test -o GATED_DELTA_NET,GATED_DELTA_NET_PRECISION -b CUDA0 -j 4
build/bin/test-gdn-continuation

# Strix: opt in to the WMMA kernel
GGML_HIP_GDN_MMA=1 build/bin/test-backend-ops test -o GATED_DELTA_NET,GATED_DELTA_NET_PRECISION -b ROCm0 -j 4
GGML_HIP_GDN_MMA=1 build/bin/test-gdn-continuation
GGML_HIP_GDN_MMA=1 build/bin/test-backend-ops perf -o GATED_DELTA_NET -b ROCm0 -p 'head_count=16,head_size=128,n_seq_tokens=(512|2048),n_seqs=1,v_repeat=3,'

# Native comparison: same binary and options
GGML_CUDA_DISABLE_GDN_MMA=1 build/bin/test-backend-ops perf -o GATED_DELTA_NET -b ROCm0 -p 'head_count=16,head_size=128,n_seq_tokens=(512|2048),n_seqs=1,v_repeat=3,'

build/bin/llama-bench -m qwen3.6-27b-nvfp4.gguf -p 4096 -n 0 -b 4096 -ub 2048 -fa 1 -ngl 999 -r 3 -o json
GGML_CUDA_DISABLE_GDN_MMA=1 build/bin/llama-bench -m qwen3.6-27b-nvfp4.gguf -p 4096 -n 0 -b 4096 -ub 2048 -fa 1 -ngl 999 -r 3 -o json
```

Both environment controls are read once per device; restart the process when changing them. Native kernels remain available for unsupported shapes and hardware.
