// The K loop of mul_mm.comp, included twice: with MM_LOAD_BOUNDS_CHECK defined
// to false for tiles fully inside A/B (the bounds checks fold away and the loop
// stays branch-free), and to true for the ragged last tiles in M/N.
    for (uint block = start_k; block < end_k; block += BK) {
        [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) {
            load_a_to_shmem(pos_a, loadr_a, loadc_a + l, ir * BM + loadc_a + l, MM_LOAD_BOUNDS_CHECK, block, end_k);
        }
        [[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
#if !defined(MUL_MAT_ID)
            load_b_to_shmem(pos_b, loadr_b, loadc_b + l, ic * BN + loadc_b + l, MM_LOAD_BOUNDS_CHECK, block, end_k);
#else
            load_b_to_shmem(pos_b, loadr_b, loadc_b + l, ic, _ne1, MM_LOAD_BOUNDS_CHECK, block, end_k);
#endif
        }

        barrier();

        pos_a += BK / LOAD_VEC_A_EFF;
        pos_b += BK / LOAD_VEC_B_EFF;

#ifdef COOPMAT
        [[unroll]] for (uint i = 0; i < BK; i += TK) {
            [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row++) {
                // Load from shared into cache
                coopMatLoad(cache_a, buf_a, a_shmem_index(warp_r * WM + cm_row * TM, i / 2), a_shmem_stride(), gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint cm_col = 0; cm_col < cms_per_col; cm_col++) {
                    coopMatLoad(cache_b, buf_b, (warp_c * WN + cm_col * TN) * SHMEM_STRIDE + i / 2, SHMEM_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);

                    sums[cm_col * cms_per_row + cm_row] = coopMatMulAdd(cache_a, cache_b, sums[cm_col * cms_per_row + cm_row]);
                }
            }
        }
#else
        [[unroll]] for (uint i = 0; i < BK / BK_STEP; i++) {
            // Load from shared into cache
            [[unroll]] for (uint wsir = 0; wsir < WMITER; wsir++) {
                [[unroll]] for (uint j = 0; j < TM; j++) {
                #if defined(DATA_A_F32) || defined(DATA_A_F16)
                    cache_a[wsir * TM + j].xy = buf_a[(warp_r * WM + wsir * WSUBM + tiwr * TM + j) * SHMEM_STRIDE + 2 * i    ];
                    cache_a[wsir * TM + j].zw = buf_a[(warp_r * WM + wsir * WSUBM + tiwr * TM + j) * SHMEM_STRIDE + 2 * i + 1];
                #else
                    cache_a[wsir * TM + j] = buf_a[(warp_r * WM + wsir * WSUBM + tiwr * TM + j) * SHMEM_STRIDE + i];
                #endif
                }
            }

            [[unroll]] for (uint wsic = 0; wsic < WNITER; wsic++) {
                [[unroll]] for (uint cc = 0; cc < TN; cc++) {
                #if defined(DATA_A_F32) || defined(DATA_A_F16)
                    cache_b.xy = buf_b[(warp_c * WN + wsic * WSUBN + tiwc * TN + cc) * SHMEM_STRIDE + 2 * i    ];
                    cache_b.zw = buf_b[(warp_c * WN + wsic * WSUBN + tiwc * TN + cc) * SHMEM_STRIDE + 2 * i + 1];
                #else
                    cache_b = buf_b[(warp_c * WN + wsic * WSUBN + tiwc * TN + cc) * SHMEM_STRIDE + i];
                #endif

                    [[unroll]] for (uint wsir = 0; wsir < WMITER; wsir++) {
                        [[unroll]] for (uint cr = 0; cr < TM / 2; cr++) {
                            // [WNITER][TN][WMITER][TM / 2] -> [wsic][cc][wsir][cr]
                            const uint sums_idx = (wsic * TN + cc) * WMITER * (TM / 2) + wsir * (TM / 2) + cr;
                            sums[sums_idx].x = dot_product(cache_a[wsir * TM + 2 * cr    ], cache_b, sums[sums_idx].x);
                            sums[sums_idx].y = dot_product(cache_a[wsir * TM + 2 * cr + 1], cache_b, sums[sums_idx].y);
                        }
                    }
                }
            }

        }
#endif

        barrier();
    }
