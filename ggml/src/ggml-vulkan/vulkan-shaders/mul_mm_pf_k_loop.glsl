// Register-prefetch K loop of mul_mm.comp for aligned tiles inside A/B: the raw registers hold the tile at pos_a / pos_b and
// store_*_raw write it to LDS while the next tile's loads are in flight. The lane-invariant addressing is computed once.
    // The 256x256 XL tier (16 wave32 subgroups, 128 VGPRs of accumulators) stays spill-free by re-reading B for every
    // accumulator row and, except for q4_K, keeping only row 0's tile addressing in registers.
    const bool XL_TILE = BM * BN >= 256 * 256;
#if defined(DATA_A_Q4_K)
    const bool XL_IDX_BASE = false;
#else
    const bool XL_IDX_BASE = XL_TILE;
#endif
    const uint NA_RAW = (BM + loadstride_a - 1) / loadstride_a;
    const uint NB_RAW = (BN + loadstride_b - 1) / loadstride_b;
    A_RAW_T a_raw[NA_RAW];
    B_TYPE b_raw[NB_RAW];
    uint a_lane[NA_RAW];
    uint a_sidx[NA_RAW];
    uint b_sidx[NB_RAW];
    const uint a_lane0 = a_lane_off(loadr_a, loadc_a);
    const uint a_sidx0 = a_shmem_index(loadc_a, loadr_a * LOAD_VEC_A / 2);
    const uint b_sidx0 = loadc_b * SHMEM_STRIDE + loadr_b * LOAD_VEC_B / 2;
    if (!XL_IDX_BASE) {
        [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) {
            a_lane[l / loadstride_a] = a_lane_off(loadr_a, loadc_a + l);
            a_sidx[l / loadstride_a] = a_shmem_index(loadc_a + l, loadr_a * LOAD_VEC_A / 2);
        }
        [[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
            b_sidx[l / loadstride_b] = (loadc_b + l) * SHMEM_STRIDE + loadr_b * LOAD_VEC_B / 2;
        }
    }
#if defined(MUL_MAT_ID)
    uint b_off[NB_RAW];
    [[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
        b_off[l / loadstride_b] = b_row_off(loadc_b + l);
    }
#define B_FETCH_ARG(l) b_off[(l) / loadstride_b]
#else
#define B_FETCH_ARG(l) loadc_b + (l)
#endif
#define A_LANE(l) (XL_IDX_BASE ? a_lane0 + a_lane_off(0, (l)) : a_lane[(l) / loadstride_a])
#define A_SIDX(l) (XL_IDX_BASE ? a_sidx0 + a_shmem_index((l), 0) : a_sidx[(l) / loadstride_a])
#define B_SIDX(l) (XL_IDX_BASE ? b_sidx0 + (l) * SHMEM_STRIDE : b_sidx[(l) / loadstride_b])
#define PF_FETCH() \
    [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) { fetch_a(pos_a, loadr_a, loadc_a + l, A_LANE(l), a_raw[l / loadstride_a]); } \
    [[unroll]] for (uint l = 0; l < BN; l += loadstride_b) { fetch_b(pos_b, loadr_b, B_FETCH_ARG(l), b_raw[l / loadstride_b]); }
#define PF_STORE() \
    [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) { store_a_raw(pos_a, loadr_a, loadc_a + l, A_SIDX(l), a_raw[l / loadstride_a]); } \
    [[unroll]] for (uint l = 0; l < BN; l += loadstride_b) { store_b_raw(B_SIDX(l), b_raw[l / loadstride_b]); }
    a_fetch_all = true;
    PF_FETCH()
    a_fetch_all = false;
    for (uint block = start_k; block < end_k; block += BK) {
        SCACHE_REFILL(pos_a, block)
        PF_STORE()
        // past the last tile the fetch re-reads the last tile (never stored), which keeps a select off the raw registers
        const bool more = block + BK < end_k;
        pos_a += more ? BK / LOAD_VEC_A_EFF : 0;
        pos_b += more ? BK / LOAD_VEC_B_EFF : 0;
        PF_FETCH()
        barrier();

#if defined(MUL_MAT_ID)
        // the MUL_MAT_ID tiles are narrow in N: the subgroup's B fragments of a K step stay in registers
        if (warp_c < required_warp_c) {
            [[unroll]] for (uint i = 0; i < BK; i += TK) {
                coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_bs[cms_per_col];
                [[unroll]] for (uint cm_col = 0; cm_col < cms_per_col; cm_col++) {
                    coopMatLoad(cache_bs[cm_col], buf_b, (warp_c * WN + cm_col * TN) * SHMEM_STRIDE + i / 2, SHMEM_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);
                }
                [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row++) {
                    coopMatLoad(cache_a, buf_a, a_shmem_index(warp_r * WM + cm_row * TM, i / 2), a_shmem_stride(), gl_CooperativeMatrixLayoutRowMajor);
                    [[unroll]] for (uint cm_col = 0; cm_col < cms_per_col; cm_col++) {
                        sums[cm_col * cms_per_row + cm_row] = coopMatMulAdd(cache_a, cache_bs[cm_col], sums[cm_col * cms_per_row + cm_row]);
                    }
                }
            }
        }
#else
        [[unroll]] for (uint i = 0; i < BK; i += TK) {
            [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row++) {
                coopMatLoad(cache_a, buf_a, a_shmem_index(warp_r * WM + cm_row * TM, i / 2), a_shmem_stride(), gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint cm_col = 0; cm_col < cms_per_col; cm_col++) {
                    coopMatLoad(cache_b, buf_b, (warp_c * WN + cm_col * TN) * SHMEM_STRIDE + i / 2, SHMEM_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);
                    sums[cm_col * cms_per_row + cm_row] = coopMatMulAdd(cache_a, cache_b, sums[cm_col * cms_per_row + cm_row]);
                }
                if (XL_TILE) { subgroupMemoryBarrierShared(); }
            }
        }
#endif

        barrier();
    }
#undef PF_FETCH
#undef PF_STORE
#undef A_LANE
#undef A_SIDX
#undef B_SIDX
#undef B_FETCH_ARG
