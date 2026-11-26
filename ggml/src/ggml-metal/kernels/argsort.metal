#include "common.h"

constant bool FC_topk_moe_with_norm [[function_constant(FC_TOPK_MOE + 0)]];
constant int  FC_topk_moe_n_expert  [[function_constant(FC_TOPK_MOE + 1)]];
constant int  FC_topk_moe_top_k     [[function_constant(FC_TOPK_MOE + 2)]];

constant int  FC_moe_reduce_n_expert_used [[function_constant(FC_MOE_REDUCE + 0)]];

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

typedef void (argsort_merge_t)(
        constant   ggml_metal_kargs_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ggml_metal_kargs_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == GGML_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == GGML_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

template [[host_name("kernel_argsort_merge_f32_i32_asc")]]  kernel argsort_merge_t kernel_argsort_merge_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<GGML_SORT_ORDER_DESC>;

static inline uint ggml_top_k_f2ui(float x) {
    uint y = as_type<uint>(x);
    if ((y & 0x80000000u) != 0u) {
        y ^= 0xFFFFFFFFu; // negative floats: flip all bits
    } else {
        y |= 0x80000000u; // positive floats: set the sign bit
    }
    return y;
}

// the columns of a row segment starting at base
struct ggml_top_k_row {
    device const float * row;
    uint base;

    uint    key   (uint i) const { return ggml_top_k_f2ui(row[base + i]); }
    int32_t column(uint i) const { return (int32_t) (base + i); }
};

// candidate columns of a row, picked by the chunk pass of the chunked top-k
struct ggml_top_k_candidates {
    device const float   * row;
    device const int32_t * cand;

    uint    key   (uint i) const { return ggml_top_k_f2ui(row[cand[i]]); }
    int32_t column(uint i) const { return cand[i]; }
};

constexpr constant static uint GGML_TOP_K_SCAN_LANES       = 32;
constexpr constant static uint GGML_TOP_K_BUCKETS_PER_LANE = 256/GGML_TOP_K_SCAN_LANES;

// the first simdgroup finds the bucket, counted from the top, that holds the desired-th largest key and the number of
// keys above it: each lane sums 8 consecutive buckets and a prefix sum over the lanes picks the lane that holds it
inline void ggml_top_k_find_bucket(
        threadgroup atomic_uint * histo,
        uint desired,
        threadgroup uint * sh_bucket,
        threadgroup uint * sh_above,
        uint lane) {
    uint cnt[GGML_TOP_K_BUCKETS_PER_LANE];
    uint sum = 0;
    for (uint j = 0; j < GGML_TOP_K_BUCKETS_PER_LANE; ++j) {
        cnt[j] = atomic_load_explicit(&histo[255 - (lane*GGML_TOP_K_BUCKETS_PER_LANE + j)], memory_order_relaxed);
        sum += cnt[j];
    }

    uint acc = simd_prefix_exclusive_sum(sum);
    if (acc >= desired || acc + sum < desired) {
        return;
    }
    for (uint j = 0; j < GGML_TOP_K_BUCKETS_PER_LANE; ++j) {
        if (acc + cnt[j] >= desired) {
            *sh_bucket = 255 - (lane*GGML_TOP_K_BUCKETS_PER_LANE + j);
            *sh_above  = acc;
            return;
        }
        acc += cnt[j];
    }
}

// writes the columns of the top_k largest of n values to dst_row: finds the k-th largest key with 4 radix-8
// histogram passes, then emits every key above it and fills the rest from its ties, both in src order
// (deterministic). ntg_x must be a multiple of 32, at most 1024
template<typename Src>
inline void ggml_top_k_radix_select(
        Src src,
        uint n,
        uint top_k,
        device int32_t * dst_row,
        threadgroup atomic_uint * histo,
        threadgroup        uint * sh_bucket,
        threadgroup        uint * sh_above,
        threadgroup atomic_uint * out_count,
        uint tid,
        uint ntg_x) {
    uint prefix  = 0;     // fixed high bits of the threshold key
    uint desired = top_k; // count still needed from the candidate range

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += ntg_x) {
            atomic_store_explicit(&histo[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hi_mask   = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << uint(shift + 8));
        const uint prefix_hi = prefix & hi_mask;

        for (uint i = tid; i < n; i += ntg_x) {
            const uint key = src.key(i);
            if ((key & hi_mask) == prefix_hi) {
                atomic_fetch_add_explicit(&histo[(key >> uint(shift)) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < GGML_TOP_K_SCAN_LANES) {
            ggml_top_k_find_bucket(histo, desired, sh_bucket, sh_above, tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        prefix  |= *sh_bucket << uint(shift);
        desired -= *sh_above;

        // ensure every thread has consumed sh_bucket/sh_above before the next pass
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    (void) out_count; // the compaction ranks with simdgroup scans instead of an atomic counter

    // deterministic compaction (ggml-org/llama.cpp#31084): rank the kept elements by their order in src, so the
    // result does not depend on the thread scheduling. above the threshold first, then the ties
    const uint threshold = prefix;

    // each simdgroup scans a contiguous range, its lanes interleave with stride 32 to keep the reads coalesced
    const uint lane  = tid & 31u;
    const uint wg    = tid >> 5u;
    const uint nw    = ntg_x >> 5u;
    const uint wlo   = wg*((n + nw - 1) / nw);
    const uint whi   = min(wlo + (n + nw - 1) / nw, n);
    const uint nstep = wlo < whi ? (whi - wlo + 31) >> 5 : 0;

    uint ca = 0; // elements above the threshold in this lane's substream
    uint ct = 0; // elements tied with the threshold in this lane's substream
    for (uint i = wlo + lane; i < whi; i += 32u) {
        const uint key = src.key(i);
        ca += (key >  threshold) ? 1u : 0u;
        ct += (key == threshold) ? 1u : 0u;
    }

    // per-simdgroup totals into the now unused histogram, the first simdgroup turns them into exclusive bases
    const uint wca = simd_sum(ca);
    const uint wct = simd_sum(ct);
    if (lane == 0) {
        atomic_store_explicit(&histo[wg],      wca, memory_order_relaxed);
        atomic_store_explicit(&histo[32 + wg], wct, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        const bool valid = tid < nw;
        const uint sa = valid ? atomic_load_explicit(&histo[tid],      memory_order_relaxed) : 0u;
        const uint st = valid ? atomic_load_explicit(&histo[32 + tid], memory_order_relaxed) : 0u;
        const uint ea = simd_prefix_exclusive_sum(sa);
        const uint et = simd_prefix_exclusive_sum(st);
        if (valid) {
            atomic_store_explicit(&histo[tid],      ea, memory_order_relaxed);
            atomic_store_explicit(&histo[32 + tid], et, memory_order_relaxed);
        }
        const uint ta = simd_sum(sa);
        const uint tt = simd_sum(st);
        if (tid == 0) {
            *sh_bucket = ta; // total above the threshold
            *sh_above  = tt; // total tied with the threshold
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint above_total = *sh_bucket;
    const uint tie_total   = *sh_above;
    const uint base_wa = atomic_load_explicit(&histo[wg],      memory_order_relaxed);
    const uint base_wt = above_total + atomic_load_explicit(&histo[32 + wg], memory_order_relaxed);

    // the lanes write their kept elements at lane-major ranks, which does not depend on the scheduling. with more ties
    // than remaining slots, the ties are ranked in src order with a per-step cross-lane scan instead, so the lowest
    // tied indices are the ones kept
    const uint excl_a = simd_prefix_exclusive_sum(ca);
    const uint excl_t = simd_prefix_exclusive_sum(ct);

    if (tie_total <= top_k - above_total) {
        uint na = 0;
        uint nt = 0;
        for (uint i = wlo + lane; i < whi; i += 32u) {
            const uint key = src.key(i);
            if (key > threshold) {
                dst_row[base_wa + excl_a + na] = src.column(i);
                ++na;
            } else if (key == threshold) {
                dst_row[base_wt + excl_t + nt] = src.column(i);
                ++nt;
            }
        }
    } else {
        uint na = 0;
        for (uint i = wlo + lane; i < whi; i += 32u) {
            if (src.key(i) > threshold) {
                dst_row[base_wa + excl_a + na] = src.column(i);
                ++na;
            }
        }

        uint tt = 0; // tied elements in the earlier steps of this simdgroup
        for (uint step = 0; step < nstep; ++step) {
            const uint i   = wlo + 32u*step + lane;
            const bool ok  = i < whi;
            const bool t   = ok && src.key(i) == threshold;
            const uint pos = base_wt + tt + simd_prefix_exclusive_sum((uint) t);
            if (t && pos < top_k) {
                dst_row[pos] = src.column(i);
            }
            tt += simd_sum((uint) t);
        }
    }
}

inline device const float * ggml_top_k_src_row(constant ggml_metal_kargs_top_k & args, device const char * src0, uint i01, uint i02, uint i03) {
    return (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);
}

inline uint ggml_top_k_row_index(constant ggml_metal_kargs_top_k & args, uint i01, uint i02, uint i03) {
    return i01 + args.ne01*i02 + args.ne01*args.ne02*i03;
}

kernel void kernel_top_k_f32_i32(
        constant   ggml_metal_kargs_top_k & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup atomic_uint * histo     [[threadgroup(0)]],
        threadgroup        uint * sh_bucket [[threadgroup(1)]],
        threadgroup        uint * sh_above  [[threadgroup(2)]],
        threadgroup atomic_uint * out_count [[threadgroup(3)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const ggml_top_k_row src = { ggml_top_k_src_row(args, src0, tgpig[0], tgpig[1], tgpig[2]), 0 };

    device int32_t * dst_row = dst + args.top_k*ggml_top_k_row_index(args, tgpig[0], tgpig[1], tgpig[2]);

    ggml_top_k_radix_select(src, args.ne00, args.top_k, dst_row, histo, sh_bucket, sh_above, out_count, tpitg.x, ntg.x);
}

// chunked top-k, pass 1: the top_k columns of each chunk of a row (the last chunk also takes the remainder)
kernel void kernel_top_k_chunk_f32_i32(
        constant   ggml_metal_kargs_top_k & args,
        device   const char * src0,
        device      int32_t * cand,
        threadgroup atomic_uint * histo     [[threadgroup(0)]],
        threadgroup        uint * sh_bucket [[threadgroup(1)]],
        threadgroup        uint * sh_above  [[threadgroup(2)]],
        threadgroup atomic_uint * out_count [[threadgroup(3)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const uint chunk = tgpig[0] % args.n_chunks;
    const uint i01   = tgpig[0] / args.n_chunks;

    const uint c0 = chunk*args.chunk;
    const uint n  = chunk == uint(args.n_chunks - 1) ? args.ne00 - c0 : args.chunk;

    const ggml_top_k_row src = { ggml_top_k_src_row(args, src0, i01, tgpig[1], tgpig[2]), c0 };

    device int32_t * cand_row = cand + args.top_k*(chunk + args.n_chunks*ggml_top_k_row_index(args, i01, tgpig[1], tgpig[2]));

    ggml_top_k_radix_select(src, n, args.top_k, cand_row, histo, sh_bucket, sh_above, out_count, tpitg.x, ntg.x);
}

// chunked top-k, pass 2: the top_k columns of a row among the n_chunks*top_k candidates of pass 1
kernel void kernel_top_k_merge_chunks_f32_i32(
        constant   ggml_metal_kargs_top_k & args,
        device   const char * src0,
        device      int32_t * dst,
        device const int32_t * cand,
        threadgroup atomic_uint * histo     [[threadgroup(0)]],
        threadgroup        uint * sh_bucket [[threadgroup(1)]],
        threadgroup        uint * sh_above  [[threadgroup(2)]],
        threadgroup atomic_uint * out_count [[threadgroup(3)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const uint row = ggml_top_k_row_index(args, tgpig[0], tgpig[1], tgpig[2]);
    const uint n   = args.n_chunks*args.top_k;

    const ggml_top_k_candidates src = { ggml_top_k_src_row(args, src0, tgpig[0], tgpig[1], tgpig[2]), cand + n*row };

    ggml_top_k_radix_select(src, n, args.top_k, dst + args.top_k*row, histo, sh_bucket, sh_above, out_count, tpitg.x, ntg.x);
}

// fused SOFT_MAX + top-k + GET_ROWS (+ optional norm/scale) for MoE routing.
// One SIMDgroup handles one token row; n_expert is limited to 1024 by the host.
kernel void kernel_topk_moe_f32(
        constant   ggml_metal_kargs_topk_moe & args,
        device const char * src0,
        device       float * weights,
        device      int32_t * ids,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    const int row = (int) tgpig.x;
    if (row >= args.ne01) {
        return;
    }

    const int n_expert   = FC_topk_moe_n_expert;
    const int top_k      = FC_topk_moe_top_k;
    const int lane       = (int) tiisg;
    const int n_per_lane = (n_expert + 31) / 32;

    device const float * logits_row = (device const float *) (src0 + row * args.nb01);
    device       float * weights_row = weights + row * top_k;
    device      int32_t * ids_row   = ids + row * (args.nb1_ids / sizeof(int32_t));

    float wt[32];
    float output_weights[32];
    FOR_UNROLL (int i = 0; i < 32; ++i) {
        wt[i]            = -INFINITY;
        output_weights[i] = 0.0f;
    }

    for (int i = lane; i < n_expert; i += 32) {
        const float v = logits_row[i];
        wt[i / 32] = isnan(v) ? -FLT_MAX : v;
    }

    // softmax over the expert logits
    float max_val = -INFINITY;
    FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
        max_val = max(max_val, wt[i]);
    }
    max_val = simd_max(max_val);

    float sum_val = 0.0f;
    FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
        wt[i] = exp(wt[i] - max_val);
        sum_val += wt[i];
    }
    sum_val = simd_sum(sum_val);

    const float inv_sum = 1.0f / sum_val;
    FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
        wt[i] *= inv_sum;
    }

    float wt_sum = 0.0f;

    for (int k = 0; k < top_k; ++k) {
        float best_val = -INFINITY;
        int   best_expert = -1;

        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
            const int expert = lane + i * 32;
            if (expert < n_expert && (wt[i] > best_val || (wt[i] == best_val && expert < best_expert))) {
                best_val    = wt[i];
                best_expert = expert;
            }
        }

        FOR_UNROLL (int mask = 16; mask > 0; mask >>= 1) {
            const float val    = simd_shuffle_xor(best_val, mask);
            const int   expert = simd_shuffle_xor(best_expert, mask);
            if (val > best_val || (val == best_val && expert < best_expert)) {
                best_val    = val;
                best_expert = expert;
            }
        }

        if ((best_expert & 31) == lane) {
            wt[best_expert / 32] = -INFINITY;
        }

        if ((k & 31) == lane) {
            output_weights[k / 32] = best_val;
        }

        if ((best_expert & 31) == lane) {
            ids_row[k] = best_expert;
            if (FC_topk_moe_with_norm) {
                wt_sum += best_val;
            }
        }
    }

    if (FC_topk_moe_with_norm) {
        wt_sum = simd_sum(wt_sum);
        wt_sum = max(wt_sum, args.clamp);
        const float inv = 1.0f / wt_sum;
        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
            output_weights[i] *= inv;
        }
    }

    FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
        const int idx = i * 32 + lane;
        if (idx < top_k) {
            weights_row[idx] = output_weights[i] * args.scale;
        }
    }
}

// fused MoE expert weighting + reduction: weighted = sum(experts[e] * weights[e]).
// The host guarantees all tensors are contiguous F32.
kernel void kernel_moe_reduce_f32(
        constant   ggml_metal_kargs_moe_reduce & args,
        device const float * experts,
        device const float * weights,
        device       float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t token = tgpig.x;
    const int64_t col   = (int64_t) tgpig.y * ntg.x + tpitg.x;
    if (token >= args.ne02 || col >= args.ne00) {
        return;
    }

    const int n_expert_used = FC_moe_reduce_n_expert_used;

    const int64_t base = token * (int64_t) n_expert_used * args.ne00 + col;
    float sum = 0.0f;
    FOR_UNROLL (int e = 0; e < n_expert_used; ++e) {
        sum += experts[base + e * args.ne00] * weights[token * n_expert_used + e];
    }
    dst[token * args.ne00 + col] = sum;
}
