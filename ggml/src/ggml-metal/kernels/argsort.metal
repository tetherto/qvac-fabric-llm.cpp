#include "common.h"

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

// fused softmax -> top-k -> get_rows for MoE decode (see ggml-cuda/topk-moe.cu)
// one simdgroup per token, 4 tokens per threadgroup
template<int n_experts>
static inline void topk_moe_softmax(thread float * vals, ushort lane) {
    constexpr int experts_per_thread = (n_experts > N_SIMDWIDTH) ? n_experts / N_SIMDWIDTH : 1;

    float max_val = -INFINITY;
    FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * N_SIMDWIDTH;
        if (n_experts % N_SIMDWIDTH == 0 || idx < n_experts) {
            max_val = max(max_val, vals[i]);
        }
    }
    max_val = simd_max(max_val);

    float sum = 0.f;
    FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * N_SIMDWIDTH;
        if (n_experts % N_SIMDWIDTH == 0 || idx < n_experts) {
            const float val = exp(vals[i] - max_val);
            vals[i] = val;
            sum += val;
        } else {
            // keep pads below the -FLT_MAX that NaN logits become, so they are never selected
            vals[i] = -INFINITY;
        }
    }
    sum = simd_sum(sum);
    const float inv_sum = 1.0f / sum;
    FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * N_SIMDWIDTH;
        if (n_experts % N_SIMDWIDTH == 0 || idx < n_experts) {
            vals[i] *= inv_sum;
        }
    }
}

template<int n_experts>
kernel void kernel_topk_moe(
        constant ggml_metal_kargs_topk_moe & args,
        device const float * logits,
        device       float * weights,
        device     int32_t * ids,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NW = N_SIMDWIDTH;
    constexpr int experts_per_thread = (n_experts > NW) ? n_experts / NW : 1;

    const int row = tgpig.x * 4 + sgitg;
    if (row >= args.n_rows) {
        return;
    }

    device const float * row_logits  = logits  + (int64_t) n_experts * row;
    device       float * row_weights = weights + (int64_t) args.n_expert_used * row;
    device     int32_t * row_ids     = ids     + (int64_t) n_experts * row;

    float wt[experts_per_thread];
    FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
    FOR_UNROLL (int i = 0; i < n_experts; i += NW) {
        const int expert = i + tiisg;
        wt[i / NW] = (n_experts % NW == 0 || expert < n_experts) ? row_logits[expert] : -INFINITY;
    }

    topk_moe_softmax<n_experts>(wt, tiisg);

    FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
        if (isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
    FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < args.n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = tiisg;

        FOR_UNROLL (int i = 1; i < experts_per_thread; i++) {
            const int expert = tiisg + i * NW;
            if ((n_experts % NW == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }

        for (int mask = NW / 2; mask > 0; mask /= 2) {
            const float val    = simd_shuffle_xor(max_val, mask);
            const int   expert = simd_shuffle_xor(max_expert, mask);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((max_expert & (NW - 1)) == tiisg) {
            wt[max_expert / NW] = -INFINITY;
        }

        if ((k & (NW - 1)) == tiisg) {
            output_weights[k / NW] = max_val;
        }

        if ((max_expert & (NW - 1)) == tiisg) {
            row_ids[k] = max_expert;
            if (args.with_norm != 0) {
                wt_sum += max_val;
            }
        }
    }

    if (args.with_norm != 0) {
        wt_sum = simd_sum(wt_sum);
        wt_sum = max(wt_sum, args.clamp_val);
        const float inv_sum = 1.0f / wt_sum;
        FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    FOR_UNROLL (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * NW + tiisg;
        if (idx < args.n_expert_used) {
            row_weights[idx] = output_weights[i] * args.scale_val;
        }
    }
}

typedef decltype(kernel_topk_moe<8>) kernel_topk_moe_t;

template [[host_name("kernel_topk_moe_8")]]   kernel kernel_topk_moe_t kernel_topk_moe<8>;
template [[host_name("kernel_topk_moe_16")]]  kernel kernel_topk_moe_t kernel_topk_moe<16>;
template [[host_name("kernel_topk_moe_32")]]  kernel kernel_topk_moe_t kernel_topk_moe<32>;
template [[host_name("kernel_topk_moe_64")]]  kernel kernel_topk_moe_t kernel_topk_moe<64>;
template [[host_name("kernel_topk_moe_128")]] kernel kernel_topk_moe_t kernel_topk_moe<128>;
template [[host_name("kernel_topk_moe_256")]] kernel kernel_topk_moe_t kernel_topk_moe<256>;

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

    const uint ncols = args.ne00;
    const uint top_k = args.top_k;
    const uint i01   = tgpig[0];
    const uint i02   = tgpig[1];
    const uint i03   = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    device int32_t * dst_row = dst + top_k*(i01 + args.ne01*i02 + args.ne01*args.ne02*i03);

    const uint tid = tpitg.x;
    const uint ntg_x = ntg.x;

    uint prefix  = 0;     // fixed high bits of the threshold key
    uint desired = top_k; // count still needed from the candidate range

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += ntg_x) {
            atomic_store_explicit(&histo[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hi_mask   = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << uint(shift + 8));
        const uint prefix_hi = prefix & hi_mask;

        for (uint i = tid; i < ncols; i += ntg_x) {
            const uint key = ggml_top_k_f2ui(src0_row[i]);
            if ((key & hi_mask) == prefix_hi) {
                atomic_fetch_add_explicit(&histo[(key >> uint(shift)) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // top-down scan for the bucket holding the k-th value
        if (tid == 0) {
            uint acc = 0;
            uint b   = 0;
            for (int bb = 255; bb >= 0; --bb) {
                const uint c = atomic_load_explicit(&histo[bb], memory_order_relaxed);
                if (acc + c >= desired) {
                    b = uint(bb);
                    break;
                }
                acc += c;
            }
            *sh_bucket = b;
            *sh_above  = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        prefix  |= *sh_bucket << uint(shift);
        desired -= *sh_above;

        // ensure every thread has consumed sh_bucket/sh_above before the next pass
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        atomic_store_explicit(out_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // emit everything above the threshold, then fill the rest from ties
    const uint threshold = prefix;

    for (uint i = tid; i < ncols; i += ntg_x) {
        if (ggml_top_k_f2ui(src0_row[i]) > threshold) {
            const uint pos = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            dst_row[pos] = (int32_t) i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < ncols; i += ntg_x) {
        if (ggml_top_k_f2ui(src0_row[i]) == threshold) {
            const uint pos = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            if (pos < top_k) {
                dst_row[pos] = (int32_t) i;
            }
        }
    }
}
