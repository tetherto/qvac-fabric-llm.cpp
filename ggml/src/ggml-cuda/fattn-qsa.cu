#include "cp-async.cuh"
#include "fattn-qsa.cuh"

#include <mma.h>

#include <cstdlib>
#include <cstring>

#if defined(__linux__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#    include <dlfcn.h>
#endif

namespace wmma = nvcuda::wmma;

namespace {

constexpr int   QSA_HEAD_DIM = 256;
constexpr int   QSA_GROUP    = 12;
constexpr int   QSA_BLOCK_M  = 16;
constexpr int   QSA_BLOCK_N  = 32;
constexpr int   QSA_D_TILES  = QSA_HEAD_DIM / 16;
constexpr float LOG2E_F      = 1.4426950408889634f;

constexpr int QSA_TRITON_SHARED = 24640;

constexpr int          QSA_TRITON_VARIANTS                          = 2;
constexpr int          QSA_TRITON_TOP_K[QSA_TRITON_VARIANTS]        = { 2048, 2051 };
constexpr const char * QSA_TRITON_CUBIN_SYMBOL[QSA_TRITON_VARIANTS] = {
    "qsa_ggml_c8f6d621_0d1d2d3d4d_cubin",
    "qsa_ggml_36391803_0d1d2d3d4d_cubin",
};

#if defined(__linux__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct qsa_triton_aot_state {
    void *       handle                                                = nullptr;
    const void * cubins[QSA_TRITON_VARIANTS]                           = {};
    CUmodule     modules[QSA_TRITON_VARIANTS][GGML_CUDA_MAX_DEVICES]   = {};
    CUfunction   functions[QSA_TRITON_VARIANTS][GGML_CUDA_MAX_DEVICES] = {};
    std::mutex   mutex;

    qsa_triton_aot_state() {
        const char * path = std::getenv("GGML_CUDA_QSA_TRITON_LIB");
        if (path == nullptr || path[0] == '\0') {
            return;
        }

        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            std::fprintf(stderr, "ggml_cuda: failed to load QSA Triton AOT library %s: %s\n", path, dlerror());
            return;
        }

        int found = 0;
        for (int variant = 0; variant < QSA_TRITON_VARIANTS; ++variant) {
            cubins[variant] = dlsym(handle, QSA_TRITON_CUBIN_SYMBOL[variant]);
            found += cubins[variant] != nullptr;
        }
        if (found == 0) {
            std::fprintf(stderr, "ggml_cuda: QSA Triton AOT library contains no supported cubin\n");
            return;
        }

        std::fprintf(stderr, "ggml_cuda: using Triton SM90 QSA AOT from %s (%s%s)\n", path,
                     cubins[0] != nullptr ? "top-k 2048" : "",
                     cubins[1] != nullptr ? (cubins[0] != nullptr ? ", top-k 2051" : "top-k 2051") : "");
    }
};

static qsa_triton_aot_state & qsa_triton_state() {
    static qsa_triton_aot_state state;
    return state;
}

static void qsa_cu_check(CUresult result, const char * operation) {
    if (result == CUDA_SUCCESS) {
        return;
    }

    const char * name = nullptr;
    const char * text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    GGML_ABORT("%s failed: %s (%s)", operation, name != nullptr ? name : "unknown", text != nullptr ? text : "unknown");
}

static int qsa_triton_variant(int n_kv_max) {
    for (int variant = 0; variant < QSA_TRITON_VARIANTS; ++variant) {
        if (QSA_TRITON_TOP_K[variant] == n_kv_max) {
            return variant;
        }
    }
    return -1;
}

static CUfunction qsa_triton_function(int device, int variant) {
    qsa_triton_aot_state & state = qsa_triton_state();
    if (variant < 0 || state.cubins[variant] == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.functions[variant][device] == nullptr) {
        qsa_cu_check(cuModuleLoadData(&state.modules[variant][device], state.cubins[variant]),
                     "cuModuleLoadData(qsa_ggml)");
        qsa_cu_check(cuModuleGetFunction(&state.functions[variant][device], state.modules[variant][device], "qsa_ggml"),
                     "cuModuleGetFunction(qsa_ggml)");
    }
    return state.functions[variant][device];
}
#endif

static __device__ __forceinline__ float qsa_warp_max(float value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, offset));
    }
    return value;
}

static __device__ __forceinline__ float qsa_exp2(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

// Qwen4-Exp QSA prefill: one warp owns one (query, KV head) pair. The 12 GQA
// heads are padded to the native 16-row MMA tile, while selected K/V rows are
// consumed 16 at a time. This mirrors the sparse-GQA schedule used by SGLang
// and, unlike the generic 8-head FlashAttention path, gathers each K/V tile
// only once for all 12 query heads.
__launch_bounds__(WARP_SIZE) static __global__ void qsa_sparse_gqa_f16(const char * __restrict__ q_ptr,
                                                                       const char * __restrict__ k_ptr,
                                                                       const char * __restrict__ v_ptr,
                                                                       const char * __restrict__ mask_ptr,
                                                                       const int * __restrict__ indices_ptr,
                                                                       float * __restrict__ dst,
                                                                       float   scale,
                                                                       int     n_queries,
                                                                       int     n_q_heads,
                                                                       int     n_kv,
                                                                       int     n_kv_heads,
                                                                       int     n_sequences,
                                                                       int     n_kv_max,
                                                                       int     mask_sequences,
                                                                       int     index_sequences,
                                                                       int64_t q_nb1,
                                                                       int64_t q_nb2,
                                                                       int64_t q_nb3,
                                                                       int64_t k_nb1,
                                                                       int64_t k_nb2,
                                                                       int64_t k_nb3,
                                                                       int64_t v_nb1,
                                                                       int64_t v_nb2,
                                                                       int64_t v_nb3,
                                                                       int64_t mask_nb1,
                                                                       int64_t mask_nb3,
                                                                       int64_t index_nb1,
                                                                       int64_t index_nb3) {
#if __CUDA_ARCH__ >= 700
    const int lane     = threadIdx.x;
    const int query    = blockIdx.x;
    const int sequence = blockIdx.y / n_kv_heads;
    const int kv_head  = blockIdx.y - sequence * n_kv_heads;
    if (query >= n_queries || sequence >= n_sequences) {
        return;
    }

    __shared__ __align__(16) half q_shared[QSA_BLOCK_M * QSA_HEAD_DIM];
    __shared__ __align__(16) half kv_shared[QSA_BLOCK_N * QSA_HEAD_DIM];
    __shared__ __align__(16) half p_shared[QSA_BLOCK_M * QSA_BLOCK_N];
    __shared__ __align__(16) float score_shared[QSA_BLOCK_M * QSA_BLOCK_N];
    __shared__ int                 selected[QSA_BLOCK_N];
    __shared__ float               bias_log2[QSA_BLOCK_N];
    __shared__ float               normalizer[QSA_GROUP];

    const int   q_head0      = kv_head * QSA_GROUP;
    const float q_scale_log2 = scale * LOG2E_F;
    for (int i = lane; i < QSA_BLOCK_M * QSA_HEAD_DIM; i += WARP_SIZE) {
        const int head  = i / QSA_HEAD_DIM;
        const int d     = i - head * QSA_HEAD_DIM;
        float     value = 0.0f;
        if (head < QSA_GROUP) {
            const float * q_row = reinterpret_cast<const float *>(
                q_ptr + int64_t(sequence) * q_nb3 + int64_t(q_head0 + head) * q_nb2 + int64_t(query) * q_nb1);
            value = q_row[d] * q_scale_log2;
        }
        q_shared[i] = __float2half_rn(value);
    }
    if (lane < QSA_GROUP) {
        normalizer[lane] = 0.0f;
    }
    __syncwarp();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag[QSA_D_TILES];
#    pragma unroll
    for (int tile = 0; tile < QSA_D_TILES; ++tile) {
        wmma::fill_fragment(out_frag[tile], 0.0f);
    }

    const int   index_sequence = sequence % index_sequences;
    const int   mask_sequence  = sequence % mask_sequences;
    const int * index_row      = reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(indices_ptr) + int64_t(index_sequence) * index_nb3 + int64_t(query) * index_nb1);
    const half * mask_row =
        reinterpret_cast<const half *>(mask_ptr + int64_t(mask_sequence) * mask_nb3 + int64_t(query) * mask_nb1);

    float running_max = -INFINITY;
    for (int start = 0; start < n_kv_max; start += QSA_BLOCK_N) {
        for (int col = lane; col < QSA_BLOCK_N; col += WARP_SIZE) {
            const int slot  = start + col;
            int       token = slot < n_kv_max ? index_row[slot] : -1;
            float     bias  = -INFINITY;
            if (token >= 0 && token < n_kv) {
                bias = __half2float(mask_row[token]);
                if (!isfinite(bias)) {
                    token = -1;
                }
            }
            selected[col]  = token;
            bias_log2[col] = bias * LOG2E_F;
        }
        __syncwarp();

        int4 * kv_shared_vec = reinterpret_cast<int4 *>(kv_shared);
#    pragma unroll
        for (int row = 0; row < QSA_BLOCK_N; ++row) {
            const int token   = selected[row];
            int4 *    dst_vec = kv_shared_vec + row * WARP_SIZE + lane;
            if (token >= 0) {
                const int4 * k_row = reinterpret_cast<const int4 *>(k_ptr + int64_t(sequence) * k_nb3 +
                                                                    int64_t(kv_head) * k_nb2 + int64_t(token) * k_nb1);
                cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(dst_vec), k_row + lane);
            } else {
                *dst_vec = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_wait_all();
        __syncwarp();

        for (int n0 = 0; n0 < QSA_BLOCK_N; n0 += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> qk_frag;
            wmma::fill_fragment(qk_frag, 0.0f);
#    pragma unroll
            for (int d0 = 0; d0 < QSA_HEAD_DIM; d0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
                wmma::load_matrix_sync(q_frag, q_shared + d0, QSA_HEAD_DIM);
                wmma::load_matrix_sync(k_frag, kv_shared + n0 * QSA_HEAD_DIM + d0, QSA_HEAD_DIM);
                wmma::mma_sync(qk_frag, q_frag, k_frag, qk_frag);
            }
            wmma::store_matrix_sync(score_shared + n0, qk_frag, QSA_BLOCK_N, wmma::mem_row_major);
        }
        __syncwarp();

        float tile_max = -INFINITY;
        for (int i = lane; i < QSA_GROUP * QSA_BLOCK_N; i += WARP_SIZE) {
            const int   col   = i % QSA_BLOCK_N;
            const float score = score_shared[i] + bias_log2[col];
            score_shared[i]   = score;
            tile_max          = fmaxf(tile_max, score);
        }
        tile_max             = qsa_warp_max(tile_max);
        const float next_max = fmaxf(running_max, tile_max);
        const float alpha =
            next_max == -INFINITY ? 1.0f : (running_max == -INFINITY ? 0.0f : qsa_exp2(running_max - next_max));

#    pragma unroll
        for (int tile = 0; tile < QSA_D_TILES; ++tile) {
#    pragma unroll
            for (int i = 0; i < out_frag[tile].num_elements; ++i) {
                out_frag[tile].x[i] *= alpha;
            }
        }

        for (int i = lane; i < QSA_BLOCK_M * QSA_BLOCK_N; i += WARP_SIZE) {
            const int head        = i / QSA_BLOCK_N;
            float     probability = 0.0f;
            if (head < QSA_GROUP && next_max != -INFINITY) {
                probability = qsa_exp2(score_shared[i] - next_max);
            }
            score_shared[i] = probability;
            p_shared[i]     = __float2half_rn(probability);
        }
        __syncwarp();

        if (lane < QSA_GROUP) {
            float row_sum = 0.0f;
#    pragma unroll
            for (int col = 0; col < QSA_BLOCK_N; ++col) {
                row_sum += score_shared[lane * QSA_BLOCK_N + col];
            }
            normalizer[lane] = normalizer[lane] * alpha + row_sum;
        }
        __syncwarp();

#    pragma unroll
        for (int row = 0; row < QSA_BLOCK_N; ++row) {
            const int token   = selected[row];
            int4 *    dst_vec = kv_shared_vec + row * WARP_SIZE + lane;
            if (token >= 0) {
                const int4 * v_row = reinterpret_cast<const int4 *>(v_ptr + int64_t(sequence) * v_nb3 +
                                                                    int64_t(kv_head) * v_nb2 + int64_t(token) * v_nb1);
                cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(dst_vec), v_row + lane);
            } else {
                *dst_vec = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_wait_all();
        __syncwarp();

#    pragma unroll
        for (int tile = 0; tile < QSA_D_TILES; ++tile) {
#    pragma unroll
            for (int n0 = 0; n0 < QSA_BLOCK_N; n0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, p_shared + n0, QSA_BLOCK_N);
                wmma::load_matrix_sync(v_frag, kv_shared + n0 * QSA_HEAD_DIM + tile * 16, QSA_HEAD_DIM);
                wmma::mma_sync(out_frag[tile], p_frag, v_frag, out_frag[tile]);
            }
        }
        running_max = next_max;
        __syncwarp();
    }

#    pragma unroll
    for (int tile = 0; tile < QSA_D_TILES; ++tile) {
        wmma::store_matrix_sync(score_shared, out_frag[tile], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < QSA_GROUP * 16; i += WARP_SIZE) {
            const int     head              = i / 16;
            const int     d                 = tile * 16 + i % 16;
            const float   denom             = normalizer[head];
            const int64_t dst_row           = (int64_t(sequence) * n_queries + query) * n_q_heads + q_head0 + head;
            dst[dst_row * QSA_HEAD_DIM + d] = denom > 0.0f ? score_shared[i] / denom : 0.0f;
        }
        __syncwarp();
    }
#else
    GGML_UNUSED_VARS(q_ptr, k_ptr, v_ptr, mask_ptr, indices_ptr, dst, scale, n_queries, n_q_heads, n_kv, n_kv_heads,
                     n_sequences, n_kv_max, mask_sequences, index_sequences, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3,
                     v_nb1, v_nb2, v_nb3, mask_nb1, mask_nb3, index_nb1, index_nb3);
#endif
}

static bool qsa_runtime_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_QSA_FATTN");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool qsa_triton_supported(const int device, const ggml_tensor * dst) {
#if defined(__linux__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (dst == nullptr || qsa_triton_state().handle == nullptr ||
        ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_HOPPER) {
        return false;
    }

    const char * block_top_k = std::getenv("LLAMA_QSA_BLOCK_TOPK");
    if (block_top_k == nullptr || block_top_k[0] == '\0' || std::strcmp(block_top_k, "0") == 0) {
        return false;
    }

    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sinks   = dst->src[4];
    const ggml_tensor * indices = dst->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr || indices == nullptr || sinks != nullptr) {
        return false;
    }

    float scale         = 0.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    const int     variant  = qsa_triton_variant(n_kv_max);

    return q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
           dst->type == GGML_TYPE_F32 && mask->type == GGML_TYPE_F16 && indices->type == GGML_TYPE_I32 &&
           scale == 0.0625f && max_bias == 0.0f && logit_softcap == 0.0f && variant >= 0 &&
           qsa_triton_state().cubins[variant] != nullptr && q->ne[0] == 256 && q->ne[1] >= 64 && q->ne[2] == 12 &&
           q->ne[3] == 1 && k->ne[0] == 256 && k->ne[1] > n_kv_max && k->ne[2] == 1 && k->ne[3] == 1 &&
           v->ne[0] == 256 && v->ne[1] == k->ne[1] && v->ne[2] == 1 && v->ne[3] == 1 && dst->ne[0] == 256 &&
           dst->ne[1] == 12 && dst->ne[2] == q->ne[1] && dst->ne[3] == 1 && indices->ne[0] == n_kv_max &&
           indices->ne[1] == q->ne[1] && indices->ne[2] == 1 && indices->ne[3] == 1 && q->nb[0] == sizeof(float) &&
           q->nb[1] == 3072 * sizeof(float) && q->nb[2] == 256 * sizeof(float) && k->nb[0] == sizeof(ggml_fp16_t) &&
           k->nb[1] == 256 * sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t) &&
           v->nb[1] == 256 * sizeof(ggml_fp16_t) && dst->nb[0] == sizeof(float) && dst->nb[1] == 256 * sizeof(float) &&
           dst->nb[2] == 3072 * sizeof(float) && indices->nb[0] == sizeof(int32_t) &&
           indices->nb[1] == n_kv_max * sizeof(int32_t);
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif
}

static void qsa_triton_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(__linux__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const int     device   = ggml_cuda_get_device();
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    const int     variant  = qsa_triton_variant(n_kv_max);
    CUfunction    function = qsa_triton_function(device, variant);
    GGML_ASSERT(function != nullptr);

    CUdeviceptr q               = (CUdeviceptr) dst->src[0]->data;
    CUdeviceptr k               = (CUdeviceptr) dst->src[1]->data;
    CUdeviceptr v               = (CUdeviceptr) dst->src[2]->data;
    CUdeviceptr out             = (CUdeviceptr) dst->data;
    CUdeviceptr indices         = (CUdeviceptr) dst->src[5]->data;
    int32_t     n_queries       = dst->src[0]->ne[1];
    CUdeviceptr global_scratch  = 0;
    CUdeviceptr profile_scratch = 0;
    void *      args[]          = { &q, &k, &v, &out, &indices, &n_queries, &global_scratch, &profile_scratch };

    qsa_cu_check(
        cuLaunchKernel(function, n_queries, 1, 1, 32, 1, 1, QSA_TRITON_SHARED, (CUstream) ctx.stream(), args, nullptr),
        "cuLaunchKernel(qsa_ggml)");

    static bool announced[GGML_CUDA_MAX_DEVICES] = {};
    if (!announced[device]) {
        announced[device] = true;
        std::fprintf(stderr, "ggml_cuda: launched Triton QSA AOT kernel on device %d (queries=%d, top-k=%d)\n", device,
                     n_queries, n_kv_max);
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("Triton QSA AOT is unavailable on this platform");
#endif
}

}  // namespace

bool ggml_cuda_flash_attn_ext_qsa_f16_supported(int device, const ggml_tensor * dst) {
    if (qsa_triton_supported(device, dst)) {
        return true;
    }

    if (!qsa_runtime_enabled() || ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_HOPPER || dst == nullptr) {
        return false;
    }

    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sinks   = dst->src[4];
    const ggml_tensor * indices = dst->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr || indices == nullptr || sinks != nullptr) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    return q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
           dst->type == GGML_TYPE_F32 && mask->type == GGML_TYPE_F16 && indices->type == GGML_TYPE_I32 &&
           q->ne[0] == QSA_HEAD_DIM && k->ne[0] == QSA_HEAD_DIM && v->ne[0] == QSA_HEAD_DIM && q->ne[1] >= 64 &&
           q->ne[2] == k->ne[2] * QSA_GROUP && k->ne[2] == v->ne[2] && q->ne[3] == k->ne[3] && q->ne[3] == v->ne[3] &&
           mask->ne[0] == k->ne[1] && mask->ne[1] >= q->ne[1] && mask->ne[2] == 1 && indices->ne[0] == n_kv_max &&
           indices->ne[1] == q->ne[1] && indices->ne[2] == 1 && q->ne[3] % mask->ne[3] == 0 &&
           q->ne[3] % indices->ne[3] == 0 && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
           q->nb[0] == sizeof(float) && k->nb[0] == sizeof(half) && v->nb[0] == sizeof(half) &&
           mask->nb[0] == sizeof(half) && indices->nb[0] == sizeof(int32_t) && ggml_is_contiguous(indices) &&
           ggml_is_contiguous(dst) && int64_t(k->ne[2]) * q->ne[3] <= 65535;
}

void ggml_cuda_flash_attn_ext_qsa_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (qsa_triton_supported(ggml_cuda_get_device(), dst)) {
        qsa_triton_launch(ctx, dst);
        return;
    }

    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * indices = dst->src[5];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    const dim3                           blocks(q->ne[1], k->ne[2] * q->ne[3], 1);
    const dim3                           threads(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(
        qsa_sparse_gqa_f16, launch_params, (const char *) q->data, (const char *) k->data, (const char *) v->data,
        (const char *) mask->data, (const int *) indices->data, (float *) dst->data, scale, int(q->ne[1]),
        int(q->ne[2]), int(k->ne[1]), int(k->ne[2]), int(q->ne[3]), n_kv_max, int(mask->ne[3]), int(indices->ne[3]),
        int64_t(q->nb[1]), int64_t(q->nb[2]), int64_t(q->nb[3]), int64_t(k->nb[1]), int64_t(k->nb[2]),
        int64_t(k->nb[3]), int64_t(v->nb[1]), int64_t(v->nb[2]), int64_t(v->nb[3]), int64_t(mask->nb[1]),
        int64_t(mask->nb[3]), int64_t(indices->nb[1]), int64_t(indices->nb[3]));
    CUDA_CHECK(cudaGetLastError());
}
