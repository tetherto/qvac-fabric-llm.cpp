#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace wmma = nvcuda::wmma;

// FLA-style WY chunking for the Qwen prefill shape. A block owns 32 value
// columns for one (sequence, head), keeps that state tile resident, and advances
// it 32 tokens at a time. BF16 tensor-core inputs with FP32 accumulation mirror
// FLA's inference path while the original recurrent kernel remains the exact
// fallback.
static constexpr int GDN_CHUNK_S  = 128;
static constexpr int GDN_CHUNK_BT = 32;
static constexpr int GDN_CHUNK_BV = 32;
static constexpr int GDN_CHUNK_THREADS = 256;

using gdn_flashinfer_aot_launch_t = int (*)(
    const void *, const void *, const void *, void *,
    const float *, const float *, float *, const float *,
    void *, const int64_t *, int32_t, int32_t, int32_t, int32_t, int32_t,
    cudaStream_t);

static gdn_flashinfer_aot_launch_t get_gdn_flashinfer_aot_launch() {
#if defined(__linux__)
    static const gdn_flashinfer_aot_launch_t launch = [] {
        const char * path = std::getenv("GGML_CUDA_GDN_AOT_LIB");
        if (path == nullptr || path[0] == '\0') {
            return (gdn_flashinfer_aot_launch_t) nullptr;
        }

        void * handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            std::fprintf(stderr, "ggml_cuda: failed to load GDN AOT library %s: %s\n", path, dlerror());
            return (gdn_flashinfer_aot_launch_t) nullptr;
        }

        void * symbol = dlsym(handle, "flashinfer_gdn_sm90_aot_launch");
        if (symbol == nullptr) {
            std::fprintf(stderr, "ggml_cuda: GDN AOT library is missing flashinfer_gdn_sm90_aot_launch: %s\n", dlerror());
            return (gdn_flashinfer_aot_launch_t) nullptr;
        }

        std::fprintf(stderr, "ggml_cuda: using FlashInfer SM90 AOT GDN from %s\n", path);
        return reinterpret_cast<gdn_flashinfer_aot_launch_t>(symbol);
    }();
    return launch;
#else
    return nullptr;
#endif
}

__global__ void gated_delta_net_flashinfer_prepare_cuda(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        __nv_bfloat16 * q_bf16,
        __nv_bfloat16 * k_bf16,
        __nv_bfloat16 * v_bf16,
        float * alpha,
        float * beta_packed,
        int64_t S,
        int64_t H_q,
        int64_t H_v,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sv1,
        int64_t sv2,
        int64_t sv3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        int64_t rq3) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_seqs * n_tokens * H_v * S;
    if (idx >= n) {
        return;
    }

    const int64_t d = idx % S;
    const int64_t ih = (idx / S) % H_v;
    const int64_t it = (idx / (S * H_v)) % n_tokens;
    const int64_t is = idx / (S * H_v * n_tokens);

    v_bf16[idx] = __float2bfloat16_rn(v[is * sv3 + it * sv2 + ih * sv1 + d]);
    if (d == 0) {
        const int64_t gate_idx = is * n_tokens * H_v + it * H_v + ih;
        const int64_t src_gate_idx = is * sb3 + it * sb2 + ih * sb1;
        alpha[gate_idx] = expf(g[src_gate_idx]);
        beta_packed[gate_idx] = beta[src_gate_idx];
    }

    // Preserve GGML's existing grouped-value head mapping (h_v % H_q).
    // FlashInfer's native GVA convention groups adjacent value heads instead,
    // so materialize equal-head Q/K in the transient BF16 buffers.
    const int64_t iq3 = is / rq3;
    const int64_t iq1 = ih % H_q;
    const int64_t q_idx = ((is * n_tokens + it) * H_v + ih) * S + d;
    const int64_t src_idx = iq3 * sq3 + it * sq2 + iq1 * sq1 + d;
    q_bf16[q_idx] = __float2bfloat16_rn(q[src_idx]);
    k_bf16[q_idx] = __float2bfloat16_rn(k[src_idx]);
}

__global__ void gated_delta_net_flashinfer_unpack_cuda(
        const __nv_bfloat16 * src, float * dst, int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

__global__ void gated_delta_net_flashinfer_cu_seqlens_cuda(
        int64_t * cu_seqlens, int64_t n_tokens, int64_t n_seqs) {
    const int64_t idx = threadIdx.x;
    if (idx <= n_seqs) {
        cu_seqlens[idx] = idx * n_tokens;
    }
}

static bool launch_gated_delta_net_flashinfer_aot(
        ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t H_q, int64_t rq3, cudaStream_t stream) {
    const gdn_flashinfer_aot_launch_t launch = get_gdn_flashinfer_aot_launch();
    const int device = ggml_cuda_get_device();
    const auto & info = ggml_cuda_info().devices[device];
    if (launch == nullptr || info.cc != GGML_CUDA_CC_HOPPER || S_v != 128 ||
        n_tokens < 64 || H_q <= 0 || H % H_q != 0 || n_seqs > 1023 ||
        n_tokens * n_seqs > INT32_MAX || H > INT32_MAX || H_q > INT32_MAX) {
        return false;
    }

    const size_t v_elements = (size_t) n_seqs * n_tokens * H * S_v;
    const size_t q_elements = v_elements;
    const size_t gate_elements = (size_t) n_seqs * n_tokens * H;
    const size_t tensormaps_bytes = (size_t) info.nsm * 128;

    ggml_cuda_pool_alloc<__nv_bfloat16> q_bf16(ctx.pool(), q_elements);
    ggml_cuda_pool_alloc<__nv_bfloat16> k_bf16(ctx.pool(), q_elements);
    ggml_cuda_pool_alloc<__nv_bfloat16> v_bf16(ctx.pool(), v_elements);
    ggml_cuda_pool_alloc<__nv_bfloat16> out_bf16(ctx.pool(), v_elements);
    ggml_cuda_pool_alloc<float> alpha(ctx.pool(), gate_elements);
    ggml_cuda_pool_alloc<float> beta(ctx.pool(), gate_elements);
    ggml_cuda_pool_alloc<int64_t> cu_seqlens(ctx.pool(), n_seqs + 1);
    ggml_cuda_pool_alloc<uint8_t> tensormaps(ctx.pool(), tensormaps_bytes);

    const int threads = 256;
    ggml_cuda_kernel_launch_params prepare_params(
        dim3((v_elements + threads - 1) / threads, 1, 1), dim3(threads, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_flashinfer_prepare_cuda, prepare_params,
        q_d, k_d, v_d, g_d, b_d,
        q_bf16.get(), k_bf16.get(), v_bf16.get(), alpha.get(), beta.get(),
        S_v, H_q, H, n_tokens, n_seqs,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, rq3);

    ggml_cuda_kernel_launch_params cu_params(
        dim3(1, 1, 1), dim3(n_seqs + 1, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_flashinfer_cu_seqlens_cuda, cu_params,
        cu_seqlens.get(), n_tokens, n_seqs);

    const int rc = launch(
        q_bf16.get(), k_bf16.get(), v_bf16.get(), out_bf16.get(),
        alpha.get(), beta.get(), state_d, s_d, tensormaps.get(), cu_seqlens.get(),
        (int32_t) (n_tokens * n_seqs), (int32_t) H, (int32_t) H, (int32_t) n_seqs,
        (int32_t) tensormaps_bytes, stream);
    if (rc != 0) {
        std::fprintf(stderr, "ggml_cuda: FlashInfer SM90 AOT GDN launch failed with status %d\n", rc);
        return false;
    }

    ggml_cuda_kernel_launch_params unpack_params(
        dim3((v_elements + threads - 1) / threads, 1, 1), dim3(threads, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_flashinfer_unpack_cuda, unpack_params,
        out_bf16.get(), dst_d, (int64_t) v_elements);
    return true;
}

struct __align__(16) gdn_wy_smem {
    __nv_bfloat16 k[GDN_CHUNK_BT * GDN_CHUNK_S];
    __nv_bfloat16 w[GDN_CHUNK_BT * GDN_CHUNK_S];
    float w_f32[GDN_CHUNK_BT * GDN_CHUNK_S];
    float a[GDN_CHUNK_BT * GDN_CHUNK_BT];
    float ai_f32[GDN_CHUNK_BT * GDN_CHUNK_BT];
    __nv_bfloat16 ai[GDN_CHUNK_BT * GDN_CHUNK_BT];
    float gcum[GDN_CHUNK_BT];
    float beta[GDN_CHUNK_BT];
};

__global__ void __launch_bounds__(GDN_CHUNK_THREADS, 2)
gated_delta_net_chunked_wy_cuda(
        const float * k,
        const float * g,
        const float * beta,
        __nv_bfloat16 * w,
        __nv_bfloat16 * ai,
        int64_t H,
        int64_t n_tokens,
        int64_t n_chunks,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        int64_t neqk1,
        int64_t rq3) {
#if defined(AMPERE_MMA_AVAILABLE)
    __shared__ gdn_wy_smem sm;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int chunk    = blockIdx.x;
    const int h_idx    = blockIdx.y;
    const int sequence = blockIdx.z;
    const int token0   = chunk * GDN_CHUNK_BT;
    const int chunk_len = min(GDN_CHUNK_BT, (int) n_tokens - token0);
    const int iq1 = h_idx % neqk1;
    const int iq3 = sequence / rq3;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * g_base = g + sequence * sb3 + h_idx * sb1;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;

    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_S; idx += GDN_CHUNK_THREADS) {
        const int t = idx / GDN_CHUNK_S;
        const int d = idx % GDN_CHUNK_S;
        const float x = t < chunk_len ? k_base[(token0 + t) * sq2 + d] : 0.0f;
        sm.k[idx] = __float2bfloat16_rn(x);
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (int t = 0; t < GDN_CHUNK_BT; ++t) {
            if (t < chunk_len) {
                sum += g_base[(token0 + t) * sb2];
                sm.beta[t] = b_base[(token0 + t) * sb2];
            } else {
                sm.beta[t] = 0.0f;
            }
            sm.gcum[t] = sum;
        }
    }
    __syncthreads();

    constexpr int token_tiles = GDN_CHUNK_BT / 16;
    for (int tile = warp; tile < token_tiles * token_tiles; tile += GDN_CHUNK_THREADS / WARP_SIZE) {
        const int tile_i = tile / token_tiles;
        const int tile_j = tile % token_tiles;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int d = 0; d < GDN_CHUNK_S; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> mb;
            wmma::load_matrix_sync(ma, sm.k + tile_i * 16 * GDN_CHUNK_S + d, GDN_CHUNK_S);
            wmma::load_matrix_sync(mb, sm.k + tile_j * 16 * GDN_CHUNK_S + d, GDN_CHUNK_S);
            wmma::mma_sync(acc, ma, mb, acc);
        }
        wmma::store_matrix_sync(
            sm.a + tile_i * 16 * GDN_CHUNK_BT + tile_j * 16,
            acc, GDN_CHUNK_BT, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BT; idx += GDN_CHUNK_THREADS) {
        const int i = idx / GDN_CHUNK_BT;
        const int j = idx % GDN_CHUNK_BT;
        sm.a[idx] = (i < chunk_len && j < i)
            ? sm.a[idx] * sm.beta[i] * expf(sm.gcum[i] - sm.gcum[j])
            : 0.0f;
    }
    __syncthreads();
    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_S; idx += GDN_CHUNK_THREADS) {
        const int t = idx / GDN_CHUNK_S;
        sm.w[idx] = __float2bfloat16_rn(
            __bfloat162float(sm.k[idx]) * sm.beta[t] * expf(sm.gcum[t]));
    }
    if (tid == 0) {
        for (int idx = 0; idx < GDN_CHUNK_BT * GDN_CHUNK_BT; ++idx) {
            sm.ai_f32[idx] = 0.0f;
        }
        // Invert the two diagonal 16x16 blocks independently. The remaining
        // lower-left block is -I11*A10*I00 and is formed by the tensor cores
        // below. This is the two-block specialization of FLA's block-WY solve.
        for (int block = 0; block < 2; ++block) {
            const int begin = block * 16;
            const int end = min(begin + 16, chunk_len);
            for (int i = begin; i < end; ++i) {
                sm.ai_f32[i * GDN_CHUNK_BT + i] = 1.0f;
                for (int j = begin; j < i; ++j) {
                    float x = 0.0f;
                    for (int l = j; l < i; ++l) {
                        x += sm.a[i * GDN_CHUNK_BT + l] * sm.ai_f32[l * GDN_CHUNK_BT + j];
                    }
                    sm.ai_f32[i * GDN_CHUNK_BT + j] = -x;
                }
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BT; idx += GDN_CHUNK_THREADS) {
        sm.ai[idx] = __float2bfloat16_rn(sm.ai_f32[idx]);
        // K is no longer needed after the weighted copy above, so reuse its
        // first 32x32 elements as a BF16 view of A for the block products.
        sm.k[idx] = __float2bfloat16_rn(sm.a[idx]);
    }
    __syncthreads();

    if (warp == 0) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
        wmma::fill_fragment(acc, 0.0f);
        wmma::load_matrix_sync(ma, sm.ai + 16 * GDN_CHUNK_BT + 16, GDN_CHUNK_BT);
        wmma::load_matrix_sync(mb, sm.k + 16 * GDN_CHUNK_BT, GDN_CHUNK_BT);
        wmma::mma_sync(acc, ma, mb, acc);
        wmma::store_matrix_sync(sm.w_f32, acc, 16, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < 16 * 16; idx += GDN_CHUNK_THREADS) {
        sm.k[GDN_CHUNK_BT * GDN_CHUNK_BT + idx] = __float2bfloat16_rn(sm.w_f32[idx]);
    }
    __syncthreads();
    if (warp == 0) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
        wmma::fill_fragment(acc, 0.0f);
        wmma::load_matrix_sync(ma, sm.k + GDN_CHUNK_BT * GDN_CHUNK_BT, 16);
        wmma::load_matrix_sync(mb, sm.ai, GDN_CHUNK_BT);
        wmma::mma_sync(acc, ma, mb, acc);
        wmma::store_matrix_sync(sm.a, acc, 16, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < 16 * 16; idx += GDN_CHUNK_THREADS) {
        const int i = idx / 16;
        const int j = idx % 16;
        sm.ai[(16 + i) * GDN_CHUNK_BT + j] = __float2bfloat16_rn(-sm.a[idx]);
    }
    __syncthreads();

    constexpr int w_tiles = token_tiles * (GDN_CHUNK_S / 16);
    for (int tile = warp; tile < w_tiles; tile += GDN_CHUNK_THREADS / WARP_SIZE) {
        const int tile_i = tile / (GDN_CHUNK_S / 16);
        const int tile_j = tile % (GDN_CHUNK_S / 16);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int d = 0; d < GDN_CHUNK_BT; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
            wmma::load_matrix_sync(ma, sm.ai + tile_i * 16 * GDN_CHUNK_BT + d, GDN_CHUNK_BT);
            wmma::load_matrix_sync(mb, sm.w + d * GDN_CHUNK_S + tile_j * 16, GDN_CHUNK_S);
            wmma::mma_sync(acc, ma, mb, acc);
        }
        wmma::store_matrix_sync(
            sm.w_f32 + tile_i * 16 * GDN_CHUNK_S + tile_j * 16,
            acc, GDN_CHUNK_S, wmma::mem_row_major);
    }
    __syncthreads();

    const size_t chunk_base =
        (((size_t) sequence * H + h_idx) * n_chunks + chunk) * GDN_CHUNK_BT * GDN_CHUNK_S;
    const size_t ai_base =
        (((size_t) sequence * H + h_idx) * n_chunks + chunk) * GDN_CHUNK_BT * GDN_CHUNK_BT;
    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_S; idx += GDN_CHUNK_THREADS) {
        w[chunk_base + idx] = __float2bfloat16_rn(sm.w_f32[idx]);
    }
    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BT; idx += GDN_CHUNK_THREADS) {
        ai[ai_base + idx] = sm.ai[idx];
    }
#else
    GGML_UNUSED_VARS(k, g, beta, w, ai, H, n_tokens, n_chunks, sq1, sq2, sq3,
                     sb1, sb2, sb3, neqk1, rq3);
#endif
}

struct __align__(16) gdn_u_smem {
    __nv_bfloat16 v[GDN_CHUNK_BT * GDN_CHUNK_BV];
    float out[GDN_CHUNK_BT * GDN_CHUNK_BV];
};

__global__ void __launch_bounds__(GDN_CHUNK_THREADS, 2)
gated_delta_net_chunked_u_cuda(
        const float * v,
        const float * beta,
        const __nv_bfloat16 * ai,
        __nv_bfloat16 * u,
        int64_t H,
        int64_t n_tokens,
        int64_t n_chunks,
        int64_t sv1,
        int64_t sv2,
        int64_t sv3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3) {
#if defined(AMPERE_MMA_AVAILABLE)
    __shared__ gdn_u_smem sm;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int chunk    = blockIdx.x;
    const int h_idx    = blockIdx.y;
    const int sequence = blockIdx.z / (GDN_CHUNK_S / GDN_CHUNK_BV);
    const int v_tile   = blockIdx.z % (GDN_CHUNK_S / GDN_CHUNK_BV);
    const int col_base = v_tile * GDN_CHUNK_BV;
    const int token0   = chunk * GDN_CHUNK_BT;
    const int chunk_len = min(GDN_CHUNK_BT, (int) n_tokens - token0);
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;

    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
        const int t = idx / GDN_CHUNK_BV;
        const int c = idx % GDN_CHUNK_BV + col_base;
        const float x = t < chunk_len
            ? v_base[(token0 + t) * sv2 + c] * b_base[(token0 + t) * sb2]
            : 0.0f;
        sm.v[idx] = __float2bfloat16_rn(x);
    }
    __syncthreads();

    const size_t ai_base =
        (((size_t) sequence * H + h_idx) * n_chunks + chunk) * GDN_CHUNK_BT * GDN_CHUNK_BT;
    constexpr int u_tiles = (GDN_CHUNK_BT / 16) * (GDN_CHUNK_BV / 16);
    for (int tile = warp; tile < u_tiles; tile += GDN_CHUNK_THREADS / WARP_SIZE) {
        const int tile_i = tile / (GDN_CHUNK_BV / 16);
        const int tile_j = tile % (GDN_CHUNK_BV / 16);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int d = 0; d < GDN_CHUNK_BT; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
            wmma::load_matrix_sync(ma, ai + ai_base + tile_i * 16 * GDN_CHUNK_BT + d, GDN_CHUNK_BT);
            wmma::load_matrix_sync(mb, sm.v + d * GDN_CHUNK_BV + tile_j * 16, GDN_CHUNK_BV);
            wmma::mma_sync(acc, ma, mb, acc);
        }
        wmma::store_matrix_sync(
            sm.out + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
            acc, GDN_CHUNK_BV, wmma::mem_row_major);
    }
    __syncthreads();

    const size_t chunk_base =
        (((size_t) sequence * H + h_idx) * n_chunks + chunk) * GDN_CHUNK_BT * GDN_CHUNK_S;
    for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
        const int t = idx / GDN_CHUNK_BV;
        const int c = idx % GDN_CHUNK_BV + col_base;
        u[chunk_base + t * GDN_CHUNK_S + c] = __float2bfloat16_rn(sm.out[idx]);
    }
#else
    GGML_UNUSED_VARS(v, beta, ai, u, H, n_tokens, n_chunks, sv1, sv2, sv3, sb1, sb2, sb3);
#endif
}

struct __align__(16) gdn_state_smem {
    union {
        float state_f32[GDN_CHUNK_S * GDN_CHUNK_BV];
        __nv_bfloat16 q[GDN_CHUNK_BT * GDN_CHUNK_S];
    };
    __nv_bfloat16 state_bf16[GDN_CHUNK_S * GDN_CHUNK_BV];
    __nv_bfloat16 r[GDN_CHUNK_BT * GDN_CHUNK_BV];
    __nv_bfloat16 ks[GDN_CHUNK_BT * GDN_CHUNK_S];
    float tmp[GDN_CHUNK_BT * GDN_CHUNK_BV];
    float qk[GDN_CHUNK_BT * GDN_CHUNK_BT];
    float gcum[GDN_CHUNK_BT];
};

__global__ void __launch_bounds__(GDN_CHUNK_THREADS, 1)
gated_delta_net_chunked_state_cuda(
        const float * q,
        const float * k,
        const float * g,
        const float * curr_state,
        const __nv_bfloat16 * w,
        __nv_bfloat16 * r,
        float * dst,
        float * state,
        int64_t H,
        int64_t n_tokens,
        int64_t n_chunks,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        int64_t neqk1,
        int64_t rq3,
        float scale) {
#if defined(AMPERE_MMA_AVAILABLE)
    __shared__ gdn_state_smem sm;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int h_idx    = blockIdx.x;
    const int sequence = blockIdx.y;
    const int v_tile   = blockIdx.z;
    const int col_base = v_tile * GDN_CHUNK_BV;
    const int iq1 = h_idx % neqk1;
    const int iq3 = sequence / rq3;
    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * g_base = g + sequence * sb3 + h_idx * sb1;
    const int64_t state_offset = (sequence * H + h_idx) * GDN_CHUNK_S * GDN_CHUNK_S;
    curr_state += state_offset;
    state      += state_offset;

    for (int idx = tid; idx < GDN_CHUNK_S * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
        const int row = idx / GDN_CHUNK_BV;
        const int col = idx % GDN_CHUNK_BV + col_base;
        const float x = curr_state[col * GDN_CHUNK_S + row];
        sm.state_f32[idx] = x;
    }
    __syncthreads();

    // Each warp owns two 16x16 tiles of the 128x32 state. Keep their FP32
    // accumulators live across the chunk recurrence, as FLA does, instead of
    // round-tripping the state through shared memory after every update.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> state_acc[2];
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        const int tile = warp + slot * (GDN_CHUNK_THREADS / WARP_SIZE);
        const int tile_i = tile / (GDN_CHUNK_BV / 16);
        const int tile_j = tile % (GDN_CHUNK_BV / 16);
        wmma::load_matrix_sync(
            state_acc[slot], sm.state_f32 + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
            GDN_CHUNK_BV, wmma::mem_row_major);
    }

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int token0 = chunk * GDN_CHUNK_BT;
        const int chunk_len = min(GDN_CHUNK_BT, (int) n_tokens - token0);
        const size_t chunk_base =
            (((size_t) sequence * H + h_idx) * n_chunks + chunk) * GDN_CHUNK_BT * GDN_CHUNK_S;

        // Materialize the register state only for the BF16 snapshot consumed
        // by W*H and by the output kernel.
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const int tile = warp + slot * (GDN_CHUNK_THREADS / WARP_SIZE);
            const int tile_i = tile / (GDN_CHUNK_BV / 16);
            const int tile_j = tile % (GDN_CHUNK_BV / 16);
            wmma::store_matrix_sync(
                sm.state_f32 + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
                state_acc[slot], GDN_CHUNK_BV, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < GDN_CHUNK_S * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
            const __nv_bfloat16 x = __float2bfloat16_rn(sm.state_f32[idx]);
            sm.state_bf16[idx] = x;
        }
        __syncthreads();
        constexpr int r_tiles = (GDN_CHUNK_BT / 16) * (GDN_CHUNK_BV / 16);
        if (warp < r_tiles) {
            const int tile_i = warp / (GDN_CHUNK_BV / 16);
            const int tile_j = warp % (GDN_CHUNK_BV / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int d = 0; d < GDN_CHUNK_S; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
                wmma::load_matrix_sync(
                    ma, w + chunk_base + tile_i * 16 * GDN_CHUNK_S + d, GDN_CHUNK_S);
                wmma::load_matrix_sync(
                    mb, sm.state_bf16 + d * GDN_CHUNK_BV + tile_j * 16, GDN_CHUNK_BV);
                wmma::mma_sync(acc, ma, mb, acc);
            }
            wmma::store_matrix_sync(
                sm.tmp + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
                acc, GDN_CHUNK_BV, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
            const int t = idx / GDN_CHUNK_BV;
            const int c = idx % GDN_CHUNK_BV + col_base;
            const float x = __bfloat162float(r[chunk_base + t * GDN_CHUNK_S + c]) - sm.tmp[idx];
            sm.r[idx] = __float2bfloat16_rn(x);
            r[chunk_base + t * GDN_CHUNK_S + c] = sm.r[idx];
        }
        if (tid == 0) {
            float sum = 0.0f;
            for (int t = 0; t < GDN_CHUNK_BT; ++t) {
                if (t < chunk_len) {
                    sum += g_base[(token0 + t) * sb2];
                }
                sm.gcum[t] = sum;
            }
        }
        __syncthreads();

        // Produce this chunk's output while H_start and R are resident. The
        // state_f32/q union is safe here because the state lives in WMMA
        // accumulator registers until the next chunk boundary.
        for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_S; idx += GDN_CHUNK_THREADS) {
            const int t = idx / GDN_CHUNK_S;
            const int d = idx % GDN_CHUNK_S;
            sm.q[idx] = __float2bfloat16_rn(
                t < chunk_len ? q_base[(token0 + t) * sq2 + d] : 0.0f);
            sm.ks[idx] = __float2bfloat16_rn(
                t < chunk_len ? k_base[(token0 + t) * sq2 + d] : 0.0f);
        }
        __syncthreads();
        constexpr int out_tiles = (GDN_CHUNK_BT / 16) * (GDN_CHUNK_BV / 16);
        if (warp < out_tiles) {
            const int tile_i = warp / (GDN_CHUNK_BV / 16);
            const int tile_j = warp % (GDN_CHUNK_BV / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int d = 0; d < GDN_CHUNK_S; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
                wmma::load_matrix_sync(
                    ma, sm.q + tile_i * 16 * GDN_CHUNK_S + d, GDN_CHUNK_S);
                wmma::load_matrix_sync(
                    mb, sm.state_bf16 + d * GDN_CHUNK_BV + tile_j * 16, GDN_CHUNK_BV);
                wmma::mma_sync(acc, ma, mb, acc);
            }
            wmma::store_matrix_sync(
                sm.tmp + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
                acc, GDN_CHUNK_BV, wmma::mem_row_major);
        } else {
            const int tile = warp - out_tiles;
            const int tile_i = tile / (GDN_CHUNK_BT / 16);
            const int tile_j = tile % (GDN_CHUNK_BT / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int d = 0; d < GDN_CHUNK_S; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> mb;
                wmma::load_matrix_sync(
                    ma, sm.q + tile_i * 16 * GDN_CHUNK_S + d, GDN_CHUNK_S);
                wmma::load_matrix_sync(
                    mb, sm.ks + tile_j * 16 * GDN_CHUNK_S + d, GDN_CHUNK_S);
                wmma::mma_sync(acc, ma, mb, acc);
            }
            wmma::store_matrix_sync(
                sm.qk + tile_i * 16 * GDN_CHUNK_BT + tile_j * 16,
                acc, GDN_CHUNK_BT, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
            sm.tmp[idx] *= expf(sm.gcum[idx / GDN_CHUNK_BV]);
        }
        for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_BT; idx += GDN_CHUNK_THREADS) {
            const int i = idx / GDN_CHUNK_BT;
            const int j = idx % GDN_CHUNK_BT;
            const float x = (i < chunk_len && j <= i)
                ? sm.qk[idx] * expf(sm.gcum[i] - sm.gcum[j])
                : 0.0f;
            sm.ks[idx] = __float2bfloat16_rn(x);
        }
        __syncthreads();
        if (warp < out_tiles) {
            const int tile_i = warp / (GDN_CHUNK_BV / 16);
            const int tile_j = warp % (GDN_CHUNK_BV / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(
                acc, sm.tmp + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
                GDN_CHUNK_BV, wmma::mem_row_major);
#pragma unroll
            for (int d = 0; d < GDN_CHUNK_BT; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ma;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
                wmma::load_matrix_sync(
                    ma, sm.ks + tile_i * 16 * GDN_CHUNK_BT + d, GDN_CHUNK_BT);
                wmma::load_matrix_sync(
                    mb, sm.r + d * GDN_CHUNK_BV + tile_j * 16, GDN_CHUNK_BV);
                wmma::mma_sync(acc, ma, mb, acc);
            }
            wmma::store_matrix_sync(
                sm.tmp + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
                acc, GDN_CHUNK_BV, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < chunk_len * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
            const int t = idx / GDN_CHUNK_BV;
            const int c = idx % GDN_CHUNK_BV + col_base;
            dst[((size_t) sequence * n_tokens + token0 + t) * H * GDN_CHUNK_S +
                h_idx * GDN_CHUNK_S + c] = sm.tmp[idx] * scale;
        }

        const float decay = expf(sm.gcum[chunk_len - 1]);
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
#pragma unroll
            for (int e = 0; e < state_acc[slot].num_elements; ++e) {
                state_acc[slot].x[e] *= decay;
            }
        }
        for (int idx = tid; idx < GDN_CHUNK_BT * GDN_CHUNK_S; idx += GDN_CHUNK_THREADS) {
            const int t = idx / GDN_CHUNK_S;
            const int d = idx % GDN_CHUNK_S;
            const float x = t < chunk_len
                ? k_base[(token0 + t) * sq2 + d] * expf(sm.gcum[chunk_len - 1] - sm.gcum[t])
                : 0.0f;
            sm.ks[idx] = __float2bfloat16_rn(x);
        }
        __syncthreads();
        // Update the two resident state tiles directly; no shared-memory spill
        // or BF16 reconversion is needed before the next chunk.
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const int tile = warp + slot * (GDN_CHUNK_THREADS / WARP_SIZE);
            const int tile_i = tile / (GDN_CHUNK_BV / 16);
            const int tile_j = tile % (GDN_CHUNK_BV / 16);
#pragma unroll
            for (int d = 0; d < GDN_CHUNK_BT; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ma;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> mb;
                wmma::load_matrix_sync(
                    ma, sm.ks + d * GDN_CHUNK_S + tile_i * 16, GDN_CHUNK_S);
                wmma::load_matrix_sync(
                    mb, sm.r + d * GDN_CHUNK_BV + tile_j * 16, GDN_CHUNK_BV);
                wmma::mma_sync(state_acc[slot], ma, mb, state_acc[slot]);
            }
        }
    }
    // Store the final FP32 state in GGML's transposed cache layout.
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        const int tile = warp + slot * (GDN_CHUNK_THREADS / WARP_SIZE);
        const int tile_i = tile / (GDN_CHUNK_BV / 16);
        const int tile_j = tile % (GDN_CHUNK_BV / 16);
        wmma::store_matrix_sync(
            sm.state_f32 + tile_i * 16 * GDN_CHUNK_BV + tile_j * 16,
            state_acc[slot], GDN_CHUNK_BV, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < GDN_CHUNK_S * GDN_CHUNK_BV; idx += GDN_CHUNK_THREADS) {
        const int row = idx / GDN_CHUNK_BV;
        const int col = idx % GDN_CHUNK_BV + col_base;
        state[col * GDN_CHUNK_S + row] = sm.state_f32[idx];
    }
#else
    GGML_UNUSED_VARS(q, k, g, curr_state, w, r, dst, state, H, n_tokens, n_chunks,
                     sq1, sq2, sq3, sb1, sb2, sb3, neqk1, rq3, scale);
#endif
}

static bool launch_gated_delta_net_chunked_parallel(
        ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, cudaStream_t stream) {
    static const int mode = [] {
        const char * value = std::getenv("GGML_CUDA_GDN_CHUNKED");
        return value != nullptr ? std::atoi(value) : 0;
    }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (mode != 2 || S_v != GDN_CHUNK_S || n_tokens < GDN_CHUNK_BT ||
        !GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }

    const int64_t n_chunks = (n_tokens + GDN_CHUNK_BT - 1) / GDN_CHUNK_BT;
    const size_t chunk_elements = (size_t) n_seqs * H * n_chunks * GDN_CHUNK_BT * GDN_CHUNK_S;
    const size_t ai_elements = (size_t) n_seqs * H * n_chunks * GDN_CHUNK_BT * GDN_CHUNK_BT;
    ggml_cuda_pool_alloc<__nv_bfloat16> w_buf(ctx.pool(), chunk_elements);
    ggml_cuda_pool_alloc<__nv_bfloat16> r_buf(ctx.pool(), chunk_elements);
    ggml_cuda_pool_alloc<__nv_bfloat16> ai_buf(ctx.pool(), ai_elements);

    ggml_cuda_kernel_launch_params wy_params(
        dim3(n_chunks, H, n_seqs), dim3(GDN_CHUNK_THREADS, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_chunked_wy_cuda, wy_params,
        k_d, g_d, b_d, w_buf.get(), ai_buf.get(), H, n_tokens, n_chunks,
        sq1, sq2, sq3, sb1, sb2, sb3, neqk1, rq3);

    ggml_cuda_kernel_launch_params u_params(
        dim3(n_chunks, H, n_seqs * (GDN_CHUNK_S / GDN_CHUNK_BV)),
        dim3(GDN_CHUNK_THREADS, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_chunked_u_cuda, u_params,
        v_d, b_d, ai_buf.get(), r_buf.get(), H, n_tokens, n_chunks,
        sv1, sv2, sv3, sb1, sb2, sb3);

    ggml_cuda_kernel_launch_params state_params(
        dim3(H, n_seqs, GDN_CHUNK_S / GDN_CHUNK_BV), dim3(GDN_CHUNK_THREADS, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_chunked_state_cuda, state_params,
        q_d, k_d, g_d, s_d, w_buf.get(), r_buf.get(), dst_d, state_d,
        H, n_tokens, n_chunks, sq1, sq2, sq3, sb1, sb2, sb3, neqk1, rq3, scale);
    return true;
}

#endif

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!kda && !keep_rs &&
        launch_gated_delta_net_flashinfer_aot(
            ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            neqk1, rq3, stream)) {
        return;
    }

    if (!kda && !keep_rs &&
        launch_gated_delta_net_chunked_parallel(
            ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            neqk1, rq3, scale, stream)) {
        return;
    }
#endif

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
