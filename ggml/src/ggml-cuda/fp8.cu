#include "fp8.cuh"
#include "mmvf.cuh"

#include <map>
#include <mutex>
#include <tuple>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080

struct fp8_abs_src {
    const float * x;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t s1;
    int64_t s2;
    int64_t s3;

    __device__ float operator()(int64_t i) const {
        const int64_t i0 = i % ne0;
        i /= ne0;
        const int64_t i1 = i % ne1;
        i /= ne1;
        const int64_t i2 = i % ne2;
        const int64_t i3 = i / ne2;
        const float value = fabsf(x[i0 + i1*s1 + i2*s2 + i3*s3]);
        return isfinite(value) ? value : 448.0f;
    }
};

static __global__ void fp8_amax_partials(fp8_abs_src src, int64_t ne, float * partials) {
    float amax = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < ne; i += (int64_t) blockDim.x*gridDim.x) {
        amax = fmaxf(amax, src(i));
    }

    __shared__ float shared[WARP_SIZE];
    amax = block_reduce<block_reduce_method::MAX, 256>(amax, shared);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = amax;
    }
}

static __global__ void fp8_amax_final(const float * partials, int n, float * amax) {
    float value = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        value = fmaxf(value, partials[i]);
    }

    __shared__ float shared[WARP_SIZE];
    value = block_reduce<block_reduce_method::MAX, 256>(value, shared);
    if (threadIdx.x == 0) {
        *amax = value;
    }
}

static __global__ void quantize_fp8_e4m3(
        const float * __restrict__ x, uint8_t * __restrict__ y, const float * __restrict__ amax,
        float * __restrict__ scale, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne, int64_t s1, int64_t s2, int64_t s3) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }

    int64_t tmp = i / ne0;
    const int64_t i0 = i - tmp*ne0;
    const int64_t i1 = tmp % ne1;
    tmp /= ne1;
    const int64_t i2 = tmp % ne2;
    const int64_t i3 = tmp / ne2;

    const float d = *amax > 0.0f ? *amax / 448.0f : 1.0f;
    y[i] = ggml_cuda_fp32_to_f8_e4m3(x[i0 + i1*s1 + i2*s2 + i3*s3] / d);
    if (i == 0) {
        *scale = d;
    }
}

// x/s -> e4m3 with a known scale s, also writes s for cuBLASLt
static __global__ void quantize_fp8_e4m3_static(
        const float * __restrict__ x, uint8_t * __restrict__ y, float * __restrict__ scale, const float s,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne, int64_t s1, int64_t s2, int64_t s3) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0) {
        *scale = s;
    }
    if (i >= ne) {
        return;
    }

    int64_t tmp = i / ne0;
    const int64_t i0 = i - tmp*ne0;
    const int64_t i1 = tmp % ne1;
    tmp /= ne1;
    const int64_t i2 = tmp % ne2;
    const int64_t i3 = tmp / ne2;

    y[i] = ggml_cuda_fp32_to_f8_e4m3(x[i0 + i1*s1 + i2*s2 + i3*s3] / s);
}

static __global__ void quantize_fp8_e4m3_static_cont4(
        const float4 * __restrict__ x, uint32_t * __restrict__ y, float * __restrict__ scale, const float s, const int64_t ne4) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0) {
        *scale = s;
    }
    if (i >= ne4) {
        return;
    }
    const float4 v = x[i];
    y[i] =  (uint32_t) ggml_cuda_fp32_to_f8_e4m3(v.x / s)        | (uint32_t) ggml_cuda_fp32_to_f8_e4m3(v.y / s) << 8 |
            (uint32_t) ggml_cuda_fp32_to_f8_e4m3(v.z / s) << 16 | (uint32_t) ggml_cuda_fp32_to_f8_e4m3(v.w / s) << 24;
}

float ggml_cuda_fp8_input_scale(const ggml_tensor * src0) {
    static const bool dynamic = getenv("GGML_CUDA_FP8_DYNAMIC") != nullptr;
    if (dynamic || src0->type != GGML_TYPE_F8_E4M3 || ggml_get_op_params_i32(src0, 0) != GGML_CUDA_FP8_INPUT_SCALE_TAG) {
        return 0.0f;
    }
    return ggml_get_op_params_f32(src0, 1);
}

static bool fp8_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return fp8_mma_hardware_available(cc) && src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && src0->ne[0] % 16 == 0 && src0->ne[1] % 16 == 0 &&
        src0->nb[0] == sizeof(uint8_t) && src0->nb[1] == (size_t) src0->ne[0] && src1->nb[0] == sizeof(float) &&
        ggml_cuda_is_aligned(src0, 16) && ggml_cuda_is_aligned(dst, 16);
}

bool ggml_cuda_fp8_activation_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return fp8_mul_mat_supported(src0, src1, dst) && src1->ne[1] > MMVF_MAX_BATCH_SIZE &&
        src0->ne[2] == 1 && src0->ne[3] == 1;
}

void ggml_cuda_fp8_quantize(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const float input_scale, uint8_t * values, float * scale) {
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    cudaStream_t stream = ctx.stream();
    const int64_t ne_src1 = ggml_nelements(src1);

    if (input_scale > 0.0f) {
        if (ggml_is_contiguous(src1) && ne_src1 % 4 == 0 && (uintptr_t) src1->data % 16 == 0) {
            const int64_t ne4 = ne_src1/4;
            quantize_fp8_e4m3_static_cont4<<<(ne4 + 255)/256, 256, 0, stream>>>(
                (const float4 *) src1->data, (uint32_t *) values, scale, input_scale, ne4);
        } else {
            quantize_fp8_e4m3_static<<<(ne_src1 + 255)/256, 256, 0, stream>>>(
                (const float *) src1->data, values, scale, input_scale, ne10, ne11, ne12, ne_src1,
                nb11 / sizeof(float), nb12 / sizeof(float), nb13 / sizeof(float));
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    ggml_cuda_pool_alloc<float> src1_amax(ctx.pool(), 1);
    const fp8_abs_src abs_src = {
        (const float *) src1->data, ne10, ne11, ne12,
        (int64_t) (nb11 / sizeof(float)), (int64_t) (nb12 / sizeof(float)), (int64_t) (nb13 / sizeof(float))
    };
    const int reduce_blocks = std::min<int64_t>((ne_src1 + 255)/256, 1024);
    ggml_cuda_pool_alloc<float> reduce_tmp(ctx.pool(), reduce_blocks);
    fp8_amax_partials<<<reduce_blocks, 256, 0, stream>>>(abs_src, ne_src1, reduce_tmp.ptr);
    fp8_amax_final<<<1, 256, 0, stream>>>(reduce_tmp.ptr, reduce_blocks, src1_amax.ptr);

    quantize_fp8_e4m3<<<(ne_src1 + 255)/256, 256, 0, stream>>>(
        (const float *) src1->data, values, src1_amax.ptr, scale, ne10, ne11, ne12, ne_src1,
        nb11 / sizeof(float), nb12 / sizeof(float), nb13 / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

// cuBLASLt descriptors and the heuristic's algorithm, cached per shape; the scale pointers are set per call
struct fp8_matmul_plan {
    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t a  = nullptr;
    cublasLtMatrixLayout_t b  = nullptr;
    cublasLtMatrixLayout_t d  = nullptr;
    cublasLtMatmulAlgo_t   algo = {};
    size_t                 workspace = 0;
    bool                   ok = false;
};

static constexpr size_t FP8_MATMUL_MAX_WORKSPACE = 32*1024*1024;

static std::mutex fp8_plan_mutex;
static std::map<std::tuple<int, int64_t, int64_t, int64_t>, fp8_matmul_plan> fp8_plans;

static fp8_matmul_plan & fp8_get_plan(ggml_backend_cuda_context & ctx, int64_t k, int64_t n, int64_t m) {
    auto [it, inserted] = fp8_plans.try_emplace(std::make_tuple(ctx.device, k, n, m));
    fp8_matmul_plan & plan = it->second;
    if (!inserted) {
        return plan;
    }

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_8F_E4M3, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_8F_E4M3, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.d, CUDA_R_32F, n, m, n));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &FP8_MATMUL_MAX_WORKSPACE, sizeof(FP8_MATMUL_MAX_WORKSPACE)));
    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ctx.cublaslt_handle(), plan.op, plan.a, plan.b, plan.d, plan.d, preference, 1, &heuristic, &returned);
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (status == CUBLAS_STATUS_SUCCESS && returned > 0) {
        plan.algo      = heuristic.algo;
        plan.workspace = heuristic.workspaceSize;
        plan.ok        = true;
    }
    return plan;
}

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale, const uint8_t * act_values, const float * act_scale) {
    if (!fp8_mul_mat_supported(src0, src1, dst)) {
        return false;
    }

    GGML_ASSERT(!weight_scale || (weight_scale->type == GGML_TYPE_F32 && ggml_is_contiguous(weight_scale) && ggml_nelements(weight_scale) == 1));

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    std::lock_guard<std::mutex> lock(fp8_plan_mutex);
    fp8_matmul_plan & plan = fp8_get_plan(ctx, ne00, ne01, ne11);
    if (!plan.ok) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<uint8_t> src1_fp8(ctx.pool());
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (act_values == nullptr) {
        src1_fp8.alloc(ggml_nelements(src1));
        src1_scale.alloc(1);
        ggml_cuda_fp8_quantize(ctx, src1, ggml_cuda_fp8_input_scale(src0), src1_fp8.ptr, src1_scale.ptr);
        act_values = src1_fp8.ptr;
        act_scale  = src1_scale.ptr;
    }

    const float * a_scale = weight_scale ? (const float *) weight_scale->data : nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_scale, sizeof(act_scale)));

    // cuBLASLt wants a 256 byte aligned workspace, the pool only guarantees 128
    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    void * workspace_ptr = nullptr;
    if (plan.workspace != 0) {
        workspace.alloc(plan.workspace + 256);
        workspace_ptr = (void *) GGML_PAD((uintptr_t) workspace.get(), (uintptr_t) 256);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    for (int64_t i3 = 0; i3 < ne13; ++i3) {
        for (int64_t i2 = 0; i2 < ne12; ++i2) {
            const char * a = (const char *) src0->data + (i2/r2)*nb02 + (i3/r3)*nb03;
            const uint8_t * b = act_values + (i3*ne12 + i2)*ne11*ne10;
            float * d = (float *) ((char *) dst->data + i2*dst->nb[2] + i3*dst->nb[3]);
            CUBLAS_CHECK(cublasLtMatmul(ctx.cublaslt_handle(), plan.op, &alpha, a, plan.a, b, plan.b,
                &beta, d, plan.d, d, plan.d, &plan.algo, workspace_ptr, plan.workspace, stream));
        }
    }
    return true;
}

#else

float ggml_cuda_fp8_input_scale(const ggml_tensor * src0) {
    GGML_UNUSED(src0);
    return 0.0f;
}

bool ggml_cuda_fp8_activation_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(src0, src1, dst);
    return false;
}

void ggml_cuda_fp8_quantize(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float input_scale, uint8_t * values, float * scale) {
    GGML_UNUSED_VARS(ctx, src1, input_scale, values, scale);
    GGML_ABORT("fatal error");
}

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale, const uint8_t * act_values, const float * act_scale) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst, weight_scale, act_values, act_scale);
    return false;
}

#endif
