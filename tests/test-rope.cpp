#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#define MAX_NARGS 3

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define GGML_SILU_FP16

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static int irand(int n) {
    if (n == 0) return 0;
    return rand()%n;
}

static void get_random_dims(int64_t * dims, int ndims) {
    dims[0] = dims[1] = dims[2] = dims[3] = 1;

    for (int i = 0; i < ndims; i++) {
        dims[i] = 1 + irand(4);
    }
}

static struct ggml_tensor * get_random_tensor_f32(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[],
        float fmin,
        float fmax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };

    return result;
}

static void fill_hadamard_f32(struct ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];
    assert(n > 0 && (n & (n - 1)) == 0);
    assert(tensor->ne[1] == n);

    float * data = (float *) tensor->data;
    data[0] = 1.0f / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];
                data[(i + s)*n + j]       =  val;
                data[i*n + (j + s)]       =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }
}

static struct ggml_tensor * test_mul_mat_aux(
        struct ggml_context * ctx,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * rot) {
    const auto n = rot->ne[0];

    struct ggml_tensor * res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat(ctx, rot, res);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

static double rel_err(const struct ggml_tensor * actual, const struct ggml_tensor * expected) {
    double sum  = 0.0;
    double diff = 0.0;

    const float * actual_data   = (const float *) actual->data;
    const float * expected_data = (const float *) expected->data;

    for (int i = 0; i < ggml_nelements(actual); ++i) {
        sum  += fabs(expected_data[i]);
        diff += fabs(actual_data[i] - expected_data[i]);
    }

    return diff / sum;
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

int main(int /*argc*/, const char ** /*argv*/) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 128*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::vector<uint8_t> work_buffer;

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * x;

    // rope f32
    for (int m = 0; m < 5; ++m) {
        const int ndims = 4;

        const int64_t n_rot = 128;
        const int64_t ne[4] = { 2*n_rot, 32, 73, 1 };

        const int n_past_0 = 100;
        const int n_past_2 = 33;

        struct ggml_tensor * r0;
        struct ggml_tensor * r1;
        struct ggml_tensor * r2;
        x = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
        int mode = -1;

        if (m < 2) {
            struct ggml_tensor * p0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);
            struct ggml_tensor * p1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);
            struct ggml_tensor * p2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);

            for (int i = 0; i < ne[2]; ++i) {
                ((int32_t *) p0->data)[i] = n_past_0 + i;
                ((int32_t *) p1->data)[i] = n_past_2 - n_past_0;
                ((int32_t *) p2->data)[i] = n_past_2 + i;
            }
            // test mode 0, 2  (standard, GPT-NeoX)
            mode = m == 0 ? GGML_ROPE_TYPE_NORMAL : GGML_ROPE_TYPE_NEOX;

            // 100, 101, 102, ..., 172
            r0 = ggml_rope(ctx0, x,  p0, n_rot, mode);
            // -67, -67, -67, ..., -67
            r1 = ggml_rope(ctx0, r0, p1, n_rot, mode); // "context swap", i.e. forget n_past_0 - n_past_2 tokens

            //  33,  34,  35, ..., 105
            r2 = ggml_rope(ctx0, x,  p2, n_rot, mode);
        } else {
            // testing multi-dimension rope position embedding mode
            struct ggml_tensor * p0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2] * 4);
            struct ggml_tensor * p1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2] * 4);
            struct ggml_tensor * p2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2] * 4);

            int sections[4] = {16, 24, 24, 0};

            mode = (m == 2) ? GGML_ROPE_TYPE_MROPE : (m == 3) ? GGML_ROPE_TYPE_VISION : GGML_ROPE_TYPE_IMROPE;

            for (int i = 0; i < ne[2]; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ((int32_t *) p0->data)[i + ne[2] * j] = n_past_0 + i + j;
                    ((int32_t *) p1->data)[i + ne[2] * j] = n_past_2 - n_past_0;
                    ((int32_t *) p2->data)[i + ne[2] * j] = n_past_2 + i + j;
                }
            }

            // [[100, 101, 102, ..., 172],
            // [101, 102, 103, ..., 173],
            // [102, 103, 104, ..., 174]]
            r0 = ggml_rope_multi(
                ctx0, x, p0, nullptr,
                n_rot, sections, mode, 32768, 1000000, 1, 0, 1, 32, 1);
            // [[-67, -67, -67, ..., -67]
            // [-67, -67, -67, ..., -67]
            // [-67, -67, -67, ..., -67]]
            r1 = ggml_rope_multi(
                ctx0, r0, p1, nullptr,
                n_rot, sections, mode, 32768, 1000000, 1, 0, 1, 32, 1);

            //  [[33,  34,  35, ..., 105]
            //  [34,  35,  36, ..., 106]
            //  [35,  36,  37, ..., 107]]
            r2 = ggml_rope_multi(
                ctx0, x, p2, nullptr,
                n_rot, sections, mode, 32768, 1000000, 1, 0, 1, 32, 1);
        }

        ggml_cgraph * gf = ggml_new_graph(ctx0);

        ggml_build_forward_expand(gf, r0);
        ggml_build_forward_expand(gf, r1);
        ggml_build_forward_expand(gf, r2);

        ggml_graph_compute_helper(work_buffer, gf, 4);

        // check that r1 and r2 are the same
        {
            double sum0 = 0.0f;
            double sum1 = 0.0f;
            double diff = 0.0f;

            const float * r1_data = (float *) r1->data;
            const float * r2_data = (float *) r2->data;

            const int n_elements = ggml_nelements(r1);

            for (int i = 0; i < n_elements; ++i) {
                sum0 += fabs(r1_data[i]);
                sum1 += fabs(r2_data[i]);
                diff += fabs(r1_data[i] - r2_data[i]);
                //if (fabs(r1_data[i] - r2_data[i]) > 0.0001f) {
                //    printf("%d: %f %f\n", i, r1_data[i], r2_data[i]);
                //    printf("diff: %f\n", fabs(r1_data[i] - r2_data[i]));
                //}
            }

            //for (int i = 4096; i < 4096 + 128; ++i) {
            //    printf("%f %f\n", r1_data[i], r2_data[i]);
            //}

            printf("mode: %d\n", mode);
            printf("sum0: %f\n", sum0);
            printf("sum1: %f\n", sum1);
            printf("diff: %f\n", diff);
            printf("rel err: %f\n", diff / sum0);
            printf("rel err: %f\n", diff / sum1);

            GGML_ASSERT(diff / sum0 < 0.0001f);
            GGML_ASSERT(diff / sum1 < 0.0001f);
        }
    }

    // Explicit decoder K-shift shape for M-RoPE/iM-RoPE image grids:
    // shift t/y/x while keeping the 4th axis unused.
    static constexpr int MROPE_TEST_NDIMS = 4;
    static constexpr int MROPE_AXIS_COUNT = 4;
    static constexpr int MROPE_T_AXIS     = 0;
    static constexpr int MROPE_Y_AXIS     = 1;
    static constexpr int MROPE_X_AXIS     = 2;
    static constexpr int MROPE_OTHER_AXIS = 3;

    static constexpr int64_t MROPE_TEST_N_ROT    = 128;
    static constexpr int64_t MROPE_TEST_N_HEADS  = 2;
    static constexpr int64_t MROPE_TEST_N_TOKENS = 11;
    static constexpr int64_t MROPE_TEST_N_BATCH  = 1;
    static constexpr int64_t MROPE_TEST_K_ROW    = 256;

    static constexpr int MROPE_SECTION_T     = 16;
    static constexpr int MROPE_SECTION_Y     = 24;
    static constexpr int MROPE_SECTION_X     = 24;
    static constexpr int MROPE_SECTION_OTHER = 0;

    static constexpr int32_t IMAGE_T_ORIGIN = 100;
    static constexpr int32_t IMAGE_Y_ORIGIN = 50;
    static constexpr int32_t IMAGE_X_ORIGIN = 25;
    static constexpr int32_t IMAGE_SHIFT    = -17;
    static constexpr int32_t IMAGE_GRID_W   = 4;

    static constexpr int ROPE_CTX_ORIG    = 32768;
    static constexpr int ROPE_FREQ_BASE   = 1000000;
    static constexpr int ROPE_FREQ_SCALE  = 1;
    static constexpr int YARN_EXT_FACTOR  = 0;
    static constexpr int YARN_ATTN_FACTOR = 1;
    static constexpr int YARN_BETA_FAST   = 32;
    static constexpr int YARN_BETA_SLOW   = 1;

    static constexpr float RANDOM_MIN = -1.0f;
    static constexpr float RANDOM_MAX =  1.0f;

    int mrope_sections[MROPE_AXIS_COUNT] = {
        MROPE_SECTION_T, MROPE_SECTION_Y, MROPE_SECTION_X, MROPE_SECTION_OTHER
    };

    // rope_multi with the shared test constants; only tensor, positions, n_rot and mode vary
    auto mrope_shift = [&](struct ggml_tensor * t, struct ggml_tensor * pos, int64_t n_rot, int mode) {
        return ggml_rope_multi(
            ctx0, t, pos, nullptr,
            n_rot, mrope_sections, mode,
            ROPE_CTX_ORIG, ROPE_FREQ_BASE, ROPE_FREQ_SCALE,
            YARN_EXT_FACTOR, YARN_ATTN_FACTOR, YARN_BETA_FAST, YARN_BETA_SLOW);
    };

    // p0 = image-grid positions, pd = in-place shift delta, p1 = shifted positions (p0 + pd)
    struct shift_positions {
        struct ggml_tensor * p0;
        struct ggml_tensor * pd;
        struct ggml_tensor * p1;
    };
    auto make_shift_positions = [&](int64_t n_tokens) {
        const shift_positions p = {
            ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * MROPE_AXIS_COUNT),
            ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * MROPE_AXIS_COUNT),
            ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * MROPE_AXIS_COUNT),
        };
        for (int i = 0; i < n_tokens; ++i) {
            const int32_t old_t = IMAGE_T_ORIGIN;
            const int32_t old_y = IMAGE_Y_ORIGIN + i/IMAGE_GRID_W;
            const int32_t old_x = IMAGE_X_ORIGIN + i%IMAGE_GRID_W;

            ((int32_t *) p.p0->data)[i + n_tokens * MROPE_T_AXIS]     = old_t;
            ((int32_t *) p.p0->data)[i + n_tokens * MROPE_Y_AXIS]     = old_y;
            ((int32_t *) p.p0->data)[i + n_tokens * MROPE_X_AXIS]     = old_x;
            ((int32_t *) p.p0->data)[i + n_tokens * MROPE_OTHER_AXIS] = 0;

            ((int32_t *) p.pd->data)[i + n_tokens * MROPE_T_AXIS]     = IMAGE_SHIFT;
            ((int32_t *) p.pd->data)[i + n_tokens * MROPE_Y_AXIS]     = IMAGE_SHIFT;
            ((int32_t *) p.pd->data)[i + n_tokens * MROPE_X_AXIS]     = IMAGE_SHIFT;
            ((int32_t *) p.pd->data)[i + n_tokens * MROPE_OTHER_AXIS] = 0;

            ((int32_t *) p.p1->data)[i + n_tokens * MROPE_T_AXIS]     = old_t + IMAGE_SHIFT;
            ((int32_t *) p.p1->data)[i + n_tokens * MROPE_Y_AXIS]     = old_y + IMAGE_SHIFT;
            ((int32_t *) p.p1->data)[i + n_tokens * MROPE_X_AXIS]     = old_x + IMAGE_SHIFT;
            ((int32_t *) p.p1->data)[i + n_tokens * MROPE_OTHER_AXIS] = 0;
        }
        return p;
    };

    for (int m = 0; m < 2; ++m) {
        const int64_t ne[MROPE_TEST_NDIMS] = {
            MROPE_TEST_N_ROT, MROPE_TEST_N_HEADS, MROPE_TEST_N_TOKENS, MROPE_TEST_N_BATCH
        };

        const int mode = m == 0 ? GGML_ROPE_TYPE_MROPE : GGML_ROPE_TYPE_IMROPE;

        x = get_random_tensor_f32(ctx0, MROPE_TEST_NDIMS, ne, RANDOM_MIN, RANDOM_MAX);

        const shift_positions sp = make_shift_positions(ne[2]);

        struct ggml_tensor * r0 = mrope_shift(x,  sp.p0, MROPE_TEST_N_ROT, mode);
        struct ggml_tensor * rd = mrope_shift(r0, sp.pd, MROPE_TEST_N_ROT, mode);
        struct ggml_tensor * r1 = mrope_shift(x,  sp.p1, MROPE_TEST_N_ROT, mode);

        ggml_cgraph * gf = ggml_new_graph(ctx0);

        ggml_build_forward_expand(gf, r0);
        ggml_build_forward_expand(gf, rd);
        ggml_build_forward_expand(gf, r1);

        ggml_graph_compute_helper(work_buffer, gf, 4);

        const double err = rel_err(rd, r1);

        printf("k-shift mode: %d\n", mode);
        printf("k-shift rel err: %f\n", err);

        static constexpr float FLOAT_K_SHIFT_TOLERANCE = 0.0001f;
        GGML_ASSERT(err < FLOAT_K_SHIFT_TOLERANCE);
    }

    // Quantized K-cache decoder K-shift for M-RoPE/iM-RoPE image grids.
    // Mirrors the standard quantized cache path:
    //   quantized K -> f32 -> M-RoPE shift -> quantized K
    // and compares it with directly rotating to the shifted positions before
    // quantizing. This catches regressions where quantized M-RoPE shift graphs
    // diverge from the unquantized shift semantics.
    static constexpr ggml_type QUANTIZED_K_SHIFT_TYPES[] = {
        GGML_TYPE_Q8_0,
        GGML_TYPE_TBQ3_0,
        GGML_TYPE_TBQ3_0_64,
        GGML_TYPE_TBQ4_0,
        GGML_TYPE_TBQ4_0_64,
        GGML_TYPE_PQ3_0,
        GGML_TYPE_PQ3_0_64,
        GGML_TYPE_PQ4_0,
        GGML_TYPE_PQ4_0_64,
    };
    static constexpr float QUANTIZED_K_SHIFT_TOLERANCE = 0.125f;
    static constexpr int64_t QUANTIZED_K_SHIFT_N_ROT = 64;
    static constexpr int64_t QUANTIZED_K_SHIFT_WIDTH = 128;

    for (int m = 0; m < 2; ++m) {
        const int64_t ne[MROPE_TEST_NDIMS] = {
            QUANTIZED_K_SHIFT_WIDTH, MROPE_TEST_N_HEADS, MROPE_TEST_N_TOKENS, MROPE_TEST_N_BATCH
        };

        const int mode = m == 0 ? GGML_ROPE_TYPE_MROPE : GGML_ROPE_TYPE_IMROPE;

        const int64_t cache_ne[MROPE_TEST_NDIMS] = {
            MROPE_TEST_K_ROW, MROPE_TEST_N_HEADS, MROPE_TEST_N_TOKENS, MROPE_TEST_N_BATCH
        };

        x = get_random_tensor_f32(ctx0, MROPE_TEST_NDIMS, ne, RANDOM_MIN, RANDOM_MAX);

        auto new_quantized_cache_view = [&](ggml_type type) {
            struct ggml_tensor * cache = ggml_new_tensor(ctx0, type, MROPE_TEST_NDIMS, cache_ne);
            return ggml_view_4d(
                ctx0, cache,
                QUANTIZED_K_SHIFT_WIDTH, MROPE_TEST_N_HEADS, MROPE_TEST_N_TOKENS, MROPE_TEST_N_BATCH,
                cache->nb[1], cache->nb[2], cache->nb[3], 0);
        };

        const shift_positions sp = make_shift_positions(ne[2]);

        struct ggml_tensor * old_f32    = mrope_shift(x, sp.p0, QUANTIZED_K_SHIFT_N_ROT, mode);
        struct ggml_tensor * target_f32 = mrope_shift(x, sp.p1, QUANTIZED_K_SHIFT_N_ROT, mode);

        for (ggml_type qtype : QUANTIZED_K_SHIFT_TYPES) {
            struct ggml_tensor * old_q = ggml_cpy(ctx0, old_f32, new_quantized_cache_view(qtype));
            struct ggml_tensor * old_deq = ggml_cast(ctx0, old_q, GGML_TYPE_F32);
            struct ggml_tensor * shifted_deq = mrope_shift(old_deq, sp.pd, QUANTIZED_K_SHIFT_N_ROT, mode);
            struct ggml_tensor * shifted_q = ggml_cpy(ctx0, shifted_deq, new_quantized_cache_view(qtype));
            struct ggml_tensor * shifted_out = ggml_cast(ctx0, shifted_q, GGML_TYPE_F32);

            struct ggml_tensor * target_q = ggml_cpy(ctx0, target_f32, new_quantized_cache_view(qtype));
            struct ggml_tensor * target_out = ggml_cast(ctx0, target_q, GGML_TYPE_F32);

            ggml_cgraph * gf = ggml_new_graph(ctx0);

            ggml_build_forward_expand(gf, old_f32);
            ggml_build_forward_expand(gf, shifted_out);
            ggml_build_forward_expand(gf, target_out);

            ggml_graph_compute_helper(work_buffer, gf, 4);

            const double err = rel_err(shifted_out, target_out);

            printf("%s k-shift mode: %d\n", ggml_type_name(qtype), mode);
            printf("%s k-shift rel err: %f\n", ggml_type_name(qtype), err);

            GGML_ASSERT(err < QUANTIZED_K_SHIFT_TOLERANCE);
        }

        // TBQ/PQ K-cache shift with the attention-rotation path enabled:
        // quantized rotated K -> f32 -> rotate back -> M-RoPE shift ->
        // rotate forward -> quantized rotated K. This mirrors the branch that
        // uses ggml_mul_mat_aux() in llama_kv_cache::build_rope_shift().
        static constexpr ggml_type ATTENTION_ROT_K_SHIFT_TYPES[] = {
            GGML_TYPE_TBQ4_0,
            GGML_TYPE_PQ4_0,
        };

        struct ggml_tensor * rot = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, QUANTIZED_K_SHIFT_WIDTH, QUANTIZED_K_SHIFT_WIDTH);
        fill_hadamard_f32(rot);

        for (ggml_type qtype : ATTENTION_ROT_K_SHIFT_TYPES) {
            struct ggml_tensor * old_rot = test_mul_mat_aux(ctx0, old_f32, rot);
            struct ggml_tensor * old_q = ggml_cpy(ctx0, old_rot, new_quantized_cache_view(qtype));
            struct ggml_tensor * old_deq = ggml_cast(ctx0, old_q, GGML_TYPE_F32);
            struct ggml_tensor * old_unrot = test_mul_mat_aux(ctx0, old_deq, rot);
            struct ggml_tensor * shifted_unrot = mrope_shift(old_unrot, sp.pd, QUANTIZED_K_SHIFT_N_ROT, mode);
            struct ggml_tensor * shifted_rot = test_mul_mat_aux(ctx0, shifted_unrot, rot);
            struct ggml_tensor * shifted_q = ggml_cpy(ctx0, shifted_rot, new_quantized_cache_view(qtype));
            struct ggml_tensor * shifted_out = ggml_cast(ctx0, shifted_q, GGML_TYPE_F32);

            struct ggml_tensor * target_rot = test_mul_mat_aux(ctx0, target_f32, rot);
            struct ggml_tensor * target_q = ggml_cpy(ctx0, target_rot, new_quantized_cache_view(qtype));
            struct ggml_tensor * target_out = ggml_cast(ctx0, target_q, GGML_TYPE_F32);

            ggml_cgraph * gf = ggml_new_graph(ctx0);

            ggml_build_forward_expand(gf, shifted_out);
            ggml_build_forward_expand(gf, target_out);

            ggml_graph_compute_helper(work_buffer, gf, 4);

            const double err = rel_err(shifted_out, target_out);

            printf("%s attention-rot k-shift mode: %d\n", ggml_type_name(qtype), mode);
            printf("%s attention-rot k-shift rel err: %f\n", ggml_type_name(qtype), err);

            static constexpr float ATTENTION_ROT_K_SHIFT_TOLERANCE = 0.25f;
            GGML_ASSERT(err < ATTENTION_ROT_K_SHIFT_TOLERANCE);
        }

        // Full-cache image-token writeback: seed a quantized cache view with
        // image-grid keys, shift that same view in place, then read it back.
        // This catches regressions where the writeback to a non-contiguous KV
        // cache view succeeds as a standalone copy but fails in the full path.
        {
            static constexpr ggml_type FULL_CACHE_WRITEBACK_TYPE = GGML_TYPE_PQ4_0;

            struct ggml_tensor * cache = ggml_new_tensor(ctx0, FULL_CACHE_WRITEBACK_TYPE, MROPE_TEST_NDIMS, cache_ne);
            struct ggml_tensor * cache_view = ggml_view_4d(
                ctx0, cache,
                QUANTIZED_K_SHIFT_WIDTH, MROPE_TEST_N_HEADS, MROPE_TEST_N_TOKENS, MROPE_TEST_N_BATCH,
                cache->nb[1], cache->nb[2], cache->nb[3], 0);

            struct ggml_tensor * old_q = ggml_cpy(ctx0, old_f32, cache_view);
            struct ggml_tensor * old_deq = ggml_cast(ctx0, old_q, GGML_TYPE_F32);
            struct ggml_tensor * shifted_deq = mrope_shift(old_deq, sp.pd, QUANTIZED_K_SHIFT_N_ROT, mode);
            struct ggml_tensor * shifted_q = ggml_cpy(ctx0, shifted_deq, cache_view);
            struct ggml_tensor * shifted_out = ggml_cast(ctx0, shifted_q, GGML_TYPE_F32);

            struct ggml_tensor * target_q = ggml_cpy(ctx0, target_f32, new_quantized_cache_view(FULL_CACHE_WRITEBACK_TYPE));
            struct ggml_tensor * target_out = ggml_cast(ctx0, target_q, GGML_TYPE_F32);

            ggml_cgraph * gf = ggml_new_graph(ctx0);

            ggml_build_forward_expand(gf, shifted_out);
            ggml_build_forward_expand(gf, target_out);

            ggml_graph_compute_helper(work_buffer, gf, 4);

            const double err = rel_err(shifted_out, target_out);

            printf("%s full-cache image k-shift mode: %d\n", ggml_type_name(FULL_CACHE_WRITEBACK_TYPE), mode);
            printf("%s full-cache image k-shift rel err: %f\n", ggml_type_name(FULL_CACHE_WRITEBACK_TYPE), err);

            GGML_ASSERT(err < QUANTIZED_K_SHIFT_TOLERANCE);
        }
    }

    ggml_free(ctx0);

    return 0;
}
