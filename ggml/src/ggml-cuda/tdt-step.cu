#include "tdt-step.cuh"

// Greedy transducer step control, mirroring ggml_tdt_step_i32 in ggml-cpu/ops.cpp.
static __global__ void tdt_step_i32_kernel(
        const int * token, const int * dur_idx, const int * state, const int * dur_table,
        int * dst, const int n_dur, const int blank_id, const int max_symbols, const int rnnt) {
    const int t = state[GGML_TDT_STEP_IN_T];
    const int s = state[GGML_TDT_STEP_IN_S];
    const int n = state[GGML_TDT_STEP_IN_N];

    int t_next = t;
    int s_next = s;
    int update = 0;

    if (t < n) {
        const int raw = dur_idx[0];
        const int di  = raw < 0 ? 0 : (raw < n_dur ? raw : n_dur - 1);
        const int dur = rnnt ? 0 : dur_table[di];
        const int adv = rnnt ? 1 : (dur > 1 ? dur : 1);

        if (token[0] == blank_id) {
            t_next = t + adv;
            s_next = 0;
        } else {
            update = 1;
            s_next = s + 1;
            if ((!rnnt && dur > 0) || s_next >= max_symbols) {
                t_next = t + adv;
                s_next = 0;
            }
        }
    }

    int frame = t_next > n - 1 ? n - 1 : t_next;
    if (frame < 0) {
        frame = 0;
    }

    dst[GGML_TDT_STEP_OUT_T]      = t_next;
    dst[GGML_TDT_STEP_OUT_S]      = s_next;
    dst[GGML_TDT_STEP_OUT_N]      = n;
    dst[GGML_TDT_STEP_OUT_UPDATE] = update;
    dst[GGML_TDT_STEP_OUT_HOLD]   = 1 - update;
    dst[GGML_TDT_STEP_OUT_FRAME]  = frame;
    dst[GGML_TDT_STEP_OUT_TOKEN]  = token[0];
    dst[GGML_TDT_STEP_OUT_DUR]    = dur_idx[0];
}

void ggml_cuda_op_tdt_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * token     = dst->src[0];
    const ggml_tensor * dur_idx   = dst->src[1];
    const ggml_tensor * state     = dst->src[2];
    const ggml_tensor * dur_table = dst->src[3];

    GGML_ASSERT(token->type     == GGML_TYPE_I32);
    GGML_ASSERT(dur_idx->type   == GGML_TYPE_I32);
    GGML_ASSERT(state->type     == GGML_TYPE_I32);
    GGML_ASSERT(dur_table->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type       == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nelements(dst) == GGML_TDT_STEP_N_OUTS);

    tdt_step_i32_kernel<<<1, 1, 0, ctx.stream()>>>(
        (const int *) token->data, (const int *) dur_idx->data,
        (const int *) state->data, (const int *) dur_table->data,
        (int *) dst->data,
        (int) ggml_nelements(dur_table),
        ggml_get_op_params_i32(dst, 0), ggml_get_op_params_i32(dst, 1), ggml_get_op_params_i32(dst, 2));
}
