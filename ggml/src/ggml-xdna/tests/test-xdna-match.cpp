// Which nodes the graph planners take for the ones llama builds around a layer,
// on the shapes llama's graph code makes. Host-only: no device is opened.

#include "../xdna-match.h"

#include <cstdio>

static constexpr int64_t CONV_CHANNELS = 6144;  // a Qwen3.5 recurrent layer's conv
static constexpr int64_t CONV_HISTORY  = 3;     // the cached inputs of its kernel of 4
static constexpr int64_t STATE_FLOATS  = CONV_HISTORY * CONV_CHANNELS;
static constexpr int64_t N_CELLS       = 4;     // recurrent cache cells
static constexpr int64_t N_RS          = 2;     // the ubatch's sequence and one extra cell
static constexpr int64_t D_ATTN        = 2048;  // a Qwen3.5 attention layer's output projection input
static constexpr int64_t D_MODEL       = 1024;  // its residual width
static constexpr int     MAX_TENSORS   = 32;

struct match_case {
    const char *        what;
    const ggml_tensor * got;
    const ggml_tensor * want;
};

template <size_t N> static int count_failures(const match_case (&cases)[N]) {
    int failures = 0;
    for (const match_case & c : cases) {
        const bool ok = c.got == c.want;
        std::printf("%s: %s\n", ok ? "ok  " : "FAIL", c.what);
        failures += ok ? 0 : 1;
    }
    return failures;
}

// build_rs gathers all n_rs states at once and hands out the ubatch's rows as a view of the gather.
static int check_state_gather(ggml_context * ctx) {
    ggml_tensor * cache  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, STATE_FLOATS, N_CELLS);
    ggml_tensor * states = ggml_reshape_2d(ctx, cache, STATE_FLOATS, cache->ne[1]);
    ggml_tensor * ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_RS);
    ggml_tensor * all    = ggml_get_rows(ctx, states, ids);
    ggml_tensor * rows   = ggml_view_2d(ctx, all, STATE_FLOATS, 1, all->nb[1], 0);
    ggml_tensor * extra  = ggml_view_2d(ctx, all, STATE_FLOATS, N_RS - 1, all->nb[1], all->nb[1]);
    ggml_tensor * cell   = ggml_view_2d(ctx, states, STATE_FLOATS, 1, states->nb[1], 0);

    const match_case cases[] = {
        { "the view of the ubatch's rows reads the gather", xdna_rec_state_gather(rows), all },
        { "a gather reads itself", xdna_rec_state_gather(all), all },
        { "the extra cells' view is not the ubatch's read", xdna_rec_state_gather(extra), nullptr },
        { "a view of the cache is not a gather", xdna_rec_state_gather(cell), nullptr },
    };
    return count_failures(cases);
}

// The last layer crops the attention output and its residual to the output rows before they are added.
static int check_row_crop(ggml_context * ctx) {
    ggml_tensor * w_o    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_ATTN, D_MODEL);
    ggml_tensor * act    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_ATTN, 1);
    ggml_tensor * out    = ggml_mul_mat(ctx, w_o, act);
    ggml_tensor * inp    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, 1);
    ggml_tensor * ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * out_c  = ggml_get_rows(ctx, out, ids);
    ggml_tensor * inp_c  = ggml_get_rows(ctx, inp, ids);
    ggml_tensor * resid  = ggml_add(ctx, out_c, inp_c);
    ggml_tensor * batch  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, 2);
    ggml_tensor * pick   = ggml_get_rows(ctx, batch, ids);
    ggml_tensor * ids2   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * repeat = ggml_get_rows(ctx, inp, ids2);

    const match_case cases[] = {
        { "the cropped attention output is the output", xdna_skip_row_crop(out_c), out },
        { "the cropped residual is the residual", xdna_skip_row_crop(inp_c), inp },
        { "one row of a two-row batch is a real crop", xdna_skip_row_crop(pick), pick },
        { "two rows of a one-row tensor are not a crop", xdna_skip_row_crop(repeat), repeat },
        { "an add is itself", xdna_skip_row_crop(resid), resid },
    };
    return count_failures(cases);
}

int main(void) {
    const ggml_init_params params = { ggml_tensor_overhead() * MAX_TENSORS, nullptr, true };
    ggml_context *         ctx    = ggml_init(params);
    const int              failures = check_state_gather(ctx) + check_row_crop(ctx);
    ggml_free(ctx);
    if (failures) {
        std::printf("test-xdna-match: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-match: all passed\n");
    return 0;
}
