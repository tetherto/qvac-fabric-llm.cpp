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
static constexpr int64_t HEAD_DIM      = 256;   // a Qwen3.5 full-attention layer's head size
static constexpr int64_t N_HEAD        = 8;     // its query heads
static constexpr int64_t N_HEAD_KV     = 2;     // its K/V heads
static constexpr int64_t N_KV          = 512;   // K/V view cells
static constexpr int64_t N_BATCH       = 64;    // prefill tokens
static constexpr int64_t QK_ROW        = 256;   // one K-quant block: the shortest row a quantized weight takes
static constexpr int64_t W_ROWS        = 4;
static constexpr int     MAX_TENSORS   = 128;

// The layer shapes the fused decode runs (ggml-xdna.cpp xdna_fused_layer_shapes): Qwen3.5's convolutions of 16 and
// 32 value heads, attention heads of 256 and query widths of 8 and 16 heads.
static constexpr int64_t           CONV_KERNEL = 4;
static constexpr int64_t           HEAD_SIZE   = 256;
static constexpr xdna_fused_shapes SHAPES      = {
    { 6144, 8192, 2048 },
    HEAD_SIZE,
    { 2048, 4096, 2048 }
};
static constexpr int64_t CONV_32     = 8192;  // the 4B/9B's
static constexpr int64_t CONV_24     = 7168;  // 24 value heads: one and a half passes
static constexpr int64_t QUERY_32    = 8192;  // 32 heads of 256
static constexpr int64_t DENSE_HEAD  = 128;   // a Qwen3 dense model's
static constexpr int64_t DENSE_QUERY = 4096;  // its 32 heads of 128

struct match_case {
    const char *        what;
    const ggml_tensor * got;
    const ggml_tensor * want;
};

struct bool_case {
    const char * what;
    bool         got;
    bool         want;
};

template <typename C, size_t N> static int count_failures(const C (&cases)[N]) {
    int failures = 0;
    for (const C & c : cases) {
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

// The prefill attention kernels read K/V as the leading cells of the view, so a node
// with sinks or with K/V row lists (ggml_flash_attn_ext_rows) stays on the host.
static int check_fa_plain(ggml_context * ctx) {
    const float   scale = 1.0f / 16.0f;
    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_BATCH, N_HEAD, 1);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD_DIM, N_KV, N_HEAD_KV, 1);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD_DIM, N_KV, N_HEAD_KV, 1);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N_KV, N_BATCH);
    ggml_tensor * fa    = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);

    ggml_tensor * sinks    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_HEAD);
    ggml_tensor * fa_sinks = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(fa_sinks, sinks);

    // a row list as long as the view: every shape the kernels check still matches
    ggml_tensor * kv_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_KV, 1);
    ggml_tensor * fa_rows = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, kv_rows, scale, 0.0f, 0.0f);

    ggml_tensor * add = ggml_add(ctx, q, q);

    const bool_case cases[] = {
        { "plain attention goes to the kernels", xdna_fa_plain(fa), true },
        { "attention with sinks stays on the host", xdna_fa_plain(fa_sinks), false },
        { "attention over K/V row lists stays on the host", xdna_fa_plain(fa_rows), false },
        { "an add is not attention", xdna_fa_plain(add), false },
        { "no node is not attention", xdna_fa_plain(nullptr), false },
    };
    return count_failures(cases);
}

static bool att_rows_shape_ok(const ggml_tensor * fa) {
    return xdna_att_rows_shape_ok(fa->src[5], fa->src[3], fa->src[1]);
}

// The decode routes take a row list naming the leading cells. They scan K/V's rows of mask columns, so a list wider
// than K/V stays on the host whatever its first entries say: a visible column past those can name any cell.
static int check_att_rows(ggml_context * ctx) {
    static constexpr int64_t N_VALID  = 300;  // the cells one sequence holds
    static constexpr int64_t N_BEHIND = 200;  // the cells of a sequence ahead of it

    const float   scale   = 1.0f / 16.0f;
    ggml_tensor * q       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, 1, N_HEAD, 1);
    ggml_tensor * k       = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD_DIM, N_KV, N_HEAD_KV, 1);
    ggml_tensor * v       = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD_DIM, N_KV, N_HEAD_KV, 1);
    ggml_tensor * k_half  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD_DIM, N_KV / 2, N_HEAD_KV, 1);
    ggml_tensor * v_half  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD_DIM, N_KV / 2, N_HEAD_KV, 1);
    ggml_tensor * mask    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N_KV, 1);
    ggml_tensor * rows    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_KV, 1);
    ggml_tensor * shifted = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_KV, 1);

    int32_t leading[N_KV];
    int32_t behind[N_KV];
    for (int j = 0; j < N_KV; j++) {
        leading[j] = j < N_VALID ? j : -1;
        behind[j]  = j < N_VALID ? j + N_BEHIND : -1;
    }
    rows->data    = leading;
    shifted->data = behind;

    ggml_tensor * fa_equal   = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, scale, 0.0f, 0.0f);
    ggml_tensor * fa_shifted = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, shifted, scale, 0.0f, 0.0f);
    ggml_tensor * fa_wide    = ggml_flash_attn_ext_rows(ctx, q, k_half, v_half, mask, rows, scale, 0.0f, 0.0f);
    ggml_tensor * fa_no_list = ggml_flash_attn_ext(ctx, q, k_half, v_half, mask, scale, 0.0f, 0.0f);

    const bool_case cases[] = {
        { "a list as wide as K/V passes the shape check", att_rows_shape_ok(fa_equal), true },
        { "a list wider than K/V does not", att_rows_shape_ok(fa_wide), false },
        { "without a list a mask may be wider than K/V", att_rows_shape_ok(fa_no_list), true },
        { "an equal-width list of the leading cells is an identity", xdna_att_rows_identity(fa_equal, N_VALID), true },
        { "a list behind another sequence's cells is not", xdna_att_rows_identity(fa_shifted, N_VALID), false },
    };
    return count_failures(cases);
}

// A weight `name` of `type`, and the MUL_MAT that reads it.
static ggml_tensor * read_weight(ggml_context * ctx, ggml_type type, const char * name, ggml_tensor * x) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, QK_ROW, W_ROWS);
    ggml_set_name(w, name);
    return ggml_mul_mat(ctx, w, x);
}

// The weights the fused decode reads in its own formats, and the first it does not.
static int check_fused_weights(ggml_context * ctx) {
    ggml_tensor *       x         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QK_ROW, 1);
    ggml_tensor *       gate      = read_weight(ctx, GGML_TYPE_Q4_K, "blk.0.ffn_gate.weight", x);
    ggml_tensor *       up        = read_weight(ctx, GGML_TYPE_Q4_K, "blk.0.ffn_up.weight", x);
    ggml_tensor *       down      = read_weight(ctx, GGML_TYPE_Q6_K, "blk.0.ffn_down.weight", x);
    ggml_tensor *       so6       = read_weight(ctx, GGML_TYPE_Q6_K, "blk.0.ssm_out.weight", x);
    ggml_tensor *       so4       = read_weight(ctx, GGML_TYPE_Q4_K, "blk.1.ssm_out.weight", x);
    ggml_tensor *       qkv8      = read_weight(ctx, GGML_TYPE_Q8_0, "blk.0.attn_qkv.weight", x);
    ggml_tensor *       down8     = read_weight(ctx, GGML_TYPE_Q8_0, "blk.3.ffn_down.weight", x);
    ggml_tensor *       gate8     = read_weight(ctx, GGML_TYPE_Q8_0, "blk.3.ffn_gate.weight", x);
    const ggml_tensor * covered[] = { gate, up, down, so6, qkv8 };
    const ggml_tensor * stock[]   = { qkv8, gate8, up, down8 };
    const ggml_tensor * q4_so[]   = { gate, so4 };

    const match_case cases[] = {
        { "the Q4_K_M set reads, an 8-bit layout",      xdna_fused_uncovered(covered,   5, 1), nullptr       },
        { "a Q6_K ssm_out needs the 8-bit layout",      xdna_fused_uncovered(covered,   5, 0), so6->src[0]   },
        { "a Q4_K ssm_out reads with the 4-bit layout", xdna_fused_uncovered(q4_so,     2, 0), nullptr       },
        { "a Q4_K ssm_out needs the 4-bit layout",      xdna_fused_uncovered(q4_so,     2, 1), so4->src[0]   },
        { "a Q8_0 FFN is the first weight not read",    xdna_fused_uncovered(stock,     4, 1), gate8->src[0] },
        { "a Q8_0 FFN down is not read",                xdna_fused_uncovered(&stock[2], 2, 1), down8->src[0] },
        { "a Q8_0 attn_qkv is not a fused-set weight",  xdna_fused_uncovered(stock,     1, 1), nullptr       },
    };
    return count_failures(cases);
}

// A weight `name` of ne0 x ne1, and a node that reads it.
static ggml_tensor * read_shaped(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(w, name);
    return ggml_scale(ctx, w, 1.0f);
}

struct census_case {
    const char *        what;
    xdna_layer_census   got;
    bool                recurrent;
    bool                whole;
    const char *        foreign;  // the name the census keeps, empty when none
};

template <size_t N>
static xdna_layer_census census_of(const ggml_tensor * const (&nodes)[N], xdna_layer_census c = {}) {
    xdna_layer_census_add(c, nodes, (int) N, SHAPES);
    return c;
}

template <size_t N> static int count_census_failures(const census_case (&cases)[N]) {
    int failures = 0;
    for (const census_case & c : cases) {
        const bool ok = c.got.recurrent == c.recurrent && c.got.whole == c.whole && c.got.foreign == c.foreign;
        std::printf("%s: %s\n", ok ? "ok  " : "FAIL", c.what);
        failures += ok ? 0 : 1;
    }
    return failures;
}

// Which decode chunks show layers the fused decode runs: Qwen3.5's dense ones, and not a dense Qwen3, a mixture of
// experts or other head counts; a chunk split off a layer tells nothing until the rest comes.
static int check_layer_census(ggml_context * ctx) {
    ggml_tensor * x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QK_ROW, 1);
    ggml_tensor * conv16 = read_shaped(ctx, "blk.0.ssm_conv1d.weight", CONV_KERNEL, CONV_CHANNELS);
    ggml_tensor * conv32 = read_shaped(ctx, "blk.0.ssm_conv1d.weight", CONV_KERNEL, CONV_32);
    ggml_tensor * conv24 = read_shaped(ctx, "blk.0.ssm_conv1d.weight", CONV_KERNEL, CONV_24);
    ggml_tensor * gate   = read_weight(ctx, GGML_TYPE_Q4_K, "blk.0.ffn_gate.weight", x);
    ggml_tensor * up     = read_weight(ctx, GGML_TYPE_Q4_K, "blk.0.ffn_up.weight", x);
    ggml_tensor * down   = read_weight(ctx, GGML_TYPE_Q6_K, "blk.0.ffn_down.weight", x);
    ggml_tensor * k_norm = read_shaped(ctx, "blk.3.attn_k_norm.weight", HEAD_SIZE, 1);
    ggml_tensor * out8   = read_shaped(ctx, "blk.3.attn_output.weight", SHAPES.query.min, D_MODEL);
    ggml_tensor * out16  = read_shaped(ctx, "blk.3.attn_output.weight", SHAPES.query.max, D_MODEL);
    ggml_tensor * out32  = read_shaped(ctx, "blk.3.attn_output.weight", QUERY_32, D_MODEL);
    ggml_tensor * dk     = read_shaped(ctx, "blk.0.attn_k_norm.weight", DENSE_HEAD, 1);
    ggml_tensor * dout   = read_shaped(ctx, "blk.0.attn_output.weight", DENSE_QUERY, D_MODEL);
    ggml_tensor * router = read_weight(ctx, GGML_TYPE_F32, "blk.0.ffn_gate_inp.weight", x);
    ggml_tensor * exps   = read_shaped(ctx, "blk.0.ffn_gate_exps.weight", QK_ROW, W_ROWS);
    ggml_tensor * l_out  = ggml_scale(ctx, x, 1.0f);
    ggml_set_name(l_out, "l_out-3");

    const ggml_tensor * small[]  = { conv16, gate, up, down, k_norm, out8, l_out };
    const ggml_tensor * wide[]   = { conv32, gate, up, down, k_norm, out16, l_out };
    const ggml_tensor * dense[]  = { dk, dout, gate, up, down, l_out };
    const ggml_tensor * moe[]    = { conv32, router, exps, k_norm, out16, l_out };
    const ggml_tensor * heads[]  = { conv24, gate, up, down, l_out };
    const ggml_tensor * query[]  = { conv16, gate, up, down, k_norm, out32, l_out };
    const ggml_tensor * before[] = { k_norm, out8 };
    const ggml_tensor * after[]  = { conv16 };

    const census_case cases[] = {
        { "Qwen3.5 0.8B/2B layers are run", census_of(small), true, true, "" },
        { "Qwen3.5 4B/9B layers are run", census_of(wide), true, true, "" },
        { "a dense Qwen3 has no recurrent layer", census_of(dense), false, true, ggml_get_name(dk->src[0]) },
        { "a mixture of experts' router is not run", census_of(moe), true, true, ggml_get_name(router->src[0]) },
        { "24 value heads are not run", census_of(heads), false, true, ggml_get_name(conv24->src[0]) },
        { "32 query heads are not run", census_of(query), true, true, ggml_get_name(out32->src[0]) },
        { "a split-off attention tells nothing", census_of(before), false, false, "" },
        { "the next chunk's recurrent layer adds up", census_of(after, census_of(before)), true, false, "" },
    };
    return count_census_failures(cases);
}

// The census keeps the name of the weight it found, not the tensor: the warning reads it after the chunk's graph,
// when the tensor may be gone.
static int check_census_keeps_name(void) {
    static constexpr char   ROUTER[] = "blk.0.ffn_gate_inp.weight";
    const ggml_init_params  params   = { ggml_tensor_overhead() * W_ROWS, nullptr, true };
    ggml_context *          ctx      = ggml_init(params);
    ggml_tensor *           x        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QK_ROW, 1);
    const ggml_tensor *     nodes[]  = { read_weight(ctx, GGML_TYPE_F32, ROUTER, x) };
    const xdna_layer_census c        = census_of(nodes);
    ggml_free(ctx);
    const bool ok = c.foreign == ROUTER;
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", "the foreign weight's name outlives its graph");
    return ok ? 0 : 1;
}

struct route_case {
    const char *     what;
    xdna_model_route route;
    bool             chunk_fits;
    bool             fused;
};

static xdna_model_route route_of(bool known, bool misfit) {
    xdna_model_route r;
    r.known  = known;
    r.misfit = misfit;
    return r;
}

template <size_t N> static int count_route_failures(const route_case (&cases)[N]) {
    int failures = 0;
    for (const route_case & c : cases) {
        const bool ok = xdna_route_fused(c.route, c.chunk_fits) == c.fused;
        std::printf("%s: %s\n", ok ? "ok  " : "FAIL", c.what);
        failures += ok ? 0 : 1;
    }
    return failures;
}

// Which decode chunks the fused layers may run: one whose width and weights they take, of a model whose layers have
// not told, or told they are run; never one of a model the census rejected, whatever its own width and weights.
static int check_route_fused(void) {
    const route_case cases[] = {
        { "a fitting chunk before the layers tell is run",             route_of(false, false), true,  true  },
        { "a fitting chunk of a model whose layers fit is run",        route_of(true,  false), true,  true  },
        { "a fitting chunk of a model the census rejected is not run", route_of(true,  true),  true,  false },
        { "a chunk outside the width or weights is not run",           route_of(true,  false), false, false },
    };
    return count_route_failures(cases);
}

int main(void) {
    const ggml_init_params params = { ggml_tensor_overhead() * MAX_TENSORS, nullptr, true };
    ggml_context *         ctx    = ggml_init(params);
    const int              failures = check_state_gather(ctx) + check_row_crop(ctx) + check_fa_plain(ctx) +
                                      check_att_rows(ctx) + check_fused_weights(ctx) + check_layer_census(ctx) +
                                      check_census_keeps_name() + check_route_fused();
    ggml_free(ctx);
    if (failures) {
        std::printf("test-xdna-match: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-match: all passed\n");
    return 0;
}
