#include "audio8/graph.h"
#include "audio8/internal.h"

#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

// The codebook rows only contribute where the semantic row holds a codec
// token; elsewhere the reference zeroes them, so a per-position gate stands in
// for the reference's torch.where.
ggml_tensor * sum_codebooks(ggml_context * ctx, ggml_tensor * table, ggml_tensor * ids,
                            ggml_tensor * gate, int hidden, int codebooks, int width) {
    ggml_tensor * rows = ggml_get_rows(ctx, table, ids);
    ggml_tensor * shaped = ggml_reshape_3d(ctx, rows, hidden, codebooks, width);
    const size_t stride = static_cast<size_t>(hidden) * codebooks * sizeof(float);
    ggml_tensor * total = nullptr;
    for (int index = 0; index < codebooks; ++index) {
        ggml_tensor * slice = ggml_view_2d(ctx, shaped, hidden, width, stride,
                                           static_cast<size_t>(index) * hidden * sizeof(float));
        total = total ? ggml_add(ctx, total, slice) : ggml_cont(ctx, slice);
    }
    return ggml_mul(ctx, total, gate);
}

ggml_tensor * embed_frames(ggml_context * ctx, const lm_model & model, int width) {
    const lm_hparams & hp = model.hp;
    ggml_tensor * text = input_i32(ctx, "text_ids", width);
    ggml_tensor * codes = input_i32(ctx, "codebook_ids", width * hp.num_codebooks);
    ggml_tensor * gate = input_f32(ctx, "semantic_gate", 1, width);
    ggml_tensor * base = ggml_get_rows(ctx, model.tok_emb, text);
    ggml_tensor * stacked = sum_codebooks(ctx, model.codebook_emb, codes, gate, hp.hidden,
                                          hp.num_codebooks, width);
    return ggml_add(ctx, base, stacked);
}

ggml_tensor * feed_forward(ggml_context * ctx, const block_weights & block, ggml_tensor * x) {
    return block.w13 ? swiglu_stacked(ctx, block.w13, block.w2, x)
                     : swiglu(ctx, block.w1, block.w2, block.w3, x);
}

ggml_tensor * run_block(ggml_context * ctx, ggml_cgraph * graph, const block_weights & block,
                        ggml_tensor * x, const rope_planes & rope, const kv_cache & cache,
                        const attention_shape & shape, ggml_tensor * mask, float eps) {
    ggml_tensor * normed = rms_norm(ctx, x, block.attn.attn_norm, eps);
    ggml_tensor * hidden = ggml_add(
        ctx, x, attention(ctx, graph, block.attn, normed, rope, cache, shape, mask));
    ggml_tensor * gated = rms_norm(ctx, hidden, block.ffn_norm, eps);
    return ggml_add(ctx, hidden, feed_forward(ctx, block, gated));
}

ggml_tensor * run_blocks(ggml_context * ctx, ggml_cgraph * graph,
                         const std::vector<block_weights> & blocks, ggml_tensor * x,
                         const rope_planes & rope, const kv_cache & cache,
                         attention_shape shape, ggml_tensor * mask, float eps) {
    for (size_t index = 0; index < blocks.size(); ++index) {
        shape.layer = static_cast<int>(index);
        x = run_block(ctx, graph, blocks[index], x, rope, cache, shape, mask, eps);
    }
    return x;
}

ggml_tensor * last_column(ggml_context * ctx, ggml_tensor * x) {
    const size_t offset = static_cast<size_t>(x->ne[1] - 1) * x->nb[1];
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], offset);
}

attention_shape slow_shape(const lm_model & model, int width, int n_past) {
    const lm_hparams & hp = model.hp;
    return {hp.n_head, hp.n_kv, hp.head_dim, width, n_past, 0,
            model.precise_outputs};
}

attention_shape fast_shape(const lm_model & model, int position) {
    const lm_hparams & hp = model.hp;
    return {hp.fast_n_head, hp.fast_n_kv, hp.fast_head_dim, 1, position, 0,
            model.precise_outputs};
}

bool is_semantic(const lm_hparams & hp, int32_t token) {
    return token >= hp.semantic_begin && token <= hp.semantic_end;
}

// Row 0 carries text or semantic ids; the codebook rows are read per position
// with the reference's per-codebook offset already applied.
struct frame_inputs {
    std::vector<int32_t> text;
    std::vector<int32_t> codes;
    std::vector<float> gate;
    std::vector<float> mask;
};

void fill_frame_inputs(const lm_hparams & hp, const int32_t * frames, int width, int n_past,
                       frame_inputs & inputs) {
    const int rows = hp.num_codebooks + 1;
    inputs.text.resize(width);
    inputs.codes.resize(static_cast<size_t>(width) * hp.num_codebooks);
    inputs.gate.resize(width);
    for (int column = 0; column < width; ++column) {
        const int32_t * frame = frames + static_cast<size_t>(column) * rows;
        inputs.text[column] = frame[0];
        inputs.gate[column] = is_semantic(hp, frame[0]) ? 1.0f : 0.0f;
        for (int book = 0; book < hp.num_codebooks; ++book) {
            inputs.codes[static_cast<size_t>(column) * hp.num_codebooks + book] =
                frame[book + 1] + book * hp.codebook_size;
        }
    }
    if (width == 1) return;
    inputs.mask.resize(static_cast<size_t>(width) * (n_past + width));
    fill_causal_mask(inputs.mask.data(), n_past + width, width, n_past, /*window=*/0);
}

void set_frame_inputs(ggml_cgraph * graph, const frame_inputs & inputs) {
    write_input(graph, "text_ids", inputs.text.data(), inputs.text.size() * sizeof(int32_t));
    write_input(graph, "codebook_ids", inputs.codes.data(),
                inputs.codes.size() * sizeof(int32_t));
    write_input(graph, "semantic_gate", inputs.gate.data(), inputs.gate.size() * sizeof(float));
    if (inputs.mask.empty()) return;
    write_input(graph, "mask", inputs.mask.data(), inputs.mask.size() * sizeof(float));
}

}  // namespace

int argmax_of(const std::vector<float> & values) {
    int best = 0;
    for (size_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[best]) best = static_cast<int>(index);
    }
    return best;
}

prompt_frames build_frames(const lm_hparams & hp, const PromptSegments & segments,
                           const std::vector<int32_t> & reference_codes, int reference_len) {
    prompt_frames frames;
    frames.rows = hp.num_codebooks + 1;
    frames.width = static_cast<int>(segments.prefix.size() + segments.suffix.size()) +
                   reference_len;
    frames.values.assign(static_cast<size_t>(frames.rows) * frames.width, 0);

    int column = 0;
    for (int32_t token : segments.prefix) {
        frames.values[static_cast<size_t>(column++) * frames.rows] = token;
    }
    for (int index = 0; index < reference_len; ++index) {
        const size_t base = static_cast<size_t>(column + index) * frames.rows;
        frames.values[base] = reference_codes[index] + hp.semantic_begin;
        for (int book = 0; book < hp.num_codebooks; ++book) {
            frames.values[base + book + 1] =
                reference_codes[static_cast<size_t>(book) * reference_len + index];
        }
    }
    column += reference_len;
    for (int32_t token : segments.suffix) {
        frames.values[static_cast<size_t>(column++) * frames.rows] = token;
    }
    return frames;
}

void build_slow_graph(lm_model & model, scratch & build, int width, int n_past,
                      slow_graph_outputs & outs) {
    const lm_hparams & hp = model.hp;
    ggml_context * ctx = build.ctx;
    ggml_tensor * mask = width == 1 ? nullptr : input_f32(ctx, "mask", n_past + width, width);
    ggml_tensor * hidden = embed_frames(ctx, model, width);
    const rope_planes rope = rope_window(ctx, model.rope_cos, model.rope_sin, n_past, width);
    hidden = run_blocks(ctx, build.graph, model.blocks, hidden, rope, model.slow_kv,
                        slow_shape(model, width, n_past), mask, hp.rms_eps);

    ggml_tensor * tail = ggml_cont(ctx, last_column(ctx, hidden));
    ggml_tensor * normed = rms_norm(ctx, tail, model.norm, hp.rms_eps);
    outs.logits = mark_output(
        build.graph, multiply_mat(ctx, model.sem_head, normed, model.precise_outputs));
    outs.carried = mark_output(build.graph, hp.norm_fast_input ? normed : tail);
}

void set_slow_graph_inputs(const lm_model & model, ggml_cgraph * graph,
                           const int32_t * frames, int width, int n_past) {
    frame_inputs inputs;
    fill_frame_inputs(model.hp, frames, width, n_past, inputs);
    set_frame_inputs(graph, inputs);
}

bool slow_step(lm_model & model, const int32_t * frames, int width, int n_past,
               int n_threads, std::vector<float> & sem_logits,
               std::vector<float> & fast_input, std::string * error) {
    const lm_hparams & hp = model.hp;
    if (n_past + width > hp.max_seq_len) {
        if (error) *error = "audio8: prompt and generation exceed the model's context";
        return false;
    }
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        if (error) *error = "audio8: failed to create the slow graph context";
        return false;
    }
    slow_graph_outputs outs;
    build_slow_graph(model, build, width, n_past, outs);
    ggml_tensor * logits  = outs.logits;
    ggml_tensor * carried = outs.carried;

    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.slow_allocr,
                       build.graph, "slow", use_sched, error)) {
        return false;
    }
    set_slow_graph_inputs(model, build.graph, frames, width, n_past);
    if (!compute_graph(model.backend, model.sched, build.graph, use_sched, n_threads,
                       "slow", error)) {
        return false;
    }

    read_output(logits, sem_logits);
    read_output(carried, fast_input);
    return true;
}

namespace {

// Position 0 consumes the slow transformer's hidden state and only primes the
// cache; later positions read the previous codebook's embedding.
struct fast_source {
    const void * data = nullptr;
    size_t bytes = 0;
    bool is_code = false;
};

ggml_tensor * fast_input_tensor(ggml_context * ctx, const lm_model & model, bool is_code) {
    if (!is_code) {
        return input_f32(ctx, "input", model.hp.fast_hidden, 1);
    }
    return ggml_get_rows(ctx, model.fast_emb, input_i32(ctx, "input", 1));
}

// The graph half of fast_pass, shared with the fit projector so the priced
// graph is the executed graph by construction.
ggml_tensor * build_fast_pass_graph(lm_model & model, scratch & build, int position,
                                    bool is_code) {
    const lm_hparams & hp = model.hp;
    ggml_context * ctx = build.ctx;
    ggml_tensor * hidden = fast_input_tensor(ctx, model, is_code);
    const rope_planes rope = rope_window(ctx, model.fast_rope_cos, model.fast_rope_sin,
                                         position, 1);
    hidden = run_blocks(ctx, build.graph, model.fast_blocks, hidden, rope, model.fast_kv,
                        fast_shape(model, position), /*mask=*/nullptr, hp.rms_eps);
    return mark_output(build.graph,
                       multiply_mat(ctx, model.fast_out,
                                    rms_norm(ctx, hidden, model.fast_norm, hp.rms_eps),
                                    model.precise_outputs));
}

void drop_cached_fast_graph(lm_model::fast_graph & cached) {
    if (cached.ctx) ggml_free(cached.ctx);
    if (cached.allocr) ggml_gallocr_free(cached.allocr);
    cached = lm_model::fast_graph{};
}

// A frame walks the same ten positions every time, so each position's graph is
// built once and replayed. Nothing in it depends on the frame: the shapes come
// from the position, the weights and the fast cache are resident, and the two
// inputs are written before every run.
bool build_cached_fast_graph(lm_model & model, int position, bool is_code,
                             lm_model::fast_graph & cached, std::string * error) {
    scratch build(AUDIO8_FAST_MAX_NODES);
    if (!build.ok()) {
        if (error) *error = "audio8: failed to create the fast graph context";
        return false;
    }
    cached.logits = build_fast_pass_graph(model, build, position, is_code);
    cached.allocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!cached.allocr) {
        if (error) *error = "audio8: failed to create the fast graph allocator";
        return false;
    }
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, cached.allocr,
                       build.graph, "fast", cached.use_sched, error)) {
        ggml_gallocr_free(cached.allocr);
        cached.allocr = nullptr;
        return false;
    }
    cached.ctx = build.ctx;
    cached.graph = build.graph;
    build.release();
    return true;
}

// A cached graph is keyed by position alone, which is only sound because the
// position decides the input: position 0 always primes from the slow hidden
// state and every later one always reads a code. Stated rather than assumed,
// so a caller that broke it gets an error instead of the wrong graph.
bool source_matches_position(const fast_source & source, int position) {
    return source.is_code == (position != 0);
}

bool run_fast_graph(lm_model & model, ggml_cgraph * graph, bool use_sched,
                    const fast_source & source, ggml_tensor * logits, int n_threads,
                    std::vector<float> * logits_out, std::string * error) {
    write_input(graph, "input", source.data, source.bytes);
    if (!compute_graph(model.backend, model.sched, graph, use_sched, n_threads, "fast",
                       error)) {
        return false;
    }
    if (logits_out) read_output(logits, *logits_out);
    return true;
}

// What fast_pass did before the cache, and still does wherever a graph cannot
// be replayed.
bool fast_pass_uncached(lm_model & model, const fast_source & source, int position,
                        int n_threads, std::vector<float> * logits_out,
                        std::string * error) {
    scratch build(AUDIO8_FAST_MAX_NODES);
    if (!build.ok()) {
        if (error) *error = "audio8: failed to create the fast graph context";
        return false;
    }
    ggml_tensor * logits = build_fast_pass_graph(model, build, position, source.is_code);
    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.fast_allocr,
                       build.graph, "fast", use_sched, error)) {
        return false;
    }
    return run_fast_graph(model, build.graph, use_sched, source, logits, n_threads,
                          logits_out, error);
}

bool position_accepts_source(const lm_model & model, const fast_source & source,
                             int position, std::string * error) {
    if (position < 0 || position >= model.hp.num_codebooks) {
        if (error) *error = "audio8: fast position outside the codebook range";
        return false;
    }
    if (!source_matches_position(source, position)) {
        if (error) *error = "audio8: fast position and input kind disagree";
        return false;
    }
    return true;
}

// One transition, wherever it is discovered: give back what was built and stay
// on the per-call path for the rest of the model's life. A backend that needs
// the scheduler cannot keep a graph at all, because
// ggml_backend_sched_alloc_graph resets one shared arena and rewrites
// node->src[] in place.
void disable_fast_cache(lm_model & model) {
    for (lm_model::fast_graph & cached : model.fast_graphs) {
        drop_cached_fast_graph(cached);
    }
    model.fast_cache_off = true;
}

// Builds the position on first use. Returns null once the cache is off, which
// includes the build that discovers this backend needs the scheduler: that one
// runs through the per-call path like every one after it.
//
// Nothing here reads the force hook. prepare_graph does, which is what lets a
// test reach the branch below on a backend that supports every node -- reading
// it here instead would route past the branch and leave it unreachable. The
// cost is that flipping the hook on after a position was already replayed does
// not reach the scheduler; set it before the first frame, as the other
// sched-equivalence harnesses do.
lm_model::fast_graph * cached_fast_graph(lm_model & model, int position,
                                         bool is_code, std::string * error) {
    if (model.fast_graphs.empty()) {
        model.fast_graphs.resize(static_cast<size_t>(model.hp.num_codebooks));
    }
    lm_model::fast_graph & cached = model.fast_graphs[static_cast<size_t>(position)];
    if (cached.graph) return &cached;
    if (!build_cached_fast_graph(model, position, is_code, cached, error)) return nullptr;
    if (cached.use_sched) {
        disable_fast_cache(model);
        return nullptr;
    }
    return &cached;
}

bool fast_pass(lm_model & model, const fast_source & source, int position, int n_threads,
               std::vector<float> * logits_out, std::string * error) {
    if (!position_accepts_source(model, source, position, error)) return false;
    if (model.fast_cache_off) {
        return fast_pass_uncached(model, source, position, n_threads, logits_out, error);
    }
    std::string build_error;
    lm_model::fast_graph * cached =
        cached_fast_graph(model, position, source.is_code, &build_error);
    if (!cached) {
        if (model.fast_cache_off) {
            return fast_pass_uncached(model, source, position, n_threads, logits_out,
                                      error);
        }
        if (error) *error = build_error;
        return false;
    }
    return run_fast_graph(model, cached->graph, cached->use_sched, source,
                          cached->logits, n_threads, logits_out, error);
}

int clamp_to_codebook(const lm_hparams & hp, int semantic) {
    const int index = semantic - hp.semantic_begin;
    if (index < 0) return 0;
    return index >= hp.codebook_size ? hp.codebook_size - 1 : index;
}

// Every fast position attends to the whole frame prefix, so its causal mask is
// all zeros, and adding zero before the softmax is exact. Passing no mask keeps
// the chained graph free of ten per-position mask inputs.
ggml_tensor * fast_logits(ggml_context * ctx, ggml_cgraph * graph, lm_model & model,
                          ggml_tensor * hidden, int position) {
    const lm_hparams & hp = model.hp;
    const rope_planes rope =
        rope_window(ctx, model.fast_rope_cos, model.fast_rope_sin, position, 1);
    ggml_tensor * out =
        run_blocks(ctx, graph, model.fast_blocks, hidden, rope, model.fast_kv,
                   fast_shape(model, position), /*mask=*/nullptr, hp.rms_eps);
    return multiply_mat(ctx, model.fast_out,
                        rms_norm(ctx, out, model.fast_norm, hp.rms_eps),
                        model.precise_outputs);
}

// Position 0 primes the cache from the slow transformer's hidden state and
// produces no code; position p reads the code position p-1 chose, so the whole
// frame is one dependency chain that never leaves the backend.
std::vector<ggml_tensor *> chain_codes(ggml_context * ctx, ggml_cgraph * graph,
                                       lm_model & model, ggml_tensor * primed,
                                       ggml_tensor * first_code) {
    ggml_build_forward_expand(graph, primed);
    std::vector<ggml_tensor *> chosen;
    ggml_tensor * code = first_code;
    for (int position = 1; position < model.hp.num_codebooks; ++position) {
        ggml_tensor * hidden = ggml_get_rows(ctx, model.fast_emb, code);
        code = ggml_argmax(ctx, fast_logits(ctx, graph, model, hidden, position));
        chosen.push_back(mark_output(graph, code));
    }
    return chosen;
}

void read_chosen(const std::vector<ggml_tensor *> & chosen, std::vector<int32_t> & codes_out) {
    for (size_t index = 0; index < chosen.size(); ++index) {
        ggml_backend_tensor_get(chosen[index], &codes_out[index + 1], 0, sizeof(int32_t));
    }
}

// The graph half of fast_frame, shared with the fit projector.
std::vector<ggml_tensor *> build_fast_frame_graph(lm_model & model, scratch & build) {
    ggml_context * ctx = build.ctx;
    ggml_tensor * primed = fast_logits(
        ctx, build.graph, model, input_f32(ctx, "input", model.hp.fast_hidden, 1), 0);
    return chain_codes(ctx, build.graph, model, primed, input_i32(ctx, "first_code", 1));
}

}  // namespace

void build_fast_fit_graph(lm_model & model, scratch & build, int position, bool prime) {
    build_fast_pass_graph(model, build, position, /*is_code=*/!prime);
}

void build_fast_frame_fit_graph(lm_model & model, scratch & build) {
    build_fast_frame_graph(model, build);
}

// Greedy expansion of one frame in a single graph. The sampled path cannot do
// this: argmax is the only picker the backend can evaluate, so any other one has
// to return to the host between positions.
bool fast_frame(lm_model & model, const std::vector<float> & fast_input, int semantic,
                int n_threads, std::vector<int32_t> & codes_out, std::string * error) {
    const lm_hparams & hp = model.hp;
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        if (error) *error = "audio8: failed to create the fast frame context";
        return false;
    }
    const int32_t first_code = clamp_to_codebook(hp, semantic);
    const std::vector<ggml_tensor *> chosen = build_fast_frame_graph(model, build);

    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.frame_allocr,
                       build.graph, "fast frame", use_sched, error)) {
        return false;
    }
    write_input(build.graph, "input", fast_input.data(), fast_input.size() * sizeof(float));
    write_input(build.graph, "first_code", &first_code, sizeof(first_code));
    if (!compute_graph(model.backend, model.sched, build.graph, use_sched, n_threads,
                       "fast frame", error)) {
        return false;
    }

    codes_out.assign(hp.num_codebooks, 0);
    codes_out[0] = first_code;
    read_chosen(chosen, codes_out);
    return true;
}

bool fast_step(lm_model & model, const std::vector<float> & fast_input, int semantic,
               int n_threads, const code_picker & pick, std::vector<int32_t> & codes_out,
               std::string * error) {
    const lm_hparams & hp = model.hp;
    const fast_source prime = {fast_input.data(), fast_input.size() * sizeof(float), false};
    if (!fast_pass(model, prime, 0, n_threads, nullptr, error)) return false;

    codes_out.assign(hp.num_codebooks, 0);
    codes_out[0] = clamp_to_codebook(hp, semantic);
    std::vector<float> logits;
    for (int position = 1; position < hp.num_codebooks; ++position) {
        const int32_t token = codes_out[position - 1];
        const fast_source step = {&token, sizeof(token), true};
        if (!fast_pass(model, step, position, n_threads, &logits, error)) return false;
        codes_out[position] = pick(logits, position);
    }
    return true;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
