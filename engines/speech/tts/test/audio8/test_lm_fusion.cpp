// Model-free: the language model's load-time fusion, which stacks wq|wk|wv
// (and their biases) and w1|w3 into single weights, must not change anything
// a graph computes. A tiny LM with distinct weights, once in f32 and once
// with q8_0 matrices, is loaded fused and unfused and run through a prefill,
// decode steps and a sampled fast-head frame. Every output is compared byte
// for byte: the fused matmul computes each row as the separate ones did, so
// bit-exactness is the bar, not a tolerance. A model whose q, k and v differ
// in type must keep them, and their biases, separate and still load.

#include "audio8/graph.h"
#include "audio8/internal.h"
#include "gpu_arm.h"
#include "test_env_portable.h"
#include "tiny_lm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

constexpr int GPU_LAYERS = 99;
constexpr int N_THREADS = 2;
constexpr int PROMPT_WIDTH = 5;
constexpr int DECODE_STEPS = 3;
constexpr int FIRST_TEXT_ID = 1;

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool ok, const std::string & what) {
    if (!ok) fail(what);
}

struct run_outputs {
    std::vector<std::vector<float>> floats;
    std::vector<int32_t> codes;
};

struct loaded_lm {
    lm_model model;
    bool ok = false;
    ~loaded_lm() { free_lm(model); }
};

int n_gpu_layers() {
    return audio8_test::is_gpu_test() ? GPU_LAYERS : 0;
}

bool load(const std::string & path, bool fused, loaded_lm & lm) {
    if (!fused) setenv("AUDIO8_LM_FUSION_DISABLE", "1", 1);
    std::string error;
    lm.ok = load_lm(path, n_gpu_layers(), /*backend=*/"", lm.model, &error);
    unsetenv("AUDIO8_LM_FUSION_DISABLE");
    if (!lm.ok) fail("load_lm: " + error);
    return lm.ok;
}

bool block_is_fused(const block_weights & block) {
    return block.attn.wqkv && block.w13 && !block.attn.wq && !block.w1;
}

bool block_is_split(const block_weights & block) {
    return !block.attn.wqkv && !block.w13 && block.attn.wq && block.w1;
}

bool all_blocks(const std::vector<block_weights> & blocks, bool (*test)(const block_weights &)) {
    for (const block_weights & block : blocks) {
        if (!test(block)) return false;
    }
    return true;
}

// The fused QKV prefill slices keys out of wider projected rows. A CPY into
// persistent GPU cache storage cannot fall back to CPU on OpenCL, whose CPY
// support requires contiguous source rows. Check the graph before allocation;
// the sequence below also verifies that packing preserves the computed values.
void check_key_copy_sources(lm_model & model, const std::string & tag) {
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        fail(tag + ": could not create cache-copy graph");
        return;
    }
    slow_graph_outputs outputs;
    build_slow_graph(model, build, PROMPT_WIDTH, 0, outputs);
    size_t copies = 0;
    for (int i = 0; i < ggml_graph_n_nodes(build.graph); ++i) {
        ggml_tensor * node = ggml_graph_node(build.graph, i);
        if (node->op != GGML_OP_CPY || node->src[1]->view_src != model.slow_kv.k) continue;
        ++copies;
        expect(ggml_is_contiguous(node->src[0]), tag + ": persistent key CPY has strided source");
    }
    expect(copies == model.blocks.size(), tag + ": missing persistent key copies");
}

void check_layouts(const lm_model & fused, const lm_model & split, const std::string & tag) {
    expect(all_blocks(fused.blocks, block_is_fused) &&
               all_blocks(fused.fast_blocks, block_is_fused),
           tag + ": the default load did not fuse every block");
    expect(all_blocks(split.blocks, block_is_split) &&
               all_blocks(split.fast_blocks, block_is_split),
           tag + ": AUDIO8_LM_FUSION_DISABLE still fused a block");
    expect(fused.hp.qkv_bias == (fused.blocks.front().attn.wqkv_b != nullptr),
           tag + ": the stacked bias does not follow qkv_bias");
}

std::vector<int32_t> text_frames(const lm_hparams & hp) {
    const int rows = hp.num_codebooks + 1;
    std::vector<int32_t> frames(static_cast<size_t>(rows) * PROMPT_WIDTH, 0);
    for (int column = 0; column < PROMPT_WIDTH; ++column) {
        frames[static_cast<size_t>(column) * rows] = FIRST_TEXT_ID + column;
    }
    return frames;
}

std::vector<int32_t> semantic_frame(const lm_hparams & hp, int step) {
    std::vector<int32_t> frame(hp.num_codebooks + 1);
    frame[0] = hp.semantic_begin + step;
    for (int book = 0; book < hp.num_codebooks; ++book) {
        frame[book + 1] = (step + book) % hp.codebook_size;
    }
    return frame;
}

bool slow(lm_model & model, const std::vector<int32_t> & frames, int width, int n_past,
          run_outputs & out, std::vector<float> & fast_input) {
    std::vector<float> logits;
    std::string error;
    if (!slow_step(model, frames.data(), width, n_past, N_THREADS, logits, fast_input,
                   &error)) {
        fail("slow_step: " + error);
        return false;
    }
    out.floats.push_back(logits);
    out.floats.push_back(fast_input);
    return true;
}

bool decode_steps(lm_model & model, run_outputs & out, std::vector<float> & fast_input) {
    for (int step = 0; step < DECODE_STEPS; ++step) {
        if (!slow(model, semantic_frame(model.hp, step), 1, PROMPT_WIDTH + step, out,
                  fast_input)) {
            return false;
        }
    }
    return true;
}

bool fast_frame_codes(lm_model & model, const std::vector<float> & fast_input,
                      run_outputs & out) {
    const code_picker pick = [&out](const std::vector<float> & logits, int) {
        out.floats.push_back(logits);
        return argmax_of(logits);
    };
    std::string error;
    if (!fast_step(model, fast_input, model.hp.semantic_begin, N_THREADS, pick, out.codes,
                   &error)) {
        fail("fast_step: " + error);
        return false;
    }
    return true;
}

bool run_sequence(lm_model & model, run_outputs & out) {
    std::vector<float> fast_input;
    return slow(model, text_frames(model.hp), PROMPT_WIDTH, 0, out, fast_input) &&
           decode_steps(model, out, fast_input) && fast_frame_codes(model, fast_input, out);
}

bool same_bytes(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void compare(const run_outputs & fused, const run_outputs & split, const std::string & tag) {
    expect(fused.floats.size() == split.floats.size(), tag + ": different output counts");
    for (size_t index = 0; index < fused.floats.size() && index < split.floats.size(); ++index) {
        expect(same_bytes(fused.floats[index], split.floats[index]),
               tag + ": output " + std::to_string(index) + " differs between layouts");
    }
    expect(fused.codes == split.codes, tag + ": the fast head picked different codes");
}

size_t projection_count(ggml_cgraph * graph, const ggml_tensor * weight) {
    size_t count = 0;
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        const ggml_tensor * node = ggml_graph_node(graph, index);
        if (node->op == GGML_OP_MUL_MAT &&
            (node->src[0] == weight || node->src[0]->view_src == weight)) ++count;
    }
    return count;
}

void check_projection_width(lm_model & model, int width, const std::string & tag) {
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        fail(tag + ": could not create projection graph");
        return;
    }
    slow_graph_outputs outputs;
    build_slow_graph(model, build, width, 0, outputs);
    const auto & block = model.blocks.front();
    const bool separate = width > 1 && block.attn.wqkv->type == GGML_TYPE_F32 &&
        tts_cpp::detail::reg_name_is_cuda(tts_cpp::detail::backend_reg_name(model.backend));
    expect(projection_count(build.graph, block.attn.wqkv) == (separate ? 3u : 1u),
           tag + ": unexpected QKV projection count at width " + std::to_string(width));
    expect(projection_count(build.graph, block.w13) == (separate ? 2u : 1u),
           tag + ": unexpected FFN projection count at width " + std::to_string(width));
}

void check_projection_routes(lm_model & model, const std::string & tag) {
    for (int width : {1, 3, 5, 8}) check_projection_width(model, width, tag);
}

void check_tier(bool q8_matrices) {
    const std::string tag = q8_matrices ? "q8_0" : "f32";
    audio8_test::tiny_lm params;
    params.varied = true;
    params.q8_matrices = q8_matrices;
    const std::string path = audio8_test::write_tiny_lm_gguf(
        params, test_tmpdir() + "/test-audio8-lm-fusion-" + tag + "-" + test_process_tag() +
                    ".gguf");
    loaded_lm fused, split;
    if (!load(path, /*fused=*/true, fused) || !load(path, /*fused=*/false, split)) return;
    if (!audio8_test::check_requested_gpu("lm-fusion", fused.model.backend)) {
        fail(tag + ": the requested GPU backend did not resolve");
        return;
    }
    check_layouts(fused.model, split.model, tag);
    check_projection_routes(fused.model, tag);
    check_key_copy_sources(fused.model, tag + " fused");
    check_key_copy_sources(split.model, tag + " split");
    run_outputs fused_out, split_out;
    if (!run_sequence(fused.model, fused_out) || !run_sequence(split.model, split_out)) return;
    compare(fused_out, split_out, tag);
    std::printf("[lm-fusion] %s on %s: %zu outputs and %zu codes identical\n", tag.c_str(),
                ggml_backend_name(fused.model.backend), fused_out.floats.size(),
                fused_out.codes.size());
    std::remove(path.c_str());
}

bool attention_is_split(const block_weights & block) {
    return !block.attn.wqkv && !block.attn.wqkv_b && block.attn.wq && block.attn.wq_b &&
           block.w13;
}

// A layer whose q, k and v cannot be stacked keeps its biases separate too,
// so the load succeeds and the gate/up pair still stacks.
void check_unstackable_projections() {
    audio8_test::tiny_lm params;
    params.varied = true;
    params.q8_matrices = true;
    params.f32_keys = true;
    const std::string path = audio8_test::write_tiny_lm_gguf(
        params, test_tmpdir() + "/test-audio8-lm-fusion-mixed-" + test_process_tag() + ".gguf");
    loaded_lm fused, split;
    if (!load(path, /*fused=*/true, fused) || !load(path, /*fused=*/false, split)) return;
    expect(all_blocks(fused.model.blocks, attention_is_split),
           "mixed: q, k and v of different types were stacked, or their biases were");
    run_outputs fused_out, split_out;
    if (!run_sequence(fused.model, fused_out) || !run_sequence(split.model, split_out)) return;
    compare(fused_out, split_out, "mixed");
    std::remove(path.c_str());
}

bool load_refused(const std::string & path, bool fused, const std::string & want,
                  const std::string & tag) {
    if (!fused) setenv("AUDIO8_LM_FUSION_DISABLE", "1", 1);
    lm_model model;
    std::string error;
    const bool loaded = load_lm(path, n_gpu_layers(), /*backend=*/"", model, &error);
    unsetenv("AUDIO8_LM_FUSION_DISABLE");
    free_lm(model);
    if (loaded) {
        fail(tag + ": a file that disagrees with its head geometry loaded");
        return false;
    }
    expect(error.find(want) != std::string::npos,
           tag + ": the refusal does not name " + want + ": " + error);
    return true;
}

void check_refused(const audio8_test::tiny_lm & params, const std::string & want,
                   const std::string & tag) {
    const std::string path = audio8_test::write_tiny_lm_gguf(
        params, test_tmpdir() + "/test-audio8-lm-fusion-" + tag + "-" + test_process_tag() +
                    ".gguf");
    load_refused(path, /*fused=*/true, want, tag + " (fused)");
    load_refused(path, /*fused=*/false, want, tag + " (separate)");
    std::remove(path.c_str());
}

// The graph cuts heads by the hparams, so a file whose keys are a row too long
// is refused at load, stacked or not.
void check_mismatched_keys() {
    audio8_test::tiny_lm params;
    params.varied = true;
    params.extra_key_rows = 1;
    check_refused(params, "lm/blk/0/w", "wrong-keys");
}

// A tensor already named like a stack must neither be overwritten with the
// stacked parts nor accepted in a shape the head geometry rules out.
void check_stray_stack_name(bool q8_matrices) {
    audio8_test::tiny_lm params;
    params.varied = true;
    params.q8_matrices = q8_matrices;
    params.stray_tensor = "lm/blk/0/wqkv";
    check_refused(params, "lm/blk/0/wqkv", q8_matrices ? "stray-stack-q8_0" : "stray-stack");
}

}  // namespace

int main() {
    check_tier(/*q8_matrices=*/false);
    check_tier(/*q8_matrices=*/true);
    check_unstackable_projections();
    check_mismatched_keys();
    check_stray_stack_name(/*q8_matrices=*/false);
    check_stray_stack_name(/*q8_matrices=*/true);
    if (g_failures == 0) {
        std::printf("[lm-fusion] PASS\n");
        return 0;
    }
    std::printf("[lm-fusion] FAIL: %d case(s)\n", g_failures);
    return 1;
}
