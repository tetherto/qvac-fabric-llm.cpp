// Model-free ACE-Step loading regression tests.

#include "acestep_test_support.h"
#include "parallel_load.h"
#include "dit_gguf.h"
#include "dit_ggml.h"
#include "qwen3_block.h"
#include "stage_dump_io.h"
#include "lm_ggml.h"
#include "bpe_tokenizer.h"
#include "vae_ggml.h"
#include "vae_gguf.h"
#include "fit_measure.h"
#include "engine_backends.h"
#include "ggml-alloc.h"
#include "gguf.h"

namespace {

void test_parallel_rows() {
    using tts_cpp::acestep::parallel_rows;
    for (int n : { 0, 1, 127, 128, 129, 255, 4096, 12345 }) {
        // Workers only touch atomics; every CHECK stays on the main thread.
        std::vector<std::atomic<int>> hits((size_t) n);
        std::atomic<bool>             bounds_ok{ true };
        parallel_rows(n, [&](int begin, int end) {
            if (begin < 0 || end > n) bounds_ok = false;
            for (int i = begin; i < end && i >= 0; i++) hits[(size_t) i].fetch_add(1);
        });
        CHECK(bounds_ok);
        bool all_once = true;
        for (const auto & h : hits) all_once = all_once && h.load() == 1;
        CHECK(all_once);
    }
}

void test_convert_f32_to_f16_rows() {
    using tts_cpp::acestep::convert_f32_to_f16_rows;
    using tts_cpp::acestep::F16_CONVERT_CHUNK;
    using tts_cpp::acestep::LOAD_SERIAL_MIN_ROWS;
    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    // The last size crosses LOAD_SERIAL_MIN_ROWS chunks, exercising the
    // threaded branch with a ragged tail.
    const size_t threaded = (size_t) LOAD_SERIAL_MIN_ROWS * F16_CONVERT_CHUNK + 7;
    for (size_t count : { (size_t) 0, (size_t) 1, F16_CONVERT_CHUNK - 1, F16_CONVERT_CHUNK,
                          F16_CONVERT_CHUNK + 1, 2 * F16_CONVERT_CHUNK, (size_t) 65537, threaded }) {
        std::vector<float> src(count);
        for (auto & v : src) v = dist(rng);
        std::vector<ggml_fp16_t> serial(count);
        std::vector<ggml_fp16_t> chunked(count);
        if (count > 0) {
            ggml_fp32_to_fp16_row(src.data(), serial.data(), (int) count);
            convert_f32_to_f16_rows(src.data(), chunked.data(), count);
            CHECK(std::memcmp(serial.data(), chunked.data(), count * sizeof(ggml_fp16_t)) == 0);
        } else {
            convert_f32_to_f16_rows(src.data(), chunked.data(), count);
        }
    }
}

// lm_load_row_block copies one GGUF tensor into a row-concatenated fused tensor
// at a running byte offset; lm_load_layer_fused chains q|k|v and gate|up. A
// missing tensor must fail the load (false, offset untouched): skipping a block
// would byte-shift every later one into silently wrong weights. The tiny GGUF
// is written by the test itself; CPU buffers only.
// POSIX runners set TMPDIR; Windows runners set TEMP/TMP and have no /tmp, so
// a plain TMPDIR-or-/tmp fallback makes every file-writing test fail there.
std::string write_fused_test_gguf(const std::vector<std::pair<std::string, std::vector<float>>> & tensors, int ne0) {
    const std::string path = test_temp_dir() + "/qvac-acestep-fused-load-test.gguf";

    ggml_init_params ip{ 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    for (const auto & [name, vals] : tensors) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, (int64_t) vals.size() / ne0);
        ggml_set_name(t, name.c_str());
        std::memcpy(t->data, vals.data(), vals.size() * sizeof(float));
        gguf_add_tensor(gc, t);
    }
    CHECK(gguf_write_to_file(gc, path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);
    return path;
}

void test_fused_load_fail_closed() {
    using tts_cpp::acestep::DitGGUF;
    using tts_cpp::acestep::Qwen3Layer;
    using tts_cpp::acestep::dit_gguf_close;
    using tts_cpp::acestep::dit_gguf_open;
    using tts_cpp::acestep::lm_load_layer_fused;
    using tts_cpp::acestep::lm_load_row_block;

    const int ne0 = 4;
    auto      seq = [](int n, float base) {
        std::vector<float> v((size_t) n);
        for (int i = 0; i < n; ++i) v[(size_t) i] = base + (float) i;
        return v;
    };
    const std::string p = "model.layers.0";
    const auto q = seq(8, 0), k = seq(4, 100), v = seq(4, 200), gate = seq(8, 300), up = seq(8, 400);

    const std::string path = write_fused_test_gguf(
        { { p + ".self_attn.q_proj.weight", q },
          { p + ".self_attn.k_proj.weight", k },
          { p + ".self_attn.v_proj.weight", v },
          { p + ".mlp.gate_proj.weight", gate },
          { p + ".mlp.up_proj.weight", up } },
        ne0);

    DitGGUF g;
    CHECK(dit_gguf_open(g, path));

    ggml_init_params      ip{ 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context *        ctx = ggml_init(ip);
    ggml_tensor *         qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 4);  // q(2) + k(1) + v(1) rows
    ggml_tensor *         gu  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 4);  // gate(2) + up(2) rows
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    CHECK(buf != nullptr);

    // Happy path: all five blocks land back to back, in declaration order.
    Qwen3Layer ly{};  // norms/o/down stay null; q3_load_* skip a null dst
    CHECK(lm_load_layer_fused(g, p, ly, qkv, gu));
    std::vector<float> qkv_want(q);
    qkv_want.insert(qkv_want.end(), k.begin(), k.end());
    qkv_want.insert(qkv_want.end(), v.begin(), v.end());
    std::vector<float> gu_want(gate);
    gu_want.insert(gu_want.end(), up.begin(), up.end());
    CHECK(std::memcmp(qkv->data, qkv_want.data(), qkv_want.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(gu->data, gu_want.data(), gu_want.size() * sizeof(float)) == 0);

    // Missing tensor: false, and the offset must not advance.
    size_t off = 0;
    CHECK(lm_load_row_block(qkv, off, g, p + ".self_attn.q_proj.weight"));
    const size_t after_q = off;
    CHECK(!lm_load_row_block(qkv, off, g, p + ".self_attn.absent.weight"));
    CHECK(off == after_q);
    dit_gguf_close(g);

    // v_proj absent -> the whole fused layer load fails.
    const std::string path2 = write_fused_test_gguf(
        { { p + ".self_attn.q_proj.weight", q },
          { p + ".self_attn.k_proj.weight", k },
          { p + ".mlp.gate_proj.weight", gate },
          { p + ".mlp.up_proj.weight", up } },
        ne0);
    DitGGUF g2;
    CHECK(dit_gguf_open(g2, path2));
    CHECK(!lm_load_layer_fused(g2, p, ly, qkv, gu));
    dit_gguf_close(g2);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::remove(path.c_str());
}

std::string write_raw_stage_dump(const char * name, const std::vector<int32_t> & header, size_t n_floats) {
    const std::string path = test_temp_dir() + "/" + name;
    FILE *            f    = fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) return path;
    const std::vector<float> payload(n_floats, 1.0f);
    fwrite(header.data(), sizeof(int32_t), header.size(), f);
    fwrite(payload.data(), sizeof(float), payload.size(), f);
    fclose(f);
    return path;
}

void expect_stage_dump_rejected(const char * name, const std::vector<int32_t> & header, size_t n_floats) {
    const std::string  path = write_raw_stage_dump(name, header, n_floats);
    std::vector<float> out;
    int                d0 = 0, d1 = 0;
    bool               read  = true;
    bool               threw = false;
    try {
        read = tts_cpp::acestep::read_stage_dump("test", path.c_str(), out, &d0, &d1);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(!threw);
    CHECK(!read);
    CHECK(out.empty());
    std::remove(path.c_str());
}

void test_stage_dump_round_trip() {
    using tts_cpp::acestep::read_stage_dump;
    using tts_cpp::acestep::write_stage_dump;

    const std::string        path = test_temp_dir() + "/qvac-acestep-stage-dump.bin";
    const std::vector<float> v    = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    CHECK(write_stage_dump("test", path.c_str(), v, 3, 2));
    std::vector<float> out;
    int                d0 = 0, d1 = 0;
    CHECK(read_stage_dump("test", path.c_str(), out, &d0, &d1));
    CHECK(d0 == 3);
    CHECK(d1 == 2);
    CHECK(out == v);
    std::remove(path.c_str());
}

void test_stage_dump_rejects_malformed() {
    constexpr int32_t BIG = std::numeric_limits<int32_t>::max();
    expect_stage_dump_rejected("qvac-acestep-dump-huge.bin", { 2, BIG, BIG }, 0);
    expect_stage_dump_rejected("qvac-acestep-dump-huge-frames.bin", { 2, BIG, 64 }, 0);
    expect_stage_dump_rejected("qvac-acestep-dump-truncated.bin", { 2, 4, 64 }, 10);
    expect_stage_dump_rejected("qvac-acestep-dump-rank.bin", { 3, 2, 3 }, 6);
    expect_stage_dump_rejected("qvac-acestep-dump-zero.bin", { 2, 0, 3 }, 0);
    expect_stage_dump_rejected("qvac-acestep-dump-negative.bin", { 2, -1, 3 }, 3);
    expect_stage_dump_rejected("qvac-acestep-dump-short-header.bin", { 2, 3 }, 0);

    std::vector<float> out;
    int                d0 = 0, d1 = 0;
    const std::string  missing = test_temp_dir() + "/qvac-acestep-dump-missing.bin";
    std::remove(missing.c_str());
    CHECK(!tts_cpp::acestep::read_stage_dump("test", missing.c_str(), out, &d0, &d1));
}

void test_gguf_parse_failure_releases_mapping() {
    using namespace tts_cpp::acestep;
    const std::string path = test_temp_dir() + "/qvac-acestep-invalid-gguf-test.gguf";
    FILE * file = std::fopen(path.c_str(), "wb");
    CHECK(file != nullptr);
    if (!file) return;
    const char invalid[] = "not a GGUF model";
    CHECK(std::fwrite(invalid, 1, sizeof(invalid), file) == sizeof(invalid));
    std::fclose(file);

    for (int retry = 0; retry < 3; ++retry) {
        DitGGUF dit;
        VaeGGUF vae;
        CHECK(!dit_gguf_open(dit, path));
        CHECK(!vae_gguf_open(vae, path));
        CHECK(dit.ctx == nullptr && dit.meta == nullptr);
        CHECK(vae.ctx == nullptr && vae.meta == nullptr);
        CHECK(dit.file.data == nullptr && dit.file.size == 0);
        CHECK(vae.file.data == nullptr && vae.file.size == 0);
#ifdef _WIN32
        CHECK(dit.file.file_handle == nullptr && dit.file.map_handle == nullptr);
        CHECK(vae.file.file_handle == nullptr && vae.file.map_handle == nullptr);
#else
        CHECK(dit.file.fd == -1);
        CHECK(vae.file.fd == -1);
#endif
        dit_gguf_close(dit);
        vae_gguf_close(vae);
    }
    std::remove(path.c_str());
}

constexpr int TINY_LM_HIDDEN     = 128;
constexpr int TINY_LM_FFN        = 256;
constexpr int TINY_LM_VOCAB      = 200;
constexpr int TINY_LM_MAX_SEQ    = 64;
constexpr int TINY_LM_CHUNK_ROWS = 32;
constexpr int TINY_LM_PREFIX     = 150;

float tiny_lm_weight(int64_t i, float scale) {
    return std::sin(0.37f * (float) i + scale) * scale;
}

std::vector<float> tiny_lm_values(int64_t n, float scale) {
    std::vector<float> values((size_t) n);
    for (int64_t i = 0; i < n; ++i) values[(size_t) i] = scale == 0.0f ? 1.0f : tiny_lm_weight(i, scale);
    return values;
}

void add_tiny_lm_tensor(ggml_context * ctx, gguf_context * gc, const std::string & name, ggml_type type, int64_t ne0,
                        int64_t ne1, float scale) {
    const std::vector<float> values = tiny_lm_values(ne0 * ne1, scale);
    ggml_tensor *            t      = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(t, name.c_str());
    if (type == GGML_TYPE_F32) {
        std::memcpy(t->data, values.data(), values.size() * sizeof(float));
    } else {
        ggml_quantize_chunk(type, values.data(), t->data, 0, ne1, ne0, nullptr);
    }
    gguf_add_tensor(gc, t);
}

std::string write_tiny_lm_gguf() {
    const std::string path = test_temp_dir() + "/qvac-acestep-tiny-lm-test.gguf";
    const int         H = TINY_LM_HIDDEN, F = TINY_LM_FFN, D = 128;
    const std::string l = "model.layers.0";

    ggml_init_params ip{ 16 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    add_tiny_lm_tensor(ctx, gc, "model.embed_tokens.weight", GGML_TYPE_Q8_0, H, TINY_LM_VOCAB, 0.5f);
    add_tiny_lm_tensor(ctx, gc, "model.norm.weight", GGML_TYPE_F32, H, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".input_layernorm.weight", GGML_TYPE_F32, H, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".post_attention_layernorm.weight", GGML_TYPE_F32, H, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.q_norm.weight", GGML_TYPE_F32, D, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.k_norm.weight", GGML_TYPE_F32, D, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.q_proj.weight", GGML_TYPE_Q8_0, H, D, 0.05f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.k_proj.weight", GGML_TYPE_Q8_0, H, D, 0.04f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.v_proj.weight", GGML_TYPE_Q8_0, H, D, 0.03f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.o_proj.weight", GGML_TYPE_Q8_0, D, H, 0.02f);
    add_tiny_lm_tensor(ctx, gc, l + ".mlp.gate_proj.weight", GGML_TYPE_Q8_0, H, F, 0.06f);
    add_tiny_lm_tensor(ctx, gc, l + ".mlp.up_proj.weight", GGML_TYPE_Q8_0, H, F, 0.07f);
    add_tiny_lm_tensor(ctx, gc, l + ".mlp.down_proj.weight", GGML_TYPE_Q8_0, F, H, 0.08f);
    CHECK(gguf_write_to_file(gc, path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);
    return path;
}

bool chunk_ranges_tile(const std::vector<std::pair<int, int>> & ranges, int vocab, int max_rows) {
    int next = 0;
    for (const auto & [row0, rows] : ranges) {
        if (row0 != next || rows <= 0 || rows > max_rows) return false;
        next = row0 + rows;
    }
    return next == vocab;
}

bool chunk_ranges_start_at(const std::vector<std::pair<int, int>> & ranges, int row) {
    for (const auto & range : ranges) {
        if (range.first == row) return true;
    }
    return false;
}

void test_lm_head_chunk_ranges() {
    using tts_cpp::acestep::AUDIO_CODE_BASE;
    using tts_cpp::acestep::AUDIO_CODE_COUNT;
    using tts_cpp::acestep::lm_head_chunk_ranges;
    using tts_cpp::acestep::TOKEN_IM_END;

    const int  ace_vocab = AUDIO_CODE_BASE + AUDIO_CODE_COUNT;
    const auto ace       = lm_head_chunk_ranges(ace_vocab, 32768);
    CHECK(chunk_ranges_tile(ace, ace_vocab, 32768));
    CHECK(chunk_ranges_start_at(ace, TOKEN_IM_END));
    CHECK(chunk_ranges_start_at(ace, AUDIO_CODE_BASE));

    const std::vector<std::pair<int, int>> small = { { 0, 32 }, { 32, 32 }, { 64, 32 }, { 96, 4 } };
    CHECK(lm_head_chunk_ranges(100, 32) == small);
    CHECK(lm_head_chunk_ranges(100, 100) == (std::vector<std::pair<int, int>>{ { 0, 100 } }));
}

void test_lm_split_table_matches_whole_table() {
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::LMModel;
    using tts_cpp::acestep::lm_model_forward;
    using tts_cpp::acestep::lm_model_free;
    using tts_cpp::acestep::lm_model_load;
    using tts_cpp::acestep::lm_model_load_split_table;

    const std::string path = write_tiny_lm_gguf();
    ggml_backend_t    cpu  = backend_cpu_init();
    CHECK(cpu != nullptr);
    LMModel * whole = cpu ? lm_model_load(path, cpu, TINY_LM_MAX_SEQ, /*verbose=*/false, /*n_kv_sets=*/1) : nullptr;
    LMModel * split = cpu ? lm_model_load_split_table(path, cpu, TINY_LM_MAX_SEQ, 1, TINY_LM_CHUNK_ROWS) : nullptr;
    CHECK(whole != nullptr);
    CHECK(split != nullptr);
    if (whole && split) {
        // Splitting retains the host embedding table as well as the head
        // chunks. Its weight footprint must include both copies, even when
        // the backend can map the remaining transformer weights in place.
        const size_t table_bytes = ggml_row_size(GGML_TYPE_Q8_0, TINY_LM_HIDDEN) * TINY_LM_VOCAB;
        CHECK(tts_cpp::acestep::lm_model_weight_bytes(split) >=
              tts_cpp::acestep::lm_model_weight_bytes(whole) + table_bytes);
        const int32_t      prompt[] = { 0, 3, 77, 150, TINY_LM_VOCAB - 1 };
        std::vector<float> whole_logits, split_logits;
        CHECK(lm_model_forward(whole, prompt, 5, whole_logits));
        CHECK(lm_model_forward(split, prompt, 5, split_logits));
        CHECK(whole_logits.size() == (size_t) TINY_LM_VOCAB);
        CHECK(whole_logits == split_logits);

        const int32_t next = 42;
        CHECK(lm_model_forward(whole, &next, 1, whole_logits, 0, nullptr, TINY_LM_PREFIX));
        CHECK(lm_model_forward(split, &next, 1, split_logits, 0, nullptr, TINY_LM_PREFIX));
        CHECK(split_logits.size() == (size_t) TINY_LM_PREFIX);
        CHECK(whole_logits == split_logits);

        const int32_t outside = TINY_LM_VOCAB;
        CHECK(!lm_model_forward(split, &outside, 1, split_logits));
    }
    lm_model_free(whole);
    lm_model_free(split);
    if (cpu) ggml_backend_free(cpu);
    std::remove(path.c_str());
}

// A metadata-only VAE (weights sized, never allocated) must still yield decode
// measurement sizes. The decoder architecture is hardcoded, so any parseable
// GGUF serves; the earlier sched-based pricing aborted on the buffer-less
// weights (GGML_ASSERT(buffer_id >= 0) in ggml-alloc).
void test_vae_metadata_only_measure() {
    using tts_cpp::acestep::AcestepStageMeasure;
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::vae_model_free;
    using tts_cpp::acestep::vae_model_load_metadata_only;
    using tts_cpp::acestep::vae_model_measure_decode;

    const std::string path = test_temp_dir() + "/qvac-acestep-vae-measure-test.gguf";
    gguf_context * gc = gguf_init_empty();
    CHECK(gguf_write_to_file(gc, path.c_str(), /*only_meta=*/true));
    gguf_free(gc);

    ggml_backend_t cpu = backend_cpu_init();
    CHECK(cpu != nullptr);
    if (cpu) {
        AcestepStageMeasure w{};
        tts_cpp::acestep::VaeModel * m =
            vae_model_load_metadata_only(path, cpu, /*with_encoder=*/false, /*verbose=*/false, w);
        CHECK(m != nullptr);
        if (m) {
            CHECK(w.weights_alloc_bytes > 0);
            size_t backend_bytes = 0, host_input_bytes = 0;
            CHECK(vae_model_measure_decode(m, /*T_latent=*/8, backend_bytes, host_input_bytes));
            CHECK(backend_bytes > 0);
            CHECK(host_input_bytes == 0);  // the input is part of the CPU measure
            vae_model_free(m);
        }
        ggml_backend_free(cpu);
    }
    std::remove(path.c_str());
}


}  // namespace

int main() {
    test_parallel_rows();
    test_convert_f32_to_f16_rows();
    test_fused_load_fail_closed();
    test_stage_dump_round_trip();
    test_stage_dump_rejects_malformed();
    test_gguf_parse_failure_releases_mapping();
    test_lm_head_chunk_ranges();
    test_lm_split_table_matches_whole_table();
    test_vae_metadata_only_measure();
    std::fprintf(stderr, "[test-acestep-loading] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
