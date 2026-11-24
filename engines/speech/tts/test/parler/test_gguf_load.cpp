// Fail-closed GGUF loading for the Parler engine (src/parler/gguf.cpp).
//
// No model fixture: synthesizes tiny metadata-only GGUFs in the system temp
// directory and asserts parler_load_gguf() rejects them with an error instead
// of half-loading: a truncated / empty file, a file without the parler.arch
// marker, each class of missing required metadata (scalar key, dac.rates
// array, tokenizer payload), and a complete metadata set with no weight
// tensors.
//
// Wrong-typed metadata is asserted in process: the loader checks the stored
// GGUF type before reading, so a mistyped key fails closed with an error
// instead of aborting.
//
// An explicit backend request is checked on the same weightless GGUF: a
// missing device fails at backend selection with no fallback, while "cpu"
// gets past it to the tensor check.

#include "parler/internal.h"
#include "../test_env_portable.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using tts_cpp::parler::detail::parler_model;
using tts_cpp::parler::detail::parler_load_gguf;
using tts_cpp::parler::detail::parler_free_model;

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        ++g_failures;                                          \
    }                                                          \
} while (0)

constexpr uint32_t SCALAR_U32 = 1;
constexpr float SCALAR_F32 = 0.5f;

enum class kv_kind { u32, f32, boolean };
struct kv_spec { const char * key; kv_kind kind; };

// Every scalar the loader requires, in the order it reads them.
const kv_spec REQUIRED_SCALARS[] = {
    { "parler.t5.n_layer",                kv_kind::u32 },
    { "parler.t5.d_model",                kv_kind::u32 },
    { "parler.t5.d_ff",                   kv_kind::u32 },
    { "parler.t5.n_head",                 kv_kind::u32 },
    { "parler.t5.d_kv",                   kv_kind::u32 },
    { "parler.t5.rel_buckets",            kv_kind::u32 },
    { "parler.t5.rel_max_dist",           kv_kind::u32 },
    { "parler.t5.rms_eps",                kv_kind::f32 },
    { "parler.t5.vocab_size",             kv_kind::u32 },
    { "parler.dec.n_layer",               kv_kind::u32 },
    { "parler.dec.d_model",               kv_kind::u32 },
    { "parler.dec.n_head",                kv_kind::u32 },
    { "parler.dec.d_ff",                  kv_kind::u32 },
    { "parler.dec.n_codebooks",           kv_kind::u32 },
    { "parler.dec.vocab_size",            kv_kind::u32 },
    { "parler.dec.ln_eps",                kv_kind::f32 },
    { "parler.dec.bos_token_id",          kv_kind::u32 },
    { "parler.dec.eos_token_id",          kv_kind::u32 },
    { "parler.dec.pad_token_id",          kv_kind::u32 },
    { "parler.dec.decoder_start_token_id", kv_kind::u32 },
    { "parler.dec.max_position",          kv_kind::u32 },
    { "parler.enc_to_dec",                kv_kind::boolean },
    { "parler.gen.max_length",            kv_kind::u32 },
    { "parler.gen.min_new_tokens",        kv_kind::u32 },
    { "parler.gen.do_sample",             kv_kind::boolean },
    { "parler.gen.temperature",           kv_kind::f32 },
    { "parler.gen.top_k",                 kv_kind::u32 },
    { "parler.dac.sample_rate",           kv_kind::u32 },
    { "parler.dac.n_quantizers",          kv_kind::u32 },
    { "parler.dac.codebook_size",         kv_kind::u32 },
    { "parler.dac.latent_dim",            kv_kind::u32 },
    { "parler.dac.decoder_dim",           kv_kind::u32 },
    { "parler.dac.hop",                   kv_kind::u32 },
};

std::string fixture_path(const char * name) {
    const std::string file = std::string("test-parler-gguf-load-") +
                             test_process_tag() + "-" + name + ".gguf";
    return (fs::temp_directory_path() / file).string();
}

void add_scalars(gguf_context * g, const char * skip_key) {
    for (const kv_spec & spec : REQUIRED_SCALARS) {
        if (skip_key && std::strcmp(spec.key, skip_key) == 0) continue;
        switch (spec.kind) {
            case kv_kind::u32:     gguf_set_val_u32(g, spec.key, SCALAR_U32);   break;
            case kv_kind::f32:     gguf_set_val_f32(g, spec.key, SCALAR_F32);   break;
            case kv_kind::boolean: gguf_set_val_bool(g, spec.key, false);       break;
        }
    }
}

void add_dac_rates(gguf_context * g) {
    const int32_t rates[] = { 2 };
    gguf_set_arr_data(g, "parler.dac.rates", GGUF_TYPE_INT32, rates, 1);
}

void add_tokenizer(gguf_context * g) {
    const char * pieces[] = { "<unk>", "</s>", "a" };
    const float scores[] = { 0.0f, 0.0f, -1.0f };
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", pieces, 3);
    gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, 3);
}

void write_gguf(const std::string & path, bool with_arch, const char * skip_key,
                const char * mistyped_key) {
    gguf_context * g = gguf_init_empty();
    if (with_arch) gguf_set_val_str(g, "parler.arch", "parler");
    add_scalars(g, skip_key ? skip_key : mistyped_key);
    if (mistyped_key) gguf_set_val_str(g, mistyped_key, "not-a-number");
    if (!skip_key || std::strcmp(skip_key, "parler.dac.rates") != 0) add_dac_rates(g);
    if (!skip_key || std::strcmp(skip_key, "tokenizer.ggml.tokens") != 0) add_tokenizer(g);
    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false)) {
        fprintf(stderr, "FATAL: cannot write fixture %s\n", path.c_str());
        exit(1);
    }
    gguf_free(g);
}

void expect_load_fails(const std::string & path, const char * expected_error,
                       const std::string & backend = {}) {
    parler_model model;
    std::string err;
    const bool ok = parler_load_gguf(path, model, /*n_gpu_layers=*/ 0, &err, backend);
    if (ok) parler_free_model(model);
    CHECK(!ok, "loading %s must fail", path.c_str());
    CHECK(err.find(expected_error) != std::string::npos,
          "error for %s should mention '%s', got '%s'",
          path.c_str(), expected_error, err.c_str());
}

void test_missing_arch_marker() {
    const std::string path = fixture_path("no-arch");
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "parler");
    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false)) {
        fprintf(stderr, "FATAL: cannot write fixture %s\n", path.c_str());
        exit(1);
    }
    gguf_free(g);
    expect_load_fails(path, "not a parler GGUF");
    fs::remove(path);
}

void test_truncated_file() {
    const std::string full = fixture_path("full-meta");
    write_gguf(full, /*with_arch=*/ true, nullptr, nullptr);

    const std::string truncated = fixture_path("truncated");
    fs::copy_file(full, truncated, fs::copy_options::overwrite_existing);
    fs::resize_file(truncated, fs::file_size(truncated) / 2);
    expect_load_fails(truncated, "failed to open GGUF");

    const std::string empty = fixture_path("empty");
    fs::copy_file(full, empty, fs::copy_options::overwrite_existing);
    fs::resize_file(empty, 0);
    expect_load_fails(empty, "failed to open GGUF");

    fs::remove(full);
    fs::remove(truncated);
    fs::remove(empty);
}

void test_missing_required_metadata() {
    const std::string no_scalar = fixture_path("no-n-layer");
    write_gguf(no_scalar, true, "parler.t5.n_layer", nullptr);
    expect_load_fails(no_scalar, "missing GGUF key: parler.t5.n_layer");
    fs::remove(no_scalar);

    const std::string no_rates = fixture_path("no-dac-rates");
    write_gguf(no_rates, true, "parler.dac.rates", nullptr);
    expect_load_fails(no_rates, "missing GGUF key: parler.dac.rates");
    fs::remove(no_rates);

    const std::string no_tokenizer = fixture_path("no-tokenizer");
    write_gguf(no_tokenizer, true, "tokenizer.ggml.tokens", nullptr);
    expect_load_fails(no_tokenizer, "missing tokenizer.ggml.tokens");
    fs::remove(no_tokenizer);
}

// Complete metadata but zero weight tensors: the loader must notice every
// required tensor is absent and fail, never hand back a weightless model.
void test_missing_tensors() {
    const std::string path = fixture_path("no-tensors");
    write_gguf(path, true, nullptr, nullptr);
    expect_load_fails(path, "expected tensors missing");
    fs::remove(path);
}

void test_explicit_backend_request() {
    const std::string path = fixture_path("backend-request");
    write_gguf(path, true, nullptr, nullptr);
    expect_load_fails(path, "requested backend 'no-such-device' is not available", "no-such-device");
    expect_load_fails(path, "expected tensors missing", "cpu");
    fs::remove(path);
}

void test_hexagon_weight_types() {
    using tts_cpp::parler::detail::parler_hexagon_runs_weight_type;
    CHECK(parler_hexagon_runs_weight_type(GGML_TYPE_Q8_0), "the q8_0 tier must load on Hexagon");
    CHECK(parler_hexagon_runs_weight_type(GGML_TYPE_F16), "the f16 tier must load on Hexagon");
    CHECK(parler_hexagon_runs_weight_type(GGML_TYPE_F32), "f32 tensors must load on Hexagon");
    CHECK(!parler_hexagon_runs_weight_type(GGML_TYPE_Q6_K), "the q6_k tier must be refused on Hexagon");
}

void test_wrong_typed_metadata_fails_closed() {
    const std::string path = fixture_path("wrong-type");
    write_gguf(path, true, nullptr, "parler.t5.n_layer");
    expect_load_fails(path, "not uint32");
    fs::remove(path);
}

} // namespace

int main() {
    test_missing_arch_marker();
    test_truncated_file();
    test_missing_required_metadata();
    test_missing_tensors();
    test_explicit_backend_request();
    test_hexagon_weight_types();
    test_wrong_typed_metadata_fails_closed();

    if (g_failures) {
        fprintf(stderr, "test-parler-gguf-load: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-parler-gguf-load: all checks passed\n");
    return 0;
}
