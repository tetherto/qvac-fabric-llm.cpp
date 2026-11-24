// Unit test for cosyvoice_load_gguf() map-in-place loading (src/cosyvoice_pipeline.cpp).
//
// No model fixture: writes its own synthetic GGUF (mixed dtypes, plus a tensor
// larger than the stream reader's chunk), loads it on the CPU backend and
// asserts every tensor's bytes round-trip exactly through the mmap-backed path.
// Then truncates the data section and asserts the load fails cleanly (throws)
// instead of reading past the mapping.

#include "test_env_portable.h"
#include "cosyvoice_pipeline.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TTS_CPP_COSYVOICE_TRACK_ALLOCATIONS
#include <set>
#endif

static int g_failures = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        ++g_failures;                                          \
    }                                                          \
} while (0)

#ifdef TTS_CPP_COSYVOICE_TRACK_ALLOCATIONS
static bool g_track_load = false;
static bool g_borrowed_freed = false;
static ggml_backend_t g_borrowed_backend = nullptr;
static std::set<ggml_context *> g_load_contexts;
static std::set<ggml_backend_buffer_t> g_load_buffers;

extern "C" ggml_context * __real_ggml_init(ggml_init_params);
extern "C" void __real_ggml_free(ggml_context *);
extern "C" ggml_backend_buffer_t __real_ggml_backend_cpu_buffer_from_ptr(void *, size_t);
extern "C" ggml_backend_buffer_t __real_ggml_backend_alloc_ctx_tensors_from_buft(ggml_context *, ggml_backend_buffer_type_t);
extern "C" void __real_ggml_backend_buffer_free(ggml_backend_buffer_t);
extern "C" void __real_ggml_backend_free(ggml_backend_t);

extern "C" ggml_context * __wrap_ggml_init(ggml_init_params p) {
    ggml_context * ctx = __real_ggml_init(p);
    if (g_track_load && ctx) g_load_contexts.insert(ctx);
    return ctx;
}

extern "C" void __wrap_ggml_free(ggml_context * ctx) {
    g_load_contexts.erase(ctx);
    __real_ggml_free(ctx);
}

extern "C" ggml_backend_buffer_t __wrap_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    ggml_backend_buffer_t buffer = __real_ggml_backend_cpu_buffer_from_ptr(ptr, size);
    if (g_track_load && buffer) g_load_buffers.insert(buffer);
    return buffer;
}

extern "C" ggml_backend_buffer_t __wrap_ggml_backend_alloc_ctx_tensors_from_buft(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_t buffer = __real_ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (g_track_load && buffer) g_load_buffers.insert(buffer);
    return buffer;
}

extern "C" void __wrap_ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    g_load_buffers.erase(buffer);
    __real_ggml_backend_buffer_free(buffer);
}

extern "C" void __wrap_ggml_backend_free(ggml_backend_t backend) {
    if (g_track_load && backend == g_borrowed_backend) g_borrowed_freed = true;
    __real_ggml_backend_free(backend);
}

static void check_load_ownership(const std::string & valid, const std::string & truncated) {
    g_borrowed_backend = tts_cpp::detail::init_cpu_backend();
    CHECK(g_borrowed_backend, "failed to init borrowed CPU backend");
    if (!g_borrowed_backend) return;
    const std::string paths[] = { valid, truncated, truncated, truncated, valid };
    for (int i = 0; i < 5; ++i) {
        g_track_load = true;
        g_borrowed_freed = false;
        bool threw = false;
        try {
            model_ctx model = cosyvoice_load_gguf(paths[i], g_borrowed_backend);
            cosyvoice_free(model);
        } catch (const std::exception &) {
            threw = true;
        }
        g_track_load = false;
        CHECK(threw == (i > 0 && i < 4), "unexpected load disposition in ownership case %d", i);
        CHECK(g_load_contexts.empty(), "load case %d retained %zu contexts", i, g_load_contexts.size());
        CHECK(g_load_buffers.empty(), "load case %d retained %zu buffers", i, g_load_buffers.size());
        CHECK(!g_borrowed_freed, "load case %d freed a borrowed backend", i);
        // Release observed leaks after recording failures so the regression stays bounded.
        for (ggml_backend_buffer_t buffer : g_load_buffers) __real_ggml_backend_buffer_free(buffer);
        for (ggml_context * ctx : g_load_contexts) __real_ggml_free(ctx);
        g_load_buffers.clear();
        g_load_contexts.clear();
        if (g_borrowed_freed) { g_borrowed_backend = nullptr; break; }
    }
    ggml_backend_free(g_borrowed_backend);
    g_borrowed_backend = nullptr;
}
#endif

struct spec { const char * name; ggml_type type; int64_t ne0; int64_t ne1; };
static const spec SPECS[] = {
    { "w/small_f32", GGML_TYPE_F32, 7,    1 },
    { "w/mat_f32",   GGML_TYPE_F32, 129, 65 },
    { "w/mat_f16",   GGML_TYPE_F16, 256, 33 },
    { "w/q8",        GGML_TYPE_Q8_0, 512, 40 },
    { "w/big_f32",   GGML_TYPE_F32, 1031, 2503 },  // > 8 MiB stream chunk
};
static const size_t N_SPECS = sizeof(SPECS) / sizeof(SPECS[0]);

static std::map<std::string, std::vector<uint8_t>> g_ref;

static std::string write_fixture() {
    const std::string path = test_tmpdir() + "/test-cosyvoice-load-fixture.gguf";

    size_t total = 0;
    for (size_t i = 0; i < N_SPECS; ++i) {
        total += ggml_row_size(SPECS[i].type, SPECS[i].ne0) * (size_t) SPECS[i].ne1;
    }
    ggml_init_params p = { total + (N_SPECS + 1) * ggml_tensor_overhead(), nullptr, false };
    ggml_context * ctx = ggml_init(p);

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "cosyvoice3");

    for (size_t i = 0; i < N_SPECS; ++i) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, SPECS[i].type, SPECS[i].ne0, SPECS[i].ne1);
        ggml_set_name(t, SPECS[i].name);
        uint8_t * d = (uint8_t *) t->data;
        const size_t nb = ggml_nbytes(t);
        for (size_t j = 0; j < nb; ++j) d[j] = (uint8_t) ((j * 131 + i * 31 + 7) & 0xff);
        g_ref[SPECS[i].name] = std::vector<uint8_t>(d, d + nb);
        gguf_add_tensor(g, t);
    }

    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false)) {
        fprintf(stderr, "FATAL: cannot write fixture %s\n", path.c_str());
        exit(1);
    }
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

static void check_roundtrip(const std::string & path) {
    model_ctx m = cosyvoice_load_gguf(path);
    for (size_t i = 0; i < N_SPECS; ++i) {
        ggml_tensor * t = cosyvoice_get(m, SPECS[i].name);
        CHECK(t != nullptr, "tensor '%s' missing after load", SPECS[i].name);
        if (!t) continue;
        std::vector<uint8_t> got(ggml_nbytes(t));
        ggml_backend_tensor_get(t, got.data(), 0, got.size());
        const auto & want = g_ref.at(SPECS[i].name);
        CHECK(got.size() == want.size(), "tensor '%s' size mismatch", SPECS[i].name);
        CHECK(memcmp(got.data(), want.data(), got.size()) == 0,
              "tensor '%s' bytes differ after map-in-place load", SPECS[i].name);
    }
    cosyvoice_free(m);
}

// Copy the fixture but drop the tail so the last tensor's data runs past EOF.
static std::string write_truncated(const std::string & src) {
    FILE * in = fopen(src.c_str(), "rb");
    if (!in) { fprintf(stderr, "FATAL: cannot reopen %s\n", src.c_str()); exit(1); }
    fseek(in, 0, SEEK_END);
    const long size = ftell(in);
    fseek(in, 0, SEEK_SET);
    std::vector<uint8_t> bytes((size_t) size);
    if (fread(bytes.data(), 1, bytes.size(), in) != bytes.size()) { fclose(in); exit(1); }
    fclose(in);

    const std::string path = test_tmpdir() + "/test-cosyvoice-load-truncated.gguf";
    FILE * out = fopen(path.c_str(), "wb");
    if (!out) { fprintf(stderr, "FATAL: cannot write %s\n", path.c_str()); exit(1); }
    const size_t keep = bytes.size() - (bytes.size() / 4);  // lose ~25% of the data section
    fwrite(bytes.data(), 1, keep, out);
    fclose(out);
    return path;
}

// A one-layer Qwen LM whose fused qkv_proj holds `qkv_rows` rows.  The graph
// builder derives the expected row count from the GGUF's own n_head / n_kv /
// head_dim metadata, so writing fewer rows than those imply is exactly the
// GGUF/metadata disagreement the fused path has to refuse.
namespace fused_qkv {

constexpr int64_t kHidden  = 32;
constexpr int64_t kNHead   = 4;
constexpr int64_t kNKv     = 2;
constexpr int64_t kHeadDim = 8;
constexpr int64_t kInter   = 16;
constexpr int64_t kVocab   = 8;
constexpr int64_t kRows    = (kNHead + 2 * kNKv) * kHeadDim;

std::string write_lm(int64_t qkv_rows, const char * tag) {
    const std::string path = test_tmpdir() + "/test-cosyvoice-load-qkv-" + tag + ".gguf";
    struct t_spec { std::string name; int64_t ne0, ne1; };
    const std::vector<t_spec> specs = {
        { "lm/blk/0/in_ln/weight",     kHidden,   1 },
        { "lm/blk/0/qkv_proj/weight",  kHidden,   qkv_rows },
        { "lm/blk/0/qkv_proj/bias",    qkv_rows,  1 },
        { "lm/blk/0/o_proj/weight",    kHidden,   kHidden },
        { "lm/blk/0/post_ln/weight",   kHidden,   1 },
        { "lm/blk/0/gate/weight",      kHidden,   kInter },
        { "lm/blk/0/up/weight",        kHidden,   kInter },
        { "lm/blk/0/down/weight",      kInter,    kHidden },
        { "lm/norm/weight",            kHidden,   1 },
        { "lm/llm_decoder/weight",     kHidden,   kVocab },
    };

    size_t total = 0;
    for (const auto & s : specs) total += ggml_row_size(GGML_TYPE_F32, s.ne0) * (size_t) s.ne1;
    ggml_init_params p = { total + (specs.size() + 1) * ggml_tensor_overhead(), nullptr, false };
    ggml_context * ctx = ggml_init(p);

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "cosyvoice3");
    gguf_set_val_u32(g, "cosyvoice3.llm.depth",    1);
    gguf_set_val_u32(g, "cosyvoice3.llm.hidden",   (uint32_t) kHidden);
    gguf_set_val_u32(g, "cosyvoice3.llm.n_head",   (uint32_t) kNHead);
    gguf_set_val_u32(g, "cosyvoice3.llm.n_kv",     (uint32_t) kNKv);
    gguf_set_val_u32(g, "cosyvoice3.llm.head_dim", (uint32_t) kHeadDim);
    gguf_set_val_u32(g, "cosyvoice3.llm.inter",    (uint32_t) kInter);

    for (const auto & s : specs) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.ne0, s.ne1);
        ggml_set_name(t, s.name.c_str());
        memset(t->data, 0, ggml_nbytes(t));
        gguf_add_tensor(g, t);
    }
    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false)) {
        fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
        exit(1);
    }
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

// Builds one LM graph over `n_tokens` positions; reports whether it threw.
bool build_rejects(const std::string & path, int n_tokens) {
    bool threw = false;
    model_ctx m = cosyvoice_load_gguf(path);
    const qwen_hp hp = cosyvoice_qwen_hp(m);
    std::vector<uint8_t> buf(4u * 1024 * 1024);
    ggml_init_params p = { buf.size(), buf.data(), /*no_alloc=*/true };
    ggml_context * c = ggml_init(p);
    try {
        ggml_tensor * x    = ggml_new_tensor_2d(c, GGML_TYPE_F32, kHidden, n_tokens);
        ggml_tensor * pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
        ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_tokens, n_tokens);
        build_qwen(c, m, hp, x, pos, mask, n_tokens);
    } catch (const std::exception &) {
        threw = true;
    }
    ggml_free(c);
    cosyvoice_free(m);
    return threw;
}

} // namespace fused_qkv

int main() {
    const std::string path = write_fixture();
    check_roundtrip(path);

    const std::string trunc = write_truncated(path);
    bool threw = false;
    try {
        model_ctx m = cosyvoice_load_gguf(trunc);
        cosyvoice_free(m);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw, "loading a truncated GGUF must throw, not read past the mapping");
#ifdef TTS_CPP_COSYVOICE_TRACK_ALLOCATIONS
    check_load_ownership(path, trunc);
#endif

    // A fused qkv_proj shorter than n_head/n_kv/head_dim imply must be refused
    // rather than sliced: the Q/K/V views carry a custom token stride, so the
    // contiguous-size check inside ggml_view_3d passes for a multi-token
    // prefill while the V view runs past the projection output.
    const std::string qkv_short = fused_qkv::write_lm(fused_qkv::kRows - fused_qkv::kHeadDim, "short");
    const std::string qkv_ok    = fused_qkv::write_lm(fused_qkv::kRows, "ok");
    for (const int n_tokens : { 1, 8 }) {
        CHECK(fused_qkv::build_rejects(qkv_short, n_tokens),
              "a short fused qkv_proj must be rejected at %d token(s)", n_tokens);
        CHECK(!fused_qkv::build_rejects(qkv_ok, n_tokens),
              "a correctly sized fused qkv_proj must build at %d token(s)", n_tokens);
    }

    remove(path.c_str());
    remove(trunc.c_str());
    remove(qkv_short.c_str());
    remove(qkv_ok.c_str());

    if (g_failures) {
        fprintf(stderr, "test-cosyvoice-load: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-cosyvoice-load: all checks passed\n");
    return 0;
}
