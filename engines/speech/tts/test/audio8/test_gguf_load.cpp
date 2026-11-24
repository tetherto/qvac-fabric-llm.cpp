// Fail-closed GGUF header reading for the Audio8 codec (src/audio8/gguf.cpp).
//
// No model fixture: synthesizes tiny metadata-only GGUFs in the system temp
// directory. The point of the wrong-typed cases is that they return an error
// at all -- gguf_get_val_* and gguf_get_arr_* GGML_ABORT on a type mismatch,
// so an unguarded reader would take the process down with the test.

#include "audio8/internal.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

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

std::string fixture_path(const char * stem) {
    return (fs::temp_directory_path() /
            ("test-audio8-gguf-load-" + std::string(stem) + ".gguf")).string();
}

void write_gguf(const std::string & path, gguf_context * g) {
    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false)) {
        fprintf(stderr, "FATAL: cannot write fixture %s\n", path.c_str());
        exit(1);
    }
    gguf_free(g);
}

bool peek_fails(const std::string & path, std::string & err) {
    tts_cpp::audio8::detail::codec_header header;
    err.clear();
    return !tts_cpp::audio8::detail::peek_codec_header(path, header, &err);
}

void test_missing_file() {
    const std::string path = fixture_path("absent");
    fs::remove(path);
    std::string err;
    CHECK(peek_fails(path, err), "a missing file must not read as a header");
    CHECK(!err.empty(), "a rejected read must set an error message");
}

void test_wrong_architecture() {
    const std::string path = fixture_path("wrong-arch");
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "audio8-lm");
    write_gguf(path, g);

    std::string err;
    CHECK(peek_fails(path, err), "an audio8-lm GGUF must not read as a codec");
    CHECK(err.find("not an audio8-codec") != std::string::npos,
          "error should name the architecture mismatch, got '%s'", err.c_str());
    fs::remove(path);
}

// general.architecture stored as a number rather than a string: the reader has
// to reject it instead of asserting inside gguf_get_val_str.
void test_wrong_typed_architecture_does_not_abort() {
    const std::string path = fixture_path("arch-not-a-string");
    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "general.architecture", 7);
    write_gguf(path, g);

    std::string err;
    CHECK(peek_fails(path, err), "a non-string architecture must be rejected");
    CHECK(err.find("not an audio8-codec") != std::string::npos,
          "a mistyped architecture should read as 'not a codec', got '%s'",
          err.c_str());
    fs::remove(path);
}

// A correctly marked codec GGUF whose scalar and string metadata carry the
// wrong types: the scalars keep their defaults and `part` stays empty, so the
// header read fails on the missing part rather than aborting.
void test_wrong_typed_metadata_does_not_abort() {
    const std::string path = fixture_path("metadata-not-typed");
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "audio8-codec");
    gguf_set_val_str(g, "audio8.codec.sample_rate", "not-a-number");
    gguf_set_val_u32(g, "audio8.codec.part", 3);
    write_gguf(path, g);

    std::string err;
    CHECK(peek_fails(path, err), "wrong-typed codec metadata must be rejected");
    CHECK(!err.empty(), "a rejected read must set an error message");
    fs::remove(path);
}

}

int main() {
    test_missing_file();
    test_wrong_architecture();
    test_wrong_typed_architecture_does_not_abort();
    test_wrong_typed_metadata_does_not_abort();

    if (g_failures) {
        fprintf(stderr, "test-audio8-gguf-load: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-audio8-gguf-load: all checks passed\n");
    return 0;
}
