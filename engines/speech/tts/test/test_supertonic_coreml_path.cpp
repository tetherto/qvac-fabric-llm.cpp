// Model-free on every host: the Supertonic vocoder sidecar naming rule
// (supertonic_coreml_path.h).

#include "supertonic_coreml_path.h"

#include <cstdio>
#include <string>

using tts_cpp::supertonic::detail::coreml_vocoder_sidecar_path;

namespace {

int g_failures = 0;

void expect_path(const std::string & gguf, const std::string & want) {
    const std::string got = coreml_vocoder_sidecar_path(gguf);
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s -> %s (want %s)\n", gguf.c_str(), got.c_str(),
                     want.c_str());
        ++g_failures;
    }
    std::string encoder_want = want;
    encoder_want.replace(encoder_want.rfind("-vocoder"), 8, "-encoder");
    const auto encoder_got = tts_cpp::supertonic::detail::coreml_encoder_sidecar_path(gguf);
    if (encoder_got != encoder_want) {
        std::fprintf(stderr, "FAIL: encoder %s -> %s (want %s)\n", gguf.c_str(), encoder_got.c_str(), encoder_want.c_str());
        ++g_failures;
    }
    std::string vector_want = want;
    vector_want.replace(vector_want.rfind("-vocoder"), 8, "-vector-L128");
    const auto vector_got = tts_cpp::supertonic::detail::coreml_vector_sidecar_path(gguf, 128);
    if (vector_got != vector_want) {
        std::fprintf(stderr, "FAIL: vector %s -> %s (want %s)\n", gguf.c_str(), vector_got.c_str(), vector_want.c_str());
        ++g_failures;
    }
}

}  // namespace

int main() {
    expect_path("models/supertonic2.gguf", "models/supertonic2-vocoder.mlmodelc");
    expect_path("models/supertonic3.gguf", "models/supertonic3-vocoder.mlmodelc");
    expect_path("models/supertonic.gguf", "models/supertonic-vocoder.mlmodelc");
    expect_path("supertonic2-q8_0.gguf", "supertonic2-vocoder.mlmodelc");
    expect_path("supertonic2.q8_0.gguf", "supertonic2-vocoder.mlmodelc");
    expect_path("supertonic2-f16.gguf", "supertonic2-vocoder.mlmodelc");
    expect_path("supertonic2-F16.gguf", "supertonic2-vocoder.mlmodelc");
    expect_path("supertonic2-bf16.gguf", "supertonic2-vocoder.mlmodelc");
    expect_path("/abs/path/supertonic3-q4_0.gguf", "/abs/path/supertonic3-vocoder.mlmodelc");
    expect_path("rel\\win\\supertonic2-q5_0.gguf", "rel\\win\\supertonic2-vocoder.mlmodelc");
    // A dot inside a directory name is not an extension.
    expect_path("/x.y/supertonic2", "/x.y/supertonic2-vocoder.mlmodelc");
    // A non-quant suffix stays part of the stem.
    expect_path("models/supertonic2-multilingual.gguf",
                "models/supertonic2-multilingual-vocoder.mlmodelc");
    // No extension at all.
    expect_path("supertonic2", "supertonic2-vocoder.mlmodelc");

    if (g_failures == 0) {
        std::printf("test_supertonic_coreml_path: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_supertonic_coreml_path: %d failure(s)\n", g_failures);
    return 1;
}
