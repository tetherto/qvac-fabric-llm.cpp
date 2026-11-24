// Shared API/default contract. Model-dependent warm-up and loading belong
// in fixture tests; ordinary string/vector assignment needs no runtime test.
#include "tts-cpp/supertonic/engine.h"
#include "supertonic_internal.h"

#include <cstdio>
#include <type_traits>
#include <utility>

namespace {

using tts_cpp::supertonic::Engine;
using tts_cpp::supertonic::EngineOptions;
using namespace tts_cpp::supertonic::detail;

static_assert(std::is_same<decltype(EngineOptions::prewarm_text), std::string>::value);
static_assert(std::is_same<decltype(&Engine::warm_up),
                           void (Engine::*)(const std::string &)>::value);
static_assert(std::is_same<decltype(EngineOptions::f16_weights_deny_list),
                           std::vector<std::string>>::value);
static_assert(std::is_same<decltype(EngineOptions::kv_attn_type), int>::value);
static_assert(std::is_same<decltype(supertonic_model::kv_attn_type), kv_attn_dtype>::value);
static_assert(std::is_same<decltype(supertonic_op_dispatch_scope::prev_kv_attn_type),
                           kv_attn_dtype>::value);
static_assert(std::is_same<decltype(load_supertonic_gguf(
    std::declval<const std::string &>(), std::declval<supertonic_model &>())), bool>::value);
static_assert(std::is_same<decltype(load_supertonic_gguf(
    std::declval<const std::string &>(), std::declval<supertonic_model &>(),
    0, false, -1, supertonic_precision::F32, 0,
    std::declval<const std::vector<std::string> &>())), bool>::value);
static_assert(std::is_same<decltype(should_materialise_f16_weight(
    std::declval<const std::string &>(),
    std::declval<const std::vector<std::string> &>())), bool>::value);

int failures = 0;
#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        ++failures; \
    } \
} while (0)

void test_defaults() {
    const EngineOptions opts;
    CHECK(opts.model_gguf_path.empty());
    CHECK(opts.prewarm_text.empty());
    CHECK(opts.vulkan_device == 0);
    CHECK(opts.precision == tts_cpp::supertonic::Precision::Auto);
    CHECK(opts.f16_attn == -1);
    CHECK(opts.f16_weights == -1);
    CHECK(opts.f16_weights_deny_list.empty());
    CHECK(opts.kv_attn_type == -1);
    const supertonic_model model;
    CHECK(model.kv_attn_type == kv_attn_dtype::f32);
    CHECK(supertonic_kv_attn_type() == kv_attn_dtype::f32);
}

void test_nested_dispatch_scopes() {
    CHECK(supertonic_kv_attn_type() == kv_attn_dtype::f32);
    {
        supertonic_model outer_model;
        outer_model.kv_attn_type = kv_attn_dtype::bf16;
        const supertonic_op_dispatch_scope outer(outer_model);
        CHECK(supertonic_kv_attn_type() == kv_attn_dtype::bf16);
        {
            supertonic_model inner_model;
            inner_model.kv_attn_type = kv_attn_dtype::q8_0;
            const supertonic_op_dispatch_scope inner(inner_model);
            CHECK(supertonic_kv_attn_type() == kv_attn_dtype::q8_0);
        }
        CHECK(supertonic_kv_attn_type() == kv_attn_dtype::bf16);
    }
    CHECK(supertonic_kv_attn_type() == kv_attn_dtype::f32);
}

} // namespace

int main() {
    test_defaults();
    test_nested_dispatch_scopes();
    return failures == 0 ? 0 : 1;
}
