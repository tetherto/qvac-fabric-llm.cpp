// Unit tests for the OpenCL bring-up dispatch helpers landed in
// `supertonic_op_dispatch_scope`, the thread-local
// `supertonic_use_cpu_custom_ops()` / `supertonic_use_f16_attn()`
// queries, and the `supertonic_model::backend_is_cpu`
// + `supertonic_model::use_f16_attn` fields they mirror.
//
// No GGUF / model file required — every test instantiates a bare
// `supertonic_model` POD on the stack with the two relevant flags set
// by hand, opens an RAII scope around it, and re-asserts the
// thread-local query state matches what the scope was constructed
// with.  This is what every public `supertonic_*_forward_ggml` /
// `*_trace_ggml` entry point does, so a regression here would mean a
// regression in the *real* dispatch path.
//
// Registered with `LABEL "unit"` in CMakeLists.txt so a fresh
// checkout's `ctest` exercises this without needing any fixture.

#include "supertonic_internal.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

// Test 1 — Default thread-local state.
//
// Every thread enters with CPU custom ops enabled (the historical
// CPU-only Supertonic path keeps working unchanged) and F16 K/V
// attention disabled (the CPU CBLAS attention path is the cheaper
// choice on a CPU backend, so the auto-policy lands here).
void test_default_flags() {
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 2 — Scope mirrors a CPU model.
//
// A CPU-backend model toggles nothing: defaults already match.
// The point of this test is to catch a "scope leaked the wrong
// previous-value back into the thread-local on dtor" regression by
// also asserting the default state after teardown.
void test_scope_mirrors_cpu_model() {
    supertonic_model model;
    model.backend_is_cpu = true;
    model.use_f16_attn   = false;
    {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == cpu_pointwise_accel_compiled());
        CHECK(supertonic_use_f16_attn() == false);
    }
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 3 — Scope mirrors a GPU model + restores defaults after.
//
// A GPU-backend engine (OpenCL / CUDA / Metal / Vulkan) sets both
// flags via the dispatch scope; the cblas-backed `ggml_custom_4d`
// fast paths in the vocoder + vector estimator must see `false`
// inside the scope, then `true` again after teardown so a
// CPU-only second engine in the same thread isn't poisoned.
void test_scope_mirrors_gpu_model() {
    supertonic_model model;
    model.backend_is_cpu = false;
    model.use_f16_attn   = true;
    {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
    }
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 4 — RAII teardown on exception.
//
// The forward functions wrap the rest of their body in try / catch;
// if the body throws (e.g. invalid voice, GGML buffer alloc failure),
// the scope must still restore the previous flags so the next
// engine's call sees a clean slate.
void test_scope_unwinds_on_exception() {
    supertonic_model model;
    model.backend_is_cpu = false;
    model.use_f16_attn   = true;
    bool caught = false;
    try {
        supertonic_op_dispatch_scope scope(model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
        throw std::runtime_error("simulated forward failure");
    } catch (const std::runtime_error &) {
        caught = true;
    }
    CHECK(caught);
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 5 — Nested scopes stack and unwind correctly.
//
// This is the harness for the "host destroyed engine_a then
// immediately invoked synthesize on engine_b on the same thread"
// path the alive-id registry already covers for gallocr free.
// Here we verify the dispatch flags don't get crossed during the
// brief window where both scopes exist (e.g. one forward function
// calling another's helper synchronously).
void test_nested_scopes() {
    supertonic_model gpu_model;
    gpu_model.backend_is_cpu = false;
    gpu_model.use_f16_attn   = true;

    supertonic_model cpu_model;
    cpu_model.backend_is_cpu = true;
    cpu_model.use_f16_attn   = false;

    {
        supertonic_op_dispatch_scope outer(gpu_model);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
        {
            supertonic_op_dispatch_scope inner(cpu_model);
            CHECK(supertonic_use_cpu_custom_ops() == cpu_pointwise_accel_compiled());
            CHECK(supertonic_use_f16_attn() == false);
        }
        // After inner unwinds, outer's state restored.
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == true);
    }
    CHECK(supertonic_use_cpu_custom_ops() == true);
    CHECK(supertonic_use_f16_attn() == false);
}

// Test 6 — Independent flags.
//
// `use_f16_attn = true` on a CPU model is a valid configuration
// (the user can `--f16-attn 1` even on CPU for parity testing),
// and `use_f16_attn = false` on a GPU model is the manual opt-out.
// Make sure the two flags are mirrored independently.
void test_independent_flags() {
    supertonic_model m;
    m.backend_is_cpu = true;
    m.use_f16_attn   = true;
    {
        supertonic_op_dispatch_scope scope(m);
        CHECK(supertonic_use_cpu_custom_ops() == cpu_pointwise_accel_compiled());
        CHECK(supertonic_use_f16_attn() == true);
    }

    m.backend_is_cpu = false;
    m.use_f16_attn   = false;
    {
        supertonic_op_dispatch_scope scope(m);
        CHECK(supertonic_use_cpu_custom_ops() == false);
        CHECK(supertonic_use_f16_attn() == false);
    }
}

// Test 7 - The CPU-kernel preference tracks the compiled fast paths.
//
// The per-stage CPU fast paths (conv1d_f32, dense_matmul_time, the tail update)
// only exist behind TTS_CPP_USE_ACCELERATE / TTS_CPP_USE_CBLAS. Without one,
// preferring them would pick im2col + mul_mat over the fused and [C, T] graph
// paths, so both predicates must follow the build, not the backend identity.
void test_cpu_kernel_preference_tracks_compiled_accel() {
    supertonic_model cpu_model;
    cpu_model.backend        = nullptr;   // treated as the CPU path
    cpu_model.backend_is_cpu = true;
    CHECK(model_prefers_cpu_kernels(cpu_model) == cpu_pointwise_accel_compiled());
    {
        supertonic_op_dispatch_scope scope(cpu_model);
        CHECK(supertonic_use_cpu_custom_ops() == cpu_pointwise_accel_compiled());
    }
}

// Test 8 - Thread-count resolver.
//
// An explicit request always wins. Everything that is not the CPU backend on
// the fused one-graph path keeps the legacy cap; the fused path leaves an
// eighth of the logical CPUs unsubscribed because full subscription regresses.
void test_thread_count_resolver() {
    CHECK(resolve_supertonic_thread_count(7, 32, true)  == 7);
    CHECK(resolve_supertonic_thread_count(7, 32, false) == 7);

    CHECK(resolve_supertonic_thread_count(0, 32, false) == kLegacyCpuThreadCap);
    CHECK(resolve_supertonic_thread_count(0, 2,  false) == 2);

    CHECK(resolve_supertonic_thread_count(0, 32, true)  == 28);
    CHECK(resolve_supertonic_thread_count(0, 16, true)  == 14);
    CHECK(resolve_supertonic_thread_count(0, 8,  true)  == 7);

    // Never zero, whatever hardware_concurrency reports.
    CHECK(resolve_supertonic_thread_count(0, 1, true)  == 1);
    CHECK(resolve_supertonic_thread_count(0, 0, true)  == 1);
    CHECK(resolve_supertonic_thread_count(0, -1, true) == 1);
    CHECK(resolve_supertonic_thread_count(0, 0, false) == 1);
}

// Test 9 - supertonic_set_n_threads classifies the backend, not just the request.
//
// Test 8 covers the pure resolver. This covers the wiring in front of it: which
// arm a model lands on is decided by backend_is_cpu together with whether the
// pointwise accel is compiled, and only the CPU backend off that accel path is
// allowed past the legacy cap.
void test_set_n_threads_backend_classification() {
    const int hw = std::max(1, (int) std::thread::hardware_concurrency());
    const int fused_default  = std::max(1, hw - hw / 8);
    const int capped_default = std::min(hw, kLegacyCpuThreadCap);
    // A CPU model reaches the fused path only where no pointwise accel exists.
    const int cpu_default = cpu_pointwise_accel_compiled() ? capped_default
                                                           : fused_default;

    supertonic_model cpu_model;
    cpu_model.backend_is_cpu = true;
    supertonic_set_n_threads(cpu_model, 0);
    CHECK(cpu_model.n_threads == cpu_default);

    supertonic_model gpu_model;
    gpu_model.backend_is_cpu = false;
    supertonic_set_n_threads(gpu_model, 0);
    CHECK(gpu_model.n_threads == capped_default);

    // An explicit request wins on either classification.
    supertonic_set_n_threads(cpu_model, 3);
    CHECK(cpu_model.n_threads == 3);
    supertonic_set_n_threads(gpu_model, 3);
    CHECK(gpu_model.n_threads == 3);
}

// Test 10 - a packed Q8_0 weight kept at Q8_0 must not stage an F32 expansion.
//
// The loader stages a dequantized copy for a packed source, then uploads it.
// That is correct only when the destination is F32. A matmul weight kept at
// q8_0 needs no expansion, and staging one uploaded four bytes per element into
// a block-quantized tensor, which aborts in ggml_backend_tensor_set. Pins the
// whole decision chain, since the bug was that the branch tested the source
// type alone.
void test_packed_q8_0_weight_is_not_expanded() {
    const std::string weight = "vector_estimator:onnx::MatMul_1234";
    CHECK(is_supertonic_matmul_weight_name(weight));

    const ggml_type dst = target_supertonic_storage_type(
        weight, GGML_TYPE_Q8_0, supertonic_precision::Q8_0,
        /*backend_is_cpu=*/false, /*backend_is_vk=*/false);
    CHECK(dst == GGML_TYPE_Q8_0);

    CHECK(needs_supertonic_tensor_conversion(GGML_TYPE_Q8_0, dst) == false);
    CHECK(should_expand_supertonic_tensor(GGML_TYPE_Q8_0) == true);
    CHECK(should_stage_f32_expansion(GGML_TYPE_Q8_0, dst) == false);

    const ggml_type dst_f32 = target_supertonic_storage_type(
        weight, GGML_TYPE_Q8_0, supertonic_precision::F32,
        /*backend_is_cpu=*/false, /*backend_is_vk=*/true);
    CHECK(dst_f32 == GGML_TYPE_F32);
    CHECK(should_stage_f32_expansion(GGML_TYPE_Q8_0, dst_f32) == true);
    CHECK(should_stage_f32_expansion(GGML_TYPE_F16,  GGML_TYPE_F32) == true);
    CHECK(should_stage_f32_expansion(GGML_TYPE_F32,  GGML_TYPE_F32) == false);
    CHECK(should_stage_f32_expansion(GGML_TYPE_F16,  GGML_TYPE_F16) == false);

    const ggml_type dst_other = target_supertonic_storage_type(
        "vocoder:conv1d_kernel", GGML_TYPE_Q8_0, supertonic_precision::Q8_0,
        /*backend_is_cpu=*/false, /*backend_is_vk=*/true);
    CHECK(dst_other == GGML_TYPE_F32);

    struct storage_case {
        const char * name;
        ggml_type source;
        supertonic_precision precision;
        bool cpu;
        bool vulkan;
        ggml_type expected;
    };
    const storage_case cases[] = {
        { weight.c_str(), GGML_TYPE_Q8_0, supertonic_precision::Auto, false, true,  GGML_TYPE_F16 },
        { weight.c_str(), GGML_TYPE_F16,  supertonic_precision::Auto, false, true,  GGML_TYPE_F16 },
        { weight.c_str(), GGML_TYPE_Q8_0, supertonic_precision::Auto, false, false, GGML_TYPE_F32 },
        { weight.c_str(), GGML_TYPE_F16,  supertonic_precision::Auto, false, false, GGML_TYPE_F32 },
        { weight.c_str(), GGML_TYPE_Q8_0, supertonic_precision::Auto, true,  true,  GGML_TYPE_F32 },
        { weight.c_str(), GGML_TYPE_Q4_0, supertonic_precision::Auto, false, true,  GGML_TYPE_F32 },
        { "vocoder:conv1d_kernel", GGML_TYPE_Q8_0, supertonic_precision::Auto, false, true, GGML_TYPE_F32 },
        { weight.c_str(), GGML_TYPE_F32,  supertonic_precision::Auto, false, true,  GGML_TYPE_F32 },
        { weight.c_str(), GGML_TYPE_Q8_0, supertonic_precision::F16, false, false, GGML_TYPE_F16 },
        { weight.c_str(), GGML_TYPE_Q8_0, supertonic_precision::F16, true,  true,  GGML_TYPE_F32 },
        { weight.c_str(), GGML_TYPE_Q8_0, supertonic_precision::Q8_0, false, false, GGML_TYPE_Q8_0 },
        { weight.c_str(), GGML_TYPE_F16,  supertonic_precision::Q8_0, false, true,  GGML_TYPE_F32 },
    };
    for (const storage_case & c : cases) {
        CHECK(target_supertonic_storage_type(
            c.name, c.source, c.precision, c.cpu, c.vulkan) == c.expected);
    }
}

} // namespace

int main() {
    test_cpu_kernel_preference_tracks_compiled_accel();
    test_thread_count_resolver();
    test_set_n_threads_backend_classification();
    test_packed_q8_0_weight_is_not_expanded();
    test_default_flags();
    test_scope_mirrors_cpu_model();
    test_scope_mirrors_gpu_model();
    test_scope_unwinds_on_exception();
    test_nested_scopes();
    test_independent_flags();

    std::fprintf(stderr,
                 "test_supertonic_backend_dispatch: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
