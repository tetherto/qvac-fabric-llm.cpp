// Model-free ACE-Step backends regression tests.

#include "acestep_test_support.h"
#include "backend_registry.h"
#include "engine_backends.h"
#include "fit_pools.h"
#include "stage_placement.h"
#include "dit_gguf.h"
#include "detok_ggml.h"
#include "audiogen-cpp/acestep/engine.h"
#include "audiogen-cpp/acestep/fit.h"
#include "audiogen-cpp/acestep/vae.h"
#include "gguf.h"

namespace {

void test_backend_device_types() {
    using tts_cpp::acestep::backend_device_type_is_gpu;
    using tts_cpp::acestep::backend_reg_name_is_validated_gpu;
    using tts_cpp::acestep::parse_adreno_version;

    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_IGPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_ACCEL));

    CHECK(backend_reg_name_is_validated_gpu("Vulkan"));
    CHECK(backend_reg_name_is_validated_gpu("MTL"));
    CHECK(backend_reg_name_is_validated_gpu("Metal"));
    CHECK(backend_reg_name_is_validated_gpu("CUDA"));
    // OpenCL is deliberately absent: it is reached by its own Adreno pass in
    // backend_gpu_init(), not by this validated-backend preference.
    CHECK(!backend_reg_name_is_validated_gpu("OpenCL"));
    // The HIP/MUSA builds of ggml-cuda register under their own names and stay
    // unvalidated until measured.
    CHECK(!backend_reg_name_is_validated_gpu("ROCm"));
    CHECK(!backend_reg_name_is_validated_gpu("MUSA"));
    CHECK(!backend_reg_name_is_validated_gpu(nullptr));

    // That Adreno pass gates on the generation parsed out of the device
    // name/description, so the parse is what decides OpenCL-over-Vulkan.
    CHECK(parse_adreno_version("Adreno (TM) 740") == 740);
    CHECK(parse_adreno_version("QUALCOMM Adreno(TM) 830") == 830);
    CHECK(parse_adreno_version("Adreno X1-85") == 800);       // Snapdragon-X naming
    CHECK(parse_adreno_version("Adreno (TM) 640") == 640);    // parsed, but below the 700 gate
    // An embedded "OpenCL 3.0" must not be mistaken for the model number, and a
    // non-Adreno device must not be claimed at all.
    CHECK(parse_adreno_version("Adreno (TM) 750 (OpenCL 3.0)") == 750);
    CHECK(parse_adreno_version("Mali-G715") == -1);
    CHECK(parse_adreno_version("Apple M1 Pro") == -1);
    CHECK(parse_adreno_version("") == -1);
    CHECK(parse_adreno_version(nullptr) == -1);
}

// A CPU run after a GPU request is the symptom users report, so the reason must
// never disagree with what the registry actually returned.
void test_gpu_matmul_capability() {
    using namespace tts_cpp::acestep;
    bool inspected = false;
    CHECK(!probe_gpu_matmul([&inspected](const ggml_tensor * operation) {
        inspected = true;
        CHECK(operation->op == GGML_OP_MUL_MAT);
        CHECK(operation->type == GGML_TYPE_F32);
        CHECK(operation->src[0]->type == GGML_TYPE_F32);
        CHECK(operation->src[1]->type == GGML_TYPE_F32);
        CHECK(operation->src[0]->data == nullptr && operation->src[1]->data == nullptr);
        CHECK(operation->src[0]->ne[0] == GPU_MATMUL_PROBE_DIMENSION);
        CHECK(operation->src[1]->ne[1] == GPU_MATMUL_PROBE_COLUMNS);
        return false;
    }));
    CHECK(inspected);
    CHECK(probe_gpu_matmul([](const ggml_tensor *) { return true; }));
    CHECK(!backend_supports_gpu_matmul(nullptr));
    ggml_backend_t cpu = backend_cpu_init();
    CHECK(cpu != nullptr);
    if (cpu) {
        CHECK(backend_supports_gpu_matmul(cpu));
        CHECK(keep_usable_gpu(cpu, backend_supports_gpu_matmul) == cpu);
        ggml_backend_free(cpu);
    }
    ggml_backend_t unsupported = backend_cpu_init();
    CHECK(unsupported != nullptr);
    CHECK(keep_usable_gpu(unsupported, [](ggml_backend_t) { return false; }) == nullptr);
    CHECK(keep_usable_gpu(nullptr, [](ggml_backend_t) { return true; }) == nullptr);
}

void test_gpu_fallback_reason() {
    using tts_cpp::GpuFallbackReason;
    using tts_cpp::gpu_fallback_reason_name;

    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::none), "none") == 0);
    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::not_requested), "not-requested") == 0);
    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::no_devices), "no-devices") == 0);
    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::init_failed), "init-failed") == 0);

    // Holds on any machine: with a GPU the reason is `none`, without one it
    // names which half of the acquisition failed.
    GpuFallbackReason reason  = GpuFallbackReason::not_requested;
    ggml_backend_t    backend = tts_cpp::acestep::backend_gpu_init(&reason);
    if (backend) {
        CHECK(reason == GpuFallbackReason::none);
        ggml_backend_free(backend);
    } else {
        CHECK(reason == GpuFallbackReason::no_devices || reason == GpuFallbackReason::init_failed);
    }
}

// Speech-engine GPU selection prefers CUDA over Vulkan on the same NVIDIA
// card, and prefers discrete over integrated adapters. gpu_tier_for is the
// pure ranking that captures that policy so a synthesised device topology can
// be scored without a live ggml-backend registry.
//
// SCOPE: this test covers the classifier (what tier a device lands in) and
// the enum-constant ordering (which tier outranks which). It does NOT
// exercise backend_gpu_init's actual walk in backend_registry.h: that
// function has its own parallel Adreno-OpenCL / CUDA-first / {require_validated,
// {GPU, IGPU}} sequence and does not call gpu_tier_for today, so a reordering
// of that walk would silently pass this test. Keeping the two in sync is a
// code-review contract; the deeper fix (drive backend_gpu_init off
// gpu_tier_for as the single source of truth) is tracked separately.
void test_gpu_tier_policy() {
    using tts_cpp::acestep::GpuTier;
    using tts_cpp::acestep::gpu_tier_for;

    // Adreno 700+ OpenCL wins outright — the OpenCL kernels are the validated
    // path on Snapdragon 8 Gen 2+.
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  740) == GpuTier::AdrenoOpenCL700Plus);
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_IGPU, 740) == GpuTier::AdrenoOpenCL700Plus);
    // Adreno 6xx OpenCL is not routed to the OpenCL tier here (broken kernels)
    // — the calling code separately skips it; the ranking treats it as generic OpenCL.
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  640) != GpuTier::AdrenoOpenCL700Plus);

    // CUDA outranks Vulkan on the same NVIDIA card.
    CHECK(gpu_tier_for("CUDA",   GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::CudaDiscrete);
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::ValidatedDiscrete);
    CHECK(static_cast<int>(GpuTier::CudaDiscrete) <
          static_cast<int>(GpuTier::ValidatedDiscrete));

    // Tegra / Jetson CUDA reports IGPU on some drivers — still preferred over
    // a validated Vulkan discrete via the CUDA-first rule.
    CHECK(gpu_tier_for("CUDA", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::CudaIntegrated);
    CHECK(static_cast<int>(GpuTier::CudaIntegrated) <
          static_cast<int>(GpuTier::ValidatedDiscrete));

    // Discrete outranks integrated at every tier that has both — the
    // other-addons behavior we are matching (llm/diffusion prefer dGPU).
    CHECK(static_cast<int>(GpuTier::CudaDiscrete)      < static_cast<int>(GpuTier::CudaIntegrated));
    CHECK(static_cast<int>(GpuTier::ValidatedDiscrete) < static_cast<int>(GpuTier::ValidatedIntegrated));
    CHECK(static_cast<int>(GpuTier::OtherDiscrete)     < static_cast<int>(GpuTier::OtherIntegrated));

    // Metal is validated; MTL is the new registry name for the same backend.
    CHECK(gpu_tier_for("Metal", GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::ValidatedDiscrete);
    CHECK(gpu_tier_for("MTL",   GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::ValidatedIntegrated);

    // Non-GPU device types are not selectable — the ranking rejects them so
    // the CPU/Accel devices never leak into the GPU pass.
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("BLAS",   GGML_BACKEND_DEVICE_TYPE_ACCEL, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for(nullptr,  GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);

    // A fake "ROCm" / "MUSA" registry lands in the Other tier — unvalidated
    // but still a candidate below every validated GPU, matching the current
    // {require_validated, ...} nest in backend_gpu_init.
    CHECK(gpu_tier_for("ROCm", GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::OtherDiscrete);
    CHECK(gpu_tier_for("MUSA", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::OtherIntegrated);
}

void test_backend_request_matching() {
    using tts_cpp::acestep::backend_request_is_auto;
    using tts_cpp::acestep::backend_request_matches;
    using tts_cpp::acestep::GpuTier;
    using tts_cpp::acestep::gpu_tier_for;

    CHECK(backend_request_is_auto(""));
    CHECK(backend_request_is_auto("auto"));
    CHECK(!backend_request_is_auto("cpu"));
    CHECK(!backend_request_is_auto("hexagon"));

    CHECK(backend_request_matches("hexagon", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("hexagon", "HTP", "HTP1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("hexagon", "OpenCL", "GPUOpenCL", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("HTP1", "HTP", "HTP1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("opencl", "OpenCL", "GPUOpenCL", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("opencl", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("webgpu", "WebGPU", "WebGPU", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("webgpu", "Vulkan", "Vulkan0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("opencl", "WebGPU", "WebGPU", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("cpu", "CPU", "CPU", GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_request_matches("cpu", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("auto", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("xdna", "XDNA", "XDNA0", GGML_BACKEND_DEVICE_TYPE_ACCEL));
    CHECK(!backend_request_matches("xdna", "XDNA", "XDNA1", GGML_BACKEND_DEVICE_TYPE_ACCEL));
    CHECK(backend_request_matches("XDNA1", "XDNA", "XDNA1", GGML_BACKEND_DEVICE_TYPE_ACCEL));
    CHECK(!backend_request_matches("xdna", "CPU", "CPU", GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_request_matches("auto", "XDNA", "XDNA0", GGML_BACKEND_DEVICE_TYPE_ACCEL));
    CHECK(!backend_request_matches("Vulkan0", nullptr, nullptr, GGML_BACKEND_DEVICE_TYPE_GPU));

    CHECK(gpu_tier_for("HTP", GGML_BACKEND_DEVICE_TYPE_GPU, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("HTP", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("XDNA", GGML_BACKEND_DEVICE_TYPE_GPU, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("XDNA", GGML_BACKEND_DEVICE_TYPE_ACCEL, -1) == GpuTier::NotSelectable);
}

void test_dsp_library_path() {
    using tts_cpp::acestep::prepend_dsp_library_directory;

    CHECK(prepend_dsp_library_directory("/app/lib", "") == "/app/lib");
    CHECK(prepend_dsp_library_directory("/app/lib", "/vendor/dsp") == "/app/lib;/vendor/dsp");
    CHECK(prepend_dsp_library_directory("/app/lib", "/app/lib;/vendor/dsp") == "/app/lib;/vendor/dsp");
    CHECK(prepend_dsp_library_directory("/app/lib", "/vendor/dsp;/app/lib") == "/vendor/dsp;/app/lib");
    CHECK(prepend_dsp_library_directory("", "/vendor/dsp") == "/vendor/dsp");
}

void test_backend_requested_init() {
    using tts_cpp::GpuFallbackReason;
    using tts_cpp::acestep::backend_is_cpu_device;
    using tts_cpp::acestep::backend_requested_init;

    GpuFallbackReason reason = GpuFallbackReason::not_requested;
    ggml_backend_t    cpu    = backend_requested_init("cpu", &reason);
    CHECK(cpu != nullptr);
    CHECK(backend_is_cpu_device(cpu));
    CHECK(reason == GpuFallbackReason::none);
    if (cpu) ggml_backend_free(cpu);

    CHECK(backend_requested_init("no-such-device", &reason) == nullptr);
    CHECK(reason == GpuFallbackReason::no_devices);
}

void test_backend_init_first_matching() {
    using tts_cpp::GpuFallbackReason;
    using tts_cpp::acestep::backend_device_matches_request;
    using tts_cpp::acestep::backend_init_first_matching;
    using tts_cpp::acestep::backend_request_outcome;
    using tts_cpp::acestep::RequestedBackendInit;

    const RequestedBackendInit cpu = backend_init_first_matching("cpu");
    CHECK(cpu.backend != nullptr);
    CHECK(cpu.saw_device);
    CHECK(backend_request_outcome(cpu) == GpuFallbackReason::none);
    if (cpu.backend) {
        CHECK(backend_device_matches_request("cpu", ggml_backend_get_device(cpu.backend)));
        CHECK(!backend_device_matches_request("hexagon", ggml_backend_get_device(cpu.backend)));
        ggml_backend_free(cpu.backend);
    }
    CHECK(!backend_device_matches_request("cpu", nullptr));

    const RequestedBackendInit missing = backend_init_first_matching("no-such-device");
    CHECK(missing.backend == nullptr);
    CHECK(!missing.saw_device);
    CHECK(backend_request_outcome(missing) == GpuFallbackReason::no_devices);

    RequestedBackendInit refused;
    refused.saw_device = true;
    CHECK(backend_request_outcome(refused) == GpuFallbackReason::init_failed);
}

void test_weight_placement_on_cpu() {
    using tts_cpp::acestep::backend_requested_init;
    using tts_cpp::acestep::dit_gguf_backend_maps_weights;
    using tts_cpp::acestep::dit_gguf_weight_buffer_type;

    ggml_backend_t cpu = backend_requested_init("cpu");
    CHECK(cpu != nullptr);
    if (!cpu) return;
    CHECK(dit_gguf_backend_maps_weights(cpu));
    CHECK(dit_gguf_weight_buffer_type(cpu) == ggml_backend_get_default_buffer_type(cpu));
    ggml_backend_free(cpu);
}

void test_lm_backend_request() {
    using tts_cpp::acestep::AcestepBackends;
    using tts_cpp::acestep::BackendRequest;
    using tts_cpp::acestep::free_acestep_backends;
    using tts_cpp::acestep::resolve_acestep_backends;

    BackendRequest req;
    req.backend    = "cpu";
    req.lm_backend = "cpu";
    AcestepBackends rb;
    CHECK(resolve_acestep_backends(req, rb));
    CHECK(rb.lm == rb.backend);
    CHECK(rb.lm_extra == nullptr);
    free_acestep_backends(rb);

    req.lm_backend = "no-such-device";
    CHECK(!resolve_acestep_backends(req, rb));
    CHECK(rb.backend == nullptr);
}

constexpr uint64_t FIT_TEST_GIB    = 1ull << 30;
constexpr uint64_t FIT_TEST_MARGIN = 256ull << 20;

ggml_backend_dev_t fit_test_device(int & tag) {
    return reinterpret_cast<ggml_backend_dev_t>(&tag);
}

tts_cpp::acestep::FitPool fit_test_pool(ggml_backend_dev_t dev, bool shares_host_memory, uint64_t free_gib) {
    tts_cpp::acestep::FitPool pool;
    pool.device             = dev;
    pool.shares_host_memory = shares_host_memory;
    pool.free_bytes         = free_gib * FIT_TEST_GIB;
    pool.total_bytes        = pool.free_bytes;
    return pool;
}

void test_fit_pool_sharing() {
    using tts_cpp::acestep::backend_requested_init;
    using tts_cpp::acestep::FIT_OPENCL_SHARES_HOST_MEMORY;
    using tts_cpp::acestep::fit_device_shares_host_memory;
    using tts_cpp::acestep::fit_pool_for_device;
    using tts_cpp::acestep::fit_pool_from_memory;
    using tts_cpp::acestep::FitPool;

    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_CPU, "CPU", 32 * FIT_TEST_GIB));
    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_IGPU, "Vulkan", 8 * FIT_TEST_GIB));
    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "MTL", 24 * FIT_TEST_GIB));
    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "HTP", 0));
    CHECK(!fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", 8 * FIT_TEST_GIB));
    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "OpenCL", 4 * FIT_TEST_GIB) ==
          FIT_OPENCL_SHARES_HOST_MEMORY);

    int                      gpu_tag = 0;
    const ggml_backend_dev_t gpu     = fit_test_device(gpu_tag);

    const FitPool silent = fit_pool_from_memory(gpu, GGML_BACKEND_DEVICE_TYPE_GPU, "HTP", 0, 0, 12 * FIT_TEST_GIB,
                                                16 * FIT_TEST_GIB);
    CHECK(silent.device == gpu);
    CHECK(silent.shares_host_memory);
    CHECK(silent.free_bytes == 12 * FIT_TEST_GIB);
    CHECK(silent.total_bytes == 16 * FIT_TEST_GIB);

    const FitPool discrete = fit_pool_from_memory(gpu, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", 6 * FIT_TEST_GIB,
                                                  8 * FIT_TEST_GIB, 12 * FIT_TEST_GIB, 16 * FIT_TEST_GIB);
    CHECK(!discrete.shares_host_memory);
    CHECK(discrete.free_bytes == 6 * FIT_TEST_GIB);
    CHECK(discrete.total_bytes == 8 * FIT_TEST_GIB);

    ggml_backend_t cpu = backend_requested_init("cpu");
    CHECK(cpu != nullptr);
    if (!cpu) return;
    const FitPool pool = fit_pool_for_device(ggml_backend_get_device(cpu));
    CHECK(pool.device == ggml_backend_get_device(cpu));
    CHECK(pool.shares_host_memory);
    CHECK(pool.total_bytes > 0);
    ggml_backend_free(cpu);
}

void test_fit_pools_charge_distinct_lm_device() {
    using tts_cpp::acestep::fit_budget_checks;
    using tts_cpp::acestep::fit_charge;
    using tts_cpp::acestep::fit_checks_headroom;
    using tts_cpp::acestep::fit_checks_hold;
    using tts_cpp::acestep::fit_pool_register;
    using tts_cpp::acestep::FIT_HOST_POOL;
    using tts_cpp::acestep::FitBudgetCheck;
    using tts_cpp::acestep::FitPool;
    using tts_cpp::acestep::PoolCharge;

    int                      host_tag = 0, lm_tag = 0, unknown_tag = 0;
    const ggml_backend_dev_t host = fit_test_device(host_tag);
    const ggml_backend_dev_t lm   = fit_test_device(lm_tag);

    std::vector<FitPool> pools;
    CHECK(fit_pool_register(pools, fit_test_pool(host, true, 16)) == FIT_HOST_POOL);
    const size_t primary = fit_pool_register(pools, fit_test_pool(host, true, 16));
    const size_t lm_pool = fit_pool_register(pools, fit_test_pool(lm, false, 1));
    CHECK(primary == FIT_HOST_POOL);
    CHECK(lm_pool == 1);
    CHECK(pools.size() == 2);

    PoolCharge peak = fit_charge(pools, host, 2 * FIT_TEST_GIB);
    peak.add(fit_charge(pools, lm, 3 * FIT_TEST_GIB));
    CHECK(peak.bytes[FIT_HOST_POOL] == 2 * FIT_TEST_GIB);
    CHECK(peak.bytes[lm_pool] == 3 * FIT_TEST_GIB);
    CHECK(!fit_checks_hold(fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN)));

    pools[lm_pool].free_bytes = 8 * FIT_TEST_GIB;
    const std::vector<FitBudgetCheck> checks = fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN);
    CHECK(checks.size() == 2);
    CHECK(fit_checks_hold(checks));
    CHECK(fit_checks_headroom(checks) == 5 * FIT_TEST_GIB - FIT_TEST_MARGIN);

    CHECK(fit_charge(pools, fit_test_device(unknown_tag), 1).bytes[FIT_HOST_POOL] == 1);
}

void test_fit_pools_shared_budget() {
    using tts_cpp::acestep::fit_budget_checks;
    using tts_cpp::acestep::fit_charge;
    using tts_cpp::acestep::fit_checks_headroom;
    using tts_cpp::acestep::fit_checks_hold;
    using tts_cpp::acestep::fit_pool_register;
    using tts_cpp::acestep::fit_shared_budget_pool;
    using tts_cpp::acestep::FIT_HOST_POOL;
    using tts_cpp::acestep::FitBudgetCheck;
    using tts_cpp::acestep::FitPool;
    using tts_cpp::acestep::PoolCharge;

    int                      host_tag = 0, npu_tag = 0, metal_tag = 0;
    const ggml_backend_dev_t host  = fit_test_device(host_tag);
    const ggml_backend_dev_t npu   = fit_test_device(npu_tag);
    const ggml_backend_dev_t metal = fit_test_device(metal_tag);

    std::vector<FitPool> npu_pools;
    fit_pool_register(npu_pools, fit_test_pool(host, true, 4));
    const size_t npu_primary = fit_pool_register(npu_pools, fit_test_pool(npu, true, 0));
    CHECK(fit_shared_budget_pool(npu_pools, npu_primary) == FIT_HOST_POOL);
    PoolCharge npu_peak = fit_charge(npu_pools, host, FIT_TEST_GIB);
    npu_peak.add(fit_charge(npu_pools, npu, 2 * FIT_TEST_GIB));
    const std::vector<FitBudgetCheck> npu_checks = fit_budget_checks(npu_pools, npu_peak, npu_primary, FIT_TEST_MARGIN);
    CHECK(npu_checks.size() == 1);
    CHECK(fit_checks_hold(npu_checks));
    CHECK(fit_checks_headroom(npu_checks) == FIT_TEST_GIB - FIT_TEST_MARGIN);
    npu_peak.add(fit_charge(npu_pools, npu, FIT_TEST_GIB));
    CHECK(!fit_checks_hold(fit_budget_checks(npu_pools, npu_peak, npu_primary, FIT_TEST_MARGIN)));

    std::vector<FitPool> metal_pools;
    fit_pool_register(metal_pools, fit_test_pool(host, true, 64));
    const size_t metal_primary = fit_pool_register(metal_pools, fit_test_pool(metal, true, 2));
    CHECK(fit_shared_budget_pool(metal_pools, metal_primary) == metal_primary);
    PoolCharge metal_peak = fit_charge(metal_pools, host, FIT_TEST_GIB);
    metal_peak.add(fit_charge(metal_pools, metal, 2 * FIT_TEST_GIB));
    CHECK(!fit_checks_hold(fit_budget_checks(metal_pools, metal_peak, metal_primary, FIT_TEST_MARGIN)));
}

void test_fit_pools_check_every_discrete_device() {
    using tts_cpp::acestep::fit_budget_checks;
    using tts_cpp::acestep::fit_charge;
    using tts_cpp::acestep::fit_checks_headroom;
    using tts_cpp::acestep::fit_checks_hold;
    using tts_cpp::acestep::fit_pool_register;
    using tts_cpp::acestep::FitBudgetCheck;
    using tts_cpp::acestep::FitPool;
    using tts_cpp::acestep::PoolCharge;

    int                      host_tag = 0, dit_tag = 0, lm_tag = 0;
    const ggml_backend_dev_t host = fit_test_device(host_tag);
    const ggml_backend_dev_t dit  = fit_test_device(dit_tag);
    const ggml_backend_dev_t lm   = fit_test_device(lm_tag);

    std::vector<FitPool> pools;
    fit_pool_register(pools, fit_test_pool(host, true, 32));
    const size_t primary = fit_pool_register(pools, fit_test_pool(dit, false, 8));
    const size_t lm_pool = fit_pool_register(pools, fit_test_pool(lm, false, 2));

    PoolCharge peak = fit_charge(pools, host, FIT_TEST_GIB);
    peak.add(fit_charge(pools, dit, 4 * FIT_TEST_GIB)).add(fit_charge(pools, lm, 3 * FIT_TEST_GIB));
    const std::vector<FitBudgetCheck> tight = fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN);
    CHECK(tight.size() == 3);
    CHECK(!fit_checks_hold(tight));

    pools[lm_pool].free_bytes = 4 * FIT_TEST_GIB;
    const std::vector<FitBudgetCheck> roomy = fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN);
    CHECK(fit_checks_hold(roomy));
    CHECK(fit_checks_headroom(roomy) == FIT_TEST_GIB - FIT_TEST_MARGIN);
}

// Which backend each stage runs on decides which numerical path the generated
// audio takes, so the policy is locked here rather than only observed on a
// device lane. Mirrors the three branches Engine::create() relies on: the
// backend allowlist, the CPU fallback for everything else, and the env
// overrides layered on top.

// A backend validated for every stage except the autoregressive LM keeps the
// LM on the CPU while the detokenizer and encoders follow the GPU.
void check_gpu_backend_keeps_lm_on_cpu(const char * name, const char * device_desc) {
    using tts_cpp::acestep::PlacementOverrides;
    using tts_cpp::acestep::resolve_stage_placement;
    using tts_cpp::acestep::StagePlacement;

    const PlacementOverrides none;
    StagePlacement p = resolve_stage_placement(name, device_desc, none);
    CHECK(!p.lm_on_gpu);
    CHECK(p.detok_on_gpu);
    CHECK(p.enc_on_gpu);
}

void test_stage_placement() {
    using tts_cpp::acestep::backend_name_is_cuda;
    using tts_cpp::acestep::backend_name_is_metal;
    using tts_cpp::acestep::backend_name_is_opencl;
    using tts_cpp::acestep::backend_name_is_vulkan;
    using tts_cpp::acestep::PlacementOverrides;
    using tts_cpp::acestep::resolve_stage_placement;
    using tts_cpp::acestep::StagePlacement;

    const auto xdna = resolve_stage_placement("XDNA", "", {});
    CHECK(!xdna.enc_on_gpu && !xdna.lm_on_gpu && !xdna.detok_on_gpu);

    // -- backend name matching ------------------------------------------------
    // ggml-metal registers as "MTL"; older ggml reported "Metal". Both must
    // match or the allowlist is silently dead on one of them.
    CHECK(backend_name_is_metal("MTL"));
    CHECK(backend_name_is_metal("Metal"));
    CHECK(backend_name_is_vulkan("Vulkan"));
    // ggml-opencl's REGISTRY name; its device reports as "GPUOpenCL", which is
    // not what reaches here.
    CHECK(backend_name_is_opencl("OpenCL"));
    CHECK(!backend_name_is_opencl("GPUOpenCL"));
    // ggml-cuda's REGISTRY name; the HIP/MUSA builds of the same backend
    // register as "ROCm"/"MUSA" and must not match.
    CHECK(backend_name_is_cuda("CUDA"));
    CHECK(!backend_name_is_cuda("ROCm"));
    CHECK(!backend_name_is_cuda("MUSA"));

    // The input is the REGISTRY name, which carries no device-index suffix.
    // ggml_backend_name() would hand over "MTL0" / "Vulkan0" and match nothing.
    CHECK(!backend_name_is_metal("MTL0"));
    CHECK(!backend_name_is_vulkan("Vulkan0"));
    CHECK(!backend_name_is_opencl("OpenCL0"));
    CHECK(!backend_name_is_cuda("CUDA0"));

    // Exact compare: no case folding, no substring match, and null/empty safe.
    CHECK(!backend_name_is_metal("mtl"));
    CHECK(!backend_name_is_metal("metal"));
    CHECK(!backend_name_is_metal("MTLX"));
    CHECK(!backend_name_is_vulkan("vulkan"));
    CHECK(!backend_name_is_opencl("opencl"));
    CHECK(!backend_name_is_metal("CUDA"));
    CHECK(!backend_name_is_cuda("cuda"));
    CHECK(!backend_name_is_vulkan("MTL"));
    CHECK(!backend_name_is_metal("Vulkan"));
    CHECK(!backend_name_is_opencl("Vulkan"));
    CHECK(!backend_name_is_cuda("Vulkan"));
    CHECK(!backend_name_is_metal(""));
    CHECK(!backend_name_is_vulkan(""));
    CHECK(!backend_name_is_opencl(""));
    CHECK(!backend_name_is_cuda(""));
    CHECK(!backend_name_is_metal(nullptr));
    CHECK(!backend_name_is_vulkan(nullptr));
    CHECK(!backend_name_is_opencl(nullptr));
    CHECK(!backend_name_is_cuda(nullptr));

    const PlacementOverrides none;
    const char * const radv_desc   = "Radeon 8060S Graphics (RADV GFX1151)";
    const char * const nvidia_desc = "NVIDIA GeForce RTX 3080";

    // -- device predicate: the Vulkan LM denylist is per-device ---------------
    using tts_cpp::acestep::vulkan_device_lm_blocked;
    CHECK(vulkan_device_lm_blocked("Mali-G715"));
    CHECK(vulkan_device_lm_blocked("Mali-G78"));
    CHECK(!vulkan_device_lm_blocked(radv_desc));
    CHECK(!vulkan_device_lm_blocked(nvidia_desc));
    CHECK(!vulkan_device_lm_blocked("Samsung Xclipse 920"));
    CHECK(!vulkan_device_lm_blocked("AMD Radeon RX 7900 XTX"));  // proprietary driver
    CHECK(!vulkan_device_lm_blocked("Intel(R) Arc(tm) A770 Graphics"));
    CHECK(!vulkan_device_lm_blocked(""));
    CHECK(!vulkan_device_lm_blocked(nullptr));

    // -- Hexagon: every stage runs on the NPU ---------------------------------
    using tts_cpp::acestep::backend_name_is_hexagon;
    CHECK(backend_name_is_hexagon("HTP"));
    CHECK(!backend_name_is_hexagon("HTP0"));
    CHECK(!backend_name_is_hexagon("htp"));
    CHECK(!backend_name_is_hexagon(nullptr));
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("HTP", "Hexagon", ov);
        CHECK(!p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
    }

    // -- allowlist: Metal, OpenCL, CUDA, and Hexagon keep LM + detokenizer on GPU
    for (const char * allowed : { "MTL", "Metal", "OpenCL", "CUDA", "HTP" }) {
        StagePlacement p = resolve_stage_placement(allowed, "", none);
        CHECK(p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
        CHECK(p.enc_on_gpu);  // encoders follow the GPU on every backend
    }

    // Vulkan runs every stage on the GPU on any device that is not denylisted.
    // RADV (Strix Halo) and NVIDIA (RTX 3080) are measured against the
    // F32-dequantized reference; the rest inherit the GPU placement.
    for (const char * device : { radv_desc, nvidia_desc, "Samsung Xclipse 920",
                                 "Intel(R) Arc(tm) A770 Graphics", "" }) {
        StagePlacement p = resolve_stage_placement("Vulkan", device, none);
        CHECK(p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
        CHECK(p.enc_on_gpu);
    }

    // Mali keeps the LM on the CPU: the GPU LM collapses to repeated semantic
    // codes and truncates the song there (README "Backends").
    check_gpu_backend_keeps_lm_on_cpu("Vulkan", "Mali-G715");
    check_gpu_backend_keeps_lm_on_cpu("Vulkan", "Mali-G78");

    // -- fallback: everything else keeps the shipping CPU placement -----------
    // Unmeasured backends must not silently pick up the GPU path. "MTL0" is in
    // this list on purpose: a suffixed name is NOT the allowlisted one. "ROCm"
    // and "MUSA" are the HIP/MUSA builds of ggml-cuda: same code base, different
    // silicon, unmeasured.
    const char * const others[] = { "ROCm",     "MUSA",     "SYCL", "BLAS", "CPU",
                                    "MTL0",     "Vulkan0",  "OpenCL0", "CUDA0",
                                    "",         nullptr };
    for (const char * other : others) {
        // A RADV description must not rescue a non-Vulkan registry name either.
        StagePlacement p = resolve_stage_placement(other, radv_desc, none);
        CHECK(!p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);
        CHECK(p.enc_on_gpu);  // only the LM and the detokenizer are allowlisted
    }

    // -- env overrides: applied after the allowlist ---------------------------
    // GPU hatch lifts a non-allowlisted backend (this is how a new backend gets
    // measured without a rebuild).
    {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        StagePlacement p = resolve_stage_placement("SYCL", "", ov);
        CHECK(p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);  // the LM hatch must not move the detokenizer
    }
    {
        PlacementOverrides ov;
        ov.detok_gpu     = true;
        StagePlacement p = resolve_stage_placement("SYCL", "", ov);
        CHECK(p.detok_on_gpu);
        CHECK(!p.lm_on_gpu);
    }

    // GPU hatch also lifts a non-validated Vulkan device (how a new device gets
    // measured without a rebuild).
    {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        StagePlacement p = resolve_stage_placement("Vulkan", "Mali-G715", ov);
        CHECK(p.lm_on_gpu);
    }

    // CPU hatch demotes an allowlisted backend.
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("MTL", "", ov);
        CHECK(!p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
    }
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("Vulkan", radv_desc, ov);
        CHECK(!p.lm_on_gpu);  // CPU hatch demotes a validated RADV device too
        CHECK(p.detok_on_gpu);
    }
    {
        PlacementOverrides ov;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement("Vulkan", "", ov);
        CHECK(!p.detok_on_gpu);
        CHECK(p.lm_on_gpu);  // the detokenizer hatch leaves the LM where it was
    }

    // Precedence: CPU wins when both hatches are set for the same stage, on an
    // allowlisted backend and on a fallback one alike.
    for (const char * name : { "MTL", "Metal", "Vulkan", "OpenCL", "CUDA" }) {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        ov.lm_cpu        = true;
        ov.detok_gpu     = true;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement(name, radv_desc, ov);
        CHECK(!p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);
    }

    // The encoder hatch is independent of the LM/detokenizer allowlist.
    for (const char * name : { "MTL", "Vulkan", "OpenCL", "CUDA" }) {
        PlacementOverrides ov;
        ov.encoders_cpu  = true;
        StagePlacement p = resolve_stage_placement(name, "", ov);
        CHECK(!p.enc_on_gpu);
        CHECK(p.lm_on_gpu == (backend_name_is_metal(name) || backend_name_is_opencl(name) ||
                              backend_name_is_cuda(name) || backend_name_is_vulkan(name)));
    }
}

// Locks which variable drives which stage, and that PRESENCE is what counts:
// ACESTEP_LM_CPU=0 still forces the LM to the CPU (the getenv() semantics this
// policy inherited). Uses "0" as the "set" value so the assertion is identical
// on POSIX and Windows, where _putenv_s(k, "") removes the variable instead.
void test_vae_backend_request_env() {
    using tts_cpp::acestep::vae_backend_request_from_env;

    set_env("ACESTEP_VAE_GPU", nullptr);
    CHECK(vae_backend_request_from_env("hexagon") == "hexagon");
    CHECK(vae_backend_request_from_env("auto") == "auto");
    CHECK(vae_backend_request_from_env("xdna") == "cpu");
    CHECK(vae_backend_request_from_env("XDNA0") == "cpu");

    set_env("ACESTEP_VAE_GPU", "0");
    CHECK(vae_backend_request_from_env("hexagon") == "cpu");
    CHECK(vae_backend_request_from_env("auto") == "auto");

    set_env("ACESTEP_VAE_GPU", "1");
    CHECK(vae_backend_request_from_env("hexagon") == "hexagon");
    CHECK(vae_backend_request_from_env("xdna") == "xdna");

    set_env("ACESTEP_VAE_GPU", nullptr);
}

void test_placement_env() {
    using tts_cpp::acestep::placement_overrides_from_env;
    using tts_cpp::acestep::PlacementOverrides;

    const char * const vars[] = { "ACESTEP_LM_GPU",    "ACESTEP_LM_CPU", "ACESTEP_DETOK_GPU",
                                  "ACESTEP_DETOK_CPU", "ACESTEP_ENCODERS_CPU" };
    auto clear_all = [&] {
        for (const char * k : vars) set_env(k, nullptr);
    };

    // Nothing set -> no override.
    clear_all();
    {
        PlacementOverrides ov = placement_overrides_from_env();
        CHECK(!ov.lm_gpu);
        CHECK(!ov.lm_cpu);
        CHECK(!ov.detok_gpu);
        CHECK(!ov.detok_cpu);
        CHECK(!ov.encoders_cpu);
    }

    // One variable at a time -> exactly its own flag, nothing else.
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        clear_all();
        set_env(vars[i], "0");  // presence, not value
        PlacementOverrides ov  = placement_overrides_from_env();
        const bool         set[] = { ov.lm_gpu, ov.lm_cpu, ov.detok_gpu, ov.detok_cpu, ov.encoders_cpu };
        for (size_t j = 0; j < sizeof(set) / sizeof(set[0]); ++j) CHECK(set[j] == (i == j));
    }

    clear_all();
}

void add_gate_test_tensor(ggml_context * ctx, gguf_context * weights, const char * name, ggml_type type,
                          int64_t ne0) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, 2);
    ggml_set_name(t, name);
    std::memset(t->data, 0, ggml_nbytes(t));
    gguf_add_tensor(weights, t);
}

gguf_context * gate_test_weights(ggml_context * ctx, bool with_k_quant) {
    gguf_context * weights = gguf_init_empty();
    add_gate_test_tensor(ctx, weights, "norm.weight", GGML_TYPE_F32, 32);
    add_gate_test_tensor(ctx, weights, "proj.weight", GGML_TYPE_Q8_0, 32);
    if (with_k_quant) add_gate_test_tensor(ctx, weights, "down.weight", GGML_TYPE_Q4_K, 256);
    return weights;
}

void check_hexagon_weight_types() {
    using tts_cpp::acestep::hexagon_runs_weight_type;
    for (const ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                                  GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4 }) {
        CHECK(hexagon_runs_weight_type(type));
    }
    for (const ggml_type type : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                  GGML_TYPE_Q5_0, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS }) {
        CHECK(!hexagon_runs_weight_type(type));
    }
}

void test_hexagon_weight_gate() {
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::dit_gguf_hexagon_refusal;
    using tts_cpp::acestep::first_tensor_hexagon_cannot_run;
    using tts_cpp::acestep::hexagon_weight_refusal;

    check_hexagon_weight_types();

    ggml_init_params ip{ 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx      = ggml_init(ip);
    gguf_context *   runnable = gate_test_weights(ctx, /*with_k_quant=*/false);
    gguf_context *   refused  = gate_test_weights(ctx, /*with_k_quant=*/true);
    CHECK(first_tensor_hexagon_cannot_run(runnable) == -1);
    CHECK(first_tensor_hexagon_cannot_run(refused) == 2);

    const std::string message = hexagon_weight_refusal(refused, 2, "weights.gguf", "acestep-test");
    CHECK(message.rfind("[acestep-test] weights.gguf: down.weight is q4_K; the Hexagon NPU runs ", 0) == 0);
    CHECK(message.find("q8_0") != std::string::npos);
    CHECK(message.find("q4_K,") == std::string::npos);

    ggml_backend_t cpu = backend_cpu_init();
    CHECK(cpu != nullptr);
    if (cpu) {
        CHECK(dit_gguf_hexagon_refusal(refused, "weights.gguf", cpu, "acestep-test").empty());
        ggml_backend_free(cpu);
    }
    gguf_free(runnable);
    gguf_free(refused);
    ggml_free(ctx);
}

struct RegNameOverride {
    explicit RegNameOverride(const char * name) { tts_cpp::acestep::backend_reg_name_override() = name; }
    ~RegNameOverride() { tts_cpp::acestep::backend_reg_name_override() = nullptr; }
};

std::string write_gate_test_gguf() {
    const std::string path = test_temp_dir() + "/qvac-acestep-hexagon-gate-test.gguf";
    ggml_init_params  ip{ 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *    ctx     = ggml_init(ip);
    gguf_context *    weights = gate_test_weights(ctx, /*with_k_quant=*/true);
    CHECK(gguf_write_to_file(weights, path.c_str(), /*only_meta=*/false));
    gguf_free(weights);
    ggml_free(ctx);
    return path;
}

template <typename Load>
std::string refusal_from(Load && load) {
    try {
        load();
    } catch (const std::runtime_error & e) {
        return e.what();
    }
    return "";
}

void test_hexagon_refusal_paths() {
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::detok_model_load;
    using tts_cpp::acestep::dit_gguf_open;
    using tts_cpp::acestep::dit_gguf_require_hexagon_weights;
    using tts_cpp::acestep::DitGGUF;
    using tts_cpp::acestep::fit_params;
    using tts_cpp::acestep::FitOptions;
    using tts_cpp::acestep::FitStatus;
    using tts_cpp::acestep::Vae;
    using tts_cpp::acestep::VaeOptions;

    const std::string path    = write_gate_test_gguf();
    const std::string refused = path + ": down.weight is q4_K; the Hexagon NPU runs ";
    ggml_backend_t    cpu     = backend_cpu_init();
    CHECK(cpu != nullptr);
    if (!cpu) return;
    RegNameOverride as_hexagon("HTP");

    DitGGUF g;
    CHECK(dit_gguf_open(g, path));
    const std::string gate = refusal_from([&] { dit_gguf_require_hexagon_weights(g, path, cpu, "acestep-test"); });
    CHECK(gate.rfind("[acestep-test] " + refused, 0) == 0);
    CHECK(g.ctx == nullptr);
    CHECK(g.meta == nullptr);
    CHECK(g.file.data == nullptr);

    const std::string detok = refusal_from([&] { detok_model_load(path, cpu, /*verbose=*/false); });
    CHECK(detok.rfind("[acestep-detok] " + refused, 0) == 0);

    VaeOptions vae_opts;
    vae_opts.backend = "cpu";
    const std::string vae = refusal_from([&] { Vae::load(path, vae_opts); });
    CHECK(vae.rfind("[acestep-vae] " + refused, 0) == 0);

    FitOptions fit;
    fit.backend             = "cpu";
    fit.text_enc_model_path = path;
    fit.lm_model_path       = path;
    fit.dit_model_path      = path;
    fit.vae_model_path      = path;
    fit.duration_seconds    = 8.0f;
    fit.text_tokens         = 32;
    fit.lyric_tokens        = 16;
    const tts_cpp::acestep::FitResult result = fit_params(fit);
    CHECK(result.status == FitStatus::Error);
    CHECK(!result.fits);
    CHECK(result.reason == "measurement-failed");
    CHECK(result.report.rfind("[acestep-fit] " + refused, 0) == 0);

    ggml_backend_free(cpu);
}

tts_cpp::acestep::EngineOptions gate_test_engine_options(const std::string & path) {
    tts_cpp::acestep::EngineOptions opts;
    opts.backend             = "cpu";
    opts.lm_backend          = "cpu";
    opts.text_enc_model_path = path;
    opts.lm_model_path       = path;
    opts.dit_model_path      = path;
    opts.vae_model_path      = path;
    return opts;
}

void test_hexagon_lazy_engine_refusal() {
    using tts_cpp::acestep::Engine;
    using tts_cpp::acestep::EngineOptions;

    const std::string   path = write_gate_test_gguf();
    const EngineOptions opts = gate_test_engine_options(path);
    set_env("ACESTEP_KEEP_STAGES", nullptr);

    const std::string on_cpu = refusal_from([&] { Engine::create(opts); });
    CHECK(on_cpu == "acestep engine: bad DiT config in " + path);

    RegNameOverride   as_hexagon("HTP");
    const std::string on_hexagon = refusal_from([&] { Engine::create(opts); });
    CHECK(on_hexagon.rfind("[acestep-engine] " + path + ": down.weight is q4_K; the Hexagon NPU runs ", 0) == 0);
}


}  // namespace

int main() {
    test_backend_device_types();
    test_gpu_matmul_capability();
    test_gpu_fallback_reason();
    test_gpu_tier_policy();
    test_backend_request_matching();
    test_dsp_library_path();
    test_backend_requested_init();
    test_backend_init_first_matching();
    test_weight_placement_on_cpu();
    test_lm_backend_request();
    test_fit_pool_sharing();
    test_fit_pools_charge_distinct_lm_device();
    test_fit_pools_shared_budget();
    test_fit_pools_check_every_discrete_device();
    test_stage_placement();
    test_vae_backend_request_env();
    test_placement_env();
    test_hexagon_weight_gate();
    test_hexagon_refusal_paths();
    test_hexagon_lazy_engine_refusal();
    std::fprintf(stderr, "[test-acestep-backends] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
