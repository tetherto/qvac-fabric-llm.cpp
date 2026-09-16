// Hardware-independent tests for fitting policy and memory arithmetic.
// Covers automatic acceleration, placement overrides, shared-memory budgets,
// and context interpolation without loading model weights or a GPU backend.

#include "../common/fit.h"

#include <cstdio>
#include <cstdint>
#include <vector>

constexpr int64_t GiB = 1024LL * 1024 * 1024;

static int failures = 0;

static void expect_i64(const char * label, int64_t got, int64_t want) {
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %lld, want %lld\n", label, (long long) got, (long long) want);
        failures++;
    }
}

static void expect_u32(const char * label, uint32_t got, uint32_t want) {
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %u, want %u\n", label, got, want);
        failures++;
    }
}

static void test_automatic_acceleration() {
    // Each row starts from the CLI's automatic defaults. Exercise both model
    // types: MoE should select caching, dense should select prefetch.
    enum scenario {
        DEFAULTS, DISABLED, RESOLVED, CPU_ONLY, MULTI_GPU, UNIFIED,
        UNSUPPORTED_CACHE, NO_COPY_STREAM, NO_OFFLOAD, TRAINING,
    };
    struct test_case {
        const char * label;
        scenario config;
        bool want_cache;
        bool want_prefetch;
    };
    const test_case cases[] = {
        { "automatic defaults", DEFAULTS, true, true },
        { "explicit disable", DISABLED, false, false },
        { "already selected or explicit value", RESOLVED, false, false },
        { "CPU only", CPU_ONLY, false, false },
        { "multiple GPUs", MULTI_GPU, false, false },
        { "unified memory", UNIFIED, false, false },
        { "unsupported cache backend", UNSUPPORTED_CACHE, false, true },
        { "no copy stream", NO_COPY_STREAM, true, false },
        { "operation offload disabled", NO_OFFLOAD, false, false },
        { "training", TRAINING, false, false },
    };
    const auto mparams = llama_model_default_params();
    for (const auto & tc : cases) {
        auto cparams = llama_context_default_params();
        cparams.moe_cache_auto = tc.config != DISABLED;
        cparams.moe_cache_size = tc.config == RESOLVED ? GiB : 0;
        cparams.prefetch_weights = tc.config == RESOLVED;
        cparams.op_offload = tc.config != NO_OFFLOAD;
        cparams.training = tc.config == TRAINING;
        const size_t nd = tc.config == CPU_ONLY ? 0 : tc.config == MULTI_GPU ? 2 : 1;
        for (uint32_t n_expert : { 0U, 8U }) {
            expect_i64(tc.label, common_fit_auto_moe_cache(
                mparams, cparams, n_expert, nd, tc.config == UNIFIED, tc.config != UNSUPPORTED_CACHE),
                n_expert > 0 && tc.want_cache);
            expect_i64(tc.label, common_fit_auto_prefetch_weights(
                cparams, tc.config != DISABLED, n_expert, nd, tc.config == UNIFIED, tc.config != NO_COPY_STREAM),
                n_expert == 0 && tc.want_prefetch);
        }
    }
}

static void test_auto_cache_preserves_context_fitting() {
    auto mparams = llama_model_default_params();
    auto cparams = llama_context_default_params();
    cparams.moe_cache_auto = true;

    // Cache selection defers context reduction until a recursive fit. Explicit
    // placement cannot reach that recursion: step 3 rejects it. Those callers
    // must retain the ordinary context-reduction path instead.
    for (int n_gpu_layers : { 0, 12, 999 }) {
        mparams.n_gpu_layers = n_gpu_layers;
        expect_i64("explicit -ngl must not defer context fitting",
            common_fit_auto_moe_cache(mparams, cparams, 8, 1, false, true), false);
    }
    mparams = llama_model_default_params();
    llama_model_tensor_buft_override overrides[] = {
        { "blk\\.\\d+\\.ffn_.*_exps", nullptr },
        { nullptr, nullptr },
    };
    mparams.tensor_buft_overrides = overrides;
    expect_i64("--cpu-moe / -ot must not defer context fitting",
        common_fit_auto_moe_cache(mparams, cparams, 8, 1, false, true), false);

    // An allocated but empty overrides array is the normal CLI default.
    mparams.tensor_buft_overrides = &overrides[1];
    expect_i64("empty overrides still allow automatic cache",
        common_fit_auto_moe_cache(mparams, cparams, 8, 1, false, true), true);
}

int main() {
    test_automatic_acceleration();
    test_auto_cache_preserves_context_fitting();
    // --- common_fit_shared_pool_deficit ---

    // nd == 1, discrete GPU: device demand never counts against the host pool.
    expect_i64("discrete GPU never folds",
        common_fit_shared_pool_deficit({14 * GiB}, {false}, 8 * GiB, 11 * GiB, 1 * GiB),
        4 * GiB); // host row alone: 8 - 11 = -3 free, short of the 1 GiB margin by 4

    // nd == 1, shared device, device surplus but combined deficit — the
    // measured Gemma ngl=48 shape: device row passes on its own, the pool sum
    // does not.
    expect_i64("shared device folds into the pool",
        common_fit_shared_pool_deficit({14 * GiB}, {true}, 18 * GiB, 5 * GiB, 1 * GiB),
        2 * GiB); // 18 - 5 - 14 = -1 free, short of the 1 GiB margin by 2

    // Same shape with room to spare: no deficit.
    expect_i64("combined budget met",
        common_fit_shared_pool_deficit({10 * GiB}, {true}, 18 * GiB, 5 * GiB, 1 * GiB),
        0);

    // nd == 2, mixed shared/discrete: only the shared device is folded.
    expect_i64("mixed devices fold only the shared one",
        common_fit_shared_pool_deficit({10 * GiB, 30 * GiB}, {true, false}, 18 * GiB, 5 * GiB, 1 * GiB),
        0);

    // --- common_fit_shared_pool_target ---

    // One shared device takes the whole pool budget: 18 - 5 - 1 = 12.
    expect_i64("single shared device takes the whole budget",
        common_fit_shared_pool_target(18 * GiB, 5 * GiB, 1 * GiB, 1),
        12 * GiB);

    // Two shared devices split it, so they cannot each claim all of it.
    // Without the split both would cap at 12 GiB and together overrun the pool.
    expect_i64("two shared devices split the budget",
        common_fit_shared_pool_target(18 * GiB, 5 * GiB, 1 * GiB, 2),
        6 * GiB);

    // A negative budget stays whole: splitting would understate the shortfall.
    expect_i64("negative budget is not split",
        common_fit_shared_pool_target(6 * GiB, 8 * GiB, 1 * GiB, 2),
        -3 * GiB);

    // --- common_fit_reduced_n_ctx ---

    // Reviewer-traced inflation shape: deficit-forced entry where the target
    // still exceeds the max-context sample. Pre-fix this interpolated
    // 32768 -> 47104; the result must never exceed the training context.
    {
        const uint32_t n_ctx = common_fit_reduced_n_ctx(
            /*sum_used_target        =*/ 17 * GiB,
            /*sum_projected_used     =*/ 14 * GiB,
            /*sum_projected_used_min =*/ 13 * GiB,
            /*hp_nct                 =*/ 131072,
            /*n_ctx_min              =*/ 4096);
        if (n_ctx > 131072) {
            fprintf(stderr, "FAIL inflation clamp: n_ctx %u exceeds training context\n", n_ctx);
            failures++;
        }
        expect_u32("target beyond max sample clamps to hp_nct", n_ctx, 131072);
    }

    // Context-independent rows (n_gpu_layers == 0 keeps KV host-side):
    // used_delta == 0 exactly. Pre-fix this divided by zero — SIGFPE on
    // x86-64, silent 0 on AArch64. Must report "no reduction possible".
    expect_u32("zero delta reports no reduction, not a fault",
        common_fit_reduced_n_ctx(12 * GiB, 14 * GiB, 14 * GiB, 131072, 4096),
        0);

    // Ordinary interior reduction: target halfway between the samples lands
    // halfway between the context bounds (rounded down to a 256 multiple).
    expect_u32("interior interpolation",
        common_fit_reduced_n_ctx(13 * GiB, 14 * GiB, 12 * GiB, 65536, 4096),
        // 4096 + (65536-4096) * 1/2 = 34816, already a 256 multiple
        34816);

    // Target below the min-context sample: no context can meet it.
    expect_u32("target below min sample",
        common_fit_reduced_n_ctx(11 * GiB, 14 * GiB, 12 * GiB, 65536, 4096),
        0);

    // Result never drops below n_ctx_min.
    {
        const uint32_t n_ctx = common_fit_reduced_n_ctx(
            12 * GiB + 1, 14 * GiB, 12 * GiB, 65536, 4096);
        if (n_ctx != 0 && n_ctx < 4096) {
            fprintf(stderr, "FAIL floor: n_ctx %u below n_ctx_min\n", n_ctx);
            failures++;
        }
    }

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("OK\n");
    return 0;
}
