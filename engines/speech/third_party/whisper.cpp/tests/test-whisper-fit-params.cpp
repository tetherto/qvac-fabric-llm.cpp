// QVAC (see PATCHES.md): fit-projection parity tests for whisper_fit_params
// (include/whisper.h) -- assert that the metadata-only memory projection
// matches what a REAL load actually allocates, byte for byte, where the
// projection is exact by construction:
//
//   1. whisper_fit_params returns a projection (never Error) for a readable
//      model, with non-zero weight/kv/compute figures and a report;
//   2. byte parity -- a projection at n_decoders = 1 must equal
//      whisper_fit_actual's measurement of a real
//      whisper_init_from_file_with_params + whisper_init_state on every
//      component: weights (same buffer-type sizing loop), kv caches (same
//      from_buft sizing), the four schedulers' compute buffers
//      (ggml_backend_sched_reserve_size vs the buffers
//      ggml_backend_sched_alloc_graph committed), and the host-overflow
//      split;
//   3. n_decoders worst case -- a projection at the whisper_full default
//      (5 decoders) must price a strictly larger self-attention KV cache
//      (the (n+2)x recreation) and never a smaller decode graph, and its
//      kv_bytes must byte-match whisper_fit_actual, which reproduces the
//      runtime recreation for n_decoders > 1;
//   4. workload scaling -- longer audio grows ONLY host_bytes; the device
//      projection is init-time-fixed (whisper allocates its schedulers at
//      the model's full n_audio_ctx regardless of input length);
//   5. VAD -- with a VAD model the projection gains a non-zero vad component
//      that also matches the real VAD context byte for byte.
//
// On a GPU run the parity gates additionally check the classifier against
// the driver (see check 2b below): parity alone compares two
// whisper_fit_charge walks, so a shared device/host misclassification would
// cancel out -- the driver's own allocation counter cannot.
//
// Usage: test-whisper-fit-params [model.bin] [vad-model.bin] [gpu]
// CMake registers two forms against the committed for-tests fixtures (see
// tests/CMakeLists.txt): the CPU form under the `unit` label (runs
// model-download-free in CI) and the "gpu" form under the `gpu` label for
// the `ctest -L gpu` lanes. The "gpu" form exits 77 (ctest SKIP) when no
// GPU/IGPU device exists, so it never green-washes a CPU-only machine.
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "whisper.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t actual, const std::string & what) {
    if (projected != actual) {
        fail(what + ": projected " + std::to_string(projected) +
             " != actual " + std::to_string(actual));
    }
}

void log_callback_null(ggml_log_level, const char *, void *) {}

}  // namespace

int main(int argc, char ** argv) {
#ifndef WHISPER_FIT_TEST_MODEL
#define WHISPER_FIT_TEST_MODEL ""
#endif
#ifndef WHISPER_FIT_TEST_VAD_MODEL
#define WHISPER_FIT_TEST_VAD_MODEL ""
#endif
    std::string model_path     = WHISPER_FIT_TEST_MODEL;
    std::string vad_model_path = WHISPER_FIT_TEST_VAD_MODEL;
    bool        use_gpu        = false;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) vad_model_path = argv[2];
    if (argc > 3 && std::strcmp(argv[3], "gpu") == 0) use_gpu = true;

    if (model_path.empty()) {
        std::fprintf(stderr, "usage: %s <model.bin> [vad-model.bin] [gpu]\n", argv[0]);
        return 2;
    }

    whisper_log_set(log_callback_null, nullptr);

    // the gpu form must test a real non-CPU pool or say so: skip (ctest
    // SKIP_RETURN_CODE 77) instead of degrading to a trivial CPU-vs-CPU pass
    ggml_backend_dev_t gpu_dev = nullptr;
    if (use_gpu) {
        ggml_backend_load_all(); // same dynamic-backend discovery whisper-cli performs
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu_dev = dev;
                break;
            }
        }
        if (gpu_dev == nullptr) {
            std::printf("test-whisper-fit-params: gpu requested but no GPU/IGPU device -- skipping\n");
            return 77;
        }
    }

    whisper_fit_options opts = whisper_fit_default_options();
    opts.model_path    = model_path.c_str();
    opts.use_gpu       = use_gpu;
    opts.n_decoders    = 1; // match whisper_fit_actual's post-init resident set
    opts.audio_seconds = 30.0f;

    // 1. A readable model always yields a projection.
    whisper_fit_result fit;
    const int rc = whisper_fit_params(&opts, &fit);
    expect(rc == (int) fit.status, "return value == status (exit-code contract)");
    expect(fit.status != WHISPER_FIT_ERROR,
           std::string("whisper_fit_params returned Error (") + fit.reason + ") for a readable model");
    expect(fit.device.weights_bytes > 0, "projected weights_bytes == 0");
    expect(fit.device.kv_bytes > 0,      "projected kv_bytes == 0");
    expect(fit.device.compute_bytes > 0, "projected compute_bytes == 0");
    expect(fit.device.vad_bytes == 0,    "projected vad_bytes != 0 without a VAD model");
    expect(fit.device.total_bytes >= fit.device.weights_bytes + fit.device.kv_bytes,
           "projected total < weights + kv");
    expect(fit.host_bytes > 0,           "projected host_bytes == 0");
    expect(fit.report[0] != '\0',        "empty report");
    expect(fit.model_type[0] != '\0',    "empty model_type");
    expect(fit.device_total_bytes > 0,   "device_total_bytes == 0");
    if (g_failures) {
        return g_failures; // nothing below is meaningful without a projection
    }
    std::printf("%s", fit.report);

    // 2. Byte parity against a real load + state init.
    whisper_fit_breakdown actual;
    if (whisper_fit_actual(&opts, &actual) != 0) {
        fail("whisper_fit_actual (real load) failed");
        return g_failures;
    }
    expect_eq(fit.device.weights_bytes, actual.weights_bytes, "weights parity");
    expect_eq(fit.device.kv_bytes,      actual.kv_bytes,      "kv parity");
    expect_eq(fit.device.compute_bytes, actual.compute_bytes, "compute parity");
    expect_eq(fit.device.total_bytes,   actual.total_bytes,   "total parity");
    expect_eq(fit.device.host_overflow_bytes, actual.host_overflow_bytes,
              "host-overflow parity");

    // 2b. GPU truth gate: check 2 compares two whisper_fit_charge walks, so a
    //     misclassification shared by projection and measurement (a wrongly
    //     bucketed buffer type) shifts bytes identically on both sides and
    //     parity still passes. Anchor the classifier to a raw figure it never
    //     touches: hold a real GPU context and read the device's own
    //     free-memory counter (ggml_backend_dev_memory, i.e. the driver)
    //     before and after. The allocation delta must roughly equal the bytes
    //     the classifier charged to the device -- a whole component routed to
    //     the wrong pool (weights are ~60% of the tiny total) lands far
    //     outside the slack, while driver-side rounding and per-backend
    //     bookkeeping allocations stay well inside it.
    if (use_gpu && gpu_dev != nullptr) {
        expect(!fit.device_is_cpu, "gpu run must resolve a non-CPU main device");

        size_t free0 = 0, total0 = 0;
        ggml_backend_dev_memory(gpu_dev, &free0, &total0);

        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu    = true;
        cparams.flash_attn = opts.flash_attn;

        whisper_context * rctx  = whisper_init_from_file_with_params_no_state(model_path.c_str(), cparams);
        whisper_state *   rstate = rctx ? whisper_init_state(rctx) : nullptr;
        if (rstate == nullptr) {
            fail("gpu truth: real GPU load failed");
        } else {
            size_t free1 = 0, total1 = 0;
            ggml_backend_dev_memory(gpu_dev, &free1, &total1);
            const uint64_t delta = free0 > free1 ? (uint64_t) (free0 - free1) : 0;

            std::printf("gpu truth: driver reports %llu bytes allocated, classifier charged %llu to the device\n",
                        (unsigned long long) delta, (unsigned long long) actual.total_bytes);

            // asymmetric slack: too-small a delta means device-charged bytes
            // actually live in host RAM (the dangerous direction -- the fitter
            // would under-project VRAM), so it gets the tight bound; the upper
            // bound stays looser because drivers allocate bookkeeping of their
            // own next to our buffers (CUDA pools, cuBLAS workspaces).
            const uint64_t slack_lo = std::max<uint64_t>(actual.total_bytes / 8, 32ull  << 20);
            const uint64_t slack_hi = std::max<uint64_t>(actual.total_bytes / 4, 256ull << 20);
            expect(delta + slack_lo >= actual.total_bytes,
                   "gpu truth: driver saw fewer device bytes than charged -- device-classified buffers live in host RAM");
            expect(delta <= actual.total_bytes + slack_hi,
                   "gpu truth: driver saw far more device bytes than charged -- device buffers classified as host");
        }
        if (rstate) whisper_free_state(rstate);
        if (rctx)   whisper_free(rctx);
    }

    // 3. Worst-case decoders: the (n+2)x KV recreation must be priced in --
    //    and byte-match whisper_fit_actual, which reproduces whisper_full's
    //    recreation (whisper_kv_cache_free + (n+2)x whisper_kv_cache_init)
    //    when asked for n_decoders > 1. compute_bytes is deliberately NOT
    //    byte-compared here: the real schedulers grow only on the first
    //    decode against the larger cache, so the projection may exceed the
    //    post-init measurement (see whisper_fit_actual in whisper.h).
    {
        whisper_fit_options wopts = opts;
        wopts.n_decoders = 5; // whisper_full default best_of/beam_size
        whisper_fit_result worst;
        whisper_fit_params(&wopts, &worst);
        expect(worst.status != WHISPER_FIT_ERROR, "n_decoders=5 projection is Error");
        expect(worst.device.kv_bytes > fit.device.kv_bytes,
               "n_decoders=5 must project a larger self-attention KV cache");
        expect(worst.device.compute_bytes >= fit.device.compute_bytes,
               "n_decoders=5 must never project a smaller decode graph");
        expect(worst.device.weights_bytes == fit.device.weights_bytes,
               "n_decoders must not change the weights projection");

        whisper_fit_breakdown wactual;
        if (whisper_fit_actual(&wopts, &wactual) != 0) {
            fail("whisper_fit_actual (n_decoders=5 real load) failed");
        } else {
            expect_eq(worst.device.kv_bytes, wactual.kv_bytes, "n_decoders=5 kv parity");
            expect_eq(worst.device.weights_bytes, wactual.weights_bytes,
                      "n_decoders=5 weights parity");
        }
    }

    // 4. Longer audio grows only the host side; the device set is fixed.
    {
        whisper_fit_options lopts = opts;
        lopts.audio_seconds = 3600.0f;
        whisper_fit_result longer;
        whisper_fit_params(&lopts, &longer);
        expect(longer.status != WHISPER_FIT_ERROR, "long-audio projection is Error");
        expect(longer.device.total_bytes == fit.device.total_bytes,
               "device projection must not grow with audio_seconds");
        expect(longer.host_bytes > fit.host_bytes,
               "host_bytes must grow with audio_seconds");
    }

    // 5. VAD component, with its own byte-parity gate.
    if (!vad_model_path.empty()) {
        whisper_fit_options vopts = opts;
        vopts.vad_model_path = vad_model_path.c_str();
        whisper_fit_result vfit;
        whisper_fit_params(&vopts, &vfit);
        expect(vfit.status != WHISPER_FIT_ERROR, "VAD projection is Error");
        expect(vfit.device.vad_bytes + vfit.device.host_overflow_bytes >
               fit.device.vad_bytes + fit.device.host_overflow_bytes,
               "VAD projection added no bytes anywhere");
        expect(vfit.device.weights_bytes == fit.device.weights_bytes,
               "VAD must not change the whisper weights projection");

        whisper_fit_breakdown vactual;
        if (whisper_fit_actual(&vopts, &vactual) != 0) {
            fail("whisper_fit_actual with VAD failed");
        } else {
            expect_eq(vfit.device.vad_bytes, vactual.vad_bytes, "vad parity");
            expect_eq(vfit.device.host_overflow_bytes, vactual.host_overflow_bytes,
                      "vad host-overflow parity");
        }

        // an unreadable VAD model must be Error, never a silent no-VAD fit
        whisper_fit_options bopts = opts;
        bopts.vad_model_path = "/nonexistent/vad.bin";
        whisper_fit_result bad;
        expect(whisper_fit_params(&bopts, &bad) == (int) WHISPER_FIT_ERROR &&
               std::strcmp(bad.reason, "vad-model-unreadable") == 0,
               "missing VAD model must be Error/vad-model-unreadable");
    }

    if (g_failures == 0) {
        std::printf("test-whisper-fit-params: all checks passed\n");
    }
    return g_failures;
}
