// Parler DAC on the Core ML sidecar (src/parler/dac_coreml.h).
//
//   test-parler-dac-coreml --self-test
//   test-parler-dac-coreml --gguf MODEL.gguf [--fallback-gguf MODEL.gguf]
//                          [--ref-dir parler-ref] [--min-snr 45] [--min-speech-snr 45]
//                          [--cache-dir DIR [--units-sweep]] [--require-coreml]
//
// --self-test (no model, no Core ML; the ctest unit test
// test-parler-dac-coreml-self-test, so every build runs it), and first in
// every run with --gguf:
//   * the HF-reference waveform gate (relative SNR for speech, bounded
//     absolute error near silence);
//   * the per-chip parity floor table: exact chip match, `all` only, never
//     above --min-snr, expiry, which --units-sweep row is gated;
//   * the memory schedule's attribution of footprint samples to phases, its
//     one-failure-per-charge checks and ANECompilerService's growth.
// Every build with --gguf (fake sidecars, ggml decodes, no Core ML needed):
//   * the host quantizer the sidecar decodes from equals ggml's latent;
//   * a sidecar that fails mid-range (on its second window) is retired once
//     and the WHOLE range is redone on ggml -- bit-identical to a ggml-only
//     decode, nothing of the failed call's earlier windows survives -- and
//     later calls stay on ggml;
//   * PARLER_COREML_STRICT turns the failure into an error (for later calls
//     too), and a missing sidecar into a throw at attach;
//   * empty ranges and out-of-range codes keep their ggml-path behaviour;
//   * the pending seam (model.dac_pending, acquired once per call): a call
//     whose sidecar is not ready runs on ggml, bit-identical, and reports
//     pending; a sidecar that becomes ready between two streamed chunks makes
//     the synthesis "mixed" while each chunk stays on one backend; a failure
//     retires through the seam and the call is redone on ggml; strict turns a
//     sidecar that cannot serve, or fails, into an error.
// TTS_CPP_COREML builds with `<stem>-dac.mlmodelc` next to --gguf:
//   * fit_params refuses the projection (coreml-memory-estimate-unavailable)
//     unless PARLER_COREML_DISABLE is set;
//   * the sidecar attaches, reports a coreml-* label, and decodes whole
//     utterances (several windows plus a partial tail, and one shorter than a
//     window) and arbitrary sub-ranges within --min-snr of the ggml CPU decode;
//   * with --ref-dir holding dump-reference.py's case0 fixtures, case0 against
//     the HF wav (--min-speech-snr, 45 dB, for speech; bounded absolute error
//     for near-silence);
//   * an Engine reports the sidecar (dac_on_coreml, dac_backend, whole and
//     streamed) and PARLER_COREML_DISABLE keeps it on ggml;
//   * --fallback-gguf: a sidecar whose window cannot hold its context is
//     refused at attach (and throws under PARLER_COREML_STRICT).
// TTS_CPP_COREML builds with --cache-dir and no sidecar next to --gguf (the
// generated route, template parler-dac-w128, macOS 15 or later):
//   * an Engine with the default wait (none) generates the DAC sidecar from
//     the GGUF into the (empty) cache in the background; Engine::cancel()
//     ends a coreml_wait(-1) early, and the next coreml_wait(-1) waits until
//     it is compiled and loaded (route generated, ready); a cancel() during
//     a synthesis survives a coreml_wait(0) and a coreml_wait(10) on another
//     thread; later owners of the same GGUF adopt it (route cached) without
//     generating again;
//   * the same --min-snr / --min-speech-snr gates as above on the generated
//     sidecar (lowered under `all` only for the chips with a measured floor,
//     see GENERATED_ALL_UNITS_FLOORS; a floor that is no longer needed
//     fails), and the Engine reports it per call;
//   * fit charges the generated envelope (route generate, validated); a
//     budget only the ggml projection fits resolves use = off (fit and Engine);
//   * --units-sweep: the generated artifact's SNR against ggml for each
//     compute-unit policy, for recording per-host parity figures; reported
//     only, except that a chip with a floor under `all` holds cpu_and_ane
//     to --min-snr.
// --memory-schedule DIR (TTS_CPP_COREML, generated route): ggml decodes run
// while the sidecar is generated, compiled, loaded and warmed into DIR; the
// footprint of each phase (prepare, load and warm-up, steady state, the whole
// run) and ANECompilerService must stay within the template constant that
// charges it (the memory validation behind template.json memory; see
// memory_schedule_gates). Use a fresh DIR: an existing artifact is adopted.
// On its own (no --cache-dir) it satisfies --require-coreml.
// Exit 77 (skip) without --gguf; --require-coreml fails instead of skipping
// the Core ML gates (CI).

#include "tts-cpp/parler/engine.h"
#include "tts-cpp/parler/fit.h"
#include "parler/internal.h"
#ifdef TTS_CPP_USE_COREML
#include "parler/dac_source.h"
#endif
#include "gguf.h"
#include "coreml_sidecar_path.h"
#include "memory_probe.h"
#include "npy.h"
#include "../test_env_portable.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>

#if defined(__APPLE__)
#include <fcntl.h>
#include <sys/clonefile.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/sysctl.h>
#endif
#include <memory>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

// Fills every window with a constant, or fails on call `fail_on` (1-based).
class fake_sidecar final : public parler_dac_sidecar {
public:
    fake_sidecar(int window, int hop, int fail_on) : window_(window), hop_(hop), fail_on_(fail_on) {}
    int window() const override { return window_; }
    const char * label() const override { return "coreml-fake"; }
    bool decode(const float *, const float * valid, float * pcm) override {
        ++calls;
        if (calls == fail_on_) return false;
        for (int t = 0; t < window_; ++t) {
            if (valid[t] != 0.0f && valid[t] != 1.0f) return false;
        }
        std::fill(pcm, pcm + (size_t) window_ * hop_, 0.25f);
        return true;
    }
    int calls = 0;

private:
    int window_, hop_, fail_on_;
};

double snr_db(const std::vector<float> & got, const std::vector<float> & ref, double & max_abs) {
    double sig = 0, noise = 0;
    max_abs = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        sig += (double) ref[i] * ref[i];
        noise += d * d;
        max_abs = std::max(max_abs, std::fabs(d));
    }
    if (!std::isfinite(noise)) return NAN;
    return noise > 0 ? 10.0 * std::log10(sig / noise) : INFINITY;
}

// Relative SNR is ill-conditioned near silence: the Indic Hindi greedy
// fixture is mostly DC at -78.5 dBFS. Keep speech on the relative gate and
// bound BOTH RMS (-70 dBFS) and peak (-60 dBFS) error on quiet HF fixtures.
// Random-code parity always uses the requested relative gate.
struct waveform_quality {
    double snr = NAN;
    double peak = 0;
    double ref_rms = 0;
    double error_rms = 0;
    bool quiet = false;
    bool pass = false;
};

waveform_quality hf_quality(const std::vector<float> & pcm, const std::vector<float> & ref,
                            double min_snr) {
    waveform_quality q;
    if (pcm.empty() || pcm.size() != ref.size()) return q;
    double signal = 0, error = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(pcm[i]) || !std::isfinite(ref[i])) return q;
        const double d = (double) pcm[i] - ref[i];
        signal += (double) ref[i] * ref[i];
        error += d * d;
        q.peak = std::max(q.peak, std::fabs(d));
    }
    q.ref_rms = std::sqrt(signal / ref.size());
    q.error_rms = std::sqrt(error / ref.size());
    q.snr = error > 0 ? 10.0 * std::log10(signal / error) : INFINITY;
    q.quiet = q.ref_rms < 1e-3; // below -60 dBFS, based only on the reference
    q.pass = q.quiet ? (q.error_rms <= std::pow(10.0, -70.0 / 20) && q.peak <= 1e-3)
                     : q.snr >= min_snr;
    return q;
}

void waveform_quality_gates() {
    const std::vector<float> silence(100, 0.0f), speech(100, 0.1f);
    check(hf_quality(silence, silence, 45).pass, "quiet gate: exact silence");
    check(hf_quality(std::vector<float>(100, 1.5e-4f), silence, 45).pass,
          "quiet gate: bounded fp16 floor");
    check(!hf_quality(std::vector<float>(100, 4e-4f), silence, 45).pass,
          "quiet gate: excessive RMS rejected");
    auto spike = silence;
    spike[0] = 2e-3f;
    check(!hf_quality(spike, silence, 45).pass, "quiet gate: isolated peak rejected");
    check(!hf_quality(std::vector<float>(100, 0.101f), speech, 45).pass,
          "speech gate: relative SNR required");
    check(hf_quality(std::vector<float>(100, 0.1001f), speech, 45).pass,
          "speech gate: small error passes");
    const std::vector<float> below(100, 0.000999f), above(100, 0.001001f);
    check(hf_quality(below, below, 45).quiet && !hf_quality(above, above, 45).quiet,
          "quiet gate: reference RMS selects the branch at -60 dBFS");
    auto transient = silence;
    transient[0] = 0.0079f; // low RMS, but the quiet gate must preserve this peak
    check(!hf_quality(silence, transient, 45).pass,
          "quiet gate: erasing a quiet reference transient fails");
    check(!hf_quality({}, silence, 45).pass, "quiet gate: wrong length rejected");
    spike[0] = NAN;
    check(!hf_quality(spike, silence, 45).pass && !hf_quality(silence, spike, 45).pass,
          "quiet gate: nonfinite PCM and reference rejected");
}

// Measured parity floors of the generated sidecar under the template's
// default compute units (`all`, label coreml-all), per chip. Core ML
// partitions the program between the GPU and the Neural Engine with its own
// per-chip cost model, and the partition sets the accuracy:
//   * Apple M2 (macOS 15.7.3 and 27.2.1): every op runs on the Neural Engine
//     (`all` equals `cpu_and_ane`), 56.2 / 56.4 / 53.3 dB at 265 / 42 / 1
//     frames, so --min-snr 54 holds.
//   * Apple M4 Max (macOS 26.7.2, the self-hosted CI runner): MLComputePlan
//     puts 443 of the 493 ops (89.5% of the estimated cost) on the GPU and
//     leaves eight folded-stage convolutions and the output stage (conv_out,
//     tanh) on the Neural Engine. In that GPU-fed island the compiler fuses
//     tanh into conv_out as its activation, which measures 51.7 dB from exact
//     inputs (62.8 dB for conv_out alone), so `all` measures 51.3 / 51.4 /
//     51.8 dB (`cpu_and_ane` 54.8 / 55.0 / 54.2, `cpu_and_gpu` 68.3 / 69.2 /
//     67.7). The spellings that keep the island from fusing (tanh after the
//     output unfold, or the output stage in fp32) cost the M2's
//     all-Neural-Engine program 4 dB (52.3 dB), so the template keeps `all`
//     and the M4 Max is held to 50 dB.
// A chip that is not listed is held to --min-snr.
struct parity_floor {
    const char * chip; // sysctl machdep.cpu.brand_string
    double min_snr;
};
constexpr parity_floor GENERATED_ALL_UNITS_FLOORS[] = {{"Apple M4 Max", 50.0}};
// Each row is a measurement recorded above; a new chip needs its own.
static_assert(std::size(GENERATED_ALL_UNITS_FLOORS) == 1, "every floor row is a recorded measurement");

std::string host_chip() {
#if defined(__APPLE__)
    char brand[128] = {};
    size_t size = sizeof brand - 1;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) return brand;
#endif
    return {};
}

// --min-snr, or the lower floor recorded above for `chip` when the sidecar
// runs under `all`. It never raises --min-snr.
double generated_min_snr(const std::string & label, const std::string & chip, double min_snr) {
    if (label != "coreml-all") return min_snr;
    for (const parity_floor & floor : GENERATED_ALL_UNITS_FLOORS) {
        if (chip == floor.chip) return std::min(min_snr, floor.min_snr);
    }
    return min_snr;
}

// A one-frame decode is 512 samples of mostly masked context, so its fp16 SNR
// is ill-conditioned. On the M2 the generated sidecar under `all` (all on the
// Neural Engine; the generated weight.bin is byte-identical to the
// exporter's) measures 53.3 dB there on macOS 15.7.3 and 27.2.1, against
// 56.2 / 56.4 dB at 265 / 42 frames. That chip and placement alone hold the
// one-frame case to 53 dB; every other chip, placement, the prebuilt sidecars
// and the units sweep hold it to the multi-frame gate (the M4 Max measures
// 51.8 dB under `all`, within its 50 dB floor, and 54.2 dB under
// `cpu_and_ane`).
struct one_frame_floor {
    const char * chip;  // sysctl machdep.cpu.brand_string
    const char * label; // the sidecar's backend label
    double min_snr;
};
constexpr one_frame_floor GENERATED_ONE_FRAME_FLOORS[] = {{"Apple M2", "coreml-all", 53.0}};
static_assert(std::size(GENERATED_ONE_FRAME_FLOORS) == 1, "every one-frame row is a recorded measurement");

// The gate of the generated sidecar's parity case of `n_frames`: the chip's
// floor under `all`, lowered for one frame only where a row above records it.
double generated_gate(int n_frames, const std::string & label, const std::string & chip, double min_snr) {
    const double gate = generated_min_snr(label, chip, min_snr);
    if (n_frames != 1) return gate;
    for (const one_frame_floor & f : GENERATED_ONE_FRAME_FLOORS) {
        if (chip == f.chip && label == f.label) return std::min(gate, f.min_snr);
    }
    return gate;
}

// A floor expires once its chip no longer needs it: true when a floor
// applies (host_min_snr below --min-snr) and every multi-frame case reaches
// --min-snr, so the row must be removed.
bool floor_expired(double host_min_snr, double min_snr, const std::vector<double> & multi_frame_snrs) {
    if (!(host_min_snr < min_snr) || multi_frame_snrs.empty()) return false;
    return std::all_of(multi_frame_snrs.begin(), multi_frame_snrs.end(),
                       [min_snr](double snr) { return snr >= min_snr; });
}

// On a chip with a floor under `all`, --units-sweep holds the cpu_and_ane
// row to --min-snr, so the run keeps one full-gate check of the
// all-Neural-Engine program. Every other row is reported only.
bool sweep_row_gated(const std::string & units, const std::string & chip, double min_snr) {
    return units == "cpu_and_ane" && generated_min_snr("coreml-all", chip, min_snr) < min_snr;
}

// One parity case of a --units-sweep row holds when both decodes ran, the
// reference on ggml and the candidate on the sidecar under test (a fallback
// to ggml would compare ggml with itself and report an infinite SNR), and
// the SNR reaches the gate.
bool sweep_case_holds(bool ref_ok, const std::string & ref_backend, bool ok, const std::string & backend,
                      const std::string & label, double snr, double gate) {
    return ref_ok && ref_backend == PARLER_DAC_GGML_BACKEND && ok && backend == label && snr >= gate;
}

// The backend label a sidecar opened for `units` must report.
std::string label_for_units(const std::string & units) {
    if (units == "cpu_and_ane") return "coreml-ane";
    if (units == "cpu_and_gpu") return "coreml-gpu";
    if (units == "cpu_only") return "coreml-cpu";
    return "coreml-all";
}

// A chip with a gated row needs the sweep: a sweep that cannot find, link or
// load the artifact fails there instead of reporting nothing.
bool sweep_required(const std::string & chip, double min_snr) {
    return sweep_row_gated("cpu_and_ane", chip, min_snr);
}

void parity_floor_gates() {
    check(generated_min_snr("coreml-all", "Apple M4 Max", 54) == 50, "parity floor: M4 Max under all is 50 dB");
    check(generated_min_snr("coreml-all", "Apple M2", 54) == 54, "parity floor: an unlisted chip keeps --min-snr");
    check(generated_min_snr("coreml-ane", "Apple M4 Max", 54) == 54, "parity floor: only under all");
    check(generated_min_snr("coreml-all", "Apple M4 Max", 45) == 45, "parity floor: never raises --min-snr");
    for (const char * chip : {"Apple M4 Max (Virtual)", "Apple M4 Pro", "Apple M4", ""}) {
        check(generated_min_snr("coreml-all", chip, 54) == 54,
              std::string("parity floor: '") + chip + "' keeps --min-snr (exact chip match)");
    }
    for (const parity_floor & floor : GENERATED_ALL_UNITS_FLOORS) {
        check(floor.min_snr < 54, std::string("parity floor: the ") + floor.chip + " row is below 54 dB");
    }
    check(generated_gate(1, "coreml-all", "Apple M2", 54) == 53 && generated_gate(42, "coreml-all", "Apple M2", 54) == 54,
          "one-frame floor: the M2's generated sidecar under all holds one frame to 53 dB, longer cases to --min-snr");
    check(generated_gate(1, "coreml-ane", "Apple M2", 54) == 54 && generated_gate(1, "coreml-all", "Apple M4 Max", 54) == 50 &&
              generated_gate(1, "coreml-ane", "Apple M4 Max", 54) == 54 && generated_gate(1, "coreml-all", "", 54) == 54 &&
              generated_gate(1, "coreml-all", "Apple M2 Pro", 54) == 54,
          "one-frame floor: no other chip or placement lowers the one-frame gate");
    check(generated_gate(1, "coreml-all", "Apple M2", 45) == 45, "one-frame floor: never raises --min-snr");
    check(floor_expired(50, 54, {54.0, 56.4}), "parity floor: expires when every multi-frame case reaches --min-snr");
    check(!floor_expired(50, 54, {51.3, 56.4}) && !floor_expired(50, 54, {NAN, 56.4}),
          "parity floor: still needed while a multi-frame case is below --min-snr");
    check(!floor_expired(54, 54, {56.2, 56.4}) && !floor_expired(50, 54, {}),
          "parity floor: nothing expires without a floor or a case");
    check(sweep_case_holds(true, "ggml", true, "coreml-ane", "coreml-ane", 54.2, 54),
          "units sweep: a case decoded by the sidecar at the gate holds");
    check(!sweep_case_holds(true, "ggml", true, "ggml", "coreml-ane", INFINITY, 54) &&
              !sweep_case_holds(true, "ggml", true, "coreml-all", "coreml-ane", 60, 54) &&
              !sweep_case_holds(false, "ggml", true, "coreml-ane", "coreml-ane", 60, 54) &&
              !sweep_case_holds(true, "coreml-ane", true, "coreml-ane", "coreml-ane", INFINITY, 54) &&
              !sweep_case_holds(true, "ggml", false, "coreml-ane", "coreml-ane", 60, 54) &&
              !sweep_case_holds(true, "ggml", true, "coreml-ane", "coreml-ane", 53.9, 54),
          "units sweep: a fallback to ggml, another sidecar, a failed decode or a low SNR fails the case");
    check(label_for_units("cpu_and_ane") == "coreml-ane" && label_for_units("cpu_and_gpu") == "coreml-gpu" &&
              label_for_units("cpu_only") == "coreml-cpu" && label_for_units("all") == "coreml-all",
          "units sweep: the expected label comes from the requested units");
    check(!sweep_case_holds(true, "ggml", true, "coreml-gpu", label_for_units("cpu_and_ane"), 68.3, 54),
          "units sweep: a cpu_and_ane row decoded by a GPU sidecar fails although its SNR is high");
    check(sweep_required("Apple M4 Max", 54) && !sweep_required("Apple M2", 54) && !sweep_required("Apple M4 Max", 45),
          "units sweep: required where a row is gated");
    check(sweep_row_gated("cpu_and_ane", "Apple M4 Max", 54) && !sweep_row_gated("all", "Apple M4 Max", 54) &&
              !sweep_row_gated("cpu_and_gpu", "Apple M4 Max", 54) &&
              !sweep_row_gated("cpu_and_ane", "Apple M2", 54) &&
              !sweep_row_gated("cpu_and_ane", "Apple M4 Max", 45),
          "units sweep: only the cpu_and_ane row on a chip with a floor is gated");
}

// The memory schedule (--memory-schedule) samples this process's footprint
// while the generated sidecar is prepared, loaded and warmed in the
// background, and attributes each sample to the DAC slot's state when it was
// taken. Each phase needs 1.25 x its measurement (MEASURED_MARGIN) within the
// template constant that charges it:
//   prepare     digesting, generating and compiling: the peak over the
//               footprint just before generation began, against
//               generation_peak_bytes + 8 MiB + compile_peak_bytes;
//   load+warm   loading and the warm-up prediction: the peak over the
//               footprint just before the load began, against
//               load_peak_bytes for the resolved compute units;
//   steady      ready, Core ML decodes running: the peak over the footprint
//               just before the load began, against the template's resident
//               part (weights, 2 x I/O, workspace_bytes; the host quantizer
//               copy predates the load, so that footprint holds it already,
//               see charges_of). Generation memory the allocator still holds when the
//               load begins is the prepare transient's, which the envelope
//               charges next to the resident; the whole run checks the sum;
//   whole run   the peak over the baseline, against the envelope fit charges;
//   compiler    ANECompilerService's size while loading and warming (its
//               peak over every instance), against ane_compile_reserve_bytes
//               (when the units include the Neural Engine): no run could
//               start the service clean, and its growth over a service that
//               already holds memory understates a cold compile, so the
//               reserve charges its absolute size. The report also prints its
//               growth (the peak less its size just before the first load;
//               see qvac_coreml_memory::ane_compiler_growth) and labels that
//               baseline clean, dirty or unknown (no sample before the load).
// generate, compile, load and warm are also reported on their own, each over
// the footprint just before that state began.
// `digest` is charged with the generation (one prepare charge) but is not
// evidence that the schedule sampled the generation itself.
enum class mem_phase { ggml, digest, generate, compile, load, warm, ready };

mem_phase mem_phase_of(const std::string & state) {
    if (state == "digesting") return mem_phase::digest;
    if (state == "generating") return mem_phase::generate;
    if (state == "compiling") return mem_phase::compile;
    if (state == "loading") return mem_phase::load;
    if (state == "warming") return mem_phase::warm;
    if (state == "ready") return mem_phase::ready;
    return mem_phase::ggml;  // idle, queued, deferred(memory): nothing of the sidecar's yet
}

struct mem_sample {
    uint64_t footprint = 0;
    mem_phase phase = mem_phase::ggml;
};

// Increments in bytes; the flags say which states were sampled.
struct mem_breakdown {
    uint64_t prepare = 0, generate = 0, compile = 0, load = 0, warm = 0, load_warm = 0, steady = 0, total = 0;
    uint64_t steady_over_base = 0;  // reported only
    bool generated = false, compiled = false, loaded = false, warmed = false, ready = false;
};

mem_breakdown mem_breakdown_of(const std::vector<mem_sample> & samples, uint64_t base) {
    // 1: prepare (digest, generate, compile); 2: load and warm-up; 0: the rest.
    const auto group = [](mem_phase p) {
        if (p == mem_phase::digest || p == mem_phase::generate || p == mem_phase::compile) return 1;
        if (p == mem_phase::load || p == mem_phase::warm) return 2;
        return 0;
    };
    const auto over = [](uint64_t v, uint64_t from) { return v > from ? v - from : (uint64_t) 0; };
    mem_breakdown b;
    uint64_t prev = base, phase_start = base, group_start = base;
    // The footprint just before the first load: the steady state is measured
    // from there, so generation memory the allocator still holds when the
    // load begins counts as the prepare's, not as resident.
    std::optional<uint64_t> load_start;
    mem_phase last = mem_phase::ggml;
    // The digest and the generation share one increment (from the
    // footprint before the digest), as one state did before they were told
    // apart; only a generating sample marks the generation sampled.
    const auto merged = [](mem_phase p) { return p == mem_phase::digest ? mem_phase::generate : p; };
    for (const mem_sample & s : samples) {
        if (merged(s.phase) != merged(last)) phase_start = prev;
        if (group(s.phase) != group(last)) group_start = prev;
        if (group(s.phase) == 2 && !load_start) load_start = group_start;
        const uint64_t inc = over(s.footprint, phase_start);
        switch (s.phase) {
            case mem_phase::digest: b.generate = std::max(b.generate, inc); break;
            case mem_phase::generate: b.generated = true; b.generate = std::max(b.generate, inc); break;
            case mem_phase::compile: b.compiled = true; b.compile = std::max(b.compile, inc); break;
            case mem_phase::load: b.loaded = true; b.load = std::max(b.load, inc); break;
            case mem_phase::warm: b.warmed = true; b.warm = std::max(b.warm, inc); break;
            case mem_phase::ready:
                b.ready = true;
                b.steady = std::max(b.steady, over(s.footprint, load_start.value_or(base)));
                b.steady_over_base = std::max(b.steady_over_base, over(s.footprint, base));
                break;
            case mem_phase::ggml: break;
        }
        if (group(s.phase) == 1) b.prepare = std::max(b.prepare, over(s.footprint, group_start));
        if (group(s.phase) == 2) b.load_warm = std::max(b.load_warm, over(s.footprint, group_start));
        b.total = std::max(b.total, over(s.footprint, base));
        prev = s.footprint;
        last = s.phase;
    }
    return b;
}

// The template's charges for one run and the envelope.
struct mem_charges {
    uint64_t prepare = 0, load = 0, resident = 0, ane_reserve = 0, envelope = 0;
};

// The template's memory constants are 1.25 x the largest value measured for
// each phase over every validated host and run (template.json
// memory.measured.margin), so a run passes only when 1.25 x what it measured
// stays within the constant. The whole run is held to the envelope as
// measured: the envelope adds phases that each carry the margin.
constexpr double MEASURED_MARGIN = 1.25;

// The charges of a run of the generated route `r`. The resident charge for
// the steady state is the template's alone: the host quantizer copy
// (`adapter_bytes`, inside r.resident_bytes) is built at construction, before
// the load, so the footprint the steady state is measured from already holds
// it. The envelope keeps it, since the whole run starts before construction.
mem_charges charges_of(const parler_dac_route & r, uint64_t adapter_bytes) {
    return {r.prepare_peak_bytes, r.load_peak_bytes,
            r.resident_bytes > adapter_bytes ? r.resident_bytes - adapter_bytes : 0, r.ane_compile_reserve_bytes,
            r.envelope_bytes};
}

// One line per charge the run exceeds (with `margin` applied to every phase
// but the whole run), or per state it never sampled.
std::vector<std::string> mem_failures(const mem_breakdown & b, uint64_t ane_compiler_load_peak, const mem_charges & c,
                                      double margin = MEASURED_MARGIN) {
    std::vector<std::string> out;
    const std::pair<bool, const char *> states[] = {{b.generated, "generating"}, {b.compiled, "compiling"},
                                                    {b.loaded, "loading"},       {b.warmed, "warming"},
                                                    {b.ready, "ready"}};
    for (const auto & [seen, state] : states) {
        if (!seen) out.push_back(std::string("the schedule never sampled the slot ") + state);
    }
    const auto over = [&out](const char * what, uint64_t got, uint64_t charged, double m) {
        const uint64_t need = (uint64_t) std::ceil((double) got * m);
        if (need <= charged) return;
        char line[256];
        std::snprintf(line, sizeof(line), "%s: %llu bytes measured, x%.2f = %llu over the charged %llu", what,
                      (unsigned long long) got, m, (unsigned long long) need, (unsigned long long) charged);
        out.push_back(line);
    };
    over("prepare", b.prepare, c.prepare, margin);
    over("load and warm-up", b.load_warm, c.load, margin);
    over("steady state (resident)", b.steady, c.resident, margin);
    over("whole run (envelope)", b.total, c.envelope, 1.0);
    if (c.ane_reserve > 0) {
        over("ANECompilerService size while loading (reserve)", ane_compiler_load_peak, c.ane_reserve, margin);
    }
    return out;
}

// The label of an ANECompilerService query bracketed by two slot-state
// reads: while loading when both reads are the load or the warm-up, before
// or after it when neither is, and no label (the sample is dropped) when the
// query crossed the start or the end of the load, so a size read before the
// load never stands for one during it, nor the reverse.
std::optional<bool> ane_query_loading(mem_phase before, mem_phase after) {
    const auto loading = [](mem_phase p) { return p == mem_phase::load || p == mem_phase::warm; };
    if (loading(before) != loading(after)) return std::nullopt;
    return loading(before);
}

// One decode of the overlapped schedule: it counts as a ggml call or a Core
// ML call only when it succeeded on that backend (the Core ML one being the
// route's own label); a failed decode, or one on any other backend, is bad
// and fails the run instead of filling the call counts.
enum class schedule_call { ggml, coreml, bad };

schedule_call classify_call(bool ok, const std::string & backend, const std::string & coreml_label) {
    if (!ok) return schedule_call::bad;
    if (backend == PARLER_DAC_GGML_BACKEND) return schedule_call::ggml;
    if (backend == coreml_label) return schedule_call::coreml;
    return schedule_call::bad;
}

// What the probes failed to measure: the baseline, footprint samples, or (on
// the Neural Engine) ANECompilerService while loading. A run with any of them
// fails instead of counting the missing readings as 0.
std::vector<std::string> probe_failures(bool base_ok, size_t footprint_failures, bool ane_units,
                                        const qvac_coreml_memory::AneGrowth & ane) {
    std::vector<std::string> out;
    if (!base_ok) out.push_back("no footprint baseline (the probe failed)");
    if (footprint_failures > 0) out.push_back(std::to_string(footprint_failures) + " footprint probe(s) failed");
    if (ane_units && !qvac_coreml_memory::ane_compiler_measured(ane)) {
        out.push_back("no successful ANECompilerService query while loading or warming");
    }
    if (ane_units && ane.failed_samples > 0) {
        out.push_back(std::to_string(ane.failed_samples) + " ANECompilerService query(ies) failed");
    }
    return out;
}

void memory_schedule_gates() {
    check(mem_phase_of("digesting") == mem_phase::digest && mem_phase_of("generating") == mem_phase::generate &&
              mem_phase_of("compiling") == mem_phase::compile && mem_phase_of("loading") == mem_phase::load &&
              mem_phase_of("warming") == mem_phase::warm && mem_phase_of("ready") == mem_phase::ready,
          "memory schedule: slot states map to phases");
    check(mem_phase_of("idle") == mem_phase::ggml && mem_phase_of("queued") == mem_phase::ggml &&
              mem_phase_of("deferred(memory)") == mem_phase::ggml && mem_phase_of("") == mem_phase::ggml,
          "memory schedule: idle, queued and deferred are ggml-only");
    using P = mem_phase;
    const std::vector<mem_sample> run = {
        {100, P::ggml},    {120, P::generate}, {160, P::generate}, {140, P::compile}, {170, P::compile},
        {102, P::ggml},    {105, P::load},     {180, P::load},     {200, P::warm},    {190, P::ready},
        {195, P::ready},   {150, P::ready},
    };
    const mem_breakdown b = mem_breakdown_of(run, 100);
    check(b.generate == 60 && b.compile == 10 && b.prepare == 70, "memory schedule: prepare = generate start to compile peak");
    check(b.load == 78 && b.warm == 20 && b.load_warm == 98, "memory schedule: load+warm = load start to warm-up peak");
    check(b.steady == 93 && b.steady_over_base == 95 && b.total == 100,
          "memory schedule: steady state over the load start, whole run over the baseline");
    check(b.generated && b.compiled && b.loaded && b.warmed && b.ready, "memory schedule: every state sampled");
    const mem_charges exact{70, 98, 93, 400, 100};
    check(mem_failures(b, 400, exact, 1.0).empty(), "memory schedule: a run at every charge passes (no margin)");
    const struct {
        mem_charges c;
        uint64_t ane;
        const char * stage;
    } over[] = {
        {{69, 98, 93, 400, 100}, 400, "prepare"},          {{70, 97, 93, 400, 100}, 400, "load and warm-up"},
        {{70, 98, 92, 400, 100}, 400, "steady state"},     {{70, 98, 93, 400, 99}, 400, "whole run"},
        {{70, 98, 93, 400, 100}, 401, "ANECompilerService"},
    };
    for (const auto & o : over) {
        const std::vector<std::string> f = mem_failures(b, o.ane, o.c, 1.0);
        check(f.size() == 1 && f[0].find(o.stage) != std::string::npos,
              std::string("memory schedule: one byte over the ") + o.stage + " charge fails");
    }
    check(mem_failures(b, 999, {70, 98, 93, 0, 100}, 1.0).empty(),
          "memory schedule: no compiler check without the Neural Engine");
    {
        // The steady state is charged against the template's resident part
        // alone: 327.5 MiB needs 409.375, over the 409 MiB charged, although
        // the route's resident figure with the quantizer copy would allow it.
        constexpr uint64_t MB = 1ull << 20;
        parler_dac_route r;
        r.resident_bytes = 409 * MB + 626688;
        r.envelope_bytes = 2000 * MB;
        r.prepare_peak_bytes = 553 * MB;
        r.load_peak_bytes = 409 * MB;
        const mem_charges ch = charges_of(r, 626688);
        check(ch.resident == 409 * MB && ch.envelope == r.envelope_bytes,
              "memory schedule: the steady-state charge leaves out the quantizer copy, the envelope keeps it");
        mem_breakdown steady327{};
        steady327.generated = steady327.compiled = steady327.loaded = steady327.warmed = steady327.ready = true;
        steady327.steady = (uint64_t) (327.5 * MB);
        const std::vector<std::string> f = mem_failures(steady327, 0, ch);
        check(f.size() == 1 && f[0].find("steady state") != std::string::npos,
              "memory schedule: 327.5 MiB of steady state fails a 409 MiB resident charge");
    }
    // With the template's 1.25 margin each phase needs 1.25 x its measurement
    // (rounded up); the whole run is held to the envelope as measured.
    const mem_charges margined{88, 123, 117, 500, 100};  // 87.5, 122.5, 116.25, 500 rounded up
    check(mem_failures(b, 400, margined).empty(), "memory schedule: a run at 1.25 x every phase passes");
    check(mem_failures(b, 400, exact).size() == 4,
          "memory schedule: a run that reaches a charge without the 1.25 margin fails each margined phase");
    const struct {
        mem_charges c;
        uint64_t ane;
        const char * stage;
    } short_of_margin[] = {
        {{87, 123, 117, 500, 100}, 400, "prepare"},      {{88, 122, 117, 500, 100}, 400, "load and warm-up"},
        {{88, 123, 116, 500, 100}, 400, "steady state"}, {{88, 123, 117, 499, 100}, 400, "ANECompilerService"},
    };
    for (const auto & o : short_of_margin) {
        const std::vector<std::string> f = mem_failures(b, o.ane, o.c);
        check(f.size() == 1 && f[0].find(o.stage) != std::string::npos,
              std::string("memory schedule: one byte short of 1.25 x the ") + o.stage + " fails");
    }
    // Generation memory the allocator still holds when the load begins (the
    // M4 Max runner's 3.5 s schedule) belongs to the prepare transient; the
    // steady state charges only what the load and warm-up leave.
    const std::vector<mem_sample> held = {{100, P::ggml},     {500, P::generate}, {510, P::compile},
                                          {515, P::load},     {800, P::warm},     {810, P::ready}};
    const mem_breakdown h = mem_breakdown_of(held, 100);
    check(h.prepare == 410 && h.load_warm == 290 && h.steady == 300 && h.steady_over_base == 710 && h.total == 710,
          "memory schedule: generation memory still held at the load is the prepare's, not the resident's");
    using qvac_coreml_memory::AneSample;
    const auto growth = [](const std::vector<AneSample> & s) { return qvac_coreml_memory::ane_compiler_growth(s); };
    const auto g = growth({{700, false}, {650, false}, {900, true}, {1000, true}, {1200, false}});
    check(g.before == 650 && g.peak == 1000 && g.growth == 350,
          "memory schedule: ANECompilerService growth = peak while loading less the last size before the first load");
    check(growth({{900, true}, {950, true}}).growth == 950,
          "memory schedule: ANECompilerService growth without a sample before the load is its peak");
    check(growth({{800, false}, {700, true}}).growth == 0 && growth({}).growth == 0,
          "memory schedule: ANECompilerService shrinking or never sampled grows 0");
    {
        const auto f = growth({{700, false, true}, {0, false, false}, {900, true, true}, {0, true, false}});
        check(f.before == 700 && f.peak == 900 && f.growth == 200 && f.loading_samples == 1 && f.failed_samples == 2 &&
                  qvac_coreml_memory::ane_compiler_measured(f),
              "memory schedule: a failed ANECompilerService query is skipped, not read as 0");
        const auto none = growth({{700, false, true}, {0, true, false}, {0, true, false}});
        check(!qvac_coreml_memory::ane_compiler_measured(none) && !qvac_coreml_memory::ane_compiler_measured(growth({})),
              "memory schedule: no successful query while loading is no compiler measurement");
        check(classify_call(true, "ggml", "coreml-all") == schedule_call::ggml &&
              classify_call(true, "coreml-all", "coreml-all") == schedule_call::coreml &&
              classify_call(false, "ggml", "coreml-all") == schedule_call::bad &&
              classify_call(false, "coreml-all", "coreml-all") == schedule_call::bad &&
              classify_call(true, "coreml-gpu", "coreml-all") == schedule_call::bad &&
              classify_call(true, "mixed", "coreml-all") == schedule_call::bad,
          "memory schedule: a failed decode, or one on another backend, is not a ggml or Core ML call");
    check(probe_failures(true, 0, true, f).size() == 1 && probe_failures(true, 0, false, f).empty() &&
                  probe_failures(true, 0, true, growth({{700, false}, {900, true}})).empty(),
              "memory schedule: failed compiler queries fail a Neural Engine run");
        check(probe_failures(false, 0, false, none).size() == 1 && probe_failures(true, 3, false, none).size() == 1 &&
                  probe_failures(true, 0, true, none).size() == 2,
              "memory schedule: a failed baseline, failed footprint samples or an unmeasured compiler fail the run");
    }
    const uint64_t clean = qvac_coreml_memory::ANE_COMPILER_CLEAN_BYTES;
    using qvac_coreml_memory::AneBaseline;
    const auto baseline = [&](const std::vector<AneSample> & s) {
        return qvac_coreml_memory::ane_compiler_baseline(growth(s));
    };
    check(qvac_coreml_memory::ane_compiler_clean_start(growth({{clean, false}, {clean + 500, true}})) &&
              !qvac_coreml_memory::ane_compiler_clean_start(growth({{clean + 1, false}, {clean + 500, true}})) &&
              baseline({{clean + 1, false}, {clean + 500, true}}) == AneBaseline::dirty &&
              baseline({{0, false}, {500, true}}) == AneBaseline::clean,
          "memory schedule: a baseline above 32 MiB before the load is reported dirty (its growth understates a clean "
          "start)");
    check(baseline({{900, true}}) == AneBaseline::unknown && baseline({}) == AneBaseline::unknown &&
              !qvac_coreml_memory::ane_compiler_clean_start(growth({{900, true}})) &&
              std::string(qvac_coreml_memory::ane_baseline_name(AneBaseline::unknown)) == "unknown",
          "memory schedule: no sample before the load is reported unknown, not clean");
    {
        // A digest sampled but no generation: the generation was not covered.
        const std::vector<mem_sample> no_gen = {{100, P::ggml}, {120, P::digest}, {130, P::compile}, {150, P::load},
                                                {160, P::warm}, {170, P::ready}};
        const mem_breakdown nb = mem_breakdown_of(no_gen, 100);
        const std::vector<std::string> fg = mem_failures(nb, 0, {1000, 1000, 1000, 0, 1000});
        check(!nb.generated && nb.prepare == 30 && fg.size() == 1 && fg[0].find("generating") != std::string::npos,
              "memory schedule: a digest sample is not a generation sample (a missed generation fails)");
        const mem_breakdown both = mem_breakdown_of({{100, P::ggml}, {120, P::digest}, {150, P::generate}}, 100);
        check(both.generated && both.generate == 50 && both.prepare == 50,
              "memory schedule: digest and generation share one increment");
        check(ane_query_loading(P::load, P::load) == std::optional<bool>(true) &&
                  ane_query_loading(P::warm, P::load) == std::optional<bool>(true) &&
                  ane_query_loading(P::compile, P::compile) == std::optional<bool>(false) &&
                  !ane_query_loading(P::compile, P::load).has_value() &&
                  !ane_query_loading(P::warm, P::ready).has_value(),
              "memory schedule: a compiler query that crosses the load's start or end has no label");
    }
    const std::vector<mem_sample> no_warm = {{100, P::ggml}, {120, P::generate}, {130, P::compile}, {150, P::load},
                                             {160, P::ready}};
    const std::vector<std::string> f = mem_failures(mem_breakdown_of(no_warm, 100), 0, {1000, 1000, 1000, 0, 1000});
    check(f.size() == 1 && f[0].find("warming") != std::string::npos, "memory schedule: a state never sampled fails");
}

// The gates that need no model and no Core ML: --self-test runs only these
// (a unit test in every build); a run with --gguf runs them first.
void model_free_gates() {
    fprintf(stderr, "model-free gates\n");
    waveform_quality_gates();
    parity_floor_gates();
    memory_schedule_gates();
}

std::vector<int32_t> random_codes(const parler_hparams & hp, int n_frames, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, hp.dac_codebook_size - 1);
    std::vector<int32_t> codes((size_t) hp.dac_n_q * n_frames);
    for (int32_t & c : codes) c = code(rng);
    return codes;
}

bool decode(const parler_model & model, const std::vector<int32_t> & codes, int n_frames,
            std::vector<float> & pcm, std::vector<float> * latent = nullptr, int begin = 0,
            int end = -1, std::string * backend = nullptr, std::string * err = nullptr) {
    return parler_dac_decode(model, codes.data(), n_frames, 2, pcm, latent, begin, end, err, backend);
}

void fake_sidecar_gates(parler_model & model) {
    fprintf(stderr, "fake sidecar gates\n");
    const parler_hparams & hp = model.hparams;
    const int rf = parler_dac_rf_frames(model);
    const int window = 2 * rf + 8;
    const int n_frames = 3 * (window - 2 * rf) + 5;
    const std::vector<int32_t> codes = random_codes(hp, n_frames, 11);

    std::vector<float> ref, ref_lat;
    std::string backend;
    check(decode(model, codes, n_frames, ref, &ref_lat, 0, -1, &backend) && backend == "ggml",
          "ggml reference decode reports ggml");

    // host quantizer == ggml's latent; the fake's pcm is its constant
    auto ok_fake = std::make_shared<fake_sidecar>(window, hp.dac_hop, 0);
    check(parler_attach_dac_sidecar(model, ok_fake), "attach a fake sidecar");
    std::vector<float> pcm, lat;
    check(decode(model, codes, n_frames, pcm, &lat, 0, -1, &backend) && backend == "coreml-fake",
          "sidecar decode reports the sidecar label");
    check(ok_fake->calls == 4, "sidecar walks ceil(n / (W - 2 rf)) windows (" +
          std::to_string(ok_fake->calls) + ")");
    double max_abs = 0;
    for (size_t i = 0; i < lat.size() && i < ref_lat.size(); ++i) {
        max_abs = std::max(max_abs, (double) std::fabs(lat[i] - ref_lat[i]));
    }
    check(lat.size() == ref_lat.size() && max_abs < 1e-4,
          "host RVQ latent == ggml latent (max_abs " + std::to_string(max_abs) + ")");
    check(pcm.size() == ref.size() && pcm.front() == 0.25f && pcm.back() == 0.25f,
          "sidecar pcm covers exactly the range");
    check(decode(model, codes, n_frames, pcm, nullptr, 7, 7) && pcm.empty(), "empty range");
    std::vector<int32_t> bad = codes;
    bad[3] = hp.dac_codebook_size;
    check(!decode(model, bad, n_frames, pcm) && model.dac_sidecar, "out-of-range code rejected before the sidecar");

    // fails on its second window: retired once, whole range redone on ggml
    auto failing = std::make_shared<fake_sidecar>(window, hp.dac_hop, 2);
    check(parler_attach_dac_sidecar(model, failing), "attach a failing fake sidecar");
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    check(decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend) && backend == "ggml",
          "mid-range failure uses captured non-strict policy despite later strict environment");
    check(pcm == ref, "fallback output is bit-identical to the ggml-only decode");
    check(!model.dac_sidecar && failing->calls == 2, "failing sidecar retired after one failure");
    check(decode(model, codes, n_frames, pcm, nullptr, 5, n_frames - 3, &backend) && backend == "ggml" &&
          std::equal(pcm.begin(), pcm.end(), ref.begin() + 5 * hp.dac_hop),
          "later ranges stay on ggml");

    // strict: a failure is an error, a missing sidecar throws at attach
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    auto strict_fail = std::make_shared<fake_sidecar>(window, hp.dac_hop, 1);
    parler_attach_dac_sidecar(model, strict_fail);
    unsetenv(PARLER_COREML_STRICT_ENV);
    std::string err;
    check(!decode(model, codes, n_frames, pcm, nullptr, 0, -1, nullptr, &err) && pcm.empty() &&
          err.find(PARLER_COREML_STRICT_ENV) != std::string::npos && !model.dac_sidecar,
          "strict: captured policy fails the call despite later environment removal");
    check(!decode(model, codes, n_frames, pcm) && pcm.empty(), "strict: later calls keep failing after a retirement");
    bool threw = false;
    try {
        parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf"),
                                  parler_resolve_coreml_policy("cpu_only", true));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "strict: no sidecar throws at attach");
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    auto explicit_fallback = std::make_shared<fake_sidecar>(window, hp.dac_hop, 1);
    check(parler_attach_dac_sidecar(model, explicit_fallback,
                                   parler_resolve_coreml_policy("cpu_only", false)),
          "explicit false attaches independently of strict environment");
    check(decode(model, codes, n_frames, pcm) && pcm == ref,
          "explicit false recovers on ggml despite strict environment");
    unsetenv(PARLER_COREML_STRICT_ENV);
    check(decode(model, codes, n_frames, pcm) && pcm == ref, "retired non-strict sidecar stays on ggml");
    check(!parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf")),
          "no sidecar: attach reports ggml");
    setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
    threw = false;
    try {
        parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf"),
                                  parler_resolve_coreml_policy("cpu_only", true));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "disabled plus strict retains legacy missing-sidecar failure");
    check(!parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf"),
                                    parler_resolve_coreml_policy("cpu_only", false)),
          "global disable permits ggml with explicit non-strict policy");
    unsetenv(PARLER_COREML_DISABLE_ENV);
}

// A pending seam: `ready_after` calls return nothing (pending), then the
// sidecar; `strict` makes acquire throw while pending. `block_until_cancel`
// makes acquire wait, like a Core ML load that is never ready, until the
// caller's cancel flag is set.
class fake_pending final : public parler_dac_pending {
public:
    fake_pending(std::shared_ptr<parler_dac_sidecar> sidecar, int ready_after, bool strict)
        : sidecar_(std::move(sidecar)), ready_after_(ready_after), strict_(strict) {}
    parler_dac_call acquire(int, const std::atomic<bool> * cancel) override {
        ++acquires;
        if (block_until_cancel) {
            saw_cancel_flag = cancel != nullptr;
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (cancel && !cancel->load() && std::chrono::steady_clock::now() < give_up) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        parler_dac_call call;
        if (retired || acquires <= ready_after_) {
            if (strict_) throw std::runtime_error("fake: not ready; strict");
            call.pending = !retired;
            return call;
        }
        call.sidecar = sidecar_;
        call.retire = [this](const std::string & why) {
            retired = true;
            retire_reason = why;
        };
        return call;
    }
    bool strict() const override { return strict_; }
    int acquires = 0;
    bool retired = false;
    bool block_until_cancel = false;
    bool saw_cancel_flag = false;
    std::string retire_reason;

private:
    std::shared_ptr<parler_dac_sidecar> sidecar_;
    int ready_after_;
    bool strict_;
};

void pending_seam_gates(parler_model & model) {
    fprintf(stderr, "pending seam gates\n");
    const parler_hparams & hp = model.hparams;
    const int rf = parler_dac_rf_frames(model);
    const int window = 2 * rf + 8;
    const int n_frames = 3 * (window - 2 * rf) + 5;
    const std::vector<int32_t> codes = random_codes(hp, n_frames, 23);
    model.dac_sidecar.reset();
    model.dac_sidecar_retired = false;
    model.dac_pending.reset();
    std::vector<float> ref;
    check(decode(model, codes, n_frames, ref), "ggml reference decode");
    if (!parler_dac_rvq_load(model, model.dac_rvq, nullptr)) {
        check(false, "host quantizer loads");
        return;
    }

    // pending: ggml, bit-equal, reported
    auto sidecar = std::make_shared<fake_sidecar>(window, hp.dac_hop, 0);
    auto seam = std::make_shared<fake_pending>(sidecar, 1, false);
    model.dac_pending = seam;
    std::vector<float> pcm;
    std::string backend, err;
    bool pending = false;
    check(parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend, &pending) &&
              backend == "ggml" && pending && pcm == ref && sidecar->calls == 0,
          "pending: the call runs on ggml, bit-identical, and reports pending");

    // ready between two streamed chunks: each chunk on one backend, the
    // synthesis mixed
    const int split = n_frames / 2;
    std::string label;
    std::vector<float> chunk;
    seam->acquires = 0;
    auto seam2 = std::make_shared<fake_pending>(sidecar, 1, false);
    model.dac_pending = seam2;
    bool p1 = false, p2 = false;
    std::string b1, b2;
    const bool ok1 = parler_dac_decode(model, codes.data(), n_frames, 2, chunk, nullptr, 0, split, &err, &b1, &p1);
    const bool chunk1_ggml = ok1 && std::equal(chunk.begin(), chunk.end(), ref.begin());
    label = parler_merge_dac_backend(label, b1);
    const bool ok2 = parler_dac_decode(model, codes.data(), n_frames, 2, chunk, nullptr, split, n_frames, &err, &b2,
                                       &p2);
    const bool chunk2_sidecar =
        ok2 && !chunk.empty() && std::all_of(chunk.begin(), chunk.end(), [](float v) { return v == 0.25f; });
    label = parler_merge_dac_backend(label, b2);
    check(chunk1_ggml && p1 && b1 == "ggml" && chunk2_sidecar && !p2 && b2 == "coreml-fake" && label == "mixed" &&
              seam2->acquires == 2,
          "ready between chunks: chunk 1 ggml (pending), chunk 2 on the sidecar, synthesis " + label);

    // a failure retires through the seam; the whole call is redone on ggml
    auto failing = std::make_shared<fake_sidecar>(window, hp.dac_hop, 2);
    auto seam3 = std::make_shared<fake_pending>(failing, 0, false);
    model.dac_pending = seam3;
    check(parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend, &pending) &&
              backend == "ggml" && !pending && pcm == ref && seam3->retired,
          "failure: retired through the seam, the whole call redone on ggml (bit-identical)");
    check(parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend, &pending) &&
              backend == "ggml" && pcm == ref,
          "after a retirement later calls stay on ggml");

    // strict: not ready, and a failure, are errors
    model.dac_pending = std::make_shared<fake_pending>(sidecar, 1, true);
    err.clear();
    check(!parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend) &&
              pcm.empty() && err.find("strict") != std::string::npos,
          "strict: a sidecar that cannot serve the call fails it");
    auto strict_failing = std::make_shared<fake_sidecar>(window, hp.dac_hop, 1);
    auto seam4 = std::make_shared<fake_pending>(strict_failing, 0, true);
    model.dac_pending = seam4;
    err.clear();
    check(!parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend) &&
              pcm.empty() && seam4->retired && err.find(PARLER_COREML_STRICT_ENV) != std::string::npos,
          "strict: a failing sidecar is retired and the call fails");

    // cancellation interrupts a wait for the sidecar: the call returns soon
    // after the flag is set, reports cancellation and decodes nothing
    const int sidecar_calls = sidecar->calls;
    auto blocking = std::make_shared<fake_pending>(sidecar, 1 << 30, false);
    blocking->block_until_cancel = true;
    model.dac_pending = blocking;
    std::atomic<bool> cancel{false};
    std::thread canceller([&cancel] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cancel.store(true);
    });
    const auto t0 = std::chrono::steady_clock::now();
    err.clear();
    pcm.assign(4, 1.0f);
    const bool ok_cancel = parler_dac_decode(model, codes.data(), n_frames, 2, pcm, nullptr, 0, -1, &err, &backend,
                                             &pending, &cancel);
    const double cancel_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    canceller.join();
    check(!ok_cancel && blocking->saw_cancel_flag && err == "cancelled" && pcm.empty() && cancel_ms < 5000 &&
              sidecar->calls == sidecar_calls,
          "cancel: a blocked acquisition returns once cancelled, the call reports '" + err + "' after " +
              std::to_string((int) cancel_ms) + " ms");
    model.dac_pending.reset();
}

void coreml_gates(parler_model & model, const std::string & gguf, const std::string & ref_dir,
                  double min_snr, double min_speech_snr) {
    fprintf(stderr, "Core ML sidecar gates (%s)\n", parler_dac_sidecar_path(gguf).c_str());
    model.dac_sidecar.reset();
    if (!parler_attach_dac_sidecar(model, gguf)) {
        check(false, "sidecar attaches");
        return;
    }
    const std::string label = model.dac_sidecar->label();
    check(label.rfind("coreml-", 0) == 0, "label " + label);
    const parler_hparams & hp = model.hparams;
    const int window = model.dac_sidecar->window();
    const int keep = window - 2 * parler_dac_rf_frames(model);

    for (const int n_frames : {2 * keep + keep / 2, window / 3, 1}) {
        const std::vector<int32_t> codes = random_codes(hp, n_frames, 100 + n_frames);
        auto sidecar = model.dac_sidecar;
        model.dac_sidecar.reset();
        std::vector<float> ref, pcm;
        decode(model, codes, n_frames, ref);
        model.dac_sidecar = sidecar;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        double max_abs = 0;
        const double snr = ok && pcm.size() == ref.size() ? snr_db(pcm, ref, max_abs) : NAN;
        check(ok && backend == label && snr >= min_snr,
              std::to_string(n_frames) + " frames vs ggml: SNR " + std::to_string(snr) +
              " dB, max_abs " + std::to_string(max_abs));
        if (n_frames > keep) {
            // three sub-ranges that split windows differently stitch to the whole
            const int a = keep / 3, b = keep + keep / 2;
            std::vector<float> stitched, part;
            for (const auto & r : {std::make_pair(0, a), std::make_pair(a, b), std::make_pair(b, n_frames)}) {
                decode(model, codes, n_frames, part, nullptr, r.first, r.second);
                stitched.insert(stitched.end(), part.begin(), part.end());
            }
            const double ssnr = stitched.size() == ref.size() ? snr_db(stitched, ref, max_abs) : NAN;
            check(ssnr >= min_snr, "sub-ranges stitch: SNR " + std::to_string(ssnr) + " dB");
        }
    }
    check(model.dac_sidecar != nullptr, "sidecar still attached");

    tts_cpp::parler::FitOptions fit;
    fit.model_gguf_path = gguf;
    check(tts_cpp::parler::fit_params(fit).reason == "coreml-memory-estimate-unavailable",
          "fit refuses to project an enabled sidecar");
    setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
    const std::string reason = tts_cpp::parler::fit_params(fit).reason;
    unsetenv(PARLER_COREML_DISABLE_ENV);
    check(reason != "coreml-memory-estimate-unavailable", "PARLER_COREML_DISABLE: fit projects ggml (" + reason + ")");

    if (!ref_dir.empty() && std::ifstream(ref_dir + "/case0_codes.npy").good() &&
        std::ifstream(ref_dir + "/case0_wav_greedy.npy").good()) {
        npy_array codes_npy = npy_load(ref_dir + "/case0_codes.npy");
        npy_array wav_npy = npy_load(ref_dir + "/case0_wav_greedy.npy");
        const int n_frames = (int) codes_npy.shape[2];
        const int64_t * c64 = reinterpret_cast<const int64_t *>(codes_npy.data.data());
        std::vector<int32_t> codes(codes_npy.n_elements());
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t) c64[i];
        std::vector<float> pcm;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        const float * w = npy_as_f32(wav_npy);
        const std::vector<float> want(w, w + wav_npy.n_elements());
        const auto q = hf_quality(pcm, want, min_speech_snr);
        check(ok && backend == label && q.pass, "case0 vs HF wav (" + backend + "): SNR " + std::to_string(q.snr) +
              " dB, max_abs " + std::to_string(q.peak) + ", ref RMS " +
              std::to_string(q.ref_rms) + ", error RMS " + std::to_string(q.error_rms) +
              (q.quiet ? " (quiet absolute gate)" : " (relative gate)"));
    }
}

// The public surface: an Engine on the model attaches the sidecar and reports
// it per call, streamed or not; PARLER_COREML_DISABLE keeps it on ggml.
void engine_gates(const std::string & gguf) {
    fprintf(stderr, "Engine gates\n");
    tts_cpp::parler::EngineOptions opts;
    opts.model_gguf_path = gguf;
    opts.default_description = "A calm voice.";
    opts.max_frames = 40;
    opts.stream_chunk_frames = 12;
    // The sidecar loads in the background; wait for it at construction so
    // the first call is deterministic.
    opts.coreml.wait = tts_cpp::CoremlOptions::Wait::construct;
    for (const bool disabled : {false, true}) {
        if (disabled) setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
        tts_cpp::parler::Engine engine(opts);
        const auto whole = engine.synthesize("a b");
        const auto streamed = engine.synthesize("a b", opts.default_description,
                                                [](const float *, std::size_t, int, bool) {});
        const std::string want = disabled ? "ggml" : "coreml-";
        check(engine.dac_on_coreml() == !disabled &&
              whole.dac_backend.rfind(want, 0) == 0 && streamed.dac_backend.rfind(want, 0) == 0,
              std::string(disabled ? "disabled" : "enabled") + ": dac_on_coreml " +
              (engine.dac_on_coreml() ? "true" : "false") + ", dac_backend " + whole.dac_backend + " / " +
              streamed.dac_backend);
        if (disabled) unsetenv(PARLER_COREML_DISABLE_ENV);
    }
}

#ifdef TTS_CPP_USE_COREML
// The route-independent decode gates against ggml for whatever model.dac_pending
// serves (a generated sidecar here).
void pending_parity_gates(parler_model & model, const std::string & ref_dir, double min_snr,
                          double min_speech_snr) {
    auto seam = model.dac_pending;
    const parler_dac_call probe = seam->acquire(1, nullptr);
    if (!probe.sidecar) {
        check(false, "the generated sidecar is ready");
        return;
    }
    const std::string label = probe.sidecar->label();
    const int window = probe.sidecar->window();
    check(label.rfind("coreml-", 0) == 0, "label " + label + ", window " + std::to_string(window));
    const parler_hparams & hp = model.hparams;
    const int keep = window - 2 * parler_dac_rf_frames(model);
    const std::string chip = host_chip();
    const double host_min_snr = generated_min_snr(label, chip, min_snr);
    fprintf(stderr, "  host %s: parity gate %.1f dB (--min-snr %.1f)\n", chip.empty() ? "unknown" : chip.c_str(),
            host_min_snr, min_snr);
    std::vector<double> multi_frame_snrs;
    for (const int n_frames : {2 * keep + keep / 2, window / 3, 1}) {
        const std::vector<int32_t> codes = random_codes(hp, n_frames, 100 + n_frames);
        model.dac_pending.reset();
        std::vector<float> ref, pcm;
        decode(model, codes, n_frames, ref);
        model.dac_pending = seam;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        double max_abs = 0;
        const double snr = ok && pcm.size() == ref.size() ? snr_db(pcm, ref, max_abs) : NAN;
        if (n_frames > 1) multi_frame_snrs.push_back(snr);
        check(ok && backend == label && snr >= generated_gate(n_frames, label, chip, min_snr),
              std::to_string(n_frames) + " frames vs ggml: SNR " + std::to_string(snr) + " dB, max_abs " +
                  std::to_string(max_abs));
    }
    if (host_min_snr < min_snr) {
        check(!floor_expired(host_min_snr, min_snr, multi_frame_snrs),
              "the " + chip + " floor is still needed (once every multi-frame case reaches --min-snr, remove its row "
              "from GENERATED_ALL_UNITS_FLOORS)");
    }
    if (!ref_dir.empty() && std::ifstream(ref_dir + "/case0_codes.npy").good() &&
        std::ifstream(ref_dir + "/case0_wav_greedy.npy").good()) {
        npy_array codes_npy = npy_load(ref_dir + "/case0_codes.npy");
        npy_array wav_npy = npy_load(ref_dir + "/case0_wav_greedy.npy");
        const int n_frames = (int) codes_npy.shape[2];
        const int64_t * c64 = reinterpret_cast<const int64_t *>(codes_npy.data.data());
        std::vector<int32_t> codes(codes_npy.n_elements());
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t) c64[i];
        std::vector<float> pcm;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        const float * w = npy_as_f32(wav_npy);
        const std::vector<float> want(w, w + wav_npy.n_elements());
        const auto q = hf_quality(pcm, want, min_speech_snr);
        check(ok && backend == label && q.pass, "case0 vs HF wav (" + backend + "): SNR " + std::to_string(q.snr) +
              " dB, max_abs " + std::to_string(q.peak) + (q.quiet ? " (quiet absolute gate)" : " (relative gate)"));
    }
}

// --units-sweep: the SNR of the generated artifact in the cache against the
// ggml CPU decode for every compute-unit policy, on the parity cases' codes.
// It shows which path `all` takes on the host and whether a parity shortfall
// comes from the Neural Engine, the GPU or the reference. The rows are
// reported only, except on a chip with a floor under `all`, where the
// cpu_and_ane row is held to --min-snr (sweep_row_gated). The artifact is
// opened through the prebuilt-sidecar loader by a symlink.
void units_sweep(parler_model & model, const std::string & cache_dir, double min_snr) {
    namespace fs = std::filesystem;
    const std::string chip = host_chip();
    fprintf(stderr, "compute-unit sweep\n");
    std::error_code ec;
    std::string artifact;
    for (fs::recursive_directory_iterator it(cache_dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory() && it->path().filename() == "model.mlmodelc" &&
            it->path().string().find("/artifacts/") != std::string::npos) {
            artifact = it->path().string();
            break;
        }
    }
    // On a chip with a gated row a setup failure is a failure, not a silent skip.
    const bool required = sweep_required(chip, min_snr);
    const auto setup_failed = [&](const std::string & what) {
        fprintf(stderr, "  %s\n", what.c_str());
        if (required) check(false, "units sweep on " + chip + ": " + what);
    };
    if (artifact.empty()) {
        setup_failed("no generated artifact under " + cache_dir);
        return;
    }
    const fs::path dir = fs::temp_directory_path() /
                         ("parler-units-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir, ec);
    const std::string fake_gguf = (dir / "parler-sweep.gguf").string();
    fs::create_directory_symlink(artifact, parler_dac_sidecar_path(fake_gguf), ec);
    if (ec) {
        setup_failed("cannot link " + artifact + ": " + ec.message());
        fs::remove_all(dir, ec);
        return;
    }
    fprintf(stderr, "  artifact %s\n", artifact.c_str());
    auto saved_pending = model.dac_pending;
    model.dac_pending.reset();
    const parler_hparams & hp = model.hparams;
    for (const char * units : {"cpu_only", "cpu_and_gpu", "cpu_and_ane", "all"}) {
        const parler_coreml_policy policy = parler_resolve_coreml_policy(units, false);
        std::string why;
        auto sidecar = parler_open_dac_sidecar(fake_gguf, model, policy, &why);
        const bool gated = sweep_row_gated(units, chip, min_snr);
        if (!sidecar || !parler_attach_dac_sidecar(model, sidecar, policy, &why)) {
            fprintf(stderr, "  units %-11s: not loaded (%s)\n", units, why.c_str());
            if (gated) check(false, std::string("units ") + units + " loads on " + chip);
            continue;
        }
        const int window = sidecar->window();
        const int keep = window - 2 * parler_dac_rf_frames(model);
        // The label the requested units must give, not the one the opened
        // sidecar reports: a sidecar opened on other units fails the row.
        const std::string label = label_for_units(units);
        const std::string opened = sidecar->label();
        std::string line = opened == label ? std::string() : "  opened as " + opened;
        bool row_holds = opened == label;
        for (const int n_frames : {2 * keep + keep / 2, window / 3, 1}) {
            const std::vector<int32_t> codes = random_codes(hp, n_frames, 100 + n_frames);
            model.dac_sidecar.reset();
            std::vector<float> ref, pcm;
            std::string ref_backend, backend;
            const bool ref_ok = decode(model, codes, n_frames, ref, nullptr, 0, -1, &ref_backend);
            model.dac_sidecar = sidecar;
            const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
            double max_abs = 0;
            const double snr = ref_ok && ok && pcm.size() == ref.size() ? snr_db(pcm, ref, max_abs) : NAN;
            if (!sweep_case_holds(ref_ok, ref_backend, ok, backend, label, snr, min_snr)) {
                row_holds = false;
            }
            char buf[160];
            snprintf(buf, sizeof buf, "  %d frames %.2f dB (max_abs %.2e%s%s)", n_frames, snr, max_abs,
                     backend == label ? "" : ", decoded on ", backend == label ? "" : backend.c_str());
            line += buf;
        }
        fprintf(stderr, "  units %-11s:%s\n", units, line.c_str());
        if (gated) {
            check(row_holds, std::string("units ") + units + " on " + chip + " decodes every case on " + label +
                                 " and holds --min-snr " + std::to_string(min_snr) +
                                 " dB (the gate `all` has a floor for)");
        }
        parler_attach_dac_sidecar(model, std::shared_ptr<parler_dac_sidecar>{}, policy);
    }
    model.dac_pending = saved_pending;
    fs::remove_all(dir, ec);
}

bool contains(const std::string & s, const std::string & needle) { return s.find(needle) != std::string::npos; }

// prewarm none issues an empty construction group and leaves the slot idle
// until the first call: wait = construct with use = require returns without
// throwing, and wait() returns false at once instead of polling an idle slot.
// Nothing is generated or loaded (a scratch cache stays empty).
void idle_slot_gates(parler_model & model, const std::string & gguf) {
    fprintf(stderr, "prewarm none gates\n");
    if (::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf))) {
        fprintf(stderr, "  a prebuilt sidecar is next to the GGUF: skipped\n");
        return;
    }
    const std::string cache =
        (std::filesystem::temp_directory_path() /
         ("parler-dac-idle-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
            .string();
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache;
    opts.use = tts_cpp::CoremlOptions::Use::require;
    opts.wait = tts_cpp::CoremlOptions::Wait::construct;
    opts.prewarm = tts_cpp::CoremlOptions::Prewarm::none;
    const parler_dac_route route = parler_dac_plan_route(gguf, opts, parler_dac_rvq_bytes(model));
    if (route.route != "generate") {
        fprintf(stderr, "  route %s (%s: %s): skipped\n", route.route.c_str(), route.state.c_str(),
                route.reason.c_str());
        return;
    }
    model.dac_sidecar.reset();
    model.dac_pending.reset();
    std::shared_ptr<parler_dac_coreml> dac;
    std::string what;
    try {
        dac = parler_dac_coreml::create(model, gguf, opts);
    } catch (const std::exception & e) {
        what = e.what();
    }
    check(dac != nullptr, "{use: require, wait: construct, prewarm: none} constructs" +
                              (what.empty() ? std::string() : ": " + what));
    if (dac) {
        const std::string status = dac->status_json();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ready = dac->wait(-1);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check(contains(status, "\"state\":\"idle\"") && !ready && ms < 1000,
              "an idle slot: wait(-1) returns false after " + std::to_string((int) ms) + " ms; " +
                  status.substr(0, std::min<size_t>(status.size(), 200)));
    }
    model.dac_pending.reset();
    dac.reset();
    std::error_code ec;
    std::filesystem::remove_all(cache, ec);
}

// The value of the first "state" in a status JSON ("" when absent).
std::string status_state(const std::string & status) {
    const std::string key = "\"state\":\"";
    const size_t at = status.find(key);
    if (at == std::string::npos) return {};
    const size_t begin = at + key.size();
    return status.substr(begin, status.find('"', begin) - begin);
}

// The GGUF is replaced after the weights were loaded and before the sidecar
// is made: the model is loaded through a symlink, which is then pointed at a
// clone of the file (same bytes, another inode). The prepare job must not
// follow the path to the replacement: it fails with source_changed, nothing
// is generated, and with use = auto the DAC decodes on ggml.
void source_changed_gate(const std::string & gguf) {
    fprintf(stderr, "replaced GGUF gate\n");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("parler-dac-replaced-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path link = dir / "model.gguf", clone = dir / "clone.gguf", cache = dir / "cache";
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::create_symlink(fs::absolute(gguf), link, ec);
    if (ec) {
        check(false, "a symlink to the GGUF: " + ec.message());
        fs::remove_all(dir, ec);
        return;
    }
    parler_model m;
    std::string err;
    if (!parler_load_gguf(link.string(), m, 0, &err)) {
        check(false, "load through the symlink: " + err);
        fs::remove_all(dir, ec);
        return;
    }
    bool replaced = ::clonefile(fs::absolute(gguf).c_str(), clone.c_str(), 0) == 0;
    if (replaced) {
        fs::remove(link, ec);
        fs::create_symlink(clone, link, ec);
        replaced = !ec;
    }
    check(replaced, "the GGUF the model was loaded from is replaced by a clone");
    if (replaced) {
        tts_cpp::CoremlOptions opts;
        opts.cache_dir = cache.string();
        opts.wait = tts_cpp::CoremlOptions::Wait::construct;
        opts.use = tts_cpp::CoremlOptions::Use::automatic;
        std::shared_ptr<parler_dac_coreml> dac;
        try {
            dac = parler_dac_coreml::create(m, link.string(), opts);
        } catch (const std::exception & e) {
            check(false, std::string("create after the replacement: ") + e.what());
        }
        if (dac) {
            const std::string st = dac->status_json();
            std::vector<float> pcm;
            std::string backend;
            const bool decoded = decode(m, random_codes(m.hparams, 8, 7), 8, pcm, nullptr, 0, -1, &backend);
            check(!dac->wait(0) && contains(st, "source_changed") && decoded && backend == PARLER_DAC_GGML_BACKEND,
                  "a GGUF replaced after the load is not generated from (DAC on " + backend +
                      "): " + st.substr(0, std::min<size_t>(st.size(), 300)));
        }
        m.dac_pending.reset();
    }
    parler_free_model(m);
    fs::remove_all(dir, ec);
}

// The Engine's default wait (none) on an empty cache: construction returns
// at once and the DAC sidecar is generated in the background.
// Engine::coreml_wait(-1) blocks until it is ready, and Engine::cancel() from
// another thread ends that wait early with false. A second coreml_wait(-1)
// then waits the generation out, which leaves the artifact the gates below
// adopt. Needs an empty cache: a sidecar that is ready at construction has
// nothing to cancel, so a populated --cache-dir fails the gate.
void engine_wait_cancel_gate(const std::string & gguf, const std::string & cache_dir) {
    fprintf(stderr, "Engine wait and cancel (cache %s)\n", cache_dir.c_str());
    tts_cpp::parler::EngineOptions eo;
    eo.model_gguf_path = gguf;
    eo.default_description = "A calm voice.";
    eo.coreml.cache_dir = cache_dir;
    tts_cpp::parler::Engine engine(eo);
    if (engine.coreml_wait(0)) {
        check(false, "Engine::cancel() ends coreml_wait(-1): the sidecar is ready at construction, so nothing is "
                     "left to cancel (pass an empty --cache-dir)");
        return;
    }
    static constexpr int CANCEL_AFTER_MS = 200;
    std::thread canceller([&engine] {
        std::this_thread::sleep_for(std::chrono::milliseconds(CANCEL_AFTER_MS));
        engine.cancel();
    });
    const auto t0 = std::chrono::steady_clock::now();
    const bool ready = engine.coreml_wait(-1);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    canceller.join();
    const std::string state = status_state(engine.coreml_status());
    check(!ready && ms >= CANCEL_AFTER_MS - 20 && ms < 2000 && state != "ready",
          "Engine::cancel() ends coreml_wait(-1) after " + std::to_string((int) ms) + " ms (sidecar " + state +
              (ready ? ", wait returned ready" : "") + ")");
    {
        // Two blocking waits: a cancel() ends the one running, and a wait
        // started right after it neither swallows that cancel nor ends by it.
        std::atomic<bool> a_done{false}, b_done{false}, a_ready{false};
        std::thread a([&] {
            a_ready = engine.coreml_wait(-1);
            a_done = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto cancelled_at = std::chrono::steady_clock::now();
        engine.cancel();
        std::thread b([&] {
            engine.coreml_wait(-1);
            b_done = true;
        });
        while (!a_done && std::chrono::steady_clock::now() - cancelled_at < std::chrono::seconds(30)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        const double a_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cancelled_at).count();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const bool b_waiting = !b_done || status_state(engine.coreml_status()) == "ready";
        engine.cancel();  // ends b
        a.join();
        b.join();
        check(a_done && !a_ready && a_ms < 1500 && b_waiting,
              "a cancel() ends the running coreml_wait(-1) after " + std::to_string((int) a_ms) +
                  " ms although another wait starts right after it" + (a_ready ? " (it returned ready)" : "") +
                  (b_waiting ? "; the later wait kept waiting" : "; the later wait ended on the earlier cancel"));
    }
    const bool done = engine.coreml_wait(-1);
    const std::string st = engine.coreml_status();
    check(done && contains(st, "\"route\":\"generated\""),
          "the next coreml_wait(-1) waits the generation out (" + status_state(st) + "): " +
              st.substr(0, std::min<size_t>(st.size(), 200)));
}

// A wait on one thread never swallows a cancel meant for a synthesis on
// another: the synthesis is held inside its first streamed chunk (not the
// last, so it has decoder steps left), Engine::cancel() is called, then a
// readiness poll coreml_wait(0) and a short coreml_wait(10); once the chunk
// returns, the synthesis must end cancelled, and both waits report the
// ready sidecar.
void engine_wait_keeps_cancel_gate(const std::string & gguf, const tts_cpp::CoremlOptions & coreml) {
    fprintf(stderr, "Engine cancel with waits on another thread\n");
    tts_cpp::parler::EngineOptions eo;
    eo.model_gguf_path = gguf;
    eo.default_description = "A calm voice.";
    eo.max_frames = 160;
    eo.min_new_tokens = 150;
    eo.stream_chunk_frames = 24;
    eo.coreml = coreml;
    tts_cpp::parler::Engine engine(eo);
    std::mutex mu;
    std::condition_variable cv;
    bool entered = false, finished = false, released = false, first_chunk_final = false;
    std::string outcome = "completed";
    std::thread synth([&] {
        try {
            engine.synthesize("a b c", eo.default_description, [&](const float *, std::size_t, int, bool final_chunk) {
                std::unique_lock<std::mutex> lk(mu);
                if (entered) return;  // only the first chunk is held
                entered = true;
                first_chunk_final = final_chunk;
                cv.notify_all();
                cv.wait_for(lk, std::chrono::seconds(60), [&] { return released; });
            });
        } catch (const std::exception & e) {
            outcome = e.what();
        }
        std::lock_guard<std::mutex> lk(mu);
        finished = true;
        cv.notify_all();
    });
    bool held = false;
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::minutes(3), [&] { return entered || finished; });
        held = entered && !finished;
    }
    bool poll = false, short_wait = false;
    if (held) {
        engine.cancel();
        poll = engine.coreml_wait(0);
        short_wait = engine.coreml_wait(10);
    }
    {
        std::lock_guard<std::mutex> lk(mu);
        released = true;
        cv.notify_all();
    }
    synth.join();
    check(held && !first_chunk_final && contains(outcome, "cancelled") && poll && short_wait,
          std::string("cancel(), then coreml_wait(0) and coreml_wait(10) on another thread: the synthesis ends '") +
              outcome + "'" + (held ? "" : " (no chunk was held)") + (first_chunk_final ? " (the first chunk was the last)" : "") +
              ", the waits report " + (poll ? "ready" : "not ready") + " / " + (short_wait ? "ready" : "not ready"));
}

void generated_gates(parler_model & model, const std::string & gguf, const std::string & cache_dir,
                     const std::string & ref_dir, double min_snr, double min_speech_snr) {
    fprintf(stderr, "generated sidecar gates (cache %s)\n", cache_dir.c_str());
    if (::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf))) {
        check(false, "no prebuilt sidecar next to --gguf (the generated route needs none)");
        return;
    }
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache_dir;
    opts.wait = tts_cpp::CoremlOptions::Wait::construct;
    opts.use = tts_cpp::CoremlOptions::Use::require;
    const parler_dac_route route = parler_dac_plan_route(gguf, opts, parler_dac_rvq_bytes(model));
    check(route.route == "generate" && route.validated && route.envelope_bytes > 0,
          "route generate (" + route.route + " " + route.state + " " + route.reason + "), envelope " +
              std::to_string(route.envelope_bytes) + " bytes" + (route.validated ? ", validated" : ""));
    engine_wait_cancel_gate(gguf, cache_dir);
    engine_wait_keeps_cancel_gate(gguf, opts);
    source_changed_gate(gguf);
    model.dac_sidecar.reset();
    model.dac_pending.reset();
    std::shared_ptr<parler_dac_coreml> dac;
    try {
        dac = parler_dac_coreml::create(model, gguf, opts);
    } catch (const std::exception & e) {
        check(false, std::string("generate, compile and load: ") + e.what());
        return;
    }
    const std::string status = dac->status_json();
    fprintf(stderr, "  status: %s\n", status.c_str());
    check(dac->wait(0) && contains(status, "\"state\":\"ready\"") &&
              (contains(status, "\"route\":\"generated\"") || contains(status, "\"route\":\"cached\"")),
          "generated, compiled and loaded (ready)");
    pending_parity_gates(model, ref_dir, min_snr, min_speech_snr);

    // A second owner of the same GGUF, template and cache adopts the artifact.
    parler_model second;
    std::string err;
    if (parler_load_gguf(gguf, second, 0, &err)) {
        auto again = parler_dac_coreml::create(second, gguf, opts);
        const std::string st2 = again->status_json();
        check(again->wait(0) && contains(st2, "\"route\":\"cached\""),
              "a second engine adopts the artifact (route cached): " + st2.substr(0, std::min<size_t>(st2.size(), 160)));
        parler_free_model(second);
    } else {
        check(false, "a second engine loads " + gguf + ": " + err);
    }

    tts_cpp::parler::FitOptions fit;
    fit.model_gguf_path = gguf;
    fit.coreml = opts;
    const tts_cpp::FitResult fr = tts_cpp::parler::fit_params(fit);
    check(fr.coreml.route == "generate" && fr.coreml.bytes == route.envelope_bytes && fr.coreml.validated,
          "fit charges the generated envelope (" + fr.coreml.route + ", " + std::to_string(fr.coreml.bytes) +
              " bytes, " + fr.reason + ")");

    // Near the budget: one only the ggml projection fits resolves use = off
    // (memory-degrade); a roomy one keeps Core ML.
    const uint64_t ggml_need = fr.device.total_bytes + fit.margin_bytes +
                               (fr.device_shares_host_memory ? fr.host_bytes : 0);
    tts_cpp::parler::FitOptions near = fit;
    near.coreml.memory_budget_bytes = (int64_t) (ggml_need + fr.coreml.bytes / 2);
    const tts_cpp::FitResult degraded = tts_cpp::parler::fit_params(near);
    tts_cpp::parler::FitOptions roomy = fit;
    roomy.coreml.memory_budget_bytes = (int64_t) (ggml_need + 2 * fr.coreml.bytes);
    const tts_cpp::FitResult kept = tts_cpp::parler::fit_params(roomy);
    check(degraded.coreml.disabled_reason == "memory" && kept.coreml.disabled_reason.empty(),
          "near-budget memory-degrade: '" + degraded.coreml.disabled_reason + "' below, '" +
              kept.coreml.disabled_reason + "' with room");
    // Near the free memory with a budget far above it: a margin that leaves
    // room for ggml but not for ggml + Core ML degrades on the free memory,
    // and the fit stands.
    const uint64_t ggml_no_margin = ggml_need - fit.margin_bytes;
    if (fr.device_free_bytes > ggml_no_margin + fr.coreml.bytes) {
        tts_cpp::parler::FitOptions tight = fit;
        tight.coreml.memory_budget_bytes = (int64_t) 1 << 50;
        tight.margin_bytes = fr.device_free_bytes - ggml_no_margin - fr.coreml.bytes / 2;
        const tts_cpp::FitResult t = tts_cpp::parler::fit_params(tight);
        check(t.coreml.disabled_reason == "memory" && t.status == tts_cpp::FitStatus::Success &&
                  t.coreml.ggml_only_fits && !t.coreml.fits,
              "near the free memory, huge budget: '" + t.coreml.disabled_reason + "', " + t.reason);
    } else {
        check(false, "free memory too small for the near-free-memory memory-degrade case");
    }

    tts_cpp::parler::EngineOptions eo;
    eo.model_gguf_path = gguf;
    eo.default_description = "A calm voice.";
    eo.max_frames = 40;
    eo.stream_chunk_frames = 12;
    eo.coreml = opts;
    {
        tts_cpp::parler::Engine engine(eo);
        const auto whole = engine.synthesize("a b");
        const auto streamed =
            engine.synthesize("a b", eo.default_description, [](const float *, std::size_t, int, bool) {});
        check(engine.dac_on_coreml() && whole.dac_backend.rfind("coreml-", 0) == 0 &&
                  streamed.dac_backend.rfind("coreml-", 0) == 0 && !whole.coreml_pending &&
                  !streamed.coreml_pending,
              "Engine: dac_backend " + whole.dac_backend + " / " + streamed.dac_backend);
    }
    {
        tts_cpp::parler::EngineOptions low = eo;
        low.coreml.use = tts_cpp::CoremlOptions::Use::automatic;
        low.coreml.memory_budget_bytes = near.coreml.memory_budget_bytes;
        tts_cpp::parler::Engine engine(low);
        const std::string st = engine.coreml_status();
        const auto whole = engine.synthesize("a b");
        check(contains(st, "\"state\":\"disabled\"") && whole.dac_backend == "ggml",
              "Engine under a near budget: DAC on ggml (" + whole.dac_backend + "), " +
                  st.substr(0, std::min<size_t>(st.size(), 200)));
    }
    {
        // A fit assessed with a larger headroom than the default: the budget
        // holds ggml + Core ML with the default 256 MiB margin, but only ggml
        // with this one. The Engine, given the same workload and margin, must
        // reach the same verdict (memory-degrade) as that fit.
        tts_cpp::parler::FitOptions wide = fit;
        wide.margin_bytes = fit.margin_bytes + fr.coreml.bytes;
        wide.coreml.memory_budget_bytes = (int64_t) (ggml_need + fr.coreml.bytes + (64ull << 20));
        const tts_cpp::FitResult assessed = tts_cpp::parler::fit_params(wide);
        tts_cpp::parler::FitOptions as_default = wide;
        as_default.margin_bytes = fit.margin_bytes;
        const tts_cpp::FitResult by_default = tts_cpp::parler::fit_params(as_default);
        check(assessed.coreml.disabled_reason == "memory" && by_default.coreml.disabled_reason.empty(),
              "a budget only the larger margin degrades: '" + assessed.coreml.disabled_reason + "' with it, '" +
                  by_default.coreml.disabled_reason + "' with the default");
        tts_cpp::parler::EngineOptions low = eo;
        low.coreml.use = tts_cpp::CoremlOptions::Use::automatic;
        low.coreml.memory_budget_bytes = wide.coreml.memory_budget_bytes;
        low.fit_margin_bytes = wide.margin_bytes;
        tts_cpp::parler::Engine engine(low);
        const std::string st = engine.coreml_status();
        const auto whole = engine.synthesize("a b");
        check(contains(st, "\"state\":\"disabled\"") && whole.dac_backend == "ggml",
              "Engine with the fit's nondefault margin: memory-degrade as that fit (DAC on " + whole.dac_backend +
                  "), " + st.substr(0, std::min<size_t>(st.size(), 200)));
    }
    model.dac_pending.reset();
}

double mib(uint64_t bytes) { return bytes / 1048576.0; }

// ggml decodes run in a loop on this thread while the DAC sidecar is
// generated, compiled, loaded and warmed into `cache_dir` in the background,
// until five calls have run on Core ML. One sampler thread reads the
// footprint every 2 ms and the slot's state every 10 ms; another reads
// ANECompilerService every 20 ms. Every phase must stay within the template
// constant that charges it (memory_schedule_gates).
void memory_schedule(parler_model & model, const std::string & gguf, const std::string & cache_dir) {
    fprintf(stderr, "memory schedule (cache %s, host %s)\n", cache_dir.c_str(), host_chip().c_str());
    tts_cpp::CoremlOptions opts;
    opts.cache_dir = cache_dir;
    const parler_dac_route route = parler_dac_plan_route(gguf, opts, parler_dac_rvq_bytes(model));
    if (route.route != "generate") {
        check(false, "route generate (" + route.route + " " + route.state + " " + route.reason + ")");
        return;
    }
    const parler_hparams & hp = model.hparams;
    const int n_frames = 128;
    const std::vector<int32_t> codes = random_codes(hp, n_frames, 7);
    model.dac_sidecar.reset();
    model.dac_pending.reset();
    std::vector<float> pcm;
    for (int i = 0; i < 2; ++i) {  // warm the ggml arena
        std::string warm_backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &warm_backend);
        if (classify_call(ok, warm_backend, "") != schedule_call::ggml) {
            check(false, "memory schedule: the ggml warm-up decode succeeds (" + warm_backend + ")");
            return;
        }
    }
    uint64_t base = 0;
    const bool base_ok = qvac_coreml_memory::phys_footprint(base);
    size_t footprint_failures = 0;
    std::atomic<bool> stop{false};
    std::atomic<int> phase{(int) mem_phase::ggml};
    std::atomic<const parler_dac_coreml *> watched{nullptr};
    std::vector<qvac_coreml_memory::AneSample> ane_samples;
    std::vector<mem_sample> samples;
    samples.reserve(1 << 16);
    std::thread sampler([&] {
        for (int tick = 0; !stop.load(); ++tick) {
            const parler_dac_coreml * d = watched.load();
            if (d != nullptr && tick % 5 == 0) phase.store((int) mem_phase_of(status_state(d->status_json())));
            uint64_t fp = 0;
            if (qvac_coreml_memory::phys_footprint(fp)) {
                samples.push_back({fp, (mem_phase) phase.load()});
            } else {
                ++footprint_failures;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    // Each compiler query is bracketed by its own reads of the slot state
    // (not the footprint sampler's cached phase): a query that crossed the
    // load's start or end is dropped (counted, reported).
    size_t ane_crossing = 0;
    std::thread ane_sampler([&] {
        const auto slot_phase = [&] {
            const parler_dac_coreml * d = watched.load();
            return d ? mem_phase_of(status_state(d->status_json())) : mem_phase::ggml;
        };
        while (!stop.load()) {
            const mem_phase before = slot_phase();
            uint64_t a = 0;
            const bool ok = qvac_coreml_memory::ane_compiler_rss(a);
            const std::optional<bool> loading = ane_query_loading(before, slot_phase());
            if (loading) {
                ane_samples.push_back({a, *loading, ok});
            } else {
                ++ane_crossing;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    auto dac = parler_dac_coreml::create(model, gguf, opts);
    watched.store(dac.get());
    int ggml_calls = 0, coreml_calls = 0, bad_calls = 0;
    std::string backend, bad_backend;
    const std::string coreml_label = label_for_units(route.compute_units);
    while (std::chrono::steady_clock::now() - t0 < std::chrono::minutes(5)) {
        backend.clear();
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        const schedule_call k = classify_call(ok, backend, coreml_label);
        if (k == schedule_call::bad) {
            ++bad_calls;
            bad_backend = (ok ? "" : "failed on ") + backend;
            break;
        }
        if (k == schedule_call::ggml) {
            ++ggml_calls;
        } else if (++coreml_calls >= 5) {
            break;
        }
    }
    if (bad_calls > 0) check(false, "memory schedule: every overlapped decode succeeds on ggml or " + coreml_label +
                                        " (" + bad_backend + ")");
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop.store(true);
    sampler.join();
    ane_sampler.join();
    fprintf(stderr, "  status: %s\n", dac->status_json().c_str());

    const mem_breakdown b = mem_breakdown_of(samples, base);
    const qvac_coreml_memory::AneGrowth ane = qvac_coreml_memory::ane_compiler_growth(ane_samples);
    const qvac_coreml_memory::AneBaseline baseline = qvac_coreml_memory::ane_compiler_baseline(ane);
    const char * baseline_name = qvac_coreml_memory::ane_baseline_name(baseline);
    char before_load[96];
    if (baseline == qvac_coreml_memory::AneBaseline::unknown) {
        std::snprintf(before_load, sizeof(before_load), "no sample before the load, an unknown baseline");
    } else {
        std::snprintf(before_load, sizeof(before_load), "before the load %.1f MiB, a %s baseline", mib(ane.before),
                      baseline_name);
    }
    uint64_t ane_peak = 0;
    for (const auto & a : ane_samples) ane_peak = std::max(ane_peak, a.rss);
    const mem_charges c = charges_of(route, parler_dac_rvq_bytes(model));
    // The prediction workspace this run implies: the steady state above the
    // template's other resident charges.
    const uint64_t other_resident = c.resident - route.workspace_bytes;
    const uint64_t workspace = b.steady > other_resident ? b.steady - other_resident : 0;
    fprintf(stderr,
            "  memory (MiB; each phase over the footprint just before it began, steady over the footprint just "
            "before the load, whole run over the warm ggml baseline %.1f; a phase needs x%.2f its measurement):\n"
            "    prepare    %7.1f  needs %7.1f  charged %7.1f  (generate %.1f, compile %.1f)\n"
            "    load+warm  %7.1f  needs %7.1f  charged %7.1f  (load %.1f, warm-up %.1f)\n"
            "    steady     %7.1f  needs %7.1f  charged %7.1f  (workspace %.1f, charged %.1f; over the baseline %.1f)\n"
            "    whole run  %7.1f  needs %7.1f  charged %7.1f  (the envelope: margined phases)\n"
            "    ANECompilerService size while loading and warming %.1f, needs %.1f, reserve %.1f (growth %.1f, %s; "
            "whole run %.1f)\n",
            mib(base), MEASURED_MARGIN, mib(b.prepare), MEASURED_MARGIN * mib(b.prepare), mib(c.prepare),
            mib(b.generate), mib(b.compile), mib(b.load_warm), MEASURED_MARGIN * mib(b.load_warm), mib(c.load),
            mib(b.load), mib(b.warm), mib(b.steady), MEASURED_MARGIN * mib(b.steady), mib(c.resident),
            mib(workspace), mib(route.workspace_bytes), mib(b.steady_over_base), mib(b.total), mib(b.total),
            mib(c.envelope), mib(ane.peak), MEASURED_MARGIN * mib(ane.peak), mib(c.ane_reserve), mib(ane.growth),
            before_load, mib(ane_peak));
    if (baseline == qvac_coreml_memory::AneBaseline::dirty) {
        fprintf(stderr,
                "    note: ANECompilerService held %.1f MiB before the load, so it reused memory of earlier compiles "
                "and its growth understates a cold compile on a clean service\n",
                mib(ane.before));
    } else if (baseline == qvac_coreml_memory::AneBaseline::unknown) {
        fprintf(stderr,
                "    note: no ANECompilerService sample preceded the load, so its growth is its whole size while "
                "loading\n");
    }
    fprintf(stderr,
            "  memory report: {\"chip\":\"%s\",\"units\":\"%s\",\"base\":%llu,\"prepare\":%llu,\"generate\":%llu,"
            "\"compile\":%llu,\"load_warm\":%llu,\"load\":%llu,\"warm\":%llu,\"steady\":%llu,\"steady_over_base\":%llu,\"workspace\":%llu,"
            "\"total\":%llu,\"ane_compiler_load_peak\":%llu,\"ane_compiler_before_load\":%llu,"
            "\"ane_compiler_growth\":%llu,\"ane_compiler_peak\":%llu,\"ane_compiler_baseline\":\"%s\",\"samples\":%zu,"
            "\"ane_compiler_loading_samples\":%zu,\"ane_compiler_crossing\":%zu}\n",
            host_chip().c_str(), route.compute_units.c_str(), (unsigned long long) base,
            (unsigned long long) b.prepare, (unsigned long long) b.generate, (unsigned long long) b.compile,
            (unsigned long long) b.load_warm, (unsigned long long) b.load, (unsigned long long) b.warm,
            (unsigned long long) b.steady, (unsigned long long) b.steady_over_base, (unsigned long long) workspace, (unsigned long long) b.total,
            (unsigned long long) ane.peak, (unsigned long long) ane.before, (unsigned long long) ane.growth,
            (unsigned long long) ane_peak, baseline_name, samples.size(), ane.loading_samples, ane_crossing);
    const bool ane_units = route.compute_units == "all" || route.compute_units == "cpu_and_ane";
    std::vector<std::string> failures = probe_failures(base_ok, footprint_failures, ane_units, ane);
    for (const std::string & f : mem_failures(b, ane.peak, c)) failures.push_back(f);
    for (const std::string & f : failures) check(false, "memory schedule: " + f);
    char line[512];
    std::snprintf(line, sizeof(line),
                  "overlapped schedule: %d ggml + %d Core ML calls in %.1f s; footprint base %.1f MiB, peak +%.1f MiB "
                  "(envelope %.1f MiB, units %s); ANECompilerService %.1f MiB while loading, grew %.1f MiB (%s)",
                  ggml_calls, coreml_calls, secs, mib(base), mib(b.total), mib(route.envelope_bytes),
                  route.compute_units.c_str(), mib(ane.peak), mib(ane.growth), before_load);
    check(coreml_calls >= 5 && ggml_calls >= 1 && bad_calls == 0 && failures.empty(), line);
    model.dac_pending.reset();
}
#endif

void fallback_gates(const std::string & gguf) {
    fprintf(stderr, "short-window sidecar gates (%s)\n", parler_dac_sidecar_path(gguf).c_str());
    parler_model model;
    std::string err;
    if (!parler_load_gguf(gguf, model, 0, &err)) {
        check(false, "load " + gguf + ": " + err);
        return;
    }
    check(!parler_attach_dac_sidecar(model, gguf) && !model.dac_sidecar,
          "a window no wider than its context is refused");
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    bool threw = false;
    try {
        parler_attach_dac_sidecar(model, gguf);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    unsetenv(PARLER_COREML_STRICT_ENV);
    check(threw, "strict: refused sidecar throws");
    parler_free_model(model);
}

} // namespace

#if defined(__APPLE__)
// The GGUF at a path is swapped while parler_load_gguf runs: B replaces A
// once the metadata is parsed, and A is back before the load ends (A/B/A).
// The load reads through one descriptor, so it must hold A's weights, and
// its recorded identity must be A's.
struct swap_paths {
    std::filesystem::path path, held, other;
};
swap_paths g_swap;
int g_swap_stages = 0;  // stages the hook ran at (both must, or nothing was swapped)

void swap_during_load(const char * stage) {
    std::error_code ec;
    ++g_swap_stages;
    if (std::string(stage) == "metadata") {
        std::filesystem::rename(g_swap.path, g_swap.held, ec);
        if (!ec) std::filesystem::rename(g_swap.other, g_swap.path, ec);
    } else if (std::string(stage) == "weights") {
        std::filesystem::rename(g_swap.path, g_swap.other, ec);
        if (!ec) std::filesystem::rename(g_swap.held, g_swap.path, ec);
    }
    if (ec) fprintf(stderr, "  swap at %s: %s\n", stage, ec.message().c_str());
}

// The file offset of `tensor`'s data in the GGUF at `path`, or -1.
int64_t gguf_tensor_offset(const std::string & path, const char * tensor) {
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return -1;
    const int64_t i = gguf_find_tensor(g, tensor);
    const int64_t off = i < 0 ? -1 : (int64_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i));
    gguf_free(g);
    return off;
}

void descriptor_load_gate(const std::string & gguf) {
    fprintf(stderr, "GGUF swapped during the load (A/B/A)\n");
    namespace fs = std::filesystem;
    const char * TENSOR = "dac.dec.conv_out.bias";
    const fs::path dir = fs::temp_directory_path() /
                         ("parler-aba-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    g_swap = {dir / "model.gguf", dir / "held.gguf", dir / "other.gguf"};
    const std::string src = fs::absolute(gguf).string();
    const int64_t off = gguf_tensor_offset(src, TENSOR);
    float bias_a = 0;
    bool ready = off >= 0 && ::clonefile(src.c_str(), g_swap.path.c_str(), 0) == 0 &&
                 ::clonefile(src.c_str(), g_swap.other.c_str(), 0) == 0;
    if (ready) {
        const int fa = ::open(g_swap.path.c_str(), O_RDONLY);
        const int fb = ::open(g_swap.other.c_str(), O_RDWR);
        const float bias_b_value = 0.25f;
        ready = fa >= 0 && fb >= 0 && ::pread(fa, &bias_a, 4, off) == 4 && bias_a != bias_b_value &&
                ::pwrite(fb, &bias_b_value, 4, off) == 4;
        if (fa >= 0) ::close(fa);
        if (fb >= 0) ::close(fb);
    }
    check(ready, "two GGUFs differing only in conv_out's bias");
    if (!ready) {
        fs::remove_all(dir, ec);
        return;
    }
    struct stat sa {};
    ::stat(g_swap.path.c_str(), &sa);
    // On the CPU every weight is mapped; with the GPU every weight is
    // streamed into a device buffer. Both readers must use the descriptor.
    for (const int ngl : {0, 99}) {
        const std::string path_kind = ngl == 0 ? "CPU, mapped weights" : "GPU, streamed weights";
        parler_model m;
        std::string err;
        g_swap_stages = 0;
        parler_gguf_load_hook = &swap_during_load;
        const bool loaded = parler_load_gguf(g_swap.path.string(), m, ngl, &err);
        parler_gguf_load_hook = nullptr;
        const std::string backend = loaded && m.backend ? ggml_backend_name(m.backend) : "";
        if (ngl > 0 && loaded && backend.rfind("CPU", 0) == 0) {
            fprintf(stderr, "  %s: not run (backend %s)\n", path_kind.c_str(), backend.c_str());
            parler_free_model(m);
            continue;
        }
        struct stat now {};
        const bool a_back = ::stat(g_swap.path.c_str(), &now) == 0 && now.st_ino == sa.st_ino &&
                            fs::exists(g_swap.other) && !fs::exists(g_swap.held);
        float got = NAN;
        if (loaded && m.dac_conv_out_b) ggml_backend_tensor_get(m.dac_conv_out_b, &got, 0, sizeof(float));
        check(loaded && g_swap_stages == 2 && a_back && got == bias_a,
              path_kind + " (" + backend + "): the load reads A's weights although B held the path in between "
              "(conv_out bias " + std::to_string(got) + ", A's " + std::to_string(bias_a) + "; hook stages " +
                  std::to_string(g_swap_stages) + " of 2" + (a_back ? "" : "; the swap did not restore A") + ")" +
                  (loaded ? "" : ": " + err));
#ifdef TTS_CPP_USE_COREML
        const bool identity = m.dac_source && m.dac_source->ok && m.dac_source->stat.ino == (uint64_t) sa.st_ino &&
                              !m.dac_source->stat.changed;
        check(identity, path_kind + ": the recorded identity is the file the load read (A's inode)");
#endif
        parler_free_model(m);
    }
    fs::remove_all(dir, ec);
}
#endif

int main(int argc, char ** argv) {
    std::string gguf, fallback, ref_dir, cache_dir, memory_dir;
    double min_snr = 45.0;
    double min_speech_snr = 45.0;
    bool require_coreml = false;
    bool sweep = false;
    bool self_test = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--self-test") self_test = true;
        else if (a == "--gguf") gguf = next();
        else if (a == "--fallback-gguf") fallback = next();
        else if (a == "--ref-dir") ref_dir = next();
        else if (a == "--min-snr") min_snr = std::atof(next().c_str());
        else if (a == "--min-speech-snr") min_speech_snr = std::atof(next().c_str());
        else if (a == "--cache-dir") cache_dir = next();
        else if (a == "--memory-schedule") memory_dir = next();
        else if (a == "--require-coreml") require_coreml = true;
        else if (a == "--units-sweep") sweep = true;
    }
    if (self_test) {
        model_free_gates();
        fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
        return g_failures ? 1 : 0;
    }
    if (gguf.empty() || !std::ifstream(gguf).good()) {
        fprintf(stderr, "SKIP: no --gguf model (%s)\n", gguf.c_str());
        return 77;
    }
    unsetenv(PARLER_COREML_STRICT_ENV);
    unsetenv(PARLER_COREML_DISABLE_ENV);

    parler_model model;
    std::string err;
    if (!parler_load_gguf(gguf, model, 0, &err)) {
        fprintf(stderr, "FAIL: load %s: %s\n", gguf.c_str(), err.c_str());
        return 1;
    }
    model_free_gates();
    fake_sidecar_gates(model);
    pending_seam_gates(model);
#if defined(__APPLE__)
    if (memory_dir.empty()) descriptor_load_gate(gguf);
#endif
#ifdef TTS_CPP_USE_COREML
    idle_slot_gates(model, gguf);
    if (!memory_dir.empty()) memory_schedule(model, gguf, memory_dir);
    if (!cache_dir.empty()) {
        generated_gates(model, gguf, cache_dir, ref_dir, min_snr, min_speech_snr);
        if (sweep) units_sweep(model, cache_dir, min_snr);
    } else if (::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf))) {
        coreml_gates(model, gguf, ref_dir, min_snr, min_speech_snr);
        engine_gates(gguf);
    } else if (memory_dir.empty()) {
        fprintf(stderr, "no sidecar next to %s: Core ML gates skipped\n", gguf.c_str());
        check(!require_coreml, "--require-coreml: sidecar present");
    }
    if (!fallback.empty()) fallback_gates(fallback);
#else
    (void) ref_dir;
    (void) min_snr;
    (void) min_speech_snr;
    (void) cache_dir;
    (void) memory_dir;
    (void) fallback;
    (void) sweep;
    fprintf(stderr, "built without TTS_CPP_COREML: Core ML gates skipped\n");
    check(!require_coreml, "--require-coreml: built with TTS_CPP_COREML");
#endif
    parler_free_model(model);
    fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
    return g_failures ? 1 : 0;
}
