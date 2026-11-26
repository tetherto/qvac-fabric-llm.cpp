// Memory-fit preflight (include/parakeet/fit.h): project the model + workload
// against the device memory available right now, without reading weight data.
//
// The projection mirrors the real runtime path by construction rather than by
// formula: it drives the same loader (metadata-only), the same graph builders,
// and the same allocation policies through ggml's size-only APIs
// (ggml_backend_alloc_ctx_tensors_from_buft_size / ggml_gallocr_reserve_n_size),
// so it tracks runtime changes instead of drifting from them.

#include "parakeet/fit.h"

#include "backend_util.h"
#include "coreml_fit.h"
#include "fit_util.h"
#include "long_form.h"
#include "parakeet_coreml_path.h"
#include "parakeet_ctc.h"
#include "parakeet_diarization_v3.h"
#include "parakeet_eou.h"
#include "parakeet_log.h"
#include "parakeet_sortformer.h"
#include "parakeet_tdt.h"
#ifdef PARAKEET_USE_COREML
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#include "coreml/coreml_diarizers.h"
#include "coreml/parakeet-encoder.h"
#include "coreml/parakeet_coreml_shape.h"
#include "coreml/parakeet_coreml_sidecars.h"
#endif

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <system_error>
#include <string>
#include <vector>

namespace parakeet {

const char * fit_status_name(FitStatus status) {
    switch (status) {
        case FitStatus::Success: return "success";
        case FitStatus::Failure: return "failure";
        case FitStatus::Error:   return "error";
    }
    return "error";
}

namespace {

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

// Post-subsampling encoder frame count: the same three stride-2 conv
// recurrence run_encoder / coreml_encoder_out_frames use to size the graph.
int encoder_out_frames(const EncoderConfig & enc, long long n_mel_frames) {
    auto next = [&](long long len) {
        // _conv_out_len(len, k=3, s=2, p=1) == (len - 1) / 2 + 1
        return enc.causal_downsampling ? (len / 2 + 1) : ((len - 1) / 2 + 1);
    };
    long long t = next(next(next(n_mel_frames)));
    if (t < 1) t = 1;
    if (t > std::numeric_limits<int>::max()) t = std::numeric_limits<int>::max();
    return (int) t;
}

using fitutil::sat_add;
using fitutil::sat_mul;
using fitutil::sat_u64_from_double;

constexpr size_t kReportLineBytes = 256;
constexpr uint64_t kNemotronDiarizationSampleCopies = 2;

// On a unified-memory device (CPU, integrated GPU, Apple Metal) the "device"
// buffers and the host extras compete for the same physical RAM, so both
// count against the free figure. Saturating arithmetic: an overflow must
// surface as DOES-NOT-FIT, never wrap into a false FITS.
uint64_t apply_fit_verdict(FitResult & r, uint64_t margin_bytes) {
    uint64_t required = sat_add(r.device.total_bytes, margin_bytes);
    if (r.device_shares_host_memory) {
        required = sat_add(required, sat_add(r.host_bytes, r.coreml_bytes));
    }
    r.fits   = required <= r.device_free_bytes;
    r.status = r.fits ? FitStatus::Success : FitStatus::Failure;
    r.reason = r.fits ? "fits" : "does-not-fit";
    return required;
}

std::string model_device_report(const FitResult & r) {
    std::string s;
    char line[kReportLineBytes];
    std::snprintf(line, sizeof(line), "model:    %s%s%s%s\n",
                  r.model_type.c_str(),
                  r.model_variant.empty() ? "" : " (",
                  r.model_variant.c_str(),
                  r.model_variant.empty() ? "" : ")");
    s += line;
    std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s\n",
                  r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                  fmt_mib(r.device_free_bytes).c_str(),
                  fmt_mib(r.device_total_bytes).c_str());
    s += line;
    return s;
}

std::string device_projection_report(const FitResult & r, uint64_t margin_bytes,
                                     uint64_t required, const std::string & compute_detail) {
    std::string s = "device projection:\n";
    char line[kReportLineBytes];
    std::snprintf(line, sizeof(line), "  weights:         %14s\n",
                  fmt_mib(r.device.weights_bytes).c_str());
    s += line;
    std::snprintf(line, sizeof(line), "  encoder compute: %14s (%s)\n",
                  fmt_mib(r.device.encoder_compute_bytes).c_str(), compute_detail.c_str());
    s += line;
    std::snprintf(line, sizeof(line), "  decoder state:   %14s\n",
                  fmt_mib(r.device.decoder_state_bytes).c_str());
    s += line;
    std::snprintf(line, sizeof(line), "  decoder compute: %14s\n",
                  fmt_mib(r.device.decoder_compute_bytes).c_str());
    s += line;
    std::snprintf(line, sizeof(line), "  total:           %14s\n",
                  fmt_mib(r.device.total_bytes).c_str());
    s += line;
    std::snprintf(line, sizeof(line), "host extras:       %14s%s\n",
                  fmt_mib(r.host_bytes).c_str(),
                  r.device_shares_host_memory
                      ? " (same RAM pool as the device projection)" : "");
    s += line;
    if (r.coreml_bytes > 0) {
        std::snprintf(line, sizeof(line), "core ml sidecars:  %14s%s\n",
                      fmt_mib(r.coreml_bytes).c_str(),
                      r.device_shares_host_memory
                          ? " (same RAM pool as the device projection)" : "");
        s += line;
    }
    std::snprintf(line, sizeof(line), "margin:            %14s\n",
                  fmt_mib(margin_bytes).c_str());
    s += line;
    if (r.fits) {
        std::snprintf(line, sizeof(line), "verdict: FITS (headroom %s)\n",
                      fmt_mib(r.device_free_bytes - required).c_str());
    } else {
        std::snprintf(line, sizeof(line), "verdict: DOES NOT FIT (short by %s)\n",
                      fmt_mib(required - r.device_free_bytes).c_str());
    }
    s += line;
    return s;
}

// Clamp through double before the integer conversion: a float audio_seconds
// large enough to overflow long long must saturate (and then surface as
// workload-too-large / DOES-NOT-FIT), not hit UB.
long long total_mel_frames(float audio_seconds, const MelConfig & mel) {
    return (long long) std::min<uint64_t>(
        std::max<uint64_t>(1, sat_u64_from_double(
            std::ceil((double) audio_seconds *
                      (double) mel.sample_rate / (double) mel.hop_length))),
        (uint64_t) std::numeric_limits<long long>::max());
}

uint64_t f32_bytes(uint64_t values) {
    return sat_mul(values, sizeof(float));
}

uint64_t nemotron_diarization_host_bytes(const ParakeetCtcModel & model,
                                         const FitOptions & opts, long long total_mel,
                                         int offline_frames, uint64_t sched_input_bytes) {
    const auto & cfg = model.nemotron_diarization_cfg;
    const uint64_t samples = sat_u64_from_double(
        std::ceil((double) opts.audio_seconds * model.mel_cfg.sample_rate));
    const uint64_t stacked_frames = sat_mul((uint64_t) offline_frames, (uint64_t) cfg.subsampling_factor);
    uint64_t host = sat_mul(f32_bytes(samples), kNemotronDiarizationSampleCopies);
    host = sat_add(host, f32_bytes(sat_mul((uint64_t) total_mel, (uint64_t) model.mel_cfg.n_mels)));
    host = sat_add(host, f32_bytes(sat_mul(stacked_frames, (uint64_t) model.mel_cfg.n_mels)));
    host = sat_add(host, f32_bytes(sat_mul(stacked_frames, (uint64_t) cfg.speakers)));
    host = sat_add(host, f32_bytes(sat_mul((uint64_t) offline_frames, (uint64_t) cfg.encoder_width)));
    return sat_add(host, sched_input_bytes);
}

#ifdef PARAKEET_USE_COREML
CoremlFitUnits fit_units(pcm::ComputeUnits units) {
    switch (units) {
        case pcm::ComputeUnits::all: return CoremlFitUnits::all;
        case pcm::ComputeUnits::cpu_and_ane: return CoremlFitUnits::cpu_and_ane;
        case pcm::ComputeUnits::cpu_and_gpu: return CoremlFitUnits::cpu_and_gpu;
        case pcm::ComputeUnits::cpu_only: return CoremlFitUnits::cpu_only;
    }
    return CoremlFitUnits::all;
}

uint64_t bundle_weight_bytes(const std::string & bundle) {
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(std::filesystem::path(bundle) / "weights" / "weight.bin", ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

bool bundle_present(const std::string & bundle) {
    std::error_code ec;
    return std::filesystem::is_directory(bundle, ec);
}

#endif

// Today's diarizer charge (FitResult::coreml_bytes, part of the top-level
// verdict): the bundles the workload's calls can load, resident. Zero when
// Core ML is off for this projection (use = off, a disable variable).
uint64_t legacy_diarizer_bytes(const std::string & gguf_path, const std::vector<CoremlFrameRange> & ranges,
                               const FitOptions & opts) {
#ifdef PARAKEET_USE_COREML
    if (coreml_diarizers_disabled() || !resolve_parakeet_coreml(opts.coreml).enabled()) return 0;
    return coreml_diarizer_resident_bytes(list_coreml_diarizer_bundles(gguf_path), ranges);
#else
    (void) gguf_path;
    (void) ranges;
    (void) opts;
    return 0;
#endif
}

// A diarizer model's Core ML charge (Nemotron 3 Diarization, Sortformer
// v2.1): every capacity of the model in one ordered list (adjacent prebuilt
// bundles, and the template capacities the engine would generate, chosen by
// the same diarizer_templates_for() + diarizer_capacities() + plan() as
// discovery; the others status-only), charged by coreml_diarizers_charge for
// the calls in `ranges` and the construction prewarm. Never reads the cache's
// contents.
CoremlDiarizersCharge coreml_model_diarizers_charge(const ParakeetCtcModel & model, const FitOptions & opts,
                                                    const std::vector<CoremlFrameRange> & ranges) {
#ifdef PARAKEET_USE_COREML
    const auto resolved = resolve_parakeet_coreml(opts.coreml);
    if (coreml_diarizers_disabled() || !resolved.enabled()) return {};
    const std::vector<CoremlDiarizerTemplate> & templates =
        diarizer_templates_for(model.model_type, model.model_variant);
    CoremlSidecars sidecars(opts.coreml);
    std::vector<int> template_frames;
    std::vector<std::string> template_ids;
    for (const CoremlDiarizerTemplate & t : templates) {
        template_frames.push_back(t.frames);
        template_ids.emplace_back(t.template_id);
    }
    const std::vector<CoremlDiarizerBundle> bundles = list_coreml_diarizer_bundles(opts.model_gguf_path);
    std::vector<int> prebuilt;
    for (const CoremlDiarizerBundle & b : bundles) prebuilt.push_back(b.capacity);
    // Without generation every template capacity is status-only (the probe
    // would be wasted).
    std::vector<pcm::generation::Candidate> probed;
    if (sidecars.generation_status().supported) {
        probed = probe_diarizer_templates(sidecars, opts.model_gguf_path, template_ids);
    } else {
        probed.resize(template_ids.size());
        for (size_t i = 0; i < template_ids.size(); ++i) probed[i].template_id = template_ids[i];
    }
    std::vector<CoremlDiarizerCharge> charges = diarizer_fit_charges(
        sidecars, opts.model_gguf_path, diarizer_capacities(prebuilt, template_frames, probed));
    for (CoremlDiarizerCharge & c : charges) {
        for (const CoremlDiarizerBundle & b : bundles) {
            if (c.prebuilt && b.capacity == c.frames) c.weights_bytes = b.weights_bytes;
        }
    }
    const CoremlDiarizerPrewarm prewarm =
        resolved.prewarm == pcm::Prewarm::all ? CoremlDiarizerPrewarm::all
        : resolved.prewarm == pcm::Prewarm::none ? CoremlDiarizerPrewarm::none
                                                 : CoremlDiarizerPrewarm::minimal;
    return coreml_diarizers_charge(charges, ranges, prewarm,
                                   fit_units(resolved.units_or(pcm::ComputeUnits::cpu_and_ane)));
#else
    (void) model;
    (void) opts;
    (void) ranges;
    return {};
#endif
}

// Core ML enablement for the projection: resolved from FitOptions::coreml
// exactly as an Engine would (PARAKEET_COREML_DISABLE, use = off, ...).
bool coreml_fit_enabled(const FitOptions & opts, std::string & disabled_reason) {
#ifdef PARAKEET_USE_COREML
    const auto resolved = resolve_parakeet_coreml(opts.coreml);
    disabled_reason = resolved.enabled() ? std::string() : resolved.disabled_reason;
    return resolved.enabled();
#else
    (void) opts;
    disabled_reason = "build";
    return false;
#endif
}

uint64_t gguf_encoder_parameters(const ParakeetCtcModel & model) {
    uint64_t n = 0;
    ggml_context * ctx = model.weights_ctx();
    if (ctx == nullptr) return 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (std::strncmp(ggml_get_name(t), "encoder.", 8) == 0) n = sat_add(n, (uint64_t) ggml_nelements(t));
    }
    return n;
}


// The Core ML encoder programs a load of this model keeps resident, found
// with the loader's own sidecar discovery and compute-unit placement: the
// adjacent encoder sidecar (and Sortformer v2.1's bypass encoder); for
// TranscribePsy EU the shared multifunction bundle or the separate route
// programs, plus the on-demand flexible or precise programs. Empty with no
// route.
std::vector<CoremlFitProgram> coreml_encoder_fit_programs(const ParakeetCtcModel & model,
                                                          const FitOptions & opts, bool enabled,
                                                          long long window_mel, int window_enc) {
    std::vector<CoremlFitProgram> programs;
#ifdef PARAKEET_USE_COREML
    if (!enabled) return programs;
    const bool sortformer_v21 = model.model_type == ParakeetModelType::SORTFORMER &&
                                model.model_variant == "sortformer-streaming-v2.1-aosc";
    const bool supported =
        model.model_type == ParakeetModelType::CTC || model.model_type == ParakeetModelType::RNNT ||
        model.model_type == ParakeetModelType::TDT || model.model_type == ParakeetModelType::EOU ||
        model.model_type == ParakeetModelType::NEMOTRON || sortformer_v21;
    if (!supported) return programs;
    const auto resolved = resolve_parakeet_coreml(opts.coreml);
    const uint64_t floor = sat_mul(gguf_encoder_parameters(model), (uint64_t) 2);
    const uint64_t io = f32_bytes(sat_add(sat_mul((uint64_t) std::max<long long>(window_mel, 0),
                                                  (uint64_t) model.mel_cfg.n_mels),
                                          sat_mul((uint64_t) std::max(window_enc, 0),
                                                  (uint64_t) model.encoder_cfg.d_model)));
    auto add = [&](const std::string & bundle, uint64_t runtime_floor, int functions, CoremlFitUnits units) {
        if (!bundle_present(bundle)) return false;
        CoremlFitProgram p;
        p.weight_file_bytes = bundle_weight_bytes(bundle);
        p.runtime_floor_bytes = runtime_floor;
        // A fixed-shape export allocates its declared I/O whatever the input
        // length; a flexible one is charged the worst window.
        p.io_bytes = std::max(io, parakeet_coreml_declared_io_bytes(bundle.c_str()));
        p.functions = functions;
        p.units = units;
        programs.push_back(p);
        return true;
    };
    // The slot loader's placement: the resolved units, else the export's own default.
    auto slot_units = [&](const std::string & bundle) {
        return fit_units(resolved.units_or(parakeet_coreml_encoder_prefers_ane(bundle.c_str())
                                               ? pcm::ComputeUnits::cpu_and_ane
                                               : pcm::ComputeUnits::all));
    };
    const std::string & gguf = opts.model_gguf_path;
    if (model.model_variant == "transcribepsy-250m-eu") {
        // The fixed programs load at construction: on the Neural Engine, or on
        // the CPU when the units exclude it (then only the English one). An
        // owner load (no ggml encoder weights, or PARAKEET_COREML_ONLY) loads
        // all three whatever the units.
        const char * units = loader_units(resolved);
        const bool owner = gguf_encoder_parameters(model) == 0 || std::getenv("PARAKEET_COREML_ONLY") != nullptr;
        const bool auxiliaries = coreml_eu_loads_fixed_auxiliaries(units);
        const int fixed_functions = auxiliaries || owner ? 3 : 1;
        const CoremlFitUnits fixed = auxiliaries ? CoremlFitUnits::cpu_and_ane : CoremlFitUnits::cpu_only;
        // A shared bundle that fails its checks falls back to the separate
        // programs, so with both present the larger set is charged.
        std::vector<CoremlFitProgram> shared;
        std::swap(shared, programs);
        add(coreml_eu_shared_sidecar_path(gguf), floor, fixed_functions, fixed);
        std::swap(shared, programs);
        add(coreml_encoder_sidecar_path(gguf), floor, 1, fixed);
        if (fixed_functions == 3) {
            add(coreml_eu_other_sidecar_path(gguf), floor, 1, fixed);
            add(coreml_eu_gate_sidecar_path(gguf), 0, 1, fixed);
        }
        if (!shared.empty() && coreml_programs_charge(shared).resident >= coreml_programs_charge(programs).resident) {
            programs = shared;
        }
        // On demand, kept until replaced: one main program (a flexible CPU
        // program replaces the other one and any precise one, and a precise GPU
        // program replaces the flexible ones), the largest reachable; and the
        // flexible gate, which the precise streaming path can keep beside the
        // fixed gate.
        std::vector<CoremlFitProgram> mains;
        std::swap(mains, programs);
        add(coreml_eu_flexible_sidecar_path(gguf, false), floor, 1, CoremlFitUnits::cpu_only);
        add(coreml_eu_flexible_sidecar_path(gguf, true), floor, 1, CoremlFitUnits::cpu_only);
#if defined(__APPLE__) && TARGET_OS_IPHONE
        constexpr bool kMobilePrecise = true;
#else
        constexpr bool kMobilePrecise = false;
#endif
        if (coreml_eu_allows_gpu_precision(units, kMobilePrecise)) {
            for (int capacity : {1500, 6008}) {
                for (bool adapters : {false, true}) {
                    add(coreml_eu_precise_sidecar_path(gguf, adapters, capacity), floor, 1, CoremlFitUnits::cpu_and_gpu);
                }
            }
        }
        std::swap(mains, programs);
        const CoremlFitProgram * largest = nullptr;
        for (const CoremlFitProgram & p : mains) {
            if (largest == nullptr || coreml_programs_charge({p}).resident > coreml_programs_charge({*largest}).resident) {
                largest = &p;
            }
        }
        if (largest != nullptr) programs.push_back(*largest);
        std::string gate_flexible = coreml_eu_gate_sidecar_path(gguf);
        gate_flexible.insert(gate_flexible.size() - std::string(".mlmodelc").size(), "-flexible");
        add(gate_flexible, 0, 1, CoremlFitUnits::cpu_only);
        return programs;
    }
    const std::string encoder = coreml_encoder_sidecar_path(gguf);
    add(encoder, floor, 1, slot_units(encoder));
    if (sortformer_v21) {
        // The bypass bundle holds only the block stack, beside the encoder slot.
        const std::string bypass = coreml_bypass_encoder_sidecar_path(gguf);
        add(bypass, 0, 1, slot_units(bypass));
    }
#else
    (void) model;
    (void) opts;
    (void) enabled;
    (void) window_mel;
    (void) window_enc;
#endif
    return programs;
}

// The encoder sidecars' charge for the worst window (zero with no route).
CoremlCharge coreml_encoder_fit_charge(const ParakeetCtcModel & model, const FitOptions & opts,
                                       bool enabled, long long window_mel, int window_enc,
                                       bool & prebuilt) {
    const std::vector<CoremlFitProgram> programs =
        coreml_encoder_fit_programs(model, opts, enabled, window_mel, window_enc);
    prebuilt = !programs.empty();
    return prebuilt ? coreml_programs_charge(programs) : CoremlCharge{};
}

std::string coreml_fit_report(const FitResult & r) {
    char line[kReportLineBytes];
    std::snprintf(line, sizeof(line),
                  "coreml:   %s; encoder route %s; envelope %s; with Core ML %s, ggml only %s\n",
                  r.coreml.disabled_reason.empty() ? "enabled" : r.coreml.disabled_reason.c_str(),
                  r.coreml.encoder_route.c_str(), fmt_mib(r.coreml.bytes).c_str(),
                  r.coreml.fits ? "fits" : "does not fit", r.ggml_only_fits ? "fits" : "does not fit");
    return line;
}

struct DiarizationGraphShape {
    int mel_frames = 0;
    int state_frames = 0;
};

struct DiarizationGraphMeasure {
    NemotronDiarizationFitMeasure bytes;
    int frames = 0;
};

struct DiarizationFitGraphs {
    DiarizationGraphMeasure single_pass;
    DiarizationGraphMeasure stream;
    DiarizationGraphMeasure long_form;
};

bool measure_diarization_graph(const ParakeetCtcModel & model, const DiarizationGraphShape & shape,
                               DiarizationGraphMeasure & measure) {
    measure.frames = shape.state_frames +
                     nemotron_diarization_encoder_frames(model, shape.mel_frames);
    return measure_nemotron_diarization(model, shape.mel_frames, shape.state_frames,
                                        measure.bytes) == 0;
}

bool measure_diarization_graphs(const ParakeetCtcModel & model, int single_pass_mel,
                                bool long_form, DiarizationFitGraphs & graphs) {
    const int state = nemotron_diarization_stream_state_frames();
    const DiarizationGraphShape single_pass{single_pass_mel, 0};
    const DiarizationGraphShape stream{nemotron_diarization_stream_mel_frames(model), state};
    const DiarizationGraphShape chunk{nemotron_diarization_long_form_mel_frames(model), state};
    return measure_diarization_graph(model, single_pass, graphs.single_pass) &&
           measure_diarization_graph(model, stream, graphs.stream) &&
           (!long_form || measure_diarization_graph(model, chunk, graphs.long_form));
}

size_t largest_device_bytes(const DiarizationFitGraphs & graphs) {
    return std::max({graphs.single_pass.bytes.device_compute_bytes,
                     graphs.stream.bytes.device_compute_bytes,
                     graphs.long_form.bytes.device_compute_bytes});
}

size_t largest_host_input_bytes(const DiarizationFitGraphs & graphs) {
    return std::max({graphs.single_pass.bytes.host_input_bytes,
                     graphs.stream.bytes.host_input_bytes,
                     graphs.long_form.bytes.host_input_bytes});
}

std::string diarization_graph_report(const char * label, const DiarizationGraphMeasure & graph) {
    char line[kReportLineBytes];
    std::snprintf(line, sizeof(line), "%s%s (%d encoder frames)", label,
                  fmt_mib(graph.bytes.device_compute_bytes).c_str(), graph.frames);
    return line;
}

std::string nemotron_diarization_report(const DiarizationFitGraphs & graphs, bool long_form) {
    std::string s = "diarization: " +
                    diarization_graph_report("single-pass graph ", graphs.single_pass) +
                    diarization_graph_report("; stream graph ", graphs.stream);
    if (long_form) {
        s += diarization_graph_report("; long-form chunk graph ", graphs.long_form);
    }
    return s + "\n";
}

std::vector<CoremlFrameRange> nemotron_coreml_ranges(const DiarizationFitGraphs & graphs,
                                                     bool long_form) {
    const CoremlFrameRange stream{1, graphs.stream.frames};
    if (long_form) return {CoremlFrameRange{1, graphs.long_form.frames}, stream};
    return {CoremlFrameRange{graphs.single_pass.frames, graphs.single_pass.frames}, stream};
}

long long nemotron_workload_mel_frames(const ParakeetCtcModel & model, float audio_seconds) {
    const uint64_t samples = std::min<uint64_t>(
        sat_u64_from_double(std::ceil((double) audio_seconds * model.mel_cfg.sample_rate)),
        (uint64_t) std::numeric_limits<long long>::max());
    return nemotron_diarization_mel_frames(model, (long long) samples);
}

// A long-form diarization's passes demand every capacity up to the largest
// (observed: a 75 s call loads the 1125-frame bundle), so the Core ML charge
// counts them all; today's charge keeps the block's range.
std::vector<CoremlFrameRange> long_form_charge_ranges(std::vector<CoremlFrameRange> ranges, bool long_form) {
    if (long_form) ranges.push_back(CoremlFrameRange{1, std::numeric_limits<int>::max()});
    return ranges;
}

void project_nemotron_diarization(const ParakeetCtcModel & model, const GgufLoadMeasure & lm,
                                  const FitOptions & opts, FitResult & r) {
    const long long total_mel = nemotron_workload_mel_frames(model, opts.audio_seconds);
    const int offline_frames = nemotron_diarization_encoder_frames(model, total_mel);
    const int window = nemotron_diarization_long_form_frames(model, opts.long_form_window_frames);
    const bool long_form = nemotron_diarization_uses_long_form(
        model, opts.long_form_window_frames, total_mel);
    if (!long_form && offline_frames > model.nemotron_diarization_cfg.position_limit) {
        r.reason = "workload-too-large";
        return;
    }
    const long long window_mel =
        (long long) window * model.nemotron_diarization_cfg.subsampling_factor;
    DiarizationFitGraphs graphs;
    if (!measure_diarization_graphs(model, (int) (long_form ? window_mel : total_mel),
                                    long_form, graphs)) {
        r.reason = "measurement-failed";
        return;
    }
    r.device.weights_bytes         = lm.weights_bytes + lm.repack_bytes;
    r.device.encoder_compute_bytes = largest_device_bytes(graphs);
    r.device.total_bytes = sat_add(r.device.weights_bytes, r.device.encoder_compute_bytes);
    r.host_bytes = nemotron_diarization_host_bytes(
        model, opts, total_mel, offline_frames, largest_host_input_bytes(graphs));
    const std::vector<CoremlFrameRange> ranges = nemotron_coreml_ranges(graphs, long_form);
    r.coreml_bytes = legacy_diarizer_bytes(opts.model_gguf_path, ranges, opts);
    const uint64_t required = apply_fit_verdict(r, opts.margin_bytes);
    std::string disabled_reason;
    const bool enabled = coreml_fit_enabled(opts, disabled_reason);
    // The instance also serves shorter calls than this workload, and keeps the
    // capacities they load: any input up to the single-pass graph (the
    // long-form window past the long-form threshold) runs as one pass, so the
    // charge covers single passes from 1 frame up to it as well.
    std::vector<CoremlFrameRange> demand = long_form_charge_ranges(ranges, long_form);
    demand.push_back(CoremlFrameRange{1, graphs.single_pass.frames});
    const CoremlDiarizersCharge diarizers = coreml_model_diarizers_charge(model, opts, demand);
    // A charge holding a prebuilt bundle is today's estimate, not a validated
    // envelope; one of generated capacities alone is.
    apply_coreml_verdicts(r, required, enabled, disabled_reason,
                          diarizers.prebuilt ? diarizers.charge : CoremlCharge{}, /*encoder_prebuilt=*/false,
                          CoremlCharge{}, diarizers.prebuilt ? CoremlCharge{} : diarizers.charge);
    r.report = model_device_report(r) + nemotron_diarization_report(graphs, long_form) +
               device_projection_report(r, opts.margin_bytes, required,
                                        "every diarization graph shares one buffer") +
               coreml_fit_report(r);
}

}  // namespace

FitResult fit_params(const FitOptions & opts) {
    FitResult r;

    if (opts.model_gguf_path.empty() || !(opts.audio_seconds > 0.0f)) {
        r.reason = "invalid-arguments";
        return r;
    }

    if (!opts.backends_dir.empty()) {
        set_backends_directory(opts.backends_dir);
    }

    // ── Metadata-only load: backend resolution + tensor wiring + weight sizing
    ParakeetCtcModel model;
    GgufLoadMeasure  lm;
    {
        const int rc = load_from_gguf_metadata_only(
            opts.model_gguf_path, model, opts.n_threads, opts.n_gpu_layers,
            opts.verbose, lm, opts.backend);
        // 10: no CPU backend registered at all; 11: the explicit backend is
        // unavailable or failed to initialize.
        if (rc == 10 || rc == 11) {
            r.reason = "no-backend-device";
            return r;
        }
        if (rc != 0) {
            r.reason = "model-unreadable";
            return r;
        }
    }

    r.model_type    = model_type_name(model.model_type);
    r.model_variant = model.model_variant;
    r.device_name   = model_active_device_name(model);

    const bool is_nemotron = model.model_type == ParakeetModelType::NEMOTRON;

    // Nemotron: resolve the streaming operating point (chunk_ms -> trained
    // right-context frames, the same table Engine::stream_start consults) up
    // front so a bad request fails before any measurement. The default (0)
    // picks the operating point with the LARGEST right context: its step
    // graph and caches bound every other operating point, so the default
    // verdict stays safe whichever chunk_ms the host later streams with.
    int nemotron_rc_frames = -1;
    int nemotron_chunk_ms  = 0;
    if (is_nemotron) {
        const auto & chunks = model.nemotron_cfg.allowed_chunk_ms;
        const auto & rcs    = model.nemotron_cfg.allowed_right_context_frames;
        if (chunks.empty() || chunks.size() != rcs.size()) {
            // validate_nemotron_model guarantees a well-formed table on any
            // loadable GGUF; a mismatch here means the model is not usable.
            r.reason = "model-unreadable";
            return r;
        }
        if (opts.nemotron_chunk_ms == 0) {
            size_t worst = 0;
            for (size_t i = 1; i < rcs.size(); ++i) {
                if (rcs[i] > rcs[worst]) worst = i;
            }
            nemotron_rc_frames = rcs[worst];
            nemotron_chunk_ms  = chunks[worst];
        } else {
            for (size_t i = 0; i < chunks.size(); ++i) {
                if (chunks[i] == opts.nemotron_chunk_ms) {
                    nemotron_rc_frames = rcs[i];
                    nemotron_chunk_ms  = chunks[i];
                    break;
                }
            }
            if (nemotron_rc_frames < 0) {
                r.reason = "invalid-arguments";  // mirrors Engine::stream_start
                return r;
            }
        }
    }

    ggml_backend_t backend = model_active_backend(model);
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        r.reason = "no-backend-device";
        return r;
    }
    r.device_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    r.device_shares_host_memory = backend_shares_host_memory(backend);
    {
        size_t free_b = 0, total_b = 0;
        backend_device_memory(backend, free_b, total_b);
        r.device_free_bytes  = opts.device_free_bytes.value_or(free_b);
        r.device_total_bytes = total_b;
    }

    if (model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION) {
        project_nemotron_diarization(model, lm, opts, r);
        return r;
    }

    // ── Workload → worst-case single-encode window ─────────────────────────
    const MelConfig & mel = model.mel_cfg;
    const long long total_mel = total_mel_frames(opts.audio_seconds, mel);

    // The transcribe paths bound device memory by sliding the encoder over
    // long inputs (resolve_long_form_plan + run_encoder_windowed). Offline
    // diarize (Sortformer) runs the encoder AND the O(T^2) head over the full
    // input in one graph, so its projection must use the full length.
    //
    // The runtime does not hold ONE encoder graph: run_encoder keeps an LRU
    // cache of up to Impl::k_encoder_graph_cache_max (3) graphs keyed by mel
    // length, each owning its own persistent compute buffer. A windowed
    // transcribe touches exactly three distinct lengths -- first =
    // center + ctx (no left context), middles = center + 2*ctx, last =
    // remainder + ctx <= first -- so all three buffers are resident together
    // for the rest of the engine's life. Project the resident SET, bounding
    // the input-dependent last window by the first so the projection is
    // conservative and saturates with audio length. (EngineOptions::prewarm
    // would occupy one more slot with a ~1 s graph; the cache cap keeps that
    // within the same three buffers, evicting the smallest-value entry.)
    std::vector<long long> resident_mel;
    long long worst_window_mel = total_mel;
    if (model.model_type != ParakeetModelType::SORTFORMER) {
        int requested_window  = opts.long_form_window_frames;
        int requested_context = opts.long_form_context_frames;
        apply_long_form_hint(model.encoder_cfg.long_form_window_hint,
                             model.encoder_cfg.long_form_context_hint,
                             requested_window, requested_context);
        const LongFormPlan plan = resolve_long_form_plan_frames(
            requested_window, requested_context,
            model.encoder_cfg.pos_emb_max_len, model.encoder_cfg.subsampling_factor,
            total_mel);
        if (plan.enabled) {
            // Widen before multiplying: with pos_emb_max_len == 0 nothing
            // clamps an absurd --window-frames, and center_frames * sub in int
            // would be signed-overflow UB. Out-of-range projects as
            // workload-too-large, mirroring the worst_window_mel guard below.
            const long long center_mel_ll = (long long) plan.center_frames  * plan.sub;
            const long long ctx_mel_ll    = (long long) plan.context_frames * plan.sub;
            if (center_mel_ll > std::numeric_limits<int>::max() ||
                ctx_mel_ll    > std::numeric_limits<int>::max()) {
                r.reason = "workload-too-large";
                return r;
            }
            const int center_mel = (int) center_mel_ll;
            const int ctx_mel    = (int) ctx_mel_ll;
            const int n_units    = (int) std::min<long long>(
                total_mel, std::numeric_limits<int>::max());
            const size_t n_windows =
                plan_long_form_windows(n_units, center_mel, ctx_mel).size();
            const long long first_mel  = std::min<long long>(
                total_mel, (long long) center_mel + ctx_mel);
            const long long middle_mel = (long long) center_mel + 2 * ctx_mel;
            worst_window_mel = n_windows >= 3 ? middle_mel : first_mel;
            // Equal lengths kept adjacent so the measurement memoizer below
            // prices each distinct length once.
            resident_mel.push_back(first_mel);
            if (n_windows >= 2) resident_mel.push_back(first_mel);  // last, bounded by first
            if (n_windows >= 3) resident_mel.push_back(middle_mel);
        }
    }
    if (resident_mel.empty()) {
        resident_mel.push_back(total_mel);
    }
    if (worst_window_mel > std::numeric_limits<int>::max()) {
        r.reason = "workload-too-large";  // a single window this size is not projectable
        return r;
    }
    const int window_mel      = (int) worst_window_mel;
    const int window_enc      = encoder_out_frames(model.encoder_cfg, window_mel);
    const int total_enc       = encoder_out_frames(model.encoder_cfg, total_mel);

    // ── Encoder compute: build + size every graph the cache will hold ──────
    uint64_t enc_compute = 0;
    {
        // Memoize by length: the resident set repeats the first-window length.
        long long memo_mel   = -1;
        size_t    memo_bytes = 0;
        for (long long len : resident_mel) {
            if (len != memo_mel) {
                size_t bytes = 0;
                if (measure_encoder_compute(model, (int) len, mel.n_mels, bytes) != 0) {
                    r.reason = "measurement-failed";
                    return r;
                }
                memo_mel   = len;
                memo_bytes = bytes;
            }
            enc_compute = sat_add(enc_compute, memo_bytes);
        }
    }

    // ── Nemotron: prompt-projection graph + one live streaming session ─────
    // The locale-prompt projection graph is cached one-at-a-time keyed by
    // frame count; the offline path builds it at the FULL stitched length
    // (windowing bounds the encoder graphs, not the prompt conditioning), so
    // -- unlike every other transcription family -- Nemotron's device
    // projection keeps growing with audio_seconds. A live stream session
    // keys the same slot at right_context + 1 frames; the larger of the two
    // bounds both since only one graph is ever resident.
    NemotronStreamFitMeasure nsm;
    uint64_t nemotron_prompt_bytes = 0;
    if (is_nemotron) {
        const int stream_enc_frames = nemotron_rc_frames + 1;
        const int prompt_frames = std::max(total_enc, stream_enc_frames);
        size_t prompt_bytes = 0;
        if (measure_nemotron_prompt_compute(model, prompt_frames,
                                            prompt_bytes) != 0) {
            r.reason = "measurement-failed";
            return r;
        }
        nemotron_prompt_bytes = prompt_bytes;
        if (nemotron_measure_stream(model, nemotron_rc_frames, nsm) != 0) {
            r.reason = "measurement-failed";
            return r;
        }
        enc_compute = sat_add(enc_compute, nemotron_prompt_bytes);
        enc_compute = sat_add(enc_compute, nsm.device_step_graph_bytes);
        enc_compute = sat_add(enc_compute, nsm.device_subsampling_bytes);
    }

    // ── Decoder-side buffers, per model type ───────────────────────────────
    DecoderFitMeasure dm;
    uint64_t extra_host = 0;  // decoder-side buffers that live in host RAM
    switch (model.model_type) {
        case ParakeetModelType::CTC:
            break;  // the CTC head lives inside the encoder graph, already measured
        case ParakeetModelType::NEMOTRON_DIARIZATION:
            break;
        case ParakeetModelType::RNNT:
        case ParakeetModelType::TDT:
        case ParakeetModelType::NEMOTRON:  // shares the RNN-T runtime
                                           // (tdt_prepare_runtime handles the
                                           // nemotron dims + weights)
            if (tdt_measure_runtime(model, window_enc, dm) != 0) {
                r.reason = "measurement-failed";
                return r;
            }
            break;
        case ParakeetModelType::EOU:
            if (eou_measure_runtime(model, window_enc, dm) != 0) {
                r.reason = "measurement-failed";
                return r;
            }
            break;
        case ParakeetModelType::SORTFORMER: {
            size_t head_active = 0, head_host_input = 0;
            if (sortformer_measure_head(model, window_enc,
                                        head_active, head_host_input) != 0) {
                r.reason = "measurement-failed";
                return r;
            }
            // On Mali-Vulkan the head is force-routed to CPU: its compute
            // buffer is host RAM, like the head weight copies already counted
            // in lm.sortformer_cpu_bytes. On GPU runs the scheduler keeps the
            // head's graph-input original in a CPU-side buffer (host RAM).
            if (model_sortformer_on_cpu(model)) {
                extra_host = sat_add(extra_host, sat_add(head_active, head_host_input));
            } else {
                dm.device_compute_bytes = head_active;
                extra_host = sat_add(extra_host, head_host_input);
            }
            break;
        }
    }

    // ── Host-side extras (scale with the FULL input, unlike the windowed
    //    device buffers): input samples, mel slab, stitched encoder output,
    //    CTC logits, encoder-graph host mirrors, CPU-path decoder weights. ──
    {
        // Saturating throughout: host_bytes feeds the verdict on
        // unified-memory devices, so a wrap here could flip a real
        // DOES-NOT-FIT into FITS.
        uint64_t host = 0;
        host = sat_add(host, sat_mul(sat_u64_from_double(std::ceil(
                   (double) opts.audio_seconds * mel.sample_rate)),
                   sizeof(float)));                                   // input samples
        host = sat_add(host, sat_mul(sat_mul((uint64_t) total_mel, (uint64_t) mel.n_mels),
                                     sizeof(float)));                 // mel spectrogram
        host = sat_add(host, sat_mul(sat_mul((uint64_t) total_enc,
                                             (uint64_t) model.encoder_cfg.d_model),
                                     sizeof(float)));                 // encoder_out
        if (model.model_type == ParakeetModelType::CTC) {
            host = sat_add(host, sat_mul(sat_mul((uint64_t) total_enc,
                                                 (uint64_t) model.vocab_size),
                                         sizeof(float)));             // logits
        }
        // EncoderGraph host mirrors -- relative positional encoding and
        // (chunked-attention models) the T x T attention mask -- are owned per
        // cached graph, so one worst-window mirror per resident graph.
        uint64_t mirrors = sat_mul((uint64_t) (2 * (long long) window_enc - 1),
                                   sat_mul((uint64_t) model.encoder_cfg.d_model,
                                           sizeof(float)));
        if (model.encoder_cfg.att_chunked_limited &&
            model.encoder_cfg.att_context_left  >= 0 &&
            model.encoder_cfg.att_context_right >= 0) {
            mirrors = sat_add(mirrors,
                              sat_mul(sat_mul((uint64_t) window_enc, (uint64_t) window_enc),
                                      sizeof(float)));
        }
        host = sat_add(host, sat_mul(mirrors, resident_mel.size()));
        host = sat_add(host, dm.host_bytes);           // CPU decode path: dequantised f32 weights
        host = sat_add(host, lm.sortformer_cpu_bytes); // Mali-Vulkan CPU head copy (host RAM)
        host = sat_add(host, lm.tdt_cpu_bytes);        // OpenCL: CPU-resident decoder weights (host RAM)
        host = sat_add(host, extra_host);
        if (is_nemotron) {
            // Offline prompt conditioning stages the one-hot concat input and
            // the projected output on the host at the full stitched length
            // (run_nemotron_prompt_projection).
            host = sat_add(host, sat_mul(sat_mul((uint64_t) total_enc,
                       sat_add((uint64_t) model.nemotron_cfg.prompt_input_width,
                               (uint64_t) model.encoder_cfg.d_model)),
                       sizeof(float)));
            // One live streaming session: per-layer caches, graph host
            // mirrors, chunk staging, decode state, sched CPU fallback.
            host = sat_add(host, nsm.host_bytes);
        }
        r.host_bytes = host;
    }
    // Sortformer diarizer demand: the offline pass over the whole input and,
    // for v2.1, every AOSC step slab of a default stream.
    std::vector<CoremlFrameRange> diarizer_demand;
    if (model.model_type == ParakeetModelType::SORTFORMER) {
        r.coreml_bytes = legacy_diarizer_bytes(opts.model_gguf_path,
                                               {CoremlFrameRange{total_enc, total_enc}}, opts);
        const int encoder_frame_ms = 1000 * model.mel_cfg.hop_length * model.encoder_cfg.subsampling_factor /
                                     std::max(1, model.mel_cfg.sample_rate);
        diarizer_demand = sortformer_diarizer_demand(
            total_enc, model.model_variant == "sortformer-streaming-v2.1-aosc"
                           ? sortformer_default_aosc_slab_frames(encoder_frame_ms)
                           : 0);
    }

    r.device.weights_bytes         = lm.weights_bytes + lm.repack_bytes;
    r.device.encoder_compute_bytes = enc_compute;
    r.device.decoder_state_bytes   = dm.device_state_bytes;
    r.device.decoder_compute_bytes = dm.device_compute_bytes;
    r.device.total_bytes =
        sat_add(sat_add(r.device.weights_bytes, r.device.encoder_compute_bytes),
                sat_add(r.device.decoder_state_bytes, r.device.decoder_compute_bytes));

    const uint64_t required = apply_fit_verdict(r, opts.margin_bytes);
    {
        std::string disabled_reason;
        const bool enabled = coreml_fit_enabled(opts, disabled_reason);
        bool encoder_prebuilt = false;
        const CoremlCharge encoder = coreml_encoder_fit_charge(model, opts, enabled, window_mel,
                                                               window_enc, encoder_prebuilt);
        // A charge holding a prebuilt bundle is today's estimate, not a
        // validated envelope; one of generated capacities alone is.
        const CoremlDiarizersCharge diarizers =
            model.model_type == ParakeetModelType::SORTFORMER
                ? coreml_model_diarizers_charge(model, opts, diarizer_demand)
                : CoremlDiarizersCharge{};
        apply_coreml_verdicts(r, required, enabled, disabled_reason,
                              diarizers.prebuilt ? diarizers.charge : CoremlCharge{}, encoder_prebuilt, encoder,
                              diarizers.prebuilt ? CoremlCharge{} : diarizers.charge);
    }

    // ── Report ─────────────────────────────────────────────────────────────
    {
        std::string s = model_device_report(r);
        char line[kReportLineBytes];
        std::snprintf(line, sizeof(line),
                      "workload: %.1f s audio -> worst window %d mel frames (%d encoder frames)\n",
                      (double) opts.audio_seconds, window_mel, window_enc);
        s += line;
        if (is_nemotron) {
            std::snprintf(line, sizeof(line),
                          "nemotron: prompt projection %s (%d frames); "
                          "stream %d ms -> step graph %s + pre-encode %s\n",
                          fmt_mib(nemotron_prompt_bytes).c_str(),
                          std::max(total_enc, nemotron_rc_frames + 1),
                          nemotron_chunk_ms,
                          fmt_mib(nsm.device_step_graph_bytes).c_str(),
                          fmt_mib(nsm.device_subsampling_bytes).c_str());
            s += line;
        }
        std::snprintf(line, sizeof(line), "%zu resident graph%s",
                      resident_mel.size(), resident_mel.size() == 1 ? "" : "s");
        s += device_projection_report(r, opts.margin_bytes, required, line);
        s += coreml_fit_report(r);
        r.report = std::move(s);
    }

    return r;
}

}  // namespace parakeet
