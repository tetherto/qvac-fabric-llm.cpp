#pragma once

// Core ML part of the Parakeet fit projection. Pure and
// portable: the route comes only from options and adjacent sidecar presence,
// never from the Core ML cache or warm hints, so the projection is identical
// before and after the first load (test-parakeet-coreml-route "fit-stable").

#include "fit_util.h"
#include "parakeet/diarization.h"
#include "parakeet/fit.h"
#include "parakeet_coreml_path.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace parakeet {

// The resolved compute units of the encoder load (the runtime's
// ComputeUnits, mirrored here so the projection stays portable).
enum class CoremlFitUnits { all, cpu_and_ane, cpu_and_gpu, cpu_only };

// Inputs of the encoder sidecar envelope.
struct CoremlEncoderEnvelopeInputs {
    uint64_t       weight_file_bytes = 0;   // <sidecar>/weights/weight.bin
    uint64_t       encoder_parameters = 0;  // encoder.* parameters in the GGUF
    uint64_t       io_bytes = 0;            // worst-window input + output, f32
    CoremlFitUnits units = CoremlFitUnits::all;
};

// The client-side load transient (`load_peak`) by compute-unit class, until
// the model's own measurement replaces it:
//  - cpu_and_ane: 0.25 x weight file (cold ANE load peaked at 0.19x);
//  - all, cpu_and_gpu: 1.0 x weight file (Parakeet TDT 0.6B fp16 measured
//    +1.08 GiB over a 1.2 GB weight file on an M2, ~0.9x, and the palettized
//    prebuilt ~0.95x);
//  - cpu_only: 1.0 x as well. The CPU load builds the same expanded weights
//    and nothing has measured it lower, so it is charged like `all`.
inline uint64_t coreml_encoder_load_transient_bytes(uint64_t weight_file_bytes, CoremlFitUnits units) {
    switch (units) {
        case CoremlFitUnits::cpu_and_ane: return weight_file_bytes / 4;
        case CoremlFitUnits::all:
        case CoremlFitUnits::cpu_and_gpu:
        case CoremlFitUnits::cpu_only: return weight_file_bytes;
    }
    return weight_file_bytes;
}

// Defaults until the model's own measurements replace them:
//  - resident runtime weights at their expanded fp16 geometry (a palettized
//    prebuilt is never charged at its compressed file size): the larger of
//    the weight file and 2 bytes per GGUF encoder parameter;
//  - I/O backing, input and output: 2 x io_bytes;
//  - one load transient by unit class (above);
//  - margin x1.25 on resident + transient.
// One Core ML program the encoder stage keeps resident. Programs a model
// keeps loaded together (TranscribePsy EU's route programs, Sortformer's
// encoder and bypass bundles) are charged together.
struct CoremlFitProgram {
    uint64_t       weight_file_bytes = 0;    // <bundle>/weights/weight.bin
    uint64_t       runtime_floor_bytes = 0;  // expanded fp16 floor of a full-encoder program, else 0
    uint64_t       io_bytes = 0;             // worst-window input + output, f32
    int            functions = 1;            // functions loaded from the bundle, each charged its weights
    CoremlFitUnits units = CoremlFitUnits::all;
};

// Every resident program (runtime weights per loaded function, since loaded
// functions are not assumed to share weight memory until measured, plus
// 2 x I/O backing), one load transient (the largest: the runtime runs one
// Core ML load at a time), and the 1.25 margin on the sum.
inline uint64_t coreml_programs_envelope_bytes(const std::vector<CoremlFitProgram> & programs) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    uint64_t resident = 0;
    uint64_t transient = 0;
    for (const CoremlFitProgram & p : programs) {
        const uint64_t weights = std::max(p.weight_file_bytes, p.runtime_floor_bytes);
        resident = sat_add(resident, sat_mul(weights, (uint64_t) std::max(1, p.functions)));
        resident = sat_add(resident, sat_mul(p.io_bytes, (uint64_t) 2));
        transient = std::max(transient, coreml_encoder_load_transient_bytes(p.weight_file_bytes, p.units));
    }
    const uint64_t total = sat_add(resident, transient);
    return sat_add(total, total / 4);
}

inline uint64_t coreml_encoder_envelope_bytes(const CoremlEncoderEnvelopeInputs & in) {
    CoremlFitProgram p;
    p.weight_file_bytes = in.weight_file_bytes;
    p.runtime_floor_bytes = fitutil::sat_mul(in.encoder_parameters, (uint64_t) 2);
    p.io_bytes = in.io_bytes;
    p.units = in.units;
    return coreml_programs_envelope_bytes({p});
}

// The encoder route fit charges: prebuilt when Core ML is enabled and the
// adjacent sidecar directory exists (today's path rules), none otherwise.
struct CoremlEncoderFitRoute {
    bool prebuilt = false;
    uint64_t weight_file_bytes = 0;
};

inline CoremlEncoderFitRoute coreml_encoder_fit_route(const std::string & sidecar_dir,
                                                      bool coreml_enabled) {
    CoremlEncoderFitRoute r;
    std::error_code ec;
    if (!coreml_enabled || !std::filesystem::is_directory(sidecar_dir, ec)) return r;
    r.prebuilt = true;
    const uintmax_t size = std::filesystem::file_size(
        std::filesystem::path(sidecar_dir) / "weights" / "weight.bin", ec);
    if (!ec) r.weight_file_bytes = static_cast<uint64_t>(size);
    return r;
}

// A stage's Core ML charge in two parts: what stays resident, and the one
// load transient it adds while a load runs. The runtime runs one Core ML job
// per library image, so a model charges its stages' residents together and
// only the largest transient among them.
struct CoremlCharge {
    uint64_t resident = 0;
    uint64_t transient = 0;
};

// The encoder programs' charge, both parts with the 1.25 margin.
inline CoremlCharge coreml_programs_charge(const std::vector<CoremlFitProgram> & programs) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    CoremlCharge c;
    for (const CoremlFitProgram & p : programs) {
        const uint64_t weights = std::max(p.weight_file_bytes, p.runtime_floor_bytes);
        const uint64_t functions = (uint64_t) std::max(1, p.functions);
        c.resident = sat_add(c.resident, sat_mul(weights, functions));
        c.resident = sat_add(c.resident, sat_mul(sat_mul(p.io_bytes, (uint64_t) 2), functions));
        c.transient = std::max(c.transient, coreml_encoder_load_transient_bytes(p.weight_file_bytes, p.units));
    }
    c.resident = sat_add(c.resident, c.resident / 4);
    c.transient = sat_add(c.transient, c.transient / 4);
    return c;
}

// One diarizer capacity of a model, as the fit charges it: a prebuilt bundle
// (`prebuilt`, with its weight.bin size), a generated capacity (`generated`,
// with its template's unmargined parts for the resolved units and its
// margin), or neither (a template capacity that is not generated: it never
// loads, and its calls go to the next capacity).
struct CoremlDiarizerCharge {
    int      frames = 0;
    bool     generated = false;
    uint64_t resident_bytes = 0;
    uint64_t transient_bytes = 0;
    double   margin = 1.0;
    bool     prebuilt = false;
    uint64_t weights_bytes = 0;
};

// The diarizers' Core ML charge, from one ordered list of every capacity
// (prebuilt, generated and status-only), the way the runtime picks them:
//  - selected: the capacities calls in `ranges` can load
//    (coreml_diarizer_loaded_capacities over the whole list), plus what
//    construction loads (`prewarm`: the smallest capacity that can serve for
//    the minimal set, every one that can for all);
//  - a selected capacity that fails, or never loads, hands its calls to the
//    next larger one that can serve, up to the next selected capacity; so
//    each selected capacity holds one of the serving capacities in
//    [itself, the next selected one) at a time, never two of them.
// Every combination of those choices is a resident set the runtime can
// reach; the charge is the largest: prebuilt bundles at today's estimate
// (coreml_diarizer_bundle_set_resident_bytes, and each bundle's load
// transient for `units`), generated capacities at their templates'
// envelope (residents + their largest transient, times their margin), and
// one transient, the larger of the two kinds. `prebuilt` is set when the
// chosen set holds a prebuilt bundle (not a validated envelope).
struct CoremlDiarizersCharge {
    CoremlCharge charge;
    bool prebuilt = false;
};

inline CoremlDiarizersCharge coreml_diarizers_charge(std::vector<CoremlDiarizerCharge> capacities,
                                                     const std::vector<CoremlFrameRange> & ranges,
                                                     CoremlDiarizerPrewarm prewarm, CoremlFitUnits units) {
    using fitutil::sat_add;
    std::sort(capacities.begin(), capacities.end(),
              [](const CoremlDiarizerCharge & a, const CoremlDiarizerCharge & b) { return a.frames < b.frames; });
    auto serves = [](const CoremlDiarizerCharge & c) { return c.prebuilt || c.generated; };
    std::vector<int> sorted;
    for (const CoremlDiarizerCharge & c : capacities) sorted.push_back(c.frames);
    std::vector<int> selected = coreml_diarizer_loaded_capacities(sorted, ranges);
    for (const CoremlDiarizerCharge & c : capacities) {
        if (!serves(c) || prewarm == CoremlDiarizerPrewarm::none) continue;
        selected.push_back(c.frames);
        if (prewarm == CoremlDiarizerPrewarm::minimal) break;  // the smallest that can serve
    }
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    // The choices of each selected capacity.
    std::vector<std::vector<const CoremlDiarizerCharge *>> slots;
    for (size_t i = 0; i < selected.size(); ++i) {
        const int below = i + 1 < selected.size() ? selected[i + 1] : 0;
        std::vector<const CoremlDiarizerCharge *> options;
        for (const CoremlDiarizerCharge & c : capacities) {
            if (serves(c) && c.frames >= selected[i] && (below == 0 || c.frames < below)) options.push_back(&c);
        }
        if (!options.empty()) slots.push_back(std::move(options));
    }
    CoremlDiarizersCharge worst;
    uint64_t worst_total = 0;
    std::vector<size_t> pick(slots.size(), 0);
    for (;;) {
        int bundles = 0;
        uint64_t weights = 0, pre_transient = 0, gen_resident = 0, gen_transient = 0;
        double margin = 1.0;
        for (size_t s = 0; s < slots.size(); ++s) {
            const CoremlDiarizerCharge & c = *slots[s][pick[s]];
            if (c.prebuilt) {
                ++bundles;
                weights = sat_add(weights, c.weights_bytes);
                pre_transient = std::max(pre_transient, coreml_encoder_load_transient_bytes(c.weights_bytes, units));
            } else {
                gen_resident = sat_add(gen_resident, c.resident_bytes);
                gen_transient = std::max(gen_transient, c.transient_bytes);
                margin = std::max(margin, c.margin);
            }
        }
        const uint64_t gen_total = sat_add(gen_resident, gen_transient);
        const uint64_t envelope =
            gen_total == 0 ? 0 : fitutil::sat_u64_from_double(std::ceil(static_cast<double>(gen_total) * margin));
        const uint64_t gen_res = std::min(
            envelope, fitutil::sat_u64_from_double(std::ceil(static_cast<double>(gen_resident) * margin)));
        CoremlCharge set;
        set.resident = sat_add(coreml_diarizer_bundle_set_resident_bytes(bundles, weights), gen_res);
        set.transient = std::max(pre_transient, envelope - gen_res);
        const uint64_t total = sat_add(set.resident, set.transient);
        if (total > worst_total) {
            worst_total = total;
            worst.charge = set;
            worst.prebuilt = bundles > 0;
        }
        size_t s = 0;
        for (; s < slots.size(); ++s) {
            if (++pick[s] < slots[s].size()) break;
            pick[s] = 0;
        }
        if (s == slots.size()) break;
    }
    return worst;
}

// The largest AOSC step slab of a Sortformer v2.1 stream with the default
// SortformerStreamingOptions, in encoder frames of `encoder_frame_ms`: the
// full speaker cache and FIFO plus one chunk with its left (80 ms, the
// stream's default for v2.1) and right context; 409 at 80 ms.
inline int sortformer_default_aosc_slab_frames(int encoder_frame_ms) {
    constexpr int kDefaultLeftContextMs = 80;
    const SortformerStreamingOptions d;
    const int ms = std::max(1, encoder_frame_ms);
    return d.spkcache_len + d.fifo_len + kDefaultLeftContextMs / ms + d.chunk_ms / ms + d.chunk_right_context_ms / ms;
}

// The Sortformer diarizer calls a fit projection charges: offline passes of
// any length up to the whole input (`offline_frames` encoder frames; the
// instance also serves shorter calls, and keeps the capacities they load)
// and, for a model that streams with the speaker cache (v2.1 AOSC;
// `aosc_slab_frames` > 0), every step slab from the first chunk up to the
// full one: the slab grows with the speaker cache and FIFO, so each capacity
// on the way loads, whatever the input's length.
inline std::vector<CoremlFrameRange> sortformer_diarizer_demand(int offline_frames, int aosc_slab_frames) {
    std::vector<CoremlFrameRange> out = {CoremlFrameRange{1, std::max(1, offline_frames)}};
    if (aosc_slab_frames > 0) out.push_back(CoremlFrameRange{1, aosc_slab_frames});
    return out;
}

// The generated capacities' charge alone (capacities that are neither
// prebuilt nor generated only shape the selection), every capacity loaded at
// construction when `every_capacity` (prewarm = all), else none.
inline CoremlCharge coreml_generated_diarizer_charge(std::vector<CoremlDiarizerCharge> capacities,
                                                     const std::vector<CoremlFrameRange> & ranges,
                                                     bool every_capacity = false) {
    for (CoremlDiarizerCharge & c : capacities) c.prebuilt = false;  // uncharged here
    return coreml_diarizers_charge(capacities, ranges,
                                   every_capacity ? CoremlDiarizerPrewarm::all : CoremlDiarizerPrewarm::none,
                                   CoremlFitUnits::cpu_and_ane)
        .charge;
}

// The generated charge as one number: resident + transient.
inline uint64_t coreml_generated_diarizer_envelope(const std::vector<CoremlDiarizerCharge> & capacities,
                                                   const std::vector<CoremlFrameRange> & ranges,
                                                   bool every_capacity = false) {
    const CoremlCharge c = coreml_generated_diarizer_charge(capacities, ranges, every_capacity);
    return fitutil::sat_add(c.resident, c.transient);
}

// Fills the two verdicts. `required` is today's verdict charge (with the
// device sharing host memory it includes today's diarizer charge,
// r.coreml_bytes); the ggml-only total excludes that. The Core ML-inclusive
// total adds the residents of the stages Core ML would activate (`diarizers`
// and `encoder`, counted only on shared memory for the diarizers as today)
// and the largest single transient. `generated` is the generated diarizer
// capacities' charge (validated templates only), counted like the other
// stages. `coreml.validated` holds when generated capacities are charged and
// nothing else is: a prebuilt charge in the bytes is not a validated
// envelope. The top-level `fits` is left as it is: prebuilt routes keep
// today's fit results until their model's PR validates them, and a generated
// route never turns a working ggml verdict into a refusal (memory-degrade
// resolves use = off).
inline void apply_coreml_verdicts(FitResult & r, uint64_t required, bool coreml_enabled,
                                  const std::string & disabled_reason, const CoremlCharge & diarizers,
                                  bool encoder_prebuilt, const CoremlCharge & encoder,
                                  const CoremlCharge & generated = {}) {
    using fitutil::sat_add;
    const uint64_t legacy_diarizers = r.device_shares_host_memory ? r.coreml_bytes : 0;
    r.ggml_required_bytes = required >= legacy_diarizers ? required - legacy_diarizers : 0;
    r.ggml_only_fits = r.ggml_required_bytes <= r.device_free_bytes;
    r.coreml.encoder_route = encoder_prebuilt ? "prebuilt" : "none";
    r.coreml.validated = false;
    if (!coreml_enabled) {
        r.coreml.bytes = 0;
        r.coreml.disabled_reason = disabled_reason.empty() ? std::string("use=off") : disabled_reason;
        r.coreml.fits = r.ggml_only_fits;
        return;
    }
    const CoremlCharge diar = r.device_shares_host_memory ? diarizers : CoremlCharge{};
    const CoremlCharge enc = encoder_prebuilt ? encoder : CoremlCharge{};
    r.coreml.disabled_reason.clear();
    r.coreml.bytes = sat_add(sat_add(sat_add(diar.resident, enc.resident), generated.resident),
                             std::max({diar.transient, enc.transient, generated.transient}));
    const bool generated_charged = generated.resident > 0 || generated.transient > 0;
    const bool other_charged = diar.resident > 0 || diar.transient > 0 || encoder_prebuilt;
    r.coreml.validated = generated_charged && !other_charged;
    r.coreml.fits = sat_add(r.ggml_required_bytes, r.coreml.bytes) <= r.device_free_bytes;
}

}  // namespace parakeet
