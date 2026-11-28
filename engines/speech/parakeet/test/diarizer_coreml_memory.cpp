// Memory validation of the generated Core ML diarizers (Nemotron 3
// Diarization, Sortformer v2.1) under overlapped schedules: diarizer passes,
// one pass size per capacity of the model's templates (the clip alone for
// the smallest capacity, then the clip after cache rows that fill each larger
// capacity exactly), run in a loop while the runtime
// prepares (generates and compiles), loads and warms the capacities in the
// background (prewarm = all, wait = none). A sampler thread reads this
// process's phys_footprint every 2 ms and the slots' states every 10 ms; a
// second thread reads ANECompilerService's resident size every 100 ms. The
// report attributes the peaks to phases:
//   - prepare (digest, generate, compile) of a capacity while passes run,
//     on ggml or on a capacity that is already ready;
//   - load and warm of a capacity while passes run;
// and records the resident growth each capacity leaves behind: the settled
// footprint (the median sample) from the moment the capacity is ready to the
// moment the next one is (or the run ends), less the same level for the
// capacity ready before it, or less the ggml-only run's peak for the first.
// The steps add up to the run's steady growth over ggml. It also records
// ANECompilerService's growth during the loads (its peak less its size when
// the first load starts, that baseline labelled clean, dirty or unknown), and compares the run's peak footprint with the
// Core ML bytes fit charges for the same options, as an increment over a
// ggml-only run of the same passes (--baseline-bytes, from --ggml-only).
// A sample that could not be read (task_info, or ps failing) is dropped and
// counted in the report's sampler_errors, so a report with missing
// measurements is never taken for a quiet one.
//
//   diarizer-coreml-memory --gguf M.gguf --wav A.wav --cache-dir DIR
//       [--ggml-only] [--baseline-bytes N] [--passes-after-ready N] [--json OUT]
//
// The generated schedule needs a GGUF with no adjacent diarizer bundle and
// an empty DIR; the prebuilt schedule a GGUF beside its bundles. A fresh
// process identity (scripts/coreml/fresh-host.sh) and fresh paths make every
// ANE load cold. Every pass's interval is recorded, with the ggml passes that
// overlapped a prepare and a load or warm. Exit status: 0 when every capacity
// became ready, each served --passes-after-ready Core ML passes after that
// (counted by the capacity that served them), a ggml pass ran during a
// prepare and one during a load or warm, and the peak increment stays within
// the charged envelope (always 0 with --ggml-only); 1 otherwise; 2 on usage or setup
// errors. The ANECompilerService size is also recorded at the start of each
// load, so the checker can hold each load's own compile growth to the reserve.

#include "mel_preprocess.h"
#include "parakeet/fit.h"
#include "parakeet_ctc.h"
#include "parakeet_diarization_v3.h"
#include "parakeet_sortformer.h"

#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kThreads = 4;
constexpr int kGpuLayers = 99;
constexpr double kMiB = 1048576.0;
constexpr double kTraceSeconds = 0.1;  // the bucket of the report's footprint trace
// ANECompilerService at or below this size before the first load counts as a
// clean start: its growth then stands for a cold compile on a clean host. A
// service that already holds more (a dirty start) reuses some of it, so its
// growth understates one.
constexpr uint64_t kAneCompilerCleanBytes = 32ull * 1024 * 1024;

uint64_t footprint_now() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}

// ANECompilerService's resident size (0 when it is not running); false when
// ps could not be run or failed.
bool ane_compiler_rss(uint64_t & total) {
    total = 0;
    FILE * f = popen("ps -axo rss=,comm= 2>/dev/null", "r");
    if (!f) return false;
    char line[1024];
    bool any = false;
    while (fgets(line, sizeof(line), f)) {
        any = true;
        if (strstr(line, "ANECompilerService")) total += (uint64_t) std::strtoull(line, nullptr, 10) * 1024;
    }
    return pclose(f) == 0 && any;
}

// stage -> state, from the status JSON (stage objects start at "artifact_key").
std::map<std::string, std::string> stage_states(const std::string & status) {
    std::map<std::string, std::string> out;
    const std::string start = "{\"artifact_key\"";
    auto value = [](const std::string & s, const char * key) {
        const std::string k = std::string("\"") + key + "\":\"";
        const size_t at = s.find(k);
        if (at == std::string::npos) return std::string();
        const size_t from = at + k.size();
        return s.substr(from, s.find('"', from) - from);
    };
    for (size_t at = status.find(start); at != std::string::npos;) {
        const size_t next = status.find(start, at + 1);
        const std::string obj = status.substr(at, next == std::string::npos ? std::string::npos : next - at);
        at = next;
        const std::string stage = value(obj, "stage");
        if (stage.rfind("diarizer-", 0) == 0) out[stage] = value(obj, "state");
    }
    return out;
}

const char * phase_of(const std::string & state) {
    if (state == "digesting" || state == "generating") return "generate";
    if (state == "compiling") return "compile";
    if (state == "loading" || state == "warming") return "load";
    return nullptr;
}

struct Sample {
    double t = 0;
    uint64_t footprint = 0;
};
struct StateSample {
    double t = 0;
    std::map<std::string, std::string> states;
};
struct AneSample {
    double t = 0;
    uint64_t rss = 0;
};

// One phase of one stage: [t0, t1).
struct Segment {
    std::string stage, phase;
    double t0 = 0, t1 = 0;
};

// One diarizer pass: [t0, t1), and whether it ran on Core ML.
struct PassInterval {
    double t0 = 0, t1 = 0;
    bool coreml = false;
};

// The ggml passes that ran while a segment of one of `phases` did.
int ggml_passes_overlapping(const std::vector<PassInterval> & passes, const std::vector<Segment> & segments,
                            std::initializer_list<const char *> phases) {
    int n = 0;
    for (const PassInterval & p : passes) {
        if (p.coreml) continue;
        for (const Segment & seg : segments) {
            const bool phase = std::any_of(phases.begin(), phases.end(), [&](const char * x) { return seg.phase == x; });
            if (phase && p.t0 < seg.t1 && seg.t0 < p.t1) {
                ++n;
                break;
            }
        }
    }
    return n;
}

std::vector<Segment> segments_of(const std::vector<StateSample> & states, double t_end) {
    std::vector<Segment> out;
    std::map<std::string, Segment> open;
    for (size_t i = 0; i < states.size(); ++i) {
        const double t = states[i].t;
        for (auto it = open.begin(); it != open.end();) {
            auto st = states[i].states.find(it->first);
            const char * p = st == states[i].states.end() ? nullptr : phase_of(st->second);
            if (p == nullptr || it->second.phase != p) {
                it->second.t1 = t;
                out.push_back(it->second);
                it = open.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto & [stage, state] : states[i].states) {
            const char * p = phase_of(state);
            if (p != nullptr && open.find(stage) == open.end()) open[stage] = Segment{stage, p, t, 0};
        }
    }
    for (auto & [stage, seg] : open) {
        seg.t1 = t_end;
        out.push_back(seg);
    }
    std::sort(out.begin(), out.end(), [](const Segment & a, const Segment & b) { return a.t0 < b.t0; });
    return out;
}

bool load_model(const std::string & gguf, const parakeet::CoremlOptions & options, bool disable_coreml,
                parakeet::ParakeetCtcModel & model) {
    if (disable_coreml) setenv("PARAKEET_COREML_DISABLE", "1", 1);
    const int status = parakeet::load_from_gguf(gguf, model, kThreads, kGpuLayers, false, "auto", options);
    if (disable_coreml) unsetenv("PARAKEET_COREML_DISABLE");
    return status == 0;
}

// The capacities of the model's embedded diarizer templates, ascending:
// Nemotron 3 Diarization's three, or the Sortformer v2.1 AOSC variant's one.
std::vector<int> template_capacities(const parakeet::ParakeetCtcModel & model) {
    if (model.model_type == parakeet::ParakeetModelType::NEMOTRON_DIARIZATION) return {358, 745, 1125};
    if (model.model_type == parakeet::ParakeetModelType::SORTFORMER &&
        model.model_variant == "sortformer-streaming-v2.1-aosc") {
        return {410};
    }
    return {};
}

// The clip's encoder frames and the rows a pass feeds the diarizer.
struct Clip {
    int frames = 0;              // encoder frames of the clip alone
    int width = 0;               // encoder width
    std::vector<float> cache;    // Nemotron: speaker cache rows (filled to the largest capacity)
    std::vector<float> rows;     // Sortformer: the clip's pre-encode rows, tiled to the largest capacity
};

bool prepare_clip(parakeet::ParakeetCtcModel & model, const std::vector<float> & mel, int mel_frames,
                  int largest, Clip & clip) {
    if (model.model_type == parakeet::ParakeetModelType::NEMOTRON_DIARIZATION) {
        clip.frames = parakeet::nemotron_diarization_encoder_frames(model, mel_frames);
        clip.width = model.nemotron_diarization_cfg.encoder_width;
        clip.cache.assign(static_cast<size_t>(largest) * clip.width, 0.01f);
        return true;
    }
    std::vector<float> pre_encode;
    if (parakeet::run_subsampling(model, mel.data(), mel_frames, model.mel_cfg.n_mels, pre_encode,
                                  clip.frames) != 0 || clip.frames <= 0) {
        return false;
    }
    clip.width = model.encoder_cfg.sortformer_fc_d_model;
    clip.rows.resize(static_cast<size_t>(largest) * clip.width);
    for (int row = 0; row < largest; ++row) {
        std::copy_n(pre_encode.begin() + static_cast<size_t>(row % clip.frames) * clip.width, clip.width,
                    clip.rows.begin() + static_cast<size_t>(row) * clip.width);
    }
    return true;
}

// One diarizer pass of `frames` encoder frames: on the covering capacity
// when it is ready, on ggml otherwise. Nemotron runs the clip after
// frames - clip.frames cache rows; Sortformer the first `frames` rows, as an
// AOSC step does (a sidecar, else the ggml encoder and speaker head).
// Returns 0 and sets *coreml, or the failing status.
int run_pass(parakeet::ParakeetCtcModel & model, const std::vector<float> & mel, int mel_frames, const Clip & clip,
             int frames, bool * coreml) {
    if (model.model_type == parakeet::ParakeetModelType::NEMOTRON_DIARIZATION) {
        const int cache_rows = frames - clip.frames;
        parakeet::NemotronDiarizationChunk chunk;
        const int rc = parakeet::run_nemotron_diarization_chunk(
            model, mel.data(), mel_frames, cache_rows > 0 ? clip.cache.data() : nullptr, cache_rows, chunk);
        *coreml = chunk.used_coreml;
        return rc;
    }
    parakeet::SortformerDiarizationResult result;
    if (parakeet::sortformer_diarize_coreml(model, clip.rows.data(), frames, clip.width, {}, result)) {
        *coreml = true;
        return 0;
    }
    *coreml = false;
    parakeet::EncoderOutputs encoded;
    if (const int rc = parakeet::run_encoder_bypass_pre_encode(model, clip.rows.data(), frames, clip.width, encoded);
        rc != 0) {
        return rc;
    }
    return parakeet::sortformer_diarize_ggml(model, encoded.encoder_out.data(), frames, clip.width, {}, result);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, wav, cache_dir, json_out;
    bool ggml_only = false;
    uint64_t baseline = 0;
    int passes_after_ready = 6;
    int ggml_passes = 30;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--gguf") gguf = next();
        else if (a == "--wav") wav = next();
        else if (a == "--cache-dir") cache_dir = next();
        else if (a == "--json") json_out = next();
        else if (a == "--ggml-only") ggml_only = true;
        else if (a == "--baseline-bytes") baseline = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--passes-after-ready") passes_after_ready = std::atoi(next().c_str());
        else if (a == "--ggml-passes") ggml_passes = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty() || wav.empty() || (!ggml_only && cache_dir.empty())) {
        std::fprintf(stderr, "usage: %s --gguf M.gguf --wav A.wav --cache-dir DIR [--ggml-only] "
                             "[--baseline-bytes N] [--passes-after-ready N] [--ggml-passes N] [--json OUT]\n",
                     argv[0]);
        return 2;
    }

    parakeet::CoremlOptions options;
    options.cache_dir = cache_dir;
    options.prewarm = parakeet::CoremlOptions::Prewarm::all;
    options.wait = parakeet::CoremlOptions::Wait::none;

    // The charge fit makes for these options (no workload-dependent part
    // under prewarm = all for generated capacities).
    parakeet::FitOptions fit;
    fit.model_gguf_path = gguf;
    fit.n_gpu_layers = kGpuLayers;
    fit.coreml = options;
    const parakeet::FitResult fr = parakeet::fit_params(fit);

    const auto t0 = Clock::now();
    auto now_s = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    std::vector<Sample> samples;
    std::vector<StateSample> states;
    std::vector<AneSample> ane;
    std::mutex mu;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> sampler_errors{0};
    parakeet::ParakeetCtcModel model;
    std::atomic<bool> loaded{false};

    std::thread sampler([&] {
        int tick = 0;
        while (!stop.load()) {
            const Sample s{now_s(), footprint_now()};
            if (s.footprint == 0) {
                ++sampler_errors;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            std::map<std::string, std::string> st;
            const bool poll = !ggml_only && loaded.load() && tick % 5 == 0;
            if (poll) st = stage_states(parakeet::model_coreml_status_json(model));
            {
                std::lock_guard<std::mutex> lock(mu);
                samples.push_back(s);
                if (poll) states.push_back(StateSample{s.t, std::move(st)});
            }
            ++tick;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::thread ane_sampler([&] {
        while (!stop.load()) {
            AneSample a{now_s(), 0};
            if (!ane_compiler_rss(a.rss)) {
                ++sampler_errors;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(mu);
                ane.push_back(a);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    auto finish = [&] {
        stop.store(true);
        sampler.join();
        ane_sampler.join();
    };

    if (!load_model(gguf, options, ggml_only, model)) {
        finish();
        std::fprintf(stderr, "cannot load %s\n", gguf.c_str());
        return 2;
    }
    loaded.store(true);
    const double t_loaded = now_s();

    std::vector<float> samples_pcm;
    int sample_rate = 0;
    std::vector<float> mel;
    int mel_frames = 0;
    if (parakeet::load_wav_mono_f32(wav, samples_pcm, sample_rate) != 0 ||
        parakeet::compute_log_mel(samples_pcm.data(), static_cast<int>(samples_pcm.size()), model.mel_cfg, mel,
                                  mel_frames) != 0) {
        finish();
        std::fprintf(stderr, "cannot read %s\n", wav.c_str());
        return 2;
    }
    const std::vector<int> capacities = template_capacities(model);
    Clip clip;
    if (capacities.empty() || !prepare_clip(model, mel, mel_frames, capacities.back(), clip)) {
        finish();
        std::fprintf(stderr, "%s has no diarizer template, or its clip could not be prepared\n", gguf.c_str());
        return 2;
    }
    if (clip.frames > capacities.front()) {
        finish();
        std::fprintf(stderr, "%s is longer than the smallest capacity (%d encoder frames)\n", wav.c_str(),
                     clip.frames);
        return 2;
    }
    // One pass size per capacity: the clip alone, and the clip after cache
    // rows that fill each larger capacity exactly.
    std::vector<int> sizes = {clip.frames};
    for (size_t k = 1; k < capacities.size(); ++k) sizes.push_back(capacities[k]);

    int passes = 0, coreml_passes = 0, ggml_after_ready = 0;
    // Core ML passes after every stage is ready, by the capacity that served
    // them (a larger capacity standing in for a failed one counts as itself).
    std::map<int, int> coreml_by_capacity;
    std::vector<PassInterval> intervals;
    double t_all_ready = -1;
    const auto deadline = Clock::now() + std::chrono::minutes(10);
    for (;;) {
        const int size = sizes[passes % sizes.size()];
        bool used_coreml = false;
        if (!ggml_only) parakeet::model_coreml_reset_call_stats(model);
        const double pass_t0 = now_s();
        if (run_pass(model, mel, mel_frames, clip, size, &used_coreml) != 0) {
            finish();
            std::fprintf(stderr, "a diarizer pass failed\n");
            return 2;
        }
        intervals.push_back(PassInterval{pass_t0, now_s(), used_coreml});
        ++passes;
        if (used_coreml) {
            ++coreml_passes;
        }
        if (ggml_only) {
            if (passes >= ggml_passes) break;
            continue;
        }
        const std::map<std::string, std::string> st = stage_states(parakeet::model_coreml_status_json(model));
        const bool all_ready = st.size() == capacities.size() &&
                               std::all_of(st.begin(), st.end(), [](const auto & kv) {
                                   return kv.second == "ready";
                               });
        if (all_ready && t_all_ready < 0) t_all_ready = now_s();
        if (all_ready && !used_coreml) ++ggml_after_ready;
        if (all_ready && used_coreml) ++coreml_by_capacity[parakeet::model_coreml_call_stats(model).capacity];
        const bool enough = all_ready && std::all_of(capacities.begin(), capacities.end(), [&](int k) {
            return coreml_by_capacity[k] >= passes_after_ready;
        });
        if (enough || Clock::now() > deadline) break;
    }
    const double t_end = now_s();
    const std::string status = ggml_only ? std::string("{}") : parakeet::model_coreml_status_json(model);
    finish();

    // ---- report
    uint64_t peak = 0, first_pass = 0;
    for (const Sample & s : samples) peak = std::max(peak, s.footprint);
    for (const Sample & s : samples) {
        if (s.t >= t_loaded) {
            first_pass = s.footprint;
            break;
        }
    }
    auto peak_in = [&](double a, double b) {
        uint64_t p = 0;
        for (const Sample & s : samples) {
            if (s.t >= a && s.t < b) p = std::max(p, s.footprint);
        }
        return p;
    };
    auto at = [&](double t) {
        uint64_t f = 0;
        for (const Sample & s : samples) {
            if (s.t > t) break;
            f = s.footprint;
        }
        return f;
    };
    auto ane_in = [&](double a, double b) {
        uint64_t p = 0;
        for (const AneSample & s : ane) {
            if (s.t >= a && s.t < b + 0.1) p = std::max(p, s.rss);
        }
        return p;
    };
    // The median footprint sample in [a, b), or the sample at `a` when the
    // window holds none.
    auto settled = [&](double a, double b) {
        std::vector<uint64_t> v;
        for (const Sample & s : samples) {
            if (s.t >= a && s.t < b) v.push_back(s.footprint);
        }
        if (v.empty()) return at(a);
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    uint64_t ane_peak = 0, ane_min = UINT64_MAX;
    for (const AneSample & s : ane) {
        ane_peak = std::max(ane_peak, s.rss);
        ane_min = std::min(ane_min, s.rss);
    }
    if (ane.empty()) ane_min = 0;
    const std::vector<Segment> segments = segments_of(states, t_end);
    // ANECompilerService before this process's first load: its last sample at
    // or before the load starts (0 when it was not running).
    double first_load = -1;
    for (const Segment & seg : segments) {
        if (seg.phase == "load" && (first_load < 0 || seg.t0 < first_load)) first_load = seg.t0;
    }
    uint64_t ane_before_loads = 0;
    bool ane_sampled_before = false;
    for (const AneSample & s : ane) {
        if (first_load < 0 || s.t > first_load) break;
        ane_before_loads = s.rss;
        ane_sampled_before = true;
    }
    // The baseline the growth is measured from: clean, dirty, or unknown (no
    // sample at or before the first load).
    const char * ane_baseline = !ane_sampled_before                         ? "unknown"
                                : ane_before_loads <= kAneCompilerCleanBytes ? "clean"
                                                                             : "dirty";
    // Resident steps, in the order the loads end (a capacity is ready).
    std::vector<size_t> ready;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].phase == "load") ready.push_back(i);
    }
    std::sort(ready.begin(), ready.end(), [&](size_t a, size_t b) { return segments[a].t1 < segments[b].t1; });
    std::map<size_t, std::pair<uint64_t, uint64_t>> level_step;  // segment -> (settled level, step)
    uint64_t level = baseline, resident_max = 0;
    for (size_t k = 0; k < ready.size(); ++k) {
        const double from = segments[ready[k]].t1;
        const double to = k + 1 < ready.size() ? segments[ready[k + 1]].t1 : t_end;
        const uint64_t l = settled(from, to);
        const uint64_t step = baseline > 0 && l > level ? l - level : 0;
        level_step[ready[k]] = {l, step};
        resident_max = std::max(resident_max, step);
        level = l;
    }
    const uint64_t steady_increment = baseline > 0 && !ready.empty() && level > baseline ? level - baseline : 0;

    std::string json = "{";
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "\"mode\":\"%s\",\"passes\":%d,\"coreml_passes\":%d,\"ggml_passes_after_ready\":%d,"
                  "\"seconds\":%.2f,\"all_ready_s\":%.2f,\"peak_footprint\":%llu,\"footprint_after_load\":%llu,"
                  "\"baseline_bytes\":%llu,\"fit_coreml_bytes\":%llu,\"fit_coreml_validated\":%d,"
                  "\"fit_ggml_required\":%llu,\"ane_compiler_peak\":%llu,\"ane_compiler_min\":%llu,"
                  "\"samples\":%zu,\"state_polls\":%zu,\"sampler_errors\":%llu,\"segments\":[",
                  ggml_only ? "ggml-only" : "coreml", passes, coreml_passes, ggml_after_ready, t_end, t_all_ready,
                  (unsigned long long) peak, (unsigned long long) first_pass, (unsigned long long) baseline,
                  (unsigned long long) fr.coreml.bytes, fr.coreml.validated ? 1 : 0,
                  (unsigned long long) fr.ggml_required_bytes, (unsigned long long) ane_peak,
                  (unsigned long long) ane_min, samples.size(), states.size(),
                  (unsigned long long) sampler_errors.load());
    json += buf;
    bool first = true;
    uint64_t compile_inc = 0, load_inc = 0, generate_inc = 0, ane_load_peak = 0, ane_load_growth = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const Segment & seg = segments[i];
        const uint64_t before = at(seg.t0);
        const uint64_t p = std::max(before, peak_in(seg.t0, seg.t1));
        const uint64_t a = ane_in(seg.t0, seg.t1);
        const uint64_t inc = p - before;
        if (seg.phase == "compile") compile_inc = std::max(compile_inc, inc);
        if (seg.phase == "generate") generate_inc = std::max(generate_inc, inc);
        std::string load_fields;
        if (seg.phase == "load") {
            load_inc = std::max(load_inc, inc);
            ane_load_peak = std::max(ane_load_peak, a);
            // The compiler's size when this load started: its growth during
            // this load is measured from here, so a service that shrank after
            // an earlier load cannot hide a later load's compile.
            uint64_t ane_before = 0;
            for (const AneSample & s : ane) {
                if (s.t > seg.t0) break;
                ane_before = s.rss;
            }
            ane_load_growth = std::max(ane_load_growth, a > ane_before ? a - ane_before : 0);
            char extra[320];
            std::snprintf(extra, sizeof(extra),
                          ",\"settled_after_ready\":%llu,\"resident_increment\":%llu,\"ane_compiler_before\":%llu",
                          (unsigned long long) level_step[i].first, (unsigned long long) level_step[i].second,
                          (unsigned long long) ane_before);
            load_fields = extra;
        }
        std::snprintf(buf, sizeof(buf),
                      "%s{\"stage\":\"%s\",\"phase\":\"%s\",\"t0\":%.3f,\"t1\":%.3f,\"before\":%llu,\"peak\":%llu,"
                      "\"increment\":%llu,\"ane_compiler_peak\":%llu%s}",
                      first ? "" : ",", seg.stage.c_str(), seg.phase.c_str(), seg.t0, seg.t1,
                      (unsigned long long) before, (unsigned long long) p, (unsigned long long) inc,
                      (unsigned long long) a, load_fields.c_str());
        json += buf;
        first = false;
    }
    const uint64_t increment = baseline > 0 && peak > baseline ? peak - baseline : 0;
    std::string by_capacity;
    for (const auto & [k, n] : coreml_by_capacity) {
        by_capacity += (by_capacity.empty() ? "\"" : ",\"") + std::to_string(k) + "\":" + std::to_string(n);
    }
    const uint64_t ane_growth = ane_load_peak > ane_before_loads ? ane_load_peak - ane_before_loads : 0;
    // The schedule is overlapped only when inference ran during the
    // background work: a ggml pass during a prepare (generate or compile) and
    // one during a load or warm (one pass spanning both counts for both).
    const int overlap_prepare = ggml_passes_overlapping(intervals, segments, {"generate", "compile"});
    const int overlap_load = ggml_passes_overlapping(intervals, segments, {"load"});
    std::snprintf(buf, sizeof(buf), "],\"ggml_passes_during_prepare\":%d,\"ggml_passes_during_load\":%d,"
                  "\"pass_intervals\":[", overlap_prepare, overlap_load);
    json += buf;
    first = true;
    for (const PassInterval & p : intervals) {
        std::snprintf(buf, sizeof(buf), "%s[%.4f,%.4f,%d]", first ? "" : ",", p.t0, p.t1, p.coreml ? 1 : 0);
        json += buf;
        first = false;
    }
    std::snprintf(buf, sizeof(buf),
                  "],\"max_generate_increment\":%llu,\"max_compile_increment\":%llu,\"max_load_increment\":%llu,"
                  "\"max_resident_increment\":%llu,\"steady_increment_over_baseline\":%llu,"
                  "\"ane_compiler_peak_during_load\":%llu,\"ane_compiler_before_loads\":%llu,"
                  "\"ane_compiler_baseline\":\"%s\",\"ane_compiler_growth\":%llu,"
                  "\"ane_compiler_max_load_growth\":%llu,"
                  "\"peak_increment_over_baseline\":%llu,\"coreml_passes_by_capacity\":{%s},\"trace\":[",
                  (unsigned long long) generate_inc, (unsigned long long) compile_inc,
                  (unsigned long long) load_inc, (unsigned long long) resident_max,
                  (unsigned long long) steady_increment, (unsigned long long) ane_load_peak,
                  (unsigned long long) ane_before_loads, ane_baseline, (unsigned long long) ane_growth,
                  (unsigned long long) ane_load_growth, (unsigned long long) increment, by_capacity.c_str());
    json += buf;
    // The footprint in kTraceSeconds buckets: [start, min, max].
    first = true;
    for (size_t i = 0; i < samples.size();) {
        const double b0 = std::floor(samples[i].t / kTraceSeconds) * kTraceSeconds;
        uint64_t lo = UINT64_MAX, hi = 0;
        for (; i < samples.size() && samples[i].t < b0 + kTraceSeconds; ++i) {
            lo = std::min(lo, samples[i].footprint);
            hi = std::max(hi, samples[i].footprint);
        }
        std::snprintf(buf, sizeof(buf), "%s[%.1f,%llu,%llu]", first ? "" : ",", b0, (unsigned long long) lo,
                      (unsigned long long) hi);
        json += buf;
        first = false;
    }
    json += "],\"status\":";
    json += status + "}";
    if (!json_out.empty()) {
        if (FILE * f = std::fopen(json_out.c_str(), "w")) {
            std::fputs(json.c_str(), f);
            std::fputc('\n', f);
            std::fclose(f);
        }
    }
    std::printf("%s: %d passes (%d Core ML) in %.1f s, all ready at %.1f s; peak footprint %.1f MiB "
                "(after load %.1f); fit Core ML %.1f MiB (validated %d)\n",
                ggml_only ? "ggml-only" : "coreml", passes, coreml_passes, t_end, t_all_ready, peak / kMiB,
                first_pass / kMiB, fr.coreml.bytes / kMiB, fr.coreml.validated ? 1 : 0);
    if (!ggml_only) {
        std::printf("increments: generate %.1f MiB, compile %.1f MiB, load+warm %.1f MiB, largest resident step "
                    "%.1f MiB (steady %.1f MiB over ggml); ANECompilerService peak %.1f MiB (min %.1f), %.1f MiB "
                    "before the loads (%s baseline), growth %.1f MiB\n",
                    generate_inc / kMiB, compile_inc / kMiB, load_inc / kMiB, resident_max / kMiB,
                    steady_increment / kMiB, ane_peak / kMiB, ane_min / kMiB, ane_before_loads / kMiB,
                    ane_baseline, ane_growth / kMiB);
        std::printf("peak increment over the ggml-only baseline %.1f MiB, envelope %.1f MiB: %s\n",
                    increment / kMiB, fr.coreml.bytes / kMiB,
                    baseline == 0 ? "no baseline" : (increment <= fr.coreml.bytes ? "WITHIN" : "OVER"));
        if (baseline > 0 && increment > fr.coreml.bytes) return 1;
        if (t_all_ready < 0) {
            std::printf("not every capacity became ready before the deadline\n");
            return 1;
        }
        std::printf("ggml passes overlapping a prepare: %d, a load or warm: %d\n", overlap_prepare, overlap_load);
        if (overlap_prepare == 0 || overlap_load == 0) {
            std::printf("no ggml pass ran during a %s: the schedule was not overlapped\n",
                        overlap_prepare == 0 ? "prepare" : "load or warm");
            return 1;
        }
        for (int k : capacities) {
            if (coreml_by_capacity[k] < passes_after_ready) {
                std::printf("capacity %d served %d Core ML passes after readiness (%d required)\n", k,
                            coreml_by_capacity[k], passes_after_ready);
                return 1;
            }
        }
    }
    return 0;
}
