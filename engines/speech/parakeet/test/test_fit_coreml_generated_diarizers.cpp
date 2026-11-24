// fit_params on a Nemotron 3 Diarization or Sortformer v2.1 GGUF with no
// adjacent diarizer bundle charges the generated capacities' Core ML
// envelope, and the charge
// never depends on the cache: an empty cache and a cache holding every
// generated capacity give the same Core ML bytes, verdict and validation.
// An Engine whose memory budget lies between the ggml-only total and the
// Core ML-inclusive total resolves Core ML off for memory; one with room for
// both keeps generation on. An Engine on compute units the templates were not
// validated for generates nothing, and every capacity keeps a status entry
// that says so; so does an Engine with generation off (generate = never).
// The generated capacities' warm hints are found and made stale: the charge
// and the returned verdicts at injected free-memory readings at the charge
// and one byte under it are those of no hint and of fresh hints.
// With prewarm = all the charge covers every generated capacity, and so does
// the default 300 s workload: an instance projected for it also runs shorter
// calls as one pass up to the long-form window (1125 frames for Nemotron),
// which keep their capacities loaded. A short workload charges less when the
// model has several capacities, the same when it has one (construction loads
// the only capacity either way).
//
// Usage: test-fit-coreml-generated-diarizers <model.gguf> <empty cache dir> [template ids]
// `template ids` are the model's diarizer templates, comma-separated and
// smallest capacity first (default Nemotron's three; Sortformer v2.1 has
// sortformer-diar-f410). The GGUF's directory must hold no diarizer
// bundle. Never skips: every check needs Core ML generation (macOS 15, the
// generator built in).

#include "parakeet/engine.h"
#include "parakeet/fit.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr int kGpuLayers = 1;
constexpr uint64_t kRoomy = 64ull * 1024 * 1024 * 1024;
// The model's diarizer templates, smallest capacity first (default
// Nemotron's 358, 745 and 1125 frames).
std::vector<std::string> g_templates = {"nemotron-diar-f358", "nemotron-diar-f745", "nemotron-diar-f1125"};
int g_capacities = 3;

bool g_passed = true;

void check(bool condition, const char * label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label);
    g_passed = g_passed && condition;
}

bool contains(const std::string & text, const char * needle) { return text.find(needle) != std::string::npos; }

int count(const std::string & text, const std::string & needle) {
    int n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) ++n;
    return n;
}

// Occurrences of `needle` in the diarizer stage objects of a status JSON
// (other stages, such as an absent Sortformer encoder sidecar, are left out).
// Stage objects have sorted keys, so each starts at its "artifact_key".
int count_in_diarizers(const std::string & status, const std::string & needle) {
    const std::string start = "{\"artifact_key\"";
    int n = 0;
    for (size_t at = status.find(start); at != std::string::npos;) {
        const size_t next = status.find(start, at + 1);
        const std::string stage = status.substr(at, next == std::string::npos ? std::string::npos : next - at);
        at = next;
        if (stage.find("\"stage\":\"diarizer-") != std::string::npos) n += count(stage, needle);
    }
    return n;
}

parakeet::CoremlOptions coreml_options(const std::string & cache_dir) {
    parakeet::CoremlOptions o;
    o.cache_dir = cache_dir;
    return o;
}

// The same FitOptions the Engine's memory-degrade check builds (the default
// workload), or a workload of `audio_seconds`.
parakeet::FitResult project(const std::string & gguf, const parakeet::CoremlOptions & coreml,
                            float audio_seconds = 0) {
    parakeet::FitOptions opts;
    opts.model_gguf_path = gguf;
    opts.n_gpu_layers = kGpuLayers;
    opts.coreml = coreml;
    if (audio_seconds > 0) opts.audio_seconds = audio_seconds;
    return parakeet::fit_params(opts);
}

// The same projection against an injected free-memory reading.
parakeet::FitResult project_free(const std::string & gguf, const parakeet::CoremlOptions & coreml, uint64_t free) {
    parakeet::FitOptions opts;
    opts.model_gguf_path = gguf;
    opts.n_gpu_layers = kGpuLayers;
    opts.coreml = coreml;
    opts.device_free_bytes = free;
    return parakeet::fit_params(opts);
}

void print(const char * what, const parakeet::FitResult & r) {
    std::printf("%s: ggml %llu, coreml %llu (fits %d, validated %d), free %llu\n", what,
                (unsigned long long) r.ggml_required_bytes, (unsigned long long) r.coreml.bytes, r.coreml.fits,
                r.coreml.validated, (unsigned long long) r.device_free_bytes);
}

// The returned verdicts at an injected free-memory reading `budget`, in the
// hint state `state`: the reading is the one used, the ggml-only verdict
// passes, and the Core ML verdict passes exactly when the budget covers
// `at_charge` (ggml-only total + Core ML charge without hints).
void check_verdicts(const std::string & gguf, const parakeet::CoremlOptions & coreml, const char * state,
                    uint64_t at_charge) {
    for (const uint64_t budget : {at_charge, at_charge - 1}) {
        const parakeet::FitResult r = project_free(gguf, coreml, budget);
        const bool covers = budget == at_charge;
        std::printf("%s, free %llu: ggml_only_fits %d, coreml.fits %d\n", state, (unsigned long long) budget,
                    r.ggml_only_fits, r.coreml.fits);
        const std::string label = std::string(state) + (covers ? ", free = the charge" : ", free = the charge - 1") +
                                  ": the returned ggml-only verdict passes and the Core ML verdict " +
                                  (covers ? "passes" : "fails");
        check(r.device_free_bytes == budget && r.ggml_only_fits && r.coreml.fits == covers, label.c_str());
    }
}

// One template's charge, from its template.json as the runtime computes it
// for cpu_and_ane: resident = runtime_weight_bytes + 2 x I/O + state +
// workspace; transient = max(generation_peak_bytes + 8 MiB +
// compile_peak_bytes, load_peak_bytes.cpu_and_ane); and its margin.
struct TemplateCharge {
    uint64_t resident = 0;
    uint64_t transient = 0;
    double margin = 0;
};

TemplateCharge template_charge(const char * id) {
    std::ifstream in(std::string(PARAKEET_TEST_TEMPLATE_DIR) + "/" + id + "/template.json");
    const nlohmann::json j = nlohmann::json::parse(in);
    const nlohmann::json & m = j.at("memory");
    uint64_t io = 0;
    for (const auto & fn : j.at("functions")) {
        for (const char * side : {"inputs", "outputs"}) {
            for (const auto & x : fn.at(side)) {
                uint64_t n = x.value("dtype", std::string("fp32")) == "fp16" ? 2 : 4;
                for (const auto & d : x.at("shape")) n *= d.get<uint64_t>();
                io += n;
            }
        }
    }
    TemplateCharge c;
    c.resident = m.at("runtime_weight_bytes").get<uint64_t>() + 2 * io + j.value("state_bytes", uint64_t{0}) +
                 m.value("workspace_bytes", uint64_t{0});
    const uint64_t prepare = j.at("generation_peak_bytes").get<uint64_t>() + 8ull * 1024 * 1024 +
                             m.at("compile_peak_bytes").get<uint64_t>();
    c.transient = std::max(prepare, m.at("load_peak_bytes").at("cpu_and_ane").get<uint64_t>());
    c.margin = m.at("margin").get<double>();
    return c;
}

// A template's capacity in encoder frames (its first input's first dimension).
int template_frames(const std::string & id) {
    std::ifstream in(std::string(PARAKEET_TEST_TEMPLATE_DIR) + "/" + id + "/template.json");
    const nlohmann::json j = nlohmann::json::parse(in);
    return j.at("functions").at(0).at("inputs").at(0).at("shape").at(0).get<int>();
}

// The warm-hint records of the model's generated capacities in `cache`, by
// the artifact each names (<template id>.<key>/model.mlmodelc).
std::vector<std::filesystem::path> template_hints(const std::string & cache) {
    std::vector<std::filesystem::path> out;
    std::error_code ec;
    for (const auto & e :
         std::filesystem::directory_iterator(std::filesystem::path(cache) / "qvac-coreml-v1" / "hints", ec)) {
        std::ifstream in(e.path());
        const nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
        if (j.is_discarded() || !j.contains("artifact") || !j["artifact"].is_string()) continue;
        const std::string artifact = j["artifact"].get<std::string>();
        for (const std::string & id : g_templates) {
            if (artifact.find("/" + id + ".") != std::string::npos) out.push_back(e.path());
        }
    }
    return out;
}

nlohmann::json read_json(const std::filesystem::path & path) {
    std::ifstream in(path);
    return nlohmann::json::parse(in, nullptr, false);
}

// The Engine's Core ML status, and its construction group in `construction`.
std::string engine_status(const std::string & gguf, parakeet::CoremlOptions coreml,
                          std::string * construction = nullptr) {
    parakeet::EngineOptions opts;
    opts.model_gguf_path = gguf;
    opts.n_gpu_layers = kGpuLayers;
    opts.coreml = std::move(coreml);
    parakeet::Engine engine(opts);
    if (opts.coreml.wait == parakeet::CoremlOptions::Wait::construct) engine.coreml_wait();
    if (construction != nullptr) *construction = engine.coreml_construction_json();
    return engine.coreml_status_json();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <empty cache dir> [template ids]\n", argv[0]);
        return 2;
    }
    if (argc == 4) {
        g_templates.clear();
        std::string ids = argv[3];
        for (size_t at = 0; at <= ids.size();) {
            const size_t comma = std::min(ids.find(',', at), ids.size());
            if (comma > at) g_templates.push_back(ids.substr(at, comma - at));
            at = comma + 1;
        }
    }
    g_capacities = static_cast<int>(g_templates.size());
    if (g_capacities < 1) {
        std::fprintf(stderr, "at least one template id is required\n");
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string cache = argv[2];
    std::error_code ec;
    std::filesystem::remove_all(cache, ec);

    const parakeet::FitResult before = project(gguf, coreml_options(cache));
    print("empty cache", before);
    check(before.status != parakeet::FitStatus::Error, "fit_params projects the GGUF");
    check(before.coreml.disabled_reason.empty(), "Core ML is enabled for the projection");
    check(before.coreml.bytes > 0 && before.coreml.validated,
          "the generated capacities are charged, validated for the resolved units");
    // The verdicts at fixed free-memory readings that straddle the charge;
    // the same readings are checked again with fresh and with stale hints.
    const uint64_t at_charge = before.ggml_required_bytes + before.coreml.bytes;
    check_verdicts(gguf, coreml_options(cache), "no hint", at_charge);

    // prewarm = all loads every capacity at construction: the charge covers
    // all of them. The default workload (300 s, long form) reaches every
    // Nemotron capacity too: its long-form chunks and stream steps load 358
    // and 745, and a shorter call of up to 1125 frames (about 90 s) runs as
    // one pass on 1125. A model with one capacity loads it at construction.
    parakeet::CoremlOptions every = coreml_options(cache);
    every.prewarm = parakeet::CoremlOptions::Prewarm::all;
    const parakeet::FitResult all = project(gguf, every);
    print("prewarm all", all);
    check(all.coreml.bytes > 0 && all.coreml.validated, "prewarm all charges every generated capacity");
    check(before.coreml.bytes == all.coreml.bytes,
          "the default workload charges every capacity a shorter single pass can load");
    const parakeet::FitResult brief = project(gguf, coreml_options(cache), 5.0f);
    print("5 s workload", brief);
    if (g_capacities > 1) {
        check(brief.coreml.bytes > 0 && brief.coreml.bytes < all.coreml.bytes,
              "a 5 s workload charges fewer capacities");
    } else {
        check(brief.coreml.bytes == all.coreml.bytes,
              "one capacity: a 5 s workload charges the capacity construction loads");
    }
    // prewarm = all loads every capacity whatever the workload: a 5 s
    // projection charges as much as the default one.
    const parakeet::FitResult all_brief = project(gguf, every, 5.0f);
    print("prewarm all, 5 s", all_brief);
    check(all_brief.coreml.bytes == all.coreml.bytes && all_brief.coreml.bytes >= brief.coreml.bytes,
          "prewarm all: a 5 s workload is charged every capacity");

    // The exact charge, recomputed from the templates' memory sections: every
    // capacity = the residents + one (the largest) transient; the 5 s
    // workload = the smallest capacity alone. The constants carry the 1.25
    // already, so the templates' envelope margin must be 1.0.
    uint64_t residents = 0, transient = 0;
    bool unit_margin = true;
    for (const std::string & id : g_templates) {
        const TemplateCharge t = template_charge(id.c_str());
        residents += t.resident;
        transient = std::max(transient, t.transient);
        unit_margin = unit_margin && t.margin == 1.0;
    }
    check(unit_margin, "the templates' memory.margin is 1.0 (the constants carry the 1.25)");
    const TemplateCharge smallest = template_charge(g_templates.front().c_str());
    const uint64_t every_bytes = residents + transient;
    const uint64_t one_bytes = smallest.resident + smallest.transient;
    std::printf("from template.json: every capacity %llu, smallest capacity %llu\n",
                (unsigned long long) every_bytes, (unsigned long long) one_bytes);
    check(all.coreml.bytes == every_bytes && before.coreml.bytes == every_bytes,
          "prewarm all and the default workload charge exactly every template's resident + one transient");
    check(brief.coreml.bytes == one_bytes, "the 5 s workload charges exactly the smallest capacity's envelope");

    // Generate every capacity (wait for construction, prewarm all).
    parakeet::CoremlOptions generate = coreml_options(cache);
    generate.wait = parakeet::CoremlOptions::Wait::construct;
    generate.prewarm = parakeet::CoremlOptions::Prewarm::all;
    const std::string generated = engine_status(gguf, generate);
    check(count_in_diarizers(generated, "\"route\":\"generated\"") == g_capacities &&
              count_in_diarizers(generated, "\"state\":\"ready\"") == g_capacities,
          "every capacity was generated into the cache and loaded");

    const parakeet::FitResult after = project(gguf, coreml_options(cache));
    print("after generation", after);
    check(after.coreml.bytes == before.coreml.bytes, "fit-stable: the same Core ML bytes after generation");
    check(after.coreml.validated == before.coreml.validated, "fit-stable: the same validation");
    check(after.ggml_required_bytes == before.ggml_required_bytes, "fit-stable: the same ggml-only total");
    if (after.device_free_bytes == before.device_free_bytes) {
        check(after.coreml.fits == before.coreml.fits && after.fits == before.fits,
              "fit-stable: the same verdicts at the same free memory");
    } else {
        check(after.coreml.fits ==
                  (after.ggml_required_bytes + after.coreml.bytes <= after.device_free_bytes),
              "fit-stable: the Core ML verdict follows the unchanged charge (free memory moved)");
    }

    // The hints the generated capacities' loads wrote: one per capacity, not
    // stale. The fit never reads them: with every hint stale (stale_until
    // ahead) the charge and the verdicts at a fixed budget are those of no
    // hint (the empty cache) and of fresh hints.
    const std::vector<std::filesystem::path> hints = template_hints(cache);
    check(hints.size() >= g_templates.size(), "every generated capacity's load wrote a warm hint");
    bool fresh = !hints.empty();
    for (const std::filesystem::path & h : hints) {
        const nlohmann::json j = read_json(h);
        fresh = fresh && !j.is_discarded() && j.contains("stale_until") && j["stale_until"].is_null();
    }
    check(fresh, "the generated capacities' hints are not stale");
    check_verdicts(gguf, coreml_options(cache), "fresh hints", at_charge);
    for (const std::filesystem::path & h : hints) {
        nlohmann::json j = read_json(h);
        j["stale_until"] = "2999-01-01T00:00:00Z";
        std::ofstream(h, std::ios::trunc) << j.dump();
    }
    bool stale = !hints.empty();
    for (const std::filesystem::path & h : hints) {
        const nlohmann::json j = read_json(h);
        stale = stale && j.contains("stale_until") && j["stale_until"] == "2999-01-01T00:00:00Z";
    }
    check(stale, "every hint now has a future stale_until");
    const parakeet::FitResult stale_fit = project(gguf, coreml_options(cache));
    print("stale hints", stale_fit);
    check(stale_fit.coreml.bytes == before.coreml.bytes && stale_fit.coreml.validated == before.coreml.validated &&
              stale_fit.ggml_required_bytes == before.ggml_required_bytes,
          "fit-stable: stale hints give the same charge, validation and ggml-only total");
    check_verdicts(gguf, coreml_options(cache), "stale hints", at_charge);
    parakeet::CoremlOptions short_by_one = coreml_options(cache);
    short_by_one.memory_budget_bytes = static_cast<int64_t>(at_charge - 1);
    check(contains(engine_status(gguf, short_by_one), "\"generation\":{\"reason\":\"memory\",\"supported\":false}"),
          "with stale hints, a budget one byte under the charge still turns Core ML off for memory");

    // Units the templates were not validated for: nothing is generated or
    // loaded from the cache, and every capacity keeps an entry that says why.
    parakeet::CoremlOptions gpu = coreml_options(cache);
    gpu.compute_units = "cpu_and_gpu";
    std::string gpu_group;
    const std::string unvalidated = engine_status(gguf, gpu, &gpu_group);
    check(contains(unvalidated, "\"generation\":{\"reason\":\"unvalidated_units\",\"supported\":false}"),
          "unvalidated compute units: generation off, reason unvalidated_units");
    check(count_in_diarizers(unvalidated, "\"state\":\"unsupported(unvalidated_units)\"") == g_capacities &&
              count_in_diarizers(unvalidated, "\"route\":\"none\"") == g_capacities,
          "every capacity reports unsupported(unvalidated_units), route none");
    check(contains(gpu_group, "\"slots\":[]") && count(gpu_group, "\"stage\":\"diarizer-") == g_capacities &&
              count(gpu_group, "unsupported(unvalidated_units)") == g_capacities,
          "the construction group loads nothing and lists every capacity as skipped");

    // Generation off for the instance: every capacity is still listed, with
    // the reason.
    parakeet::CoremlOptions never = coreml_options(cache);
    never.generate = parakeet::CoremlOptions::Generate::never;
    std::string never_group;
    const std::string off = engine_status(gguf, never, &never_group);
    check(contains(off, "\"generation\":{\"reason\":\"generate=never\",\"supported\":false}") &&
              count(off, "\"stage\":\"diarizer-") == g_capacities,
          "generate = never: generation off, every capacity keeps a status entry");
    check(contains(never_group, "\"slots\":[]") && count(never_group, "\"stage\":\"diarizer-") == g_capacities,
          "generate = never: the construction group lists every capacity as skipped");

    // A budget between the two totals: Core ML off for memory, before any load.
    parakeet::CoremlOptions tight = coreml_options(cache);
    tight.memory_budget_bytes = static_cast<int64_t>(after.ggml_required_bytes + after.coreml.bytes / 2);
    const std::string degraded = engine_status(gguf, tight);
    check(contains(degraded, "\"generation\":{\"reason\":\"memory\",\"supported\":false}"),
          "a budget between the ggml-only and Core ML-inclusive totals: generation off, reason memory");
    check(!contains(degraded, "\"route\":\"generated\"") && !contains(degraded, "\"route\":\"cached\""),
          "and no generated route");

    // Just under the full charge of the default workload (above what its
    // long-form and stream calls alone load): Core ML off for memory.
    parakeet::CoremlOptions near = coreml_options(cache);
    near.memory_budget_bytes =
        static_cast<int64_t>(after.ggml_required_bytes + all.coreml.bytes - 1024 * 1024);
    check(contains(engine_status(gguf, near), "\"generation\":{\"reason\":\"memory\",\"supported\":false}"),
          "a budget 1 MiB under the three-capacity charge: generation off, reason memory");

    // Room for both: generation stays on (the cached minimal capacity loads
    // before the engine goes away).
    parakeet::CoremlOptions roomy = coreml_options(cache);
    roomy.wait = parakeet::CoremlOptions::Wait::construct;
    roomy.memory_budget_bytes = static_cast<int64_t>(after.ggml_required_bytes + after.coreml.bytes + kRoomy);
    const std::string kept = engine_status(gguf, roomy);
    check(contains(kept, "\"generation\":{\"reason\":null,\"supported\":true}"),
          "a budget with room for Core ML keeps generation on");

    // A prebuilt bundle one capacity above the largest template (220 MiB of
    // weights) beside the GGUF, prewarm minimal, the default 300 s workload:
    // the stream loads the smallest capacity, and a shorter offline call that
    // only the bundle covers loads it beside the capacities already resident.
    // The charge holds every template capacity and the bundle at once:
    // 2 x 220 MiB + 160 MiB runtime for the bundle, the templates' residents,
    // and the largest transient (the bundle's is a quarter of its weights).
    // A budget just under that turns Core ML off for memory.
    {
        constexpr uint64_t kBundleWeights = 220ull * 1024 * 1024;
        const std::filesystem::path dir = std::filesystem::path(cache).parent_path() /
                                          (std::filesystem::path(cache).filename().string() + "-adjacent");
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir);
        const std::string linked = (dir / "model.gguf").string();
        std::filesystem::create_symlink(std::filesystem::absolute(gguf), linked, ec);
        const int bundle_frames = template_frames(g_templates.back()) + 102;
        const std::filesystem::path weights =
            dir / ("model-diarizer-" + std::to_string(bundle_frames) + ".mlmodelc") / "weights" / "weight.bin";
        std::filesystem::create_directories(weights.parent_path());
        std::ofstream(weights, std::ios::binary).put('\0');
        std::filesystem::resize_file(weights, kBundleWeights, ec);
        const parakeet::FitResult mixed = project(linked, coreml_options(cache));
        print("generated + adjacent bundle", mixed);
        const uint64_t expected = residents + 2 * kBundleWeights + 160ull * 1024 * 1024 +
                                  std::max(transient, kBundleWeights / 4);
        std::printf("expected %llu\n", (unsigned long long) expected);
        check(mixed.coreml.bytes == expected,
              "an adjacent bundle above the templates: the charge holds every capacity and the bundle together");
        const parakeet::FitResult tight_fit =
            project_free(linked, coreml_options(cache), mixed.ggml_required_bytes + expected - 1024 * 1024);
        check(tight_fit.ggml_only_fits && !tight_fit.coreml.fits,
              "free memory 1 MiB under that charge: ggml-only fits, the Core ML verdict fails");
        parakeet::CoremlOptions under = coreml_options(cache);
        under.memory_budget_bytes = static_cast<int64_t>(mixed.ggml_required_bytes + expected - 1024 * 1024);
        check(contains(engine_status(linked, under), "\"generation\":{\"reason\":\"memory\",\"supported\":false}"),
              "a budget 1 MiB under that charge: generation off, reason memory");
        std::filesystem::remove_all(dir, ec);
    }

    std::filesystem::remove_all(cache, ec);
    std::printf("%s\n", g_passed ? "PASS" : "FAIL");
    return g_passed ? 0 : 1;
}
