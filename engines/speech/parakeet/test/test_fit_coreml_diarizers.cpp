// fit_params must charge the Core ML diarizer sidecars beside a Nemotron 3
// Diarization or Sortformer v2.1 GGUF: the bundles construction loads (the
// smallest for the default prewarm, every one for prewarm all) together with
// the ones the workload can load, plus the largest load transient once. A
// short offline input loads only the smallest bundle, a long Nemotron input
// reaches every bundle up to the one covering its long-form block, a long
// Sortformer input exceeds them all and runs on ggml (only the construction
// bundle stays charged), the charge decides the verdict on unified memory,
// and PARAKEET_COREML_DIARIZER_DISABLE removes it. parakeet-fit-params --json
// reports the same figure. Exits 77 without sidecars.

#include "parakeet/cli.h"
#include "parakeet/fit.h"

#include "coreml/coreml_diarizers.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr int kSkip = 77;
constexpr int kGpuLayers = 1;
constexpr float kShortSeconds = 27.3f;
constexpr float kLongSeconds = 273.0f;
constexpr uint64_t kDefaultMargin = 256ull * 1024 * 1024;
constexpr int kAllFrames = std::numeric_limits<int>::max();
constexpr int kLongFormBlockFrames = 745;
constexpr const char * kNemotronType = "nemotron-diarization";
constexpr const char * kReportLine = "core ml sidecars:";
constexpr const char * kJsonKey = "\"coremlBytes\": ";
constexpr const char * kJsonCapture = "fit-coreml-diarizers.json";

bool g_passed = true;

void check(bool condition, const char * label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label);
    g_passed = g_passed && condition;
}

parakeet::FitResult project(const std::string & gguf, float seconds, uint64_t margin,
                            parakeet::CoremlOptions::Prewarm prewarm = parakeet::CoremlOptions::Prewarm::unset) {
    parakeet::FitOptions opts;
    opts.model_gguf_path = gguf;
    opts.audio_seconds = seconds;
    opts.n_gpu_layers = kGpuLayers;
    opts.margin_bytes = margin;
    opts.coreml.prewarm = prewarm;
    return parakeet::fit_params(opts);
}

// Today's charge (FitResult::coreml_bytes, part of the top-level verdict):
// the bundles the workload can load, resident.
uint64_t legacy(const std::vector<parakeet::CoremlDiarizerBundle> & bundles, int frames = kAllFrames) {
    return parakeet::coreml_diarizer_resident_bytes(bundles, {{1, frames}});
}

// The Core ML-inclusive charge (FitResult::coreml.bytes) of bundles resident
// together: their resident bytes plus the largest one's load transient (a
// quarter of its weights on the default CPU and Neural Engine units).
uint64_t charge(const std::vector<parakeet::CoremlDiarizerBundle> & bundles,
                int frames = kAllFrames) {
    uint64_t transient = 0;
    for (const auto & b : parakeet::coreml_diarizer_loaded_bundles(bundles, {{1, frames}})) {
        transient = std::max(transient, b.weights_bytes / 4);
    }
    return parakeet::coreml_diarizer_resident_bytes(bundles, {{1, frames}}) + transient;
}

bool reports_sidecars(const parakeet::FitResult & result) {
    return result.report.find(kReportLine) != std::string::npos;
}

uint64_t margin_between(const parakeet::FitResult & result) {
    const uint64_t without_sidecars = result.device.total_bytes + result.host_bytes;
    return result.device_free_bytes - without_sidecars - result.coreml_bytes / 2;
}

std::vector<char *> pointers_to(std::vector<std::string> & args) {
    std::vector<char *> pointers(args.size());
    std::transform(args.begin(), args.end(), pointers.begin(),
                   [](std::string & arg) { return arg.data(); });
    return pointers;
}

std::string captured_cli_json(const std::string & gguf) {
    const std::string seconds = std::to_string(kShortSeconds);
    const std::string layers = std::to_string(kGpuLayers);
    std::vector<std::string> args = {"parakeet-fit-params", "--model", gguf, "--audio-seconds",
                                     seconds, "--n-gpu-layers", layers, "--json"};
    std::vector<char *> argv = pointers_to(args);
    std::fflush(stdout);
    const int saved = dup(STDOUT_FILENO);
    if (std::freopen(kJsonCapture, "w", stdout) != nullptr) {
        parakeet_fit_cli_main(static_cast<int>(argv.size()), argv.data());
        std::fflush(stdout);
    }
    dup2(saved, STDOUT_FILENO);
    close(saved);
    std::ifstream file(kJsonCapture);
    std::stringstream text;
    text << file.rdbuf();
    std::remove(kJsonCapture);
    return text.str();
}

void check_cli_json(const std::string & gguf, uint64_t expected) {
    const std::string json = captured_cli_json(gguf);
    const std::string field = kJsonKey + std::to_string(expected) + ",";
    check(json.find(field) != std::string::npos, "fit CLI json reports coremlBytes");
}

void check_workloads(const std::string & gguf,
                     const std::vector<parakeet::CoremlDiarizerBundle> & bundles) {
    const parakeet::FitResult offline = project(gguf, kShortSeconds, kDefaultMargin);
    check(offline.coreml_bytes == legacy({bundles.front()}), "short input: today's charge is the smallest bundle");
    // Nemotron: if the smallest bundle fails to load, its calls go to a larger
    // one, which is then resident in its place, one at a time; the Core ML
    // charge is the largest such bundle alone. Sortformer: the smallest bundle.
    uint64_t worst_single = 0;
    for (const auto & b : bundles) worst_single = std::max(worst_single, charge({b}));
    const bool reaches_larger = offline.model_type == kNemotronType;
    check(offline.coreml.bytes == (reaches_larger ? worst_single : charge({bundles.front()})),
          "short input: the Core ML charge is the bundle its calls can reach that costs most, and its transient");
    check(reports_sidecars(offline), "report lists the sidecar charge");
    check_cli_json(gguf, offline.coreml_bytes);
    const parakeet::FitResult long_input = project(gguf, kLongSeconds, kDefaultMargin);
    const bool nemotron = long_input.model_type == kNemotronType;
    check(long_input.coreml_bytes == (nemotron ? legacy(bundles, kLongFormBlockFrames) : 0),
          nemotron ? "long-form input: today's charge is every bundle up to its block"
                   : "input above every bundle: today's charge is none");
    check(long_input.coreml.bytes == (nemotron ? charge(bundles) : charge({bundles.front()})),
          nemotron ? "long-form input: the Core ML charge is every bundle (its passes reach the largest)"
                   : "input above every bundle: the Core ML charge is the construction bundle");
    // prewarm all loads every capacity at construction, whatever the input;
    // today's charge does not change with prewarm.
    const parakeet::FitResult all = project(gguf, kShortSeconds, kDefaultMargin, parakeet::CoremlOptions::Prewarm::all);
    check(all.coreml.bytes == charge(bundles) && all.coreml_bytes == offline.coreml_bytes,
          "prewarm all: the Core ML charge is every bundle for a short input; today's charge is unchanged");
    const parakeet::FitResult none = project(gguf, kLongSeconds, kDefaultMargin, parakeet::CoremlOptions::Prewarm::none);
    check(none.coreml.bytes == (nemotron ? charge(bundles) : 0),
          "prewarm none: the Core ML charge is only what the workload loads");
    parakeet::FitOptions off;
    off.model_gguf_path = gguf;
    off.audio_seconds = kShortSeconds;
    off.n_gpu_layers = kGpuLayers;
    off.coreml.use = parakeet::CoremlOptions::Use::off;
    const parakeet::FitResult disabled = parakeet::fit_params(off);
    check(disabled.coreml_bytes == 0 && disabled.coreml.bytes == 0, "use = off charges no diarizer sidecar");
}

void check_verdict(const std::string & gguf) {
    const parakeet::FitResult baseline = project(gguf, kShortSeconds, kDefaultMargin);
    if (!baseline.device_shares_host_memory || !baseline.fits) return;
    const uint64_t margin = margin_between(baseline);
    check(!project(gguf, kShortSeconds, margin).fits, "sidecar charge makes a tight budget fail");
    setenv(parakeet::kCoremlDiarizerDisableEnv, "1", 1);
    const parakeet::FitResult disabled = project(gguf, kShortSeconds, margin);
    unsetenv(parakeet::kCoremlDiarizerDisableEnv);
    check(disabled.fits, "the same budget fits with the diarizers disabled");
    check(disabled.coreml_bytes == 0 && !reports_sidecars(disabled),
          "disable env removes the sidecar charge");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }
    const std::vector<parakeet::CoremlDiarizerBundle> bundles =
        parakeet::list_coreml_diarizer_bundles(argv[1]);
    if (bundles.empty()) {
        std::printf("SKIP: no Core ML diarizer beside %s\n", argv[1]);
        return kSkip;
    }
    check_workloads(argv[1], bundles);
    check_verdict(argv[1]);
    std::printf("[fit-coreml-diarizers] %s\n", g_passed ? "PASS" : "FAIL");
    return g_passed ? 0 : 1;
}
