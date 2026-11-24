#include "cosyvoice_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int      kSpeechVocabSize      = 6561;
constexpr int      kPromptTokenCount     = 48;
constexpr int      kPromptTokenStride    = 37;
constexpr int      kPromptTokenOffset    = 11;
constexpr int      kSpeechTokenCount     = 96;
constexpr int      kSpeechTokenStride    = 91;
constexpr int      kSpeechTokenOffset    = 5;
constexpr int      kTokenMelRatio        = 2;
constexpr int      kMelChannels          = 80;
constexpr int      kSpeakerEmbeddingSize = 192;
constexpr unsigned kPromptFeatSeed       = 7;
constexpr unsigned kEmbeddingSeed        = 8;
constexpr unsigned kHiftMelSeed          = 11;
constexpr float    kLogMelMean           = -4.0f;
constexpr float    kLogMelStddev         = 2.0f;
constexpr int      kHiftMelFrames        = 120;
constexpr int      kHiftSynthSeed        = 42;
constexpr int      kVoicingPeriodFrames  = 20;
constexpr float    kVoicedF0BaseHz       = 150.0f;
constexpr float    kVoicedF0SwingHz      = 30.0f;
constexpr float    kF0PhasePerFrame      = 0.2f;
constexpr double   kDefaultMinCosine     = 0.999;
constexpr double   kDefaultFlowMaxAbs    = 1.0;
constexpr double   kDefaultHiftMaxAbs    = 0.1;

struct tier_args {
    std::string stage, ref_gguf, tier_gguf;
    double min_cosine = kDefaultMinCosine;
    double max_abs    = -1.0;
    bool   ok         = false;
};

struct tier_metrics {
    double cosine = 0, max_abs = 0;
};

bool parse_bounded_arg(const char * s, double lo, double hi, double & out) {
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || !end || *end != '\0' || !(v >= lo && v <= hi)) return false;
    out = v;
    return true;
}

tier_args parse_tier_args(int argc, char ** argv) {
    tier_args args;
    bool parsed = true;
    for (int i = 1; i < argc && parsed; ++i) {
        std::string a = argv[i];
        if (a == "--stage" && i + 1 < argc) args.stage = argv[++i];
        else if (a == "--ref-gguf" && i + 1 < argc) args.ref_gguf = argv[++i];
        else if (a == "--tier-gguf" && i + 1 < argc) args.tier_gguf = argv[++i];
        else if (a == "--min-cosine" && i + 1 < argc) parsed = parse_bounded_arg(argv[++i], 0.0, 1.0, args.min_cosine);
        else if (a == "--max-abs" && i + 1 < argc) parsed = parse_bounded_arg(argv[++i], 0.0, 1e9, args.max_abs);
        else parsed = false;
    }
    args.ok = parsed && !args.ref_gguf.empty() && !args.tier_gguf.empty() &&
              (args.stage == "flow" || args.stage == "hift");
    if (args.ok && args.max_abs < 0) {
        args.max_abs = args.stage == "flow" ? kDefaultFlowMaxAbs : kDefaultHiftMaxAbs;
    }
    return args;
}

void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s --stage {flow|hift} --ref-gguf F32.gguf --tier-gguf TIER.gguf\n"
            "          [--min-cosine %.3f] [--max-abs BOUND]\n"
            "runs the stage on both GGUFs over identical synthetic inputs and gates\n"
            "the tier output against the f32 output (cosine + max absolute difference);\n"
            "the hift stage pins f0 so the gate measures weight precision, not phase\n"
            "cosine thresholds take [0,1]; the abs bound takes a nonnegative value\n",
            argv0, kDefaultMinCosine);
}

tier_metrics compute_tier_metrics(const std::vector<float> & ref, const std::vector<float> & tier) {
    tier_metrics m;
    double dot = 0, nr = 0, nt = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i], t = tier[i];
        dot += r * t; nr += r * r; nt += t * t;
        m.max_abs = std::max(m.max_abs, (double)std::fabs(r - t));
    }
    m.cosine = dot / (std::sqrt(nr) * std::sqrt(nt));
    return m;
}

std::vector<int> make_token_ids(int count, int stride, int offset) {
    std::vector<int> ids(count);
    for (int i = 0; i < count; ++i) ids[i] = (i * stride + offset) % kSpeechVocabSize;
    return ids;
}

std::vector<float> make_normal_values(size_t count, unsigned seed, float mean, float stddev) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> values(count);
    for (float & v : values) v = dist(rng);
    return values;
}

std::vector<float> make_f0_pattern(int frames) {
    std::vector<float> f0(frames);
    for (int t = 0; t < frames; ++t) {
        const bool voiced = (t / kVoicingPeriodFrames) % 2 == 0;
        f0[t] = voiced
            ? kVoicedF0BaseHz + kVoicedF0SwingHz * std::sin(kF0PhasePerFrame * (float)t)
            : 0.0f;
    }
    return f0;
}

struct flow_inputs {
    std::vector<int>   prompt_token, speech_tokens;
    std::vector<float> prompt_feat, embedding;
    int mel_len1 = 0;
};

flow_inputs make_flow_inputs() {
    flow_inputs in;
    in.prompt_token  = make_token_ids(kPromptTokenCount, kPromptTokenStride, kPromptTokenOffset);
    in.speech_tokens = make_token_ids(kSpeechTokenCount, kSpeechTokenStride, kSpeechTokenOffset);
    in.mel_len1      = kTokenMelRatio * kPromptTokenCount;
    in.prompt_feat   = make_normal_values((size_t)in.mel_len1 * kMelChannels,
                                          kPromptFeatSeed, kLogMelMean, kLogMelStddev);
    in.embedding     = make_normal_values(kSpeakerEmbeddingSize, kEmbeddingSeed, 0.0f, 1.0f);
    return in;
}

std::vector<float> run_flow(const std::string & gguf, const flow_inputs & in, int & mel_len) {
    model_ctx m = cosyvoice_load_gguf(gguf);
    std::vector<float> mel = cosyvoice_flow_run(m, in.prompt_token, in.speech_tokens,
                                                in.prompt_feat, in.mel_len1,
                                                in.embedding, mel_len);
    cosyvoice_free(m);
    return mel;
}

std::vector<float> run_hift(const std::string & gguf, const std::vector<float> & mel,
                            const std::vector<float> & f0) {
    model_ctx m = cosyvoice_load_gguf(gguf);
    std::vector<float> wav = cosyvoice_hift_synth(m, mel, kHiftMelFrames, kHiftSynthSeed,
                                                  nullptr, &f0);
    cosyvoice_free(m);
    return wav;
}

bool compare_flow_tiers(const tier_args & args, tier_metrics & metrics) {
    const flow_inputs in = make_flow_inputs();
    int ref_len = 0, tier_len = 0;
    const std::vector<float> ref  = run_flow(args.ref_gguf, in, ref_len);
    const std::vector<float> tier = run_flow(args.tier_gguf, in, tier_len);
    if (ref_len != tier_len || ref.size() != tier.size()) {
        fprintf(stderr, "FAIL: tier mel shape differs (ref %d frames, tier %d frames)\n",
                ref_len, tier_len);
        return false;
    }
    metrics = compute_tier_metrics(ref, tier);
    return true;
}

bool compare_hift_tiers(const tier_args & args, tier_metrics & metrics) {
    const std::vector<float> mel = make_normal_values((size_t)kMelChannels * kHiftMelFrames,
                                                      kHiftMelSeed, kLogMelMean, kLogMelStddev);
    const std::vector<float> f0  = make_f0_pattern(kHiftMelFrames);
    const std::vector<float> ref  = run_hift(args.ref_gguf, mel, f0);
    const std::vector<float> tier = run_hift(args.tier_gguf, mel, f0);
    if (ref.size() != tier.size()) {
        fprintf(stderr, "FAIL: tier waveform length differs (ref %zu, tier %zu)\n",
                ref.size(), tier.size());
        return false;
    }
    metrics = compute_tier_metrics(ref, tier);
    return true;
}

int report_comparison(const tier_args & args, const tier_metrics & metrics) {
    fprintf(stderr, "%s tier-vs-f32: cosine = %.6f (threshold %.4f)  max|diff| = %.4f (bound %.4f)\n",
            args.stage.c_str(), metrics.cosine, args.min_cosine, metrics.max_abs, args.max_abs);
    if (!(metrics.cosine >= args.min_cosine) || !(metrics.max_abs <= args.max_abs)) {
        fprintf(stderr, "FAIL: tier output diverges from f32\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const tier_args args = parse_tier_args(argc, argv);
    if (!args.ok) {
        print_usage(argv[0]);
        return 2;
    }
    tier_metrics metrics;
    const bool comparable = args.stage == "flow"
        ? compare_flow_tiers(args, metrics)
        : compare_hift_tiers(args, metrics);
    if (!comparable) return 1;
    return report_comparison(args, metrics);
}
