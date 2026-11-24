#include "chatterbox_coreml.h"

#include "coreml_sidecar_path.h"
#include "ggml.h"
#include "gguf.h"
#include "npy.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#include "qvac_coreml/manager.h"
#endif

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef TTS_CPP_USE_COREML
namespace pcm = QVAC_COREML_NS;

// One S3Gen call's Core ML models, acquired at its start.
struct S3genCoremlPlan {
    struct stage {
        pcm::Handle model;
        int bucket = -1;
        bool pending = false;
        bool present = false;
        // The model failed a prediction during this call (plan_retire): the
        // plan runs the stage on ggml from then on, a replay included.
        mutable std::atomic<bool> dropped{false};
        bool usable() const { return model && !dropped.load(std::memory_order_acquire); }
    };
    stage stages[tts_cpp::chatterbox::detail::S3GEN_STAGES];
    bool strict = false;
    mutable std::atomic<bool> defer{false};
};
#else
struct S3genCoremlPlan {
    bool strict = false;
    mutable std::atomic<bool> defer{false};
};
#endif

namespace tts_cpp::chatterbox::detail {
namespace {

constexpr const char * STAGE_NAMES[] = {"encoder", "cfm", "hift"};

#ifdef TTS_CPP_USE_COREML
constexpr const char * STAGE_SUFFIXES[] = {COREML_ENCODER_SUFFIX, COREML_CFM_SUFFIX, COREML_HIFT_SUFFIX};
constexpr int MEL = 80;
constexpr int EMBED = 512;
constexpr int TIME_EMBED = 1024;
constexpr int MAX_RANK = 8;
constexpr int FUNCTION_LIST_BYTES = 4096;
constexpr const char * ENGINE_NAME = "chatterbox";
constexpr const char * WEIGHTS_FILE = "/weights/weight.bin";
// ANECompilerService peaked at about 1.6x the weights compiling a 199 MB model.
constexpr double ANE_COMPILE_RESERVE_FACTOR = 1.6;

std::vector<int64_t> dims_of(moss_coreml_model * model, const char * name) {
    int64_t dims[MAX_RANK] = {};
    const int rank = moss_coreml_feature_dims(model, name, dims, MAX_RANK);
    return std::vector<int64_t>(dims, dims + std::max(rank, 0));
}

std::string dims_text(const std::vector<int64_t> & dims) {
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < dims.size(); ++i) text << (i ? ", " : "") << dims[i];
    text << ']';
    return text.str();
}

// One prediction feature: its name, the shape this call packs it as, and a
// buffer of exactly that many floats.
struct input_feature {
    const char * name;
    std::vector<int64_t> dims;
    const std::vector<float> * data;
};
struct output_feature {
    const char * name;
    std::vector<int64_t> dims;
    std::vector<float> * data;
};

// The shared runner copies every feature by the shape the model declares, so
// a sidecar declaring any other shape than the buffer packed for it would be
// read or written out of bounds. Every feature is checked first; a mismatch
// is an incompatible export and throws, which retires the function (or fails
// a strict call) before anything is copied.
void predict(moss_coreml_model * model, const std::vector<input_feature> & inputs,
             const std::vector<output_feature> & outputs, const char * stage) {
    auto check = [&](const char * name, const std::vector<int64_t> & dims, size_t size) {
        const std::vector<int64_t> declared = dims_of(model, name);
        int64_t elements = 1;
        for (int64_t d : dims) elements *= d;
        if (declared != dims || (int64_t) size != elements) {
            throw coreml_stage_failure(std::string("chatterbox: the Core ML ") + stage + " sidecar declares " + name +
                                       " " + dims_text(declared) + " where this call packs " + dims_text(dims));
        }
    };
    std::vector<const char *> in_names, out_names;
    std::vector<const float *> in_data;
    std::vector<float *> out_data;
    for (const input_feature & f : inputs) {
        check(f.name, f.dims, f.data->size());
        in_names.push_back(f.name);
        in_data.push_back(f.data->data());
    }
    for (const output_feature & f : outputs) {
        check(f.name, f.dims, f.data->size());
        out_names.push_back(f.name);
        out_data.push_back(f.data->data());
    }
    if (moss_coreml_predict(model, in_names.data(), in_data.data(), (int) in_names.size(), out_names.data(),
                            out_data.data(), (int) out_names.size()) != 0) {
        throw coreml_stage_failure(std::string("chatterbox: Core ML ") + stage + " prediction failed");
    }
}

void require_finite(const std::vector<float> & values, const char * stage) {
    if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); })) {
        throw coreml_stage_failure(std::string("chatterbox: non-finite Core ML ") + stage + " output");
    }
}

void maybe_fail(s3gen_stage stage) {
    int expected = (int) stage;
    if (coreml_test().fail_stage.compare_exchange_strong(expected, -1)) {
        throw coreml_stage_failure(std::string("chatterbox: injected Core ML ") + to_string(stage) + " failure");
    }
}

uint64_t file_bytes(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? static_cast<uint64_t>(st.st_size) : 0;
}

template <class E, class F> F map_enum(E v) {
    return static_cast<F>(static_cast<int>(v));
}

pcm::Options to_runtime_options(const CoremlOptions & o) {
    // The public enums mirror the runtime's member for member.
    static_assert(static_cast<int>(CoremlOptions::Use::require) == static_cast<int>(pcm::Use::require), "Use");
    static_assert(static_cast<int>(CoremlOptions::Generate::background) ==
                      static_cast<int>(pcm::Generate::background), "Generate");
    static_assert(static_cast<int>(CoremlOptions::Wait::call) == static_cast<int>(pcm::Wait::call), "Wait");
    static_assert(static_cast<int>(CoremlOptions::Prewarm::all) == static_cast<int>(pcm::Prewarm::all), "Prewarm");
    pcm::Options r;
    r.use = map_enum<CoremlOptions::Use, pcm::Use>(o.use);
    r.generate = map_enum<CoremlOptions::Generate, pcm::Generate>(o.generate);
    r.wait = map_enum<CoremlOptions::Wait, pcm::Wait>(o.wait);
    r.prewarm = map_enum<CoremlOptions::Prewarm, pcm::Prewarm>(o.prewarm);
    r.cache_dir = o.cache_dir;
    r.compute_units = o.compute_units;
    r.strict_wait_ms = o.strict_wait_ms;
    r.memory_budget_bytes = o.memory_budget_bytes;
    return r;
}

// Load + warm-up prediction of one function in a new process whose ANE plan
// is cached (an M2): the warm time a slot expects until its hint holds an
// observed one. The load grows with the whole multi-function program (a
// seven-width CFM function about 1.08 s, a three-width one 0.48 s); the
// warm-up prediction grows with the width.
int64_t expected_warm_load_ms(s3gen_stage stage, int width, size_t functions, bool meanflow) {
    const bool wide_set = functions >= 5;
    switch (stage) {
    case s3gen_stage::encoder:
        return (wide_set ? 155 : 80) + (int64_t) std::lround(16.6 + 0.254 * std::max(0, width - 256));
    case s3gen_stage::cfm:
        return (wide_set ? 1080 : 480) +
               (int64_t) std::lround(meanflow ? 22.9 + 0.176 * std::max(0, width - 512)
                                              : 54.0 + 0.384 * std::max(0, width - 512));
    case s3gen_stage::hift:
        return 120;
    }
    return 0;
}

// The runtime's loader for every Chatterbox function (a plain function
// pointer: it captures nothing; units arrive resolved in the request).
pcm::LoadResult load_function(const pcm::LoadRequest & req) {
    coreml_test().record_load(req.stage + "/" + req.function);
    if (const int delay = coreml_test().load_delay_ms.load(); delay > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
    while (coreml_test().hold_loads.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    pcm::LoadResult result;
    const char * units = pcm::to_string(req.units);
    moss_coreml_model * raw = req.function == "main"
        ? moss_coreml_load(req.artifact_path.c_str(), units)
        : moss_coreml_load_function(req.artifact_path.c_str(), units, req.function.c_str());
    if (raw == nullptr) {
        result.error = pcm::Error::deterministic("chatterbox: cannot load Core ML " + req.stage + " " +
                                                 req.artifact_path + " (" + req.function + ")");
        return result;
    }
    result.model = std::shared_ptr<void>(raw, [](void * m) { moss_coreml_free(static_cast<moss_coreml_model *>(m)); });
    return result;
}

// One prediction on neutral inputs right after the load, so a call's first
// prediction does not pay the Neural Engine's first-run cost (part of the
// expected warm time).
pcm::Error warm_function(void * model, const pcm::LoadRequest & req) {
    auto * m = static_cast<moss_coreml_model *>(model);
    std::vector<const char *> in_names, out_names;
    if (req.stage == "encoder") {
        in_names = {"embedding", "valid"};
        out_names = {"mu"};
    } else if (req.stage == "cfm") {
        in_names = {"x", "mu", "cond", "spks", "t_emb", "valid"};
        out_names = {"velocity"};
    } else {
        in_names = {"mel", "source", "valid"};
        out_names = {"spectrum"};
    }
    auto elements = [&](const char * name) {
        const auto dims = dims_of(m, name);
        if (dims.empty()) return (int64_t) 0;
        int64_t n = 1;
        for (int64_t d : dims) n *= d;
        return n;
    };
    std::vector<std::vector<float>> inputs, outputs;
    for (const char * name : in_names) {
        const int64_t n = elements(name);
        if (n <= 0) return pcm::Error::deterministic(std::string("chatterbox: sidecar has no input ") + name);
        inputs.emplace_back((size_t) n, std::strcmp(name, "valid") == 0 ? 1.0f : 0.0f);
    }
    for (const char * name : out_names) {
        const int64_t n = elements(name);
        if (n <= 0) return pcm::Error::deterministic(std::string("chatterbox: sidecar has no output ") + name);
        outputs.emplace_back((size_t) n, 0.0f);
    }
    std::vector<const float *> in_data;
    std::vector<float *> out_data;
    for (auto & v : inputs) in_data.push_back(v.data());
    for (auto & v : outputs) out_data.push_back(v.data());
    if (moss_coreml_predict(m, in_names.data(), in_data.data(), (int) in_names.size(), out_names.data(),
                            out_data.data(), (int) out_names.size()) != 0) {
        return pcm::Error::transient("chatterbox: Core ML " + req.stage + " warm-up prediction failed");
    }
    return {};
}

#endif

} // namespace

// ----------------------------------------------------------- GGUF facts --

s3gen_gguf_facts s3gen_read_facts(const std::string & path) {
    static std::mutex mu;
    static std::map<std::string, std::pair<std::string, s3gen_gguf_facts>> cache;
    struct stat st {};
    const std::string stamp = ::stat(path.c_str(), &st) == 0
        ? std::to_string((long long) st.st_size) + ":" + std::to_string((long long) st.st_mtime) : std::string();
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(path);
        if (it != cache.end() && it->second.first == stamp) return it->second.second;
    }
    s3gen_gguf_facts f;
    ggml_context * meta = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/&meta};
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return f;
    auto read_u32 = [&](const char * key, int & out) {
        const int64_t k = gguf_find_key(g, key);
        if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_UINT32) out = (int) gguf_get_val_u32(g, k);
    };
    read_u32("s3gen.encoder.token_mel_ratio", f.token_mel_ratio);
    read_u32("s3gen.speech_vocab_size", f.speech_vocab);
    // The loader's own limit: a valid ratio plans with exactly the value the
    // call renders with (an invalid one fails the load).
    if (f.token_mel_ratio < 1 || f.token_mel_ratio > S3GEN_MAX_TOKEN_MEL_RATIO) f.token_mel_ratio = 2;
    const int64_t k_mf = gguf_find_key(g, "s3gen.meanflow");
    if (k_mf >= 0 && gguf_get_kv_type(g, k_mf) == GGUF_TYPE_BOOL) f.meanflow = gguf_get_val_bool(g, k_mf);
    if (meta != nullptr) {
        if (ggml_tensor * t = ggml_get_tensor(meta, "s3gen/builtin/prompt_token")) {
            f.builtin_prompt_tokens = (int) ggml_nelements(t);
        }
    }
    gguf_free(g);
    if (meta != nullptr) ggml_free(meta);
    std::lock_guard<std::mutex> lock(mu);
    cache[path] = {stamp, f};
    return f;
}

s3gen_call_shape s3gen_call_shape_for(const s3gen_synthesize_opts & opts, const std::vector<int32_t> & speech_tokens,
                                      const s3gen_gguf_facts & facts) {
    const bool has_prompt_token = (opts.prompt_token_view_data && opts.prompt_token_view_size > 0) ||
                                  !opts.prompt_token_override.empty();
    const bool has_embedding = (opts.embedding_view_data && opts.embedding_view_size > 0) ||
                               !opts.embedding_override.empty();
    const bool has_prompt_feat = (opts.prompt_feat_view_data && opts.prompt_feat_view_size > 0) ||
                                 !opts.prompt_feat_override.empty();
    int prompt = 0;
    if (opts.ref_dir.empty() && !has_embedding && !has_prompt_feat && !has_prompt_token) {
        prompt = facts.builtin_prompt_tokens;
    } else if (opts.prompt_token_view_data && opts.prompt_token_view_size > 0) {
        prompt = (int) opts.prompt_token_view_size;
    } else if (!opts.prompt_token_override.empty()) {
        prompt = (int) opts.prompt_token_override.size();
    } else if (!opts.ref_dir.empty()) {
        prompt = (int) npy_load(opts.ref_dir + "/prompt_token.npy").n_elements();
    }
    int speech = 0;
    for (int32_t t : speech_tokens) speech += (t >= 0 && t < facts.speech_vocab) ? 1 : 0;
    return s3gen_shape(prompt, speech, facts.token_mel_ratio, opts.append_lookahead_silence, opts.finalize);
}

// ---------------------------------------------------------------- options --

bool coreml_valid_units(const std::string & units) {
    return units == "all" || units == "cpu_and_ane" || units == "cpu_and_gpu" || units == "cpu_only";
}

CoremlOptions merge_legacy_coreml(CoremlOptions options, const std::string & units, int strict, bool disabled) {
    if (!options.compute_units && !units.empty()) options.compute_units = units;
    // An explicit use wins; among the aliases, disabled wins over strict.
    // The CHATTERBOX_COREML_DISABLE kill switch still overrides everything
    // when the options are resolved.
    if (options.use == CoremlOptions::Use::unset) {
        if (disabled) {
            options.use = CoremlOptions::Use::off;
        } else if (strict >= 0) {
            options.use = strict > 0 ? CoremlOptions::Use::require : CoremlOptions::Use::automatic;
        }
    }
    return options;
}

CoremlOptions call_coreml_options(const s3gen_synthesize_opts & opts) {
    return merge_legacy_coreml(opts.coreml, opts.coreml_compute_units, opts.coreml_strict, opts.coreml_disabled);
}

void validate_coreml_options(const CoremlOptions & options) {
    if (options.compute_units && !coreml_valid_units(*options.compute_units)) {
        throw std::invalid_argument("chatterbox: invalid coreml_compute_units='" + *options.compute_units +
                                    "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
    }
    if (options.strict_wait_ms && *options.strict_wait_ms < 0) {
        throw std::invalid_argument("chatterbox: Core ML strict_wait_ms must not be negative");
    }
    if (options.memory_budget_bytes && *options.memory_budget_bytes < 0) {
        throw std::invalid_argument("chatterbox: Core ML memory_budget_bytes must not be negative");
    }
}

// ----------------------------------------------------------------- naming --

std::string coreml_stage_path(const std::string & s3gen_gguf, const char * suffix) {
    return ::tts_cpp::detail::coreml_sidecar_path(s3gen_gguf, suffix);
}

bool coreml_prebuilt_route(const std::string & s3gen_gguf, const CoremlOptions & options) {
    validate_coreml_options(options);
#ifdef TTS_CPP_USE_COREML
    if (s3gen_gguf.empty()) return false;
    const pcm::Resolved resolved = pcm::resolve_coreml(to_runtime_options(options), COREML_ENV_PREFIX);
    if (!resolved.enabled()) return false;
    for (const char * suffix : STAGE_SUFFIXES) {
        if (::tts_cpp::detail::coreml_sidecar_exists(coreml_stage_path(s3gen_gguf, suffix))) return true;
    }
#else
    (void) s3gen_gguf;
#endif
    return false;
}

std::vector<int> coreml_buckets(const std::string & function_names, const std::string & prefix) {
    std::vector<int> buckets;
    std::stringstream list(function_names);
    std::string name;
    while (std::getline(list, name, ',')) {
        if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) continue;
        const std::string digits = name.substr(prefix.size());
        if (!std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
        if (digits.size() > 6) continue;
        const int width = std::atoi(digits.c_str());
        if (width > 0) buckets.push_back(width);
    }
    std::sort(buckets.begin(), buckets.end());
    buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
    return buckets;
}

int coreml_pick_bucket(const std::vector<int> & buckets, int needed) {
    for (int width : buckets) {
        if (width >= needed) return width;
    }
    return -1;
}

std::vector<hift_window> coreml_hift_windows(int frames, int window, int context) {
    std::vector<hift_window> plan;
    if (frames <= 0 || window <= 2 * context) return plan;
    for (int begin = 0; begin < frames;) {
        const int start = std::max(0, begin - context);
        const int end = start + window >= frames ? frames : start + window - context;
        plan.push_back({start, begin, end});
        begin = end;
    }
    return plan;
}

int coreml_expected_speech_tokens(size_t text_bytes) {
    return (int) std::lround(1.15 * (double) text_bytes);
}

std::vector<int> coreml_stream_windows(int speech_tokens, int first_chunk, int chunk, int left_context) {
    std::vector<int> windows;
    if (speech_tokens <= 0 || chunk <= 0) return windows;
    int start = 0;
    for (int end = std::min(speech_tokens, first_chunk > 0 ? first_chunk : chunk); start < speech_tokens;
         end = std::min(speech_tokens, end + chunk)) {
        const int from = left_context > 0 ? std::max(0, start - left_context) : 0;
        windows.push_back(end - from);
        start = end;
    }
    return windows;
}

// --------------------------------------------------------- shapes and cost --

s3gen_call_shape s3gen_shape(int prompt_tokens, int speech_tokens, int token_mel_ratio, bool append_lookahead,
                             bool finalize) {
    s3gen_call_shape shape;
    shape.tokens = std::max(0, prompt_tokens) + std::max(0, speech_tokens) +
                   (append_lookahead ? S3GEN_LOOKAHEAD_TOKENS : 0);
    shape.frames = token_mel_ratio * shape.tokens - (finalize ? 0 : S3GEN_LOOKAHEAD_TOKENS * token_mel_ratio);
    shape.frames = std::max(0, shape.frames);
    return shape;
}

double s3gen_estimate_metal_ms_frames(bool meanflow, int frames) {
    // ms = a + b * frames over a sentence-length sweep on Metal (an M2), one
    // whole S3Gen call per sentence: Turbo R^2 0.996 over 566-1200 frames,
    // Multilingual R^2 0.991 over 376-820 frames (worst measured / fitted
    // 1.06 and 1.17). The intercepts are negative, so clamp below.
    const double a = meanflow ? -1347.8 : -1992.6;
    const double b = meanflow ? 2.9867 : 9.2146;
    return std::max(0.0, a + b * (double) frames);
}

const char * to_string(s3gen_stage stage) {
    return STAGE_NAMES[(int) stage];
}

// --------------------------------------------------------- CFM integration --

int cfm_integrate(std::vector<float> & z, size_t steps, bool on_coreml, const cfm_step_fn & step,
                  const std::function<void(const coreml_stage_failure &, size_t s)> & on_failure) {
    const std::vector<float> z0 = on_coreml ? z : std::vector<float>();
    for (size_t s = 0; s < steps; ++s) {
        int rc = 0;
        try {
            rc = step(s, z, on_coreml);
        } catch (const coreml_stage_failure & e) {
            if (!on_coreml) throw;
            on_failure(e, s);
            // Latch to ggml and integrate again from the initial noise: the
            // trajectory never mixes backends.
            on_coreml = false;
            z = z0;
            s = (size_t) -1;
            continue;
        }
        if (rc != 0) return rc;
    }
    return 0;
}

// ------------------------------------------------------------ test hooks --

std::vector<std::string> coreml_test_hooks::loads() const {
    std::lock_guard<std::mutex> lock(mu_);
    return loads_;
}

void coreml_test_hooks::clear_loads() {
    std::lock_guard<std::mutex> lock(mu_);
    loads_.clear();
}

void coreml_test_hooks::record_load(const std::string & what) {
    if (!tracing.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(mu_);
    loads_.push_back(what);
}

void coreml_test_hooks::trace(const std::string & event) {
    if (!tracing.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(mu_);
    events_.push_back(event);
}

std::vector<std::string> coreml_test_hooks::events() const {
    std::lock_guard<std::mutex> lock(mu_);
    return events_;
}

void coreml_test_hooks::clear_events() {
    std::lock_guard<std::mutex> lock(mu_);
    events_.clear();
}

coreml_test_hooks & coreml_test() {
    // Leaked on purpose, like the runtime's manager: the load thread reads it
    // and must never see it destroyed by static destruction at exit.
    static auto * hooks = new coreml_test_hooks;
    return *hooks;
}

// ---------------------------------------------------------------- the plan --

#ifdef TTS_CPP_USE_COREML

plan_stage_view plan_view(const S3genCoremlPlan & plan, s3gen_stage stage) {
    const auto & st = plan.stages[(int) stage];
    plan_stage_view v;
    v.present = st.present;
    v.model = st.usable();
    v.bucket = v.model ? st.bucket : -1;
    v.pending = st.pending;
    v.label = v.model ? st.model->label() : GGML_STAGE_BACKEND;
    return v;
}

bool plan_pending(const S3genCoremlPlan & plan) {
    for (const auto & st : plan.stages) {
        if (st.pending) return true;
    }
    return false;
}

void plan_retire(const S3genCoremlPlan & plan, s3gen_stage stage, const std::string & reason) {
    const auto & st = plan.stages[(int) stage];
    if (st.model) st.model->retire(reason);
    st.dropped.store(true, std::memory_order_release);
}

bool plan_encode(const S3genCoremlPlan & plan, const std::vector<float> & embedding, int tokens, int ratio,
                 std::vector<float> & mu) {
    const auto & st = plan.stages[(int) s3gen_stage::encoder];
    if (!st.usable() || tokens <= 0 || st.bucket < tokens || (int64_t) embedding.size() != (int64_t) tokens * EMBED) {
        return false;
    }
    maybe_fail(s3gen_stage::encoder);
    const int width = st.bucket;
    std::vector<float> padded((size_t) width * EMBED, 0.0f);
    std::copy(embedding.begin(), embedding.end(), padded.begin());
    std::vector<float> valid((size_t) width, 0.0f);
    std::fill(valid.begin(), valid.begin() + tokens, 1.0f);
    std::vector<float> out((size_t) width * ratio * MEL);
    {
        auto lock = st.model->lock_for_prediction();
        predict(st.model->as<moss_coreml_model>(),
                {{"embedding", {1, width, EMBED}, &padded}, {"valid", {1, 1, width}, &valid}},
                {{"mu", {1, (int64_t) width * ratio, MEL}, &out}}, "encoder");
    }
    out.resize((size_t) tokens * ratio * MEL);
    require_finite(out, "encoder");
    mu = std::move(out);
    return true;
}

bool plan_velocity(const S3genCoremlPlan & plan, size_t step, int batch, int frames,
                   const std::vector<const float *> & x, const std::vector<const float *> & mu,
                   const std::vector<const float *> & cond, const std::vector<const float *> & spks,
                   const std::vector<const float *> & t_emb, std::vector<std::vector<float>> & out) {
    const auto & st = plan.stages[(int) s3gen_stage::cfm];
    // Callers pass the CFG pair; a batch-1 (meanflow) sidecar uses the first.
    const size_t planes = (size_t) std::max(batch, 0);
    if (!st.usable() || frames <= 0 || batch <= 0 || st.bucket < frames || x.size() < planes || mu.size() < planes ||
        cond.size() < planes || spks.size() < planes || t_emb.size() < planes) {
        return false;
    }
    {
        int expected = (int) step;
        if (coreml_test().fail_cfm_step.compare_exchange_strong(expected, -1)) {
            throw coreml_stage_failure("chatterbox: injected Core ML CFM failure at step " + std::to_string(step));
        }
    }
    maybe_fail(s3gen_stage::cfm);
    const int width = st.bucket;
    const std::vector<int64_t> plane = {batch, MEL, width};
    const size_t plane_size = (size_t) MEL * width;
    auto pack = [&](const std::vector<const float *> & sources) {
        std::vector<float> packed((size_t) batch * plane_size, 0.0f);
        for (int b = 0; b < batch; ++b) {
            for (int m = 0; m < MEL; ++m) {
                std::memcpy(packed.data() + b * plane_size + (size_t) m * width, sources[b] + (size_t) m * frames,
                            (size_t) frames * sizeof(float));
            }
        }
        return packed;
    };
    auto rows = [&](const std::vector<const float *> & vectors, int width_each) {
        std::vector<float> packed((size_t) batch * width_each);
        for (int b = 0; b < batch; ++b) {
            std::memcpy(packed.data() + (size_t) b * width_each, vectors[b], (size_t) width_each * sizeof(float));
        }
        return packed;
    };
    const std::vector<float> xs = pack(x), mus = pack(mu), conds = pack(cond);
    const std::vector<float> speakers = rows(spks, MEL), times = rows(t_emb, TIME_EMBED);
    std::vector<float> valid((size_t) width, 0.0f);
    std::fill(valid.begin(), valid.begin() + frames, 1.0f);
    std::vector<float> velocity((size_t) batch * plane_size);
    {
        auto lock = st.model->lock_for_prediction();
        predict(st.model->as<moss_coreml_model>(),
                {{"x", plane, &xs}, {"mu", plane, &mus}, {"cond", plane, &conds}, {"spks", {batch, MEL}, &speakers},
                 {"t_emb", {batch, TIME_EMBED}, &times}, {"valid", {1, 1, width}, &valid}},
                {{"velocity", plane, &velocity}}, "CFM");
    }
    out.assign(batch, std::vector<float>((size_t) MEL * frames));
    for (int b = 0; b < batch; ++b) {
        for (int m = 0; m < MEL; ++m) {
            std::memcpy(out[b].data() + (size_t) m * frames, velocity.data() + b * plane_size + (size_t) m * width,
                        (size_t) frames * sizeof(float));
        }
        require_finite(out[b], "CFM");
    }
    return true;
}

bool plan_vocode(const S3genCoremlPlan & plan, const std::vector<float> & mel, int frames,
                 const std::vector<float> & source_stft, std::vector<float> & spectrum) {
    const auto & st = plan.stages[(int) s3gen_stage::hift];
    const int64_t stft = (int64_t) frames * HIFT_STFT_PER_FRAME + 1;
    if (!st.usable() || frames <= 0 || (int64_t) mel.size() != (int64_t) MEL * frames ||
        (int64_t) source_stft.size() != HIFT_STFT_BINS * stft) {
        return false;
    }
    maybe_fail(s3gen_stage::hift);
    auto * model = st.model->as<moss_coreml_model>();
    auto lock = st.model->lock_for_prediction();
    // The window is the sidecar's; predict() checks every feature against it.
    const std::vector<int64_t> mel_dims = dims_of(model, "mel");
    const int window = mel_dims.size() == 3 ? (int) mel_dims[2] : 0;
    if (window <= 2 * HIFT_CONTEXT_FRAMES) {
        throw coreml_stage_failure("chatterbox: the Core ML HiFT sidecar declares mel " + dims_text(mel_dims) +
                                   ", narrower than its " + std::to_string(2 * HIFT_CONTEXT_FRAMES) +
                                   " context frames");
    }
    const int64_t window_stft = (int64_t) window * HIFT_STFT_PER_FRAME + 1;
    const std::vector<int64_t> spectrum_dims = {1, HIFT_STFT_BINS, window_stft};
    std::vector<float> mel_in((size_t) MEL * window), source_in((size_t) HIFT_STFT_BINS * window_stft);
    std::vector<float> valid((size_t) window), out((size_t) HIFT_STFT_BINS * window_stft);
    std::vector<float> result((size_t) HIFT_STFT_BINS * stft, 0.0f);
    for (const hift_window & w : coreml_hift_windows(frames, window, HIFT_CONTEXT_FRAMES)) {
        const int count = std::min(window, frames - w.start);
        const int64_t source_count = (int64_t) count * HIFT_STFT_PER_FRAME + 1;
        const int64_t source_start = (int64_t) w.start * HIFT_STFT_PER_FRAME;
        std::fill(mel_in.begin(), mel_in.end(), 0.0f);
        std::fill(source_in.begin(), source_in.end(), 0.0f);
        std::fill(valid.begin(), valid.end(), 0.0f);
        std::fill(valid.begin(), valid.begin() + count, 1.0f);
        for (int m = 0; m < MEL; ++m) {
            std::memcpy(mel_in.data() + (size_t) m * window, mel.data() + (size_t) m * frames + w.start,
                        (size_t) count * sizeof(float));
        }
        for (int c = 0; c < HIFT_STFT_BINS; ++c) {
            std::memcpy(source_in.data() + (size_t) c * window_stft,
                        source_stft.data() + (size_t) c * stft + source_start, (size_t) source_count * sizeof(float));
        }
        predict(model,
                {{"mel", {1, MEL, window}, &mel_in}, {"source", spectrum_dims, &source_in},
                 {"valid", {1, 1, window}, &valid}},
                {{"spectrum", spectrum_dims, &out}}, "HiFT");
        const int64_t keep_begin = (int64_t) (w.keep_begin - w.start) * HIFT_STFT_PER_FRAME;
        const int64_t keep_count = (int64_t) (w.keep_end - w.keep_begin) * HIFT_STFT_PER_FRAME +
                                   (w.keep_end == frames ? 1 : 0);
        for (int c = 0; c < HIFT_STFT_BINS; ++c) {
            std::memcpy(result.data() + (size_t) c * stft + (size_t) w.keep_begin * HIFT_STFT_PER_FRAME,
                        out.data() + (size_t) c * window_stft + keep_begin, (size_t) keep_count * sizeof(float));
        }
    }
    require_finite(result, "HiFT");
    spectrum = std::move(result);
    return true;
}

#else  // !TTS_CPP_USE_COREML

plan_stage_view plan_view(const S3genCoremlPlan &, s3gen_stage) {
    plan_stage_view v;
    v.label = GGML_STAGE_BACKEND;
    return v;
}
bool plan_pending(const S3genCoremlPlan &) { return false; }
void plan_retire(const S3genCoremlPlan &, s3gen_stage, const std::string &) {}
bool plan_encode(const S3genCoremlPlan &, const std::vector<float> &, int, int, std::vector<float> &) { return false; }
bool plan_velocity(const S3genCoremlPlan &, size_t, int, int, const std::vector<const float *> &,
                   const std::vector<const float *> &, const std::vector<const float *> &,
                   const std::vector<const float *> &, const std::vector<const float *> &,
                   std::vector<std::vector<float>> &) {
    return false;
}
bool plan_vocode(const S3genCoremlPlan &, const std::vector<float> &, int, const std::vector<float> &,
                 std::vector<float> &) {
    return false;
}

#endif

void defer_coreml_failures(const S3genCoremlPlan & plan) {
    plan.defer.store(true, std::memory_order_release);
}

bool coreml_failures_deferred(const S3genCoremlPlan & plan) {
    return plan.defer.load(std::memory_order_acquire);
}

bool plan_strict(const S3genCoremlPlan & plan) {
    return plan.strict;
}

std::shared_ptr<const S3genCoremlPlan> empty_coreml_plan() {
    return std::make_shared<const S3genCoremlPlan>();
}

std::shared_ptr<const S3genCoremlPlan> replay_plan(std::shared_ptr<const S3genCoremlPlan> plan) {
    if (plan) plan->defer.store(false, std::memory_order_release);
    return plan;
}

std::string coreml_empty_group_json(const char * group) {
#ifdef TTS_CPP_USE_COREML
    pcm::PrewarmResult r;
    r.group = group;
    r.success = true;
    r.nothing_to_prepare = true;
    return pcm::to_json(r);
#else
    return std::string("{\"group\":\"") + group +
           "\",\"nothing_to_prepare\":true,\"skipped_route_none\":[],\"slots\":[],\"success\":true}";
#endif
}

std::string with_construction(std::string status, const std::string & construction) {
    // `status` is a JSON object: the construction result becomes its
    // "construction" member.
    const size_t close = status.rfind('}');
    if (close == std::string::npos) return status;
    const bool empty = status.find_first_not_of(" \t\n{", 0) == close;
    status.insert(close, std::string(empty ? "" : ",") + "\"construction\":" + construction);
    return status;
}

// ----------------------------------------------------------- the sidecars --

#ifdef TTS_CPP_USE_COREML

struct s3gen_sidecars::impl {
    struct stage_slots {
        bool present = false;  // the sidecar exists and Core ML may use it
        std::string unusable;  // present but lists no usable function: why
        std::string path;
        std::vector<int> widths;                          // empty for HiFT
        std::map<int, std::shared_ptr<pcm::Slot>> slots;  // HiFT: key 0
        std::shared_ptr<pcm::ResidentPool> pool;
    };

    std::string gguf;
    pcm::Resolved resolved;
    pcm::ComputeUnits units = pcm::ComputeUnits::cpu_and_ane;
    std::shared_ptr<pcm::Manager> manager;
    s3gen_gguf_facts facts;
    stage_slots stages[S3GEN_STAGES];
    mutable std::mutex mu;  // construction
    std::shared_ptr<pcm::Group> construction;
    std::atomic<bool> predicting{false};
    // The current call's S3Gen calls not acquired yet, in order (predict()
    // to end_call()); each acquisition drops the first.
    std::mutex ahead_mu;
    std::deque<s3gen_call_shape> ahead;

    std::vector<std::shared_ptr<pcm::Slot>> all_slots() const {
        std::vector<std::shared_ptr<pcm::Slot>> out;
        for (const auto & st : stages) {
            for (const auto & entry : st.slots) out.push_back(entry.second);
        }
        return out;
    }

    // HiFT plus, per multi-function stage, the widths with a non-stale warm
    // hint on this host, the three most recently loaded.
    std::vector<std::shared_ptr<pcm::Slot>> minimal_set() const {
        std::vector<std::shared_ptr<pcm::Slot>> out;
        for (const auto & entry : stages[(int) s3gen_stage::hift].slots) out.push_back(entry.second);
        for (s3gen_stage s : {s3gen_stage::encoder, s3gen_stage::cfm}) {
            std::vector<std::pair<int64_t, std::shared_ptr<pcm::Slot>>> hinted;
            for (const auto & entry : stages[(int) s].slots) {
                const auto hint = entry.second->hint();
                if (hint && !hint->stale) hinted.emplace_back(entry.second->hint_mtime_ns(), entry.second);
            }
            std::stable_sort(hinted.begin(), hinted.end(),
                             [](const auto & a, const auto & b) { return a.first > b.first; });
            for (size_t i = 0; i < hinted.size() && i < COREML_RESIDENT_CAP; ++i) out.push_back(hinted[i].second);
        }
        return out;
    }

    std::vector<pcm::GroupRequest> group(CoremlOptions::Prewarm scope, pcm::GroupKind kind) const {
        std::vector<pcm::GroupRequest> out;
        if (scope == CoremlOptions::Prewarm::none || scope == CoremlOptions::Prewarm::unset) return out;
        const auto minimal = minimal_set();
        auto in_minimal = [&](const std::shared_ptr<pcm::Slot> & s) {
            return std::find(minimal.begin(), minimal.end(), s) != minimal.end();
        };
        // The construction-time minimal set is `minimal`; `all` extras and
        // later prewarm groups are `prewarm`, which load after every minimal
        // request.
        for (const auto & slot : scope == CoremlOptions::Prewarm::all ? all_slots() : minimal) {
            const bool min = kind == pcm::GroupKind::construction && in_minimal(slot);
            out.push_back(pcm::GroupRequest{slot, min ? pcm::Priority::minimal : pcm::Priority::prewarm});
        }
        return out;
    }

    // Present sidecars that list no usable function, as failed slots: a
    // request group never reports success (or nothing_to_prepare) with one.
    std::vector<pcm::SlotStatus> discovery_failures() const {
        std::vector<pcm::SlotStatus> out;
        for (int s = 0; s < S3GEN_STAGES; ++s) {
            const auto & st = stages[s];
            if (!st.present || st.unusable.empty()) continue;
            pcm::SlotStatus f;
            f.stage = STAGE_NAMES[s];
            f.state = pcm::SlotState::failed;
            f.reason = st.unusable;
            f.error = st.unusable;
            f.error_class = pcm::ErrorClass::deterministic;
            f.route = pcm::StatusRoute::prebuilt;
            f.compute_units = pcm::to_string(units);
            f.label = GGML_STAGE_BACKEND;
            f.path = st.path;
            out.push_back(std::move(f));
        }
        return out;
    }

    pcm::PrewarmResult with_discovery(pcm::PrewarmResult r) const {
        const auto failures = discovery_failures();
        if (failures.empty()) return r;
        r.slots.insert(r.slots.end(), failures.begin(), failures.end());
        r.success = false;
        r.nothing_to_prepare = false;
        return r;
    }

    std::string construction_json() const {
        std::shared_ptr<pcm::Group> group;
        {
            std::lock_guard<std::mutex> lock(mu);
            group = construction;
        }
        pcm::PrewarmResult r;
        if (group) {
            r = group->result();
        } else {
            r.group = "construction";
            r.success = true;
            r.nothing_to_prepare = true;
        }
        return pcm::to_json(with_discovery(std::move(r)));
    }

    std::shared_ptr<pcm::Slot> slot(s3gen_stage s, int width) const {
        const auto & st = stages[(int) s];
        auto it = st.slots.find(width);
        return it == st.slots.end() ? nullptr : it->second;
    }

    static int need_of(s3gen_stage s, const s3gen_call_shape & shape) {
        return s == s3gen_stage::encoder ? shape.tokens : s == s3gen_stage::cfm ? shape.frames : 0;
    }

    // The exact function for a need: the narrowest width that covers it (0
    // for HiFT); -1 when none does or the sidecar is unusable (no functions
    // listed); -2 when the stage has no sidecar.
    int exact_of(s3gen_stage s, int need) const {
        const auto & st = stages[(int) s];
        if (!st.present) return -2;
        if (st.slots.empty()) return -1;
        if (s == s3gen_stage::hift) return 0;
        return coreml_pick_bucket(st.widths, need);
    }

    // Marks HiFT and the exact widths of the calls still ahead as predicted,
    // so the resident cap never releases them; a width only finished calls
    // used becomes evictable. With `request`, also requests them at
    // `current` priority, HiFT first, then the widths in call order. Called
    // under ahead_mu.
    void mark_predicted(bool request) {
        std::vector<std::shared_ptr<pcm::Slot>> predicted[S3GEN_STAGES];
        auto add = [&](s3gen_stage s, const std::shared_ptr<pcm::Slot> & slot) {
            auto & list = predicted[(int) s];
            if (slot && std::find(list.begin(), list.end(), slot) == list.end()) list.push_back(slot);
        };
        add(s3gen_stage::hift, slot(s3gen_stage::hift, 0));
        for (const auto & call : ahead) {
            for (s3gen_stage s : {s3gen_stage::encoder, s3gen_stage::cfm}) {
                const int exact = exact_of(s, need_of(s, call));
                if (exact >= 0) add(s, slot(s, exact));
            }
        }
        for (s3gen_stage s : {s3gen_stage::hift, s3gen_stage::encoder, s3gen_stage::cfm}) {
            if (stages[(int) s].pool) stages[(int) s].pool->set_predicted(predicted[(int) s]);
            if (!request) continue;
            for (const auto & sl : predicted[(int) s]) sl->request(pcm::Priority::current);
        }
    }
};

namespace {

// Leaked on purpose, like the runtime's manager: the sets are dropped by
// s3gen_unload (or the S3Gen cache's atexit release), never by static
// destruction after the runtime may be gone.
std::mutex & sets_mu() {
    static auto * mu = new std::mutex;
    return *mu;
}
std::map<std::string, std::shared_ptr<s3gen_sidecars>> & sets() {
    static auto * table = new std::map<std::string, std::shared_ptr<s3gen_sidecars>>;
    return *table;
}

std::string set_key(const std::string & gguf, const pcm::Resolved & r) {
    std::ostringstream key;
    key << gguf << '\n' << (int) r.use << '/' << (int) r.wait << '/' << (int) r.prewarm << '/'
        << (r.compute_units ? (int) *r.compute_units : -1) << '/' << (r.strict_wait_ms ? *r.strict_wait_ms : -1)
        << '\n' << r.cache_dir;
    return key.str();
}

} // namespace

std::shared_ptr<s3gen_sidecars> s3gen_sidecars::create(const std::string & s3gen_gguf, const CoremlOptions & options) {
    validate_coreml_options(options);
    if (s3gen_gguf.empty()) return nullptr;
    auto p = std::make_unique<impl>();
    p->gguf = s3gen_gguf;
    p->resolved = pcm::resolve_coreml(to_runtime_options(options), COREML_ENV_PREFIX);
    if (!p->resolved.enabled()) return nullptr;
    p->units = p->resolved.units_or(pcm::ComputeUnits::cpu_and_ane);
    bool any = false;
    for (const char * suffix : STAGE_SUFFIXES) {
        any = any || ::tts_cpp::detail::coreml_sidecar_exists(coreml_stage_path(s3gen_gguf, suffix));
    }
    if (!any) return nullptr;
    for (const std::string & w : p->resolved.warnings) std::fprintf(stderr, "chatterbox: %s\n", w.c_str());

    p->manager = pcm::Manager::instance();
    p->facts = s3gen_read_facts(s3gen_gguf);
    const pcm::SlotPolicy policy = pcm::SlotPolicy::from(p->resolved);
    const bool capped = p->resolved.prewarm != pcm::Prewarm::all;
    const char * prefixes[S3GEN_STAGES] = {COREML_ENCODER_PREFIX, COREML_CFM_PREFIX, nullptr};
    for (int s = 0; s < S3GEN_STAGES; ++s) {
        auto & st = p->stages[s];
        st.path = coreml_stage_path(s3gen_gguf, STAGE_SUFFIXES[s]);
        pcm::PlanInput in;
        in.coreml_built = true;
        in.generation_built = false;  // prebuilt sidecars only: no generation templates yet
        in.prebuilt_present = ::tts_cpp::detail::coreml_sidecar_exists(st.path);
        in.units = p->units;
        in.platform = p->manager->platform();
        in.os = p->manager->os_version();
        const pcm::Plan route = pcm::plan(in, p->resolved);
        if (route.route == pcm::Route::none) continue;
        st.present = true;

        std::vector<std::pair<int, std::string>> functions;  // (width, function name)
        if (prefixes[s] != nullptr) {
            char names[FUNCTION_LIST_BYTES] = {};
            if (moss_coreml_function_names(st.path.c_str(), names, sizeof(names)) <= 0) {
                st.unusable = "cannot list the functions of " + st.path + " (needs macOS 15 / iOS 18)";
                std::fprintf(stderr, "chatterbox: %s, retaining ggml for the %s\n", st.unusable.c_str(),
                             STAGE_NAMES[s]);
                continue;
            }
            st.widths = coreml_buckets(names, prefixes[s]);
            if (st.widths.empty()) {
                st.unusable = st.path + " has no " + prefixes[s] + "* functions";
                std::fprintf(stderr, "chatterbox: %s, retaining ggml for the %s\n", st.unusable.c_str(),
                             STAGE_NAMES[s]);
                continue;
            }
            for (int w : st.widths) functions.emplace_back(w, prefixes[s] + std::to_string(w));
            if (capped) st.pool = p->manager->make_pool(COREML_RESIDENT_CAP);
        } else {
            functions.emplace_back(0, "main");
        }
        const uint64_t reserve = pcm::units_include_ane(p->units)
            ? (uint64_t) (ANE_COMPILE_RESERVE_FACTOR * (double) file_bytes(st.path + WEIGHTS_FILE))
            : 0;
        for (const auto & [width, function] : functions) {
            pcm::SlotSpec spec;
            spec.engine = ENGINE_NAME;
            spec.stage = STAGE_NAMES[s];
            spec.function = function;
            spec.units = p->units;
            spec.route = route.route;
            spec.artifact_path = st.path;
            spec.load_fn = &load_function;
            spec.warm_fn = &warm_function;
            spec.ane_compile_reserve_bytes = reserve;
            spec.expected_warm_load_ms =
                expected_warm_load_ms((s3gen_stage) s, width, functions.size(), p->facts.meanflow);
            spec.pool = st.pool;
            st.slots.emplace(width, p->manager->make_slot(std::move(spec), policy));
        }
    }
    // A present sidecar that lists no usable function keeps the set: its
    // stage stays in every plan, so use = require refuses the call (cannot
    // serve) instead of silently running on ggml.
    bool present = false;
    for (const auto & st : p->stages) present = present || st.present;
    if (!present) return nullptr;
    return std::shared_ptr<s3gen_sidecars>(new s3gen_sidecars(std::move(p)));
}

std::shared_ptr<s3gen_sidecars> s3gen_sidecars::get(const std::string & s3gen_gguf, const CoremlOptions & options) {
    validate_coreml_options(options);
    if (s3gen_gguf.empty()) return nullptr;
    const pcm::Resolved resolved = pcm::resolve_coreml(to_runtime_options(options), COREML_ENV_PREFIX);
    if (!resolved.enabled()) return nullptr;
    const std::string key = set_key(s3gen_gguf, resolved);
    std::lock_guard<std::mutex> lock(sets_mu());
    auto it = sets().find(key);
    if (it != sets().end()) return it->second;
    auto set = create(s3gen_gguf, options);
    if (set) sets().emplace(key, set);
    return set;
}

void s3gen_sidecars::release_all() {
    std::map<std::string, std::shared_ptr<s3gen_sidecars>> dropped;
    {
        std::lock_guard<std::mutex> lock(sets_mu());
        dropped.swap(sets());
    }
}

s3gen_sidecars::s3gen_sidecars(std::unique_ptr<impl> p) : p_(std::move(p)) {}
s3gen_sidecars::~s3gen_sidecars() = default;

const s3gen_gguf_facts & s3gen_sidecars::facts() const { return p_->facts; }
bool s3gen_sidecars::strict() const { return p_->resolved.strict(); }
std::vector<int> s3gen_sidecars::buckets(s3gen_stage stage) const { return p_->stages[(int) stage].widths; }
bool s3gen_sidecars::has(s3gen_stage stage) const { return !p_->stages[(int) stage].slots.empty(); }

void s3gen_sidecars::start_construction(const std::atomic<bool> * cancel) {
    const auto scope = map_enum<pcm::Prewarm, CoremlOptions::Prewarm>(p_->resolved.prewarm);
    const auto requests = p_->group(scope, pcm::GroupKind::construction);
    const bool waited = p_->resolved.wait == pcm::Wait::construct;
    auto group = p_->manager->issue_group(pcm::GroupKind::construction, requests, waited);
    {
        std::lock_guard<std::mutex> lock(p_->mu);
        p_->construction = group;
    }
    if (waited) group->wait(std::nullopt, cancel);
}

std::string s3gen_sidecars::construction_json() const { return p_->construction_json(); }

void s3gen_sidecars::predict(const std::vector<s3gen_call_shape> & calls) {
    // Only prewarm = predicted (and all) loads a call's predictions ahead of
    // time; none and minimal load a width when a call demands it. Every scope
    // keeps the predicted widths from the resident cap's evictions.
    const bool speculative = p_->resolved.prewarm == pcm::Prewarm::predicted || p_->resolved.prewarm == pcm::Prewarm::all;
    std::lock_guard<std::mutex> lock(p_->ahead_mu);
    p_->ahead.assign(calls.begin(), calls.end());
    p_->predicting = true;
    p_->mark_predicted(speculative);
}

void s3gen_sidecars::end_call() {
    std::lock_guard<std::mutex> lock(p_->ahead_mu);
    p_->ahead.clear();
    for (const auto & st : p_->stages) {
        if (st.pool) st.pool->clear_predicted();
    }
    p_->predicting = false;
}

bool s3gen_sidecars::predicting() const { return p_->predicting.load(); }

std::string coreml_status_without_sidecars() {
    return with_construction("{}", coreml_empty_group_json("construction"));
}

std::shared_ptr<S3genCoremlPlan> s3gen_sidecars::acquire(const s3gen_call_shape & shape,
                                                         std::chrono::steady_clock::time_point deadline,
                                                         const std::atomic<bool> * cancel) {
    auto plan = std::make_shared<S3genCoremlPlan>();
    plan->strict = p_->resolved.strict();
    const s3gen_stage order[] = {s3gen_stage::encoder, s3gen_stage::cfm, s3gen_stage::hift};
    int exact[S3GEN_STAGES] = {-2, -2, -2};
    for (s3gen_stage s : order) {
        auto & out = plan->stages[(int) s];
        const int need = impl::need_of(s, shape);
        exact[(int) s] = p_->exact_of(s, need);
        if (exact[(int) s] == -2) continue;  // no sidecar for the stage
        out.present = true;
        if (exact[(int) s] == -1 && plan->strict) {
            const auto & widths = p_->stages[(int) s].widths;
            const std::string detail = widths.empty()
                ? std::string("it lists no usable function")
                : std::to_string(need) + (s == s3gen_stage::encoder ? " tokens" : " frames") + ", widest " +
                      std::to_string(widths.back());
            throw std::runtime_error(std::string("s3gen: the Core ML ") + STAGE_NAMES[(int) s] +
                                     " sidecar cannot serve this call (" + detail + ") and use = require / " +
                                     COREML_STRICT_ENV + " is set");
        }
    }
    // The narrowest ready function that covers the call serves it (the
    // padding is masked) without waiting.
    auto ready_covering = [&](s3gen_stage s) {
        auto & out = plan->stages[(int) s];
        for (const auto & [width, slot] : p_->stages[(int) s].slots) {
            if (width < exact[(int) s]) continue;
            pcm::Acquired ready;
            try {
                ready = slot->acquire(pcm::Demand::if_ready);
            } catch (const pcm::CoremlError &) {
                continue;  // strict: this one is retired or failed; try the next
            }
            if (ready.model) {
                out.model = std::move(ready.model);
                out.bucket = width;
                out.pending = false;
                return true;
            }
        }
        return false;
    };
    // Every stage first retains its ready covering function, so the exact
    // requests below can never evict one this call is about to use (a full
    // resident pool then has nothing evictable and the exact load waits for
    // room).
    bool covered[S3GEN_STAGES] = {false, false, false};
    for (s3gen_stage s : order) {
        if (exact[(int) s] >= 0) covered[(int) s] = ready_covering(s);
    }
    // Then raise every exact function to `now` (for later calls when
    // covered), so the load lane serves the uncovered ones against the one
    // shared deadline; only those wait, as the wait mode allows.
    for (s3gen_stage s : order) {
        if (exact[(int) s] >= 0) p_->slot(s, exact[(int) s])->request(pcm::Priority::now);
    }
    for (s3gen_stage s : order) {
        if (exact[(int) s] < 0 || covered[(int) s]) continue;
        auto & out = plan->stages[(int) s];
        pcm::Acquired got;
        try {
            got = p_->slot(s, exact[(int) s])->acquire(pcm::Demand::now, deadline, cancel);
        } catch (const pcm::CoremlError &) {
            // Strict: the exact function cannot serve (retired, failed, at the
            // resident cap, timed out); a covering one that became ready
            // meanwhile still can.
            if (ready_covering(s)) continue;
            throw;
        }
        if (got.model) {
            out.model = std::move(got.model);
            out.bucket = exact[(int) s];
            continue;
        }
        out.pending = got.pending;
        ready_covering(s);
    }
    // This call no longer needs its predictions: the plan holds what it
    // runs on. Only the calls still ahead stay predicted, so a width that
    // only earlier calls used can make room for a later, wider one.
    {
        std::lock_guard<std::mutex> lock(p_->ahead_mu);
        if (p_->predicting && !p_->ahead.empty()) {
            p_->ahead.pop_front();
            p_->mark_predicted(false);
        }
    }
    return plan;
}

std::string s3gen_sidecars::prewarm(CoremlOptions::Prewarm scope, bool wait, const std::atomic<bool> * cancel) {
    auto group = p_->manager->issue_group(pcm::GroupKind::prewarm, p_->group(scope, pcm::GroupKind::prewarm), wait);
    return pcm::to_json(p_->with_discovery(wait ? group->wait(std::nullopt, cancel) : group->result()));
}

bool s3gen_sidecars::wait(int64_t timeout_ms, const std::atomic<bool> * cancel) {
    std::shared_ptr<pcm::Group> group;
    {
        std::lock_guard<std::mutex> lock(p_->mu);
        group = p_->construction;
    }
    if (!group) return false;
    std::optional<pcm::Clock::duration> timeout;
    if (timeout_ms >= 0) timeout = std::chrono::milliseconds(timeout_ms);
    return p_->with_discovery(group->wait(timeout, cancel)).success;
}

std::string s3gen_sidecars::status_json() const {
    pcm::GenerationStatus generation;
    generation.supported = false;
    generation.reason = "no_template";
    pcm::EngineStatus status = p_->manager->status(ENGINE_NAME, p_->resolved.cache_dir, p_->all_slots(), generation);
    const auto failures = p_->discovery_failures();
    status.stages.insert(status.stages.end(), failures.begin(), failures.end());
    return with_construction(pcm::to_json(status), construction_json());
}

size_t s3gen_sidecars::live(s3gen_stage stage) const {
    const auto & pool = p_->stages[(int) stage].pool;
    return pool ? pool->live() : 0;
}

std::string s3gen_sidecars::state(s3gen_stage stage, int bucket) const {
    auto slot = p_->slot(stage, bucket);
    if (!slot) return {};
    const pcm::SlotStatus status = slot->status();
    return pcm::state_label(status.state, status.reason);
}

#else  // !TTS_CPP_USE_COREML

struct s3gen_sidecars::impl {
    s3gen_gguf_facts facts;
};

std::shared_ptr<s3gen_sidecars> s3gen_sidecars::create(const std::string &, const CoremlOptions & options) {
    validate_coreml_options(options);
    return nullptr;
}
std::shared_ptr<s3gen_sidecars> s3gen_sidecars::get(const std::string &, const CoremlOptions & options) {
    validate_coreml_options(options);
    return nullptr;
}
void s3gen_sidecars::release_all() {}
s3gen_sidecars::s3gen_sidecars(std::unique_ptr<impl> p) : p_(std::move(p)) {}
s3gen_sidecars::~s3gen_sidecars() = default;
const s3gen_gguf_facts & s3gen_sidecars::facts() const { return p_->facts; }
bool s3gen_sidecars::strict() const { return false; }
std::vector<int> s3gen_sidecars::buckets(s3gen_stage) const { return {}; }
bool s3gen_sidecars::has(s3gen_stage) const { return false; }
void s3gen_sidecars::start_construction(const std::atomic<bool> *) {}
std::string s3gen_sidecars::construction_json() const { return "{}"; }
void s3gen_sidecars::predict(const std::vector<s3gen_call_shape> &) {}
void s3gen_sidecars::end_call() {}
bool s3gen_sidecars::predicting() const { return false; }
std::shared_ptr<S3genCoremlPlan> s3gen_sidecars::acquire(const s3gen_call_shape &,
                                                         std::chrono::steady_clock::time_point,
                                                         const std::atomic<bool> *) {
    return nullptr;
}
std::string s3gen_sidecars::prewarm(CoremlOptions::Prewarm, bool, const std::atomic<bool> *) { return "{}"; }
bool s3gen_sidecars::wait(int64_t, const std::atomic<bool> *) { return false; }
std::string s3gen_sidecars::status_json() const { return "{}"; }
std::string coreml_status_without_sidecars() {
    return with_construction("{}", coreml_empty_group_json("construction"));
}
size_t s3gen_sidecars::live(s3gen_stage) const { return 0; }
std::string s3gen_sidecars::state(s3gen_stage, int) const { return {}; }

#endif

} // namespace tts_cpp::chatterbox::detail

bool s3gen_coreml_plan_offloads(const S3genCoremlPlan & plan) {
#ifdef TTS_CPP_USE_COREML
    using tts_cpp::chatterbox::detail::s3gen_stage;
    const auto & cfm = plan.stages[(int) s3gen_stage::cfm];
    const auto & hift = plan.stages[(int) s3gen_stage::hift];
    // `all` and `cpu_and_gpu` may put a stage on the GPU, where it would
    // contend with T3.
    return cfm.usable() && hift.usable() && cfm.model->request().units == pcm::ComputeUnits::cpu_and_ane &&
           hift.model->request().units == pcm::ComputeUnits::cpu_and_ane;
#else
    (void) plan;
    return false;
#endif
}

std::shared_ptr<const S3genCoremlPlan> s3gen_acquire_coreml_plan(const s3gen_synthesize_opts & opts,
                                                                 const std::vector<int32_t> & speech_tokens,
                                                                 std::chrono::steady_clock::time_point deadline) {
    namespace cb = tts_cpp::chatterbox::detail;
    auto sidecars = cb::s3gen_sidecars::get(opts.s3gen_gguf_path, cb::call_coreml_options(opts));
    if (!sidecars) return nullptr;
    return sidecars->acquire(cb::s3gen_call_shape_for(opts, speech_tokens, sidecars->facts()), deadline,
                             opts.cancel_flag);
}

double s3gen_estimate_metal_ms(const s3gen_synthesize_opts & opts, const std::vector<int32_t> & speech_tokens) {
    namespace cb = tts_cpp::chatterbox::detail;
    const cb::s3gen_gguf_facts facts = cb::s3gen_read_facts(opts.s3gen_gguf_path);
    return cb::s3gen_estimate_metal_ms_frames(facts.meanflow, cb::s3gen_call_shape_for(opts, speech_tokens, facts).frames);
}
