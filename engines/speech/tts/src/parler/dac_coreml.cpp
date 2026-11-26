#include "dac_coreml.h"

#include "internal.h"
#include "../coreml_sidecar_path.h"

#ifdef TTS_CPP_USE_COREML
#include "dac_source.h"
#include "../moss/coreml/runner.h"

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"
#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tts_cpp {
namespace parler {
namespace detail {
namespace {

constexpr int SINGLE_BATCH = 1;
constexpr int MONO = 1;
constexpr int FEATURE_RANK = 3;

bool read_f32(const ggml_tensor * t, int64_t n, std::vector<float> & out) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != n) return false;
    out.resize((size_t) n);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t) n * sizeof(float));
    return true;
}

#ifdef TTS_CPP_USE_COREML
class coreml_dac final : public parler_dac_sidecar {
public:
    coreml_dac(moss_coreml_model * model, int window) : model_(model), window_(window) {}
    ~coreml_dac() override { moss_coreml_free(model_); }
    coreml_dac(const coreml_dac &) = delete;
    coreml_dac & operator=(const coreml_dac &) = delete;

    int window() const override { return window_; }
    const char * label() const override { return moss_coreml_label(model_); }

    bool decode(const float * latent, const float * valid, float * pcm) override {
        const char * in_names[] = {"latent", "valid"};
        const float * in_data[] = {latent, valid};
        const char * out_names[] = {"pcm"};
        float * out_data[] = {pcm};
        return moss_coreml_predict(model_, in_names, in_data, 2, out_names, out_data, 1) == 0;
    }

private:
    moss_coreml_model * model_;
    int window_;
};

std::vector<int64_t> feature_dims(moss_coreml_model * model, const char * name) {
    int64_t dims[8] = {};
    const int rank = moss_coreml_feature_dims(model, name, dims, 8);
    return std::vector<int64_t>(dims, dims + std::max(rank, 0));
}
#endif

} // namespace

std::string parler_dac_sidecar_path(const std::string & gguf_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(gguf_path, PARLER_DAC_SIDECAR_SUFFIX);
}

bool parler_coreml_disabled() {
    return std::getenv(PARLER_COREML_DISABLE_ENV) != nullptr;
}

bool parler_coreml_strict() {
    return std::getenv(PARLER_COREML_STRICT_ENV) != nullptr;
}

bool parler_dac_sidecar_enabled(const std::string & gguf_path) {
#ifdef TTS_CPP_USE_COREML
    return !parler_coreml_disabled() &&
           ::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf_path));
#else
    (void) gguf_path;
    return false;
#endif
}

void parler_dac_rvq::latent(const int32_t * codes, int n_frames, int f0, int f1,
                            float * out, int stride) const {
    const int n = f1 - f0;
    for (int c = 0; c < latent_dim; ++c) {
        float bsum = 0.0f;
        for (int k = 0; k < n_q; ++k) bsum += bias[(size_t) k][(size_t) c];
        std::fill(out + (size_t) c * stride, out + (size_t) c * stride + n, bsum);
    }
    std::vector<float> vec((size_t) cb_dim);
    for (int k = 0; k < n_q; ++k) {
        const float * cb = codebook[(size_t) k].data();
        const float * w  = proj[(size_t) k].data();
        for (int t = 0; t < n; ++t) {
            const int32_t code = codes[(size_t) k * n_frames + f0 + t];
            std::memcpy(vec.data(), cb + (size_t) code * cb_dim, (size_t) cb_dim * sizeof(float));
            for (int c = 0; c < latent_dim; ++c) {
                const float * wr = w + (size_t) c * cb_dim;
                float acc = 0.0f;
                for (int i = 0; i < cb_dim; ++i) acc += wr[i] * vec[(size_t) i];
                out[(size_t) c * stride + t] += acc;
            }
        }
    }
}

bool parler_dac_rvq_load(const parler_model & model, parler_dac_rvq & rvq, std::string * err) {
    const parler_hparams & hp = model.hparams;
    parler_dac_rvq r;
    r.n_q           = hp.dac_n_q;
    r.codebook_size = hp.dac_codebook_size;
    r.latent_dim    = hp.dac_latent;
    if ((int) model.dac_quant.size() != r.n_q || r.n_q <= 0 || model.dac_quant[0].codebook == nullptr) {
        if (err) *err = "DAC quantizer tensors missing";
        return false;
    }
    r.cb_dim = (int) model.dac_quant[0].codebook->ne[0];
    r.codebook.resize((size_t) r.n_q);
    r.proj.resize((size_t) r.n_q);
    r.bias.resize((size_t) r.n_q);
    for (int k = 0; k < r.n_q; ++k) {
        const parler_dac_quant & q = model.dac_quant[(size_t) k];
        if (!read_f32(q.codebook, (int64_t) r.codebook_size * r.cb_dim, r.codebook[(size_t) k]) ||
            !read_f32(q.out_proj_w, (int64_t) r.latent_dim * r.cb_dim, r.proj[(size_t) k]) ||
            !read_f32(q.out_proj_b, r.latent_dim, r.bias[(size_t) k])) {
            if (err) *err = "DAC quantizer " + std::to_string(k) + " is not F32 of the expected shape";
            return false;
        }
    }
    rvq = std::move(r);
    return true;
}

std::shared_ptr<parler_dac_sidecar> parler_open_dac_sidecar(const std::string & gguf_path,
                                                            const parler_model & model,
                                                            std::string * why) {
    return parler_open_dac_sidecar(gguf_path, model, parler_resolve_coreml_policy(), why);
}

std::shared_ptr<parler_dac_sidecar> parler_open_dac_sidecar(const std::string & gguf_path,
                                                            const parler_model & model,
                                                            const parler_coreml_policy & policy,
                                                            std::string * why) {
    auto reject = [&](const std::string & reason) -> std::shared_ptr<parler_dac_sidecar> {
        if (why) *why = reason;
        return nullptr;
    };
    if (policy.disabled) return reject(std::string(PARLER_COREML_DISABLE_ENV) + " is set");
#ifdef TTS_CPP_USE_COREML
    const std::string path = parler_dac_sidecar_path(gguf_path);
    if (!::tts_cpp::detail::coreml_sidecar_exists(path)) return reject("no sidecar at " + path);
    moss_coreml_model * m = moss_coreml_load(path.c_str(), policy.compute_units.c_str());
    if (m == nullptr) return reject("Core ML could not load " + path);

    const parler_hparams & hp = model.hparams;
    const std::vector<int64_t> lat = feature_dims(m, "latent");
    const int64_t window = lat.size() == FEATURE_RANK ? lat[2] : 0;
    const bool ok = lat.size() == FEATURE_RANK && lat[0] == SINGLE_BATCH && lat[1] == hp.dac_latent &&
                    feature_dims(m, "valid") == std::vector<int64_t>{SINGLE_BATCH, MONO, window} &&
                    feature_dims(m, "pcm") == std::vector<int64_t>{SINGLE_BATCH, MONO, window * hp.dac_hop};
    if (!ok) {
        moss_coreml_free(m);
        return reject(path + " does not declare latent [1, " + std::to_string(hp.dac_latent) +
                      ", W], valid [1, 1, W] and pcm [1, 1, W * " + std::to_string(hp.dac_hop) + "]");
    }
    return std::make_shared<coreml_dac>(m, (int) window);
#else
    (void) gguf_path;
    (void) model;
    return reject("built without TTS_CPP_COREML");
#endif
}

bool parler_attach_dac_sidecar(parler_model & model, std::shared_ptr<parler_dac_sidecar> sidecar,
                               std::string * err) {
    return parler_attach_dac_sidecar(model, std::move(sidecar), parler_resolve_coreml_policy(), err);
}

bool parler_attach_dac_sidecar(parler_model & model, std::shared_ptr<parler_dac_sidecar> sidecar,
                               const parler_coreml_policy & policy, std::string * err) {
    model.dac_sidecar.reset();
    model.dac_sidecar_retired = false;
    model.dac_coreml_strict = policy.strict;
    if (!sidecar || policy.disabled) return true;
    if (!parler_dac_rvq_load(model, model.dac_rvq, err)) return false;
    model.dac_sidecar = std::move(sidecar);
    return true;
}

bool parler_attach_dac_sidecar(parler_model & model, const std::string & gguf_path) {
    return parler_attach_dac_sidecar(model, gguf_path, parler_resolve_coreml_policy());
}

bool parler_attach_dac_sidecar(parler_model & model, const std::string & gguf_path,
                               const parler_coreml_policy & policy) {
    // Reset retirement even if the new sidecar cannot be opened. Strictness
    // belongs to this attachment, never to the environment at decode time.
    parler_attach_dac_sidecar(model, std::shared_ptr<parler_dac_sidecar>{}, policy);
    if (policy.disabled) {
        if (policy.strict) {
            throw std::runtime_error("parler: Core ML is disabled but strict policy is enabled");
        }
        return false;
    }
    std::string why;
    std::shared_ptr<parler_dac_sidecar> sidecar = parler_open_dac_sidecar(gguf_path, model, policy, &why);
    if (sidecar && sidecar->window() <= 2 * parler_dac_rf_frames(model)) {
        why = "sidecar window " + std::to_string(sidecar->window()) + " leaves no frames after " +
              std::to_string(parler_dac_rf_frames(model)) + " context frames per side";
        sidecar.reset();
    }
    if (sidecar && !parler_attach_dac_sidecar(model, sidecar, policy, &why)) sidecar.reset();
    if (sidecar) {
        fprintf(stderr, "parler: DAC on Core ML (%s, %d-frame window)\n", sidecar->label(),
                sidecar->window());
        return true;
    }
    if (policy.strict) {
        throw std::runtime_error(std::string("parler: no usable Core ML DAC sidecar (") + why +
                                 "); Core ML strict policy is enabled (coreml_strict / " +
                                 PARLER_COREML_STRICT_ENV + ")");
    }
    return false;
}

bool parler_dac_decode_sidecar(const parler_model & model, parler_dac_sidecar & sidecar, const int32_t * codes,
                               int n_frames, std::vector<float> & pcm_out, std::vector<float> * latent_out,
                               int out_begin, int out_end) {
    const parler_dac_rvq & rvq = model.dac_rvq;
    const int hop = model.hparams.dac_hop;
    const int window = sidecar.window();
    const int rf = parler_dac_rf_frames(model);
    const int keep = window - 2 * rf;
    if (keep <= 0 || rvq.latent_dim != model.hparams.dac_latent) return false;

    std::vector<float> latent((size_t) rvq.latent_dim * window);
    std::vector<float> valid((size_t) window);
    std::vector<float> pcm((size_t) window * hop);
    for (int a = out_begin; a < out_end; a += keep) {
        const int b  = std::min(a + keep, out_end);
        const int w0 = std::max(0, a - rf);
        const int w1 = std::min(n_frames, w0 + window);
        const int n  = w1 - w0;
        // the unused tail is masked out, so its contents never reach the kept span
        std::fill(latent.begin(), latent.end(), 0.0f);
        rvq.latent(codes, n_frames, w0, w1, latent.data(), window);
        std::fill(valid.begin(), valid.begin() + n, 1.0f);
        std::fill(valid.begin() + n, valid.end(), 0.0f);
        if (!sidecar.decode(latent.data(), valid.data(), pcm.data())) return false;

        const size_t off = (size_t) (a - w0) * hop;
        pcm_out.insert(pcm_out.end(), pcm.begin() + (ptrdiff_t) off,
                       pcm.begin() + (ptrdiff_t) (off + (size_t) (b - a) * hop));
        if (latent_out) {
            for (int t = a; t < b; ++t) {
                for (int c = 0; c < rvq.latent_dim; ++c) {
                    latent_out->push_back(latent[(size_t) c * window + (t - w0)]);
                }
            }
        }
    }
    return true;
}

uint64_t parler_dac_rvq_bytes(const parler_model & model) {
    const parler_hparams & hp = model.hparams;
    if (model.dac_quant.empty() || model.dac_quant[0].codebook == nullptr) return 0;
    const uint64_t cb_dim = (uint64_t) model.dac_quant[0].codebook->ne[0];
    const uint64_t per_book = (uint64_t) hp.dac_codebook_size * cb_dim + (uint64_t) hp.dac_latent * cb_dim +
                              (uint64_t) hp.dac_latent;
    return (uint64_t) hp.dac_n_q * per_book * sizeof(float);
}

std::string parler_merge_dac_backend(const std::string & acc, const std::string & next) {
    if (acc.empty() || acc == next) return next;
    return PARLER_DAC_MIXED_BACKEND;
}

namespace {

constexpr const char * DAC_TEMPLATE_ID = "parler-dac-w128";
constexpr const char * ENGINE_NAME = "parler";
constexpr const char * STAGE_NAME = "dac";
constexpr const char * ENV_PREFIX = "PARLER";

template <class E, class F> F map_enum(E v) {
    return static_cast<F>(static_cast<int>(v));
}

void validate_options(const CoremlOptions & o) {
    if (o.compute_units && !parler_coreml_valid_units(*o.compute_units)) {
        throw std::invalid_argument("parler: invalid coreml.compute_units='" + *o.compute_units +
                                    "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
    }
    if (o.strict_wait_ms && *o.strict_wait_ms < 0) {
        throw std::invalid_argument("parler: coreml.strict_wait_ms must not be negative");
    }
    if (o.memory_budget_bytes && *o.memory_budget_bytes < 0) {
        throw std::invalid_argument("parler: coreml.memory_budget_bytes must not be negative");
    }
}

} // namespace

#ifdef TTS_CPP_USE_COREML

namespace pcm = QVAC_COREML_NS;

namespace {

pcm::Options to_runtime_options(const CoremlOptions & o) {
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

uint64_t file_bytes(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? (uint64_t) st.st_size : 0;
}

// Metal decodes Parler mini's DAC at about 2 ms per output frame on an M2
// (1,034 ms for a 498-step utterance, docs/parler.md); the CPU is slower. The
// bounded wait for a warm load never exceeds what the call would cost on ggml.
constexpr double GGML_DAC_MS_PER_FRAME = 2.0;
constexpr int64_t GGML_DAC_MS_FLOOR = 20;
constexpr double ANE_COMPILE_RESERVE_FACTOR = 1.6;

// The loaded model a slot holds: the Core ML model and its window.
struct dac_model {
    moss_coreml_model * model = nullptr;
    int window = 0;
    ~dac_model() {
        if (model) moss_coreml_free(model);
    }
};

std::vector<int64_t> feature_dims_of(moss_coreml_model * model, const char * name) {
    int64_t dims[8] = {};
    const int rank = moss_coreml_feature_dims(model, name, dims, 8);
    return std::vector<int64_t>(dims, dims + std::max(rank, 0));
}

// LoadFn. Takes the shared lease of a generated artifact first, then loads
// the model on the request's compute units and validates the geometry the
// constructing thread passed by value: {latent_dim, hop, rf_frames}.
pcm::LoadResult load_dac(const pcm::LoadRequest & req) {
    pcm::LoadResult r;
    if (!pcm::generation::begin_load(req, r)) return r;
    auto fail = [&](const pcm::Error & e) {
        r.model.reset();
        r.error = e;
        pcm::generation::end_load(req, r);
        return r;
    };
    if (req.geometry.size() != 3) return fail(pcm::Error::deterministic("parler: DAC load without its geometry"));
    const int64_t latent_dim = req.geometry[0], hop = req.geometry[1], rf = req.geometry[2];
    moss_coreml_model * m = moss_coreml_load(req.artifact_path.c_str(), pcm::to_string(req.units));
    if (m == nullptr) return fail(pcm::Error::transient("parler: Core ML could not load " + req.artifact_path));
    const std::vector<int64_t> lat = feature_dims_of(m, "latent");
    const int64_t window = lat.size() == FEATURE_RANK ? lat[2] : 0;
    const bool shapes = lat.size() == FEATURE_RANK && lat[0] == SINGLE_BATCH && lat[1] == latent_dim &&
                        feature_dims_of(m, "valid") == std::vector<int64_t>{SINGLE_BATCH, MONO, window} &&
                        feature_dims_of(m, "pcm") == std::vector<int64_t>{SINGLE_BATCH, MONO, window * hop};
    if (!shapes || window <= 2 * rf) {
        moss_coreml_free(m);
        const std::string why = !shapes
            ? req.artifact_path + " does not declare latent [1, " + std::to_string(latent_dim) +
                  ", W], valid [1, 1, W] and pcm [1, 1, W * " + std::to_string(hop) + "]"
            : req.artifact_path + ": window " + std::to_string(window) + " leaves no frames after " +
                  std::to_string(rf) + " context frames per side";
        return fail(pcm::Error::deterministic("parler: " + why));
    }
    auto * dm = new dac_model;
    dm->model = m;
    dm->window = (int) window;
    r.model = std::shared_ptr<void>(dm, [](void * p) { delete static_cast<dac_model *>(p); });
    pcm::generation::end_load(req, r);
    return r;
}

bool predict(dac_model & dm, const float * latent, const float * valid, float * pcm_out) {
    const char * in_names[] = {"latent", "valid"};
    const float * in_data[] = {latent, valid};
    const char * out_names[] = {"pcm"};
    float * out_data[] = {pcm_out};
    return moss_coreml_predict(dm.model, in_names, in_data, 2, out_names, out_data, 1) == 0;
}

// WarmFn: one prediction on a silent window, so a call's first prediction
// does not pay the first-run cost.
pcm::Error warm_dac(void * model, const pcm::LoadRequest & req) {
    auto & dm = *static_cast<dac_model *>(model);
    const int64_t latent_dim = req.geometry.size() == 3 ? req.geometry[0] : 0;
    const int64_t hop = req.geometry.size() == 3 ? req.geometry[1] : 0;
    std::vector<float> latent((size_t) (latent_dim * dm.window), 0.0f), valid((size_t) dm.window, 1.0f);
    std::vector<float> pcm_out((size_t) (dm.window * hop));
    if (!predict(dm, latent.data(), valid.data(), pcm_out.data())) {
        return pcm::Error::transient("parler: the Core ML DAC warm-up prediction failed");
    }
    return {};
}

// One call's sidecar: the handle acquired at the call's entry.
class handle_sidecar final : public parler_dac_sidecar {
public:
    explicit handle_sidecar(pcm::Handle h) : h_(std::move(h)), dm_(h_->as<dac_model>()) {}
    int window() const override { return dm_->window; }
    const char * label() const override { return h_->label(); }
    bool decode(const float * latent, const float * valid, float * pcm_out) override {
        auto lock = h_->lock_for_prediction();
        return predict(*dm_, latent, valid, pcm_out);
    }
    void retire(const std::string & reason) { h_->retire(reason); }

private:
    pcm::Handle h_;
    dac_model * dm_;
};

struct planned {
    parler_dac_route route;
    pcm::Plan plan;
    pcm::ComputeUnits units = pcm::ComputeUnits::all;
    std::optional<pcm::generation::Candidate> candidate;
    std::string prebuilt_path;
};

planned plan_route(const std::string & gguf_path, const pcm::Resolved & r, uint64_t adapter_bytes) {
    planned out;
    out.route.strict = r.strict();
    out.prebuilt_path = parler_dac_sidecar_path(gguf_path);
    const std::shared_ptr<pcm::Manager> mgr = pcm::Manager::instance();
    pcm::PlanInput in;
    in.coreml_built = true;
    in.generation_built = pcm::generation::built();
    in.prebuilt_present = ::tts_cpp::detail::coreml_sidecar_exists(out.prebuilt_path);
    if (in.generation_built && !in.prebuilt_present && r.enabled()) {
        std::vector<pcm::generation::Candidate> c = pcm::generation::probe({DAC_TEMPLATE_ID}, gguf_path, r);
        if (!c.empty()) out.candidate = std::move(c.front());
    }
    const bool available = out.candidate && out.candidate->available;
    out.units = r.units_or(available && out.candidate->default_units ? *out.candidate->default_units
                                                                     : pcm::ComputeUnits::all);
    out.route.compute_units = pcm::to_string(out.units);
    in.template_available = available;
    in.template_matched = available && out.candidate->matched;
    if (available) in.validated_units = out.candidate->memory.validated_units;
    in.units = out.units;
    in.platform = mgr->platform();
    in.os = mgr->os_version();
    out.plan = pcm::plan(in, r);
    out.route.route = pcm::to_string(out.plan.route);
    if (out.plan.route == pcm::Route::none) {
        out.route.state = pcm::to_string(out.plan.none_state);
        out.route.reason = out.plan.reason;
        if (out.plan.none_state == pcm::SlotState::no_template && out.candidate && !out.candidate->reason.empty()) {
            out.route.reason += ": " + out.candidate->reason;
        }
    }
    if (out.plan.route == pcm::Route::generate) {
        const pcm::generation::TemplateMemory & m = out.candidate->memory;
        // The template's envelope plus the engine-side quantizer copy, both
        // with the template's margin.
        const double margin = m.margin > 0 ? m.margin : 1.25;
        out.route.envelope_bytes = pcm::generation::envelope_bytes(m, out.units) +
                                   (uint64_t) std::ceil((double) adapter_bytes * margin);
        out.route.resident_bytes = pcm::generation::resident_bytes(m) + adapter_bytes;
        out.route.workspace_bytes = m.workspace_bytes;
        out.route.prepare_peak_bytes = pcm::generation::prepare_peak_bytes(m);
        out.route.load_peak_bytes = pcm::generation::load_peak_bytes(m, out.units);
        out.route.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(m, out.units);
        out.route.validated = std::find(m.validated_units.begin(), m.validated_units.end(), out.units) !=
                              m.validated_units.end();
    }
    return out;
}

} // namespace

struct parler_dac_coreml::impl {
    pcm::Resolved resolved;
    planned plan;
    std::shared_ptr<pcm::Manager> manager;
    std::shared_ptr<pcm::Slot> slot;
    pcm::SlotStatus none_status;
    mutable std::mutex mu;
    std::shared_ptr<pcm::Group> construction;
};

parler_dac_route parler_dac_plan_route(const std::string & gguf_path, const CoremlOptions & options,
                                       uint64_t adapter_bytes) {
    validate_options(options);
    const pcm::Resolved r = pcm::resolve_coreml(to_runtime_options(options), ENV_PREFIX);
    return plan_route(gguf_path, r, adapter_bytes).route;
}

std::shared_ptr<parler_dac_coreml> parler_dac_coreml::create(parler_model & model, const std::string & gguf_path,
                                                             const CoremlOptions & options, bool memory_degrade) {
    validate_options(options);
    auto p = std::make_unique<impl>();
    p->resolved = pcm::resolve_coreml(to_runtime_options(options), ENV_PREFIX);
    for (const std::string & w : p->resolved.warnings) fprintf(stderr, "parler: %s\n", w.c_str());
    if (memory_degrade && p->resolved.enabled()) {
        p->resolved.use = pcm::Use::off;
        p->resolved.disabled_reason = "memory";
    }
    p->manager = pcm::Manager::instance();
    p->plan = plan_route(gguf_path, p->resolved, parler_dac_rvq_bytes(model));
    const parler_hparams & hp = model.hparams;

    std::string rvq_err;
    if (p->plan.plan.route != pcm::Route::none && !parler_dac_rvq_load(model, model.dac_rvq, &rvq_err)) {
        p->plan.plan.route = pcm::Route::none;
        p->plan.plan.none_state = pcm::SlotState::unsupported;
        p->plan.plan.reason = "quantizer: " + rvq_err;
        p->plan.route.route = "none";
        p->plan.route.state = pcm::to_string(pcm::SlotState::unsupported);
        p->plan.route.reason = p->plan.plan.reason;
    }
    if (p->plan.plan.route == pcm::Route::none) {
        p->none_status.stage = STAGE_NAME;
        p->none_status.function = "main";
        p->none_status.state = p->plan.plan.none_state;
        p->none_status.reason = p->plan.plan.reason;
        p->none_status.compute_units = p->plan.route.compute_units;
        p->none_status.label = PARLER_DAC_GGML_BACKEND;
        if (p->resolved.strict()) {
            throw std::runtime_error("parler: no usable Core ML DAC sidecar (" + p->plan.route.state + ": " +
                                     p->plan.route.reason + "); Core ML strict policy is enabled (use = require / " +
                                     PARLER_COREML_STRICT_ENV + ")");
        }
        auto out = std::shared_ptr<parler_dac_coreml>(new parler_dac_coreml(std::move(p)));
        return out;
    }

    pcm::SlotSpec spec;
    spec.engine = ENGINE_NAME;
    spec.stage = STAGE_NAME;
    spec.function = "main";
    spec.units = p->plan.units;
    spec.route = p->plan.plan.route;
    spec.load_fn = &load_dac;
    spec.warm_fn = &warm_dac;
    spec.geometry = {hp.dac_latent, hp.dac_hop, parler_dac_rf_frames(model)};
    spec.shareable = true;
    if (spec.route == pcm::Route::prebuilt) {
        spec.artifact_path = p->plan.prebuilt_path;
        spec.ane_compile_reserve_bytes =
            pcm::units_include_ane(spec.units)
                ? (uint64_t) (ANE_COMPILE_RESERVE_FACTOR * (double) file_bytes(spec.artifact_path +
                                                                               "/weights/weight.bin"))
                : 0;
    } else {
        const pcm::generation::Candidate & c = *p->plan.candidate;
        // The file the weights were read from (fstat of the load's one
        // descriptor), not whatever `gguf_path` names now: prepare opens the
        // path and fails with source_changed unless it is that file, and
        // asking for another path than the load's is refused here; the DAC
        // then stays on ggml.
        pcm::SourceStat source;
        pcm::SourceStat asked;
        const bool loaded = model.dac_source && model.dac_source->ok;
        if (loaded) source = model.dac_source->stat;
        source.changed = !loaded || !pcm::generation::source_stat(gguf_path, asked) || asked.path != source.path;
        spec.tmpl = c.tmpl;
        spec.prepare = pcm::generation::prepare_request(c, source, p->resolved);
        spec.prepare_fn = &pcm::generation::prepare;
        spec.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(c.memory, spec.units);
    }
    p->slot = p->manager->make_slot(std::move(spec), pcm::SlotPolicy::from(p->resolved));

    std::vector<pcm::GroupRequest> requests;
    if (p->resolved.prewarm != pcm::Prewarm::none) requests.push_back({p->slot, pcm::Priority::minimal});
    const bool waited = p->resolved.wait == pcm::Wait::construct;
    p->construction = p->manager->issue_group(pcm::GroupKind::construction, requests, waited);
    auto out = std::shared_ptr<parler_dac_coreml>(new parler_dac_coreml(std::move(p)));
    if (waited) {
        // With prewarm none the group is empty and nothing waits; strict
        // throws only for a slot of the group that ended non-ready (terminal,
        // deferred(memory) or cancelled).
        const pcm::PrewarmResult result = out->p_->construction->wait();
        if (out->p_->resolved.strict() && !requests.empty() && !result.success) {
            const pcm::SlotStatus st = out->p_->slot->status();
            throw std::runtime_error("parler: the Core ML DAC sidecar is not ready after construction (" +
                                     pcm::state_label(st.state, st.reason) + (st.error.empty() ? "" : ": " + st.error) +
                                     "); Core ML strict policy is enabled (use = require / " +
                                     PARLER_COREML_STRICT_ENV + ")");
        }
    }
    model.dac_pending = out;
    return out;
}

parler_dac_coreml::parler_dac_coreml(std::unique_ptr<impl> p) : p_(std::move(p)) {}
parler_dac_coreml::~parler_dac_coreml() = default;

parler_dac_call parler_dac_coreml::acquire(int out_frames, const std::atomic<bool> * cancel) {
    parler_dac_call call;
    if (!p_->slot) return call;
    const int64_t budget_ms =
        std::max<int64_t>(GGML_DAC_MS_FLOOR, (int64_t) (GGML_DAC_MS_PER_FRAME * (double) std::max(out_frames, 0)));
    const auto deadline = pcm::Clock::now() + std::chrono::milliseconds(budget_ms);
    pcm::Acquired got;
    try {
        got = p_->slot->acquire(pcm::Demand::now, deadline, cancel);
    } catch (const pcm::CoremlError & e) {
        throw std::runtime_error(std::string("parler: the Core ML DAC sidecar cannot serve this call (") + e.what() +
                                 "); Core ML strict policy is enabled (use = require / " + PARLER_COREML_STRICT_ENV +
                                 ")");
    }
    call.pending = got.pending;
    if (!got.model) return call;
    auto sidecar = std::make_shared<handle_sidecar>(std::move(got.model));
    call.retire = [sidecar](const std::string & reason) { sidecar->retire(reason); };
    call.sidecar = std::move(sidecar);
    return call;
}

bool parler_dac_coreml::strict() const { return p_->resolved.strict(); }
const parler_dac_route & parler_dac_coreml::route() const { return p_->plan.route; }
bool parler_dac_coreml::active() const { return p_->slot != nullptr; }

std::string parler_dac_coreml::status_json() const {
    pcm::GenerationStatus gen;
    gen.supported = pcm::generation::built() && p_->manager->platform() == pcm::Platform::macos &&
                    pcm::os_at_least(p_->manager->os_version(), 15, 0);
    if (!gen.supported) {
        gen.reason = !pcm::generation::built() ? "build" : "os";
    } else if (p_->plan.plan.route == pcm::Route::none && p_->plan.plan.reason == "unvalidated_units") {
        gen.reason = "unvalidated_units";
    }
    if (p_->slot) return pcm::to_json(p_->manager->status(ENGINE_NAME, p_->resolved.cache_dir, {p_->slot}, gen));
    pcm::EngineStatus st;
    st.engine = ENGINE_NAME;
    st.cache_dir = p_->resolved.cache_dir;
    st.generation = gen;
    st.stages.push_back(p_->none_status);
    return pcm::to_json(st);
}

std::string parler_dac_coreml::construction_json() const {
    return p_->construction ? pcm::to_json(p_->construction->result()) : std::string("{}");
}

bool parler_dac_coreml::wait(int64_t timeout_ms, const std::atomic<uint64_t> * cancels, uint64_t since) const {
    if (!p_->slot) return false;
    const auto start = pcm::Clock::now();
    for (;;) {
        const pcm::SlotState st = p_->slot->state();
        if (st == pcm::SlotState::ready) return true;
        // Terminal, or nothing will move it without a request: idle (prewarm
        // none before the first call, or evicted) and deferred(memory).
        if (pcm::is_terminal(st) || st == pcm::SlotState::idle || st == pcm::SlotState::deferred_memory) {
            return false;
        }
        if (cancels && cancels->load() != since) return false;
        if (timeout_ms >= 0 && pcm::Clock::now() - start >= std::chrono::milliseconds(timeout_ms)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

#else  // !TTS_CPP_USE_COREML

struct parler_dac_coreml::impl {
    parler_dac_route route;
};

parler_dac_route parler_dac_plan_route(const std::string &, const CoremlOptions & options, uint64_t) {
    validate_options(options);
    parler_dac_route r;
    r.state = "unsupported";
    r.reason = "build";
    return r;
}

std::shared_ptr<parler_dac_coreml> parler_dac_coreml::create(parler_model &, const std::string &,
                                                             const CoremlOptions & options, bool) {
    validate_options(options);
    return nullptr;
}

parler_dac_coreml::parler_dac_coreml(std::unique_ptr<impl> p) : p_(std::move(p)) {}
parler_dac_coreml::~parler_dac_coreml() = default;
parler_dac_call parler_dac_coreml::acquire(int, const std::atomic<bool> *) { return {}; }
bool parler_dac_coreml::strict() const { return false; }
const parler_dac_route & parler_dac_coreml::route() const { return p_->route; }
bool parler_dac_coreml::active() const { return false; }
std::string parler_dac_coreml::status_json() const { return "{}"; }
std::string parler_dac_coreml::construction_json() const { return "{}"; }
bool parler_dac_coreml::wait(int64_t, const std::atomic<uint64_t> *, uint64_t) const { return false; }

#endif

} // namespace detail
} // namespace parler
} // namespace tts_cpp
