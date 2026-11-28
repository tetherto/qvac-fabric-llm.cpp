#pragma once

// EncoderRoute: the Core ML encoder handle one unit of work acquires once, at
// its start, and holds until it ends.
//
// A unit is one top-level call: an offline transcription, a diarization, one
// streaming window. Every plan that depends on the sidecar (the long-form
// window plan, the independent-segment cap, the Nemotron cache-aware switch)
// is chosen from the route:
//   - a route that holds a handle runs the unit on Core ML;
//   - a null route runs it on ggml with exactly today's ggml-only plan, so a
//     sidecar that is still loading never changes a ggml transcript.
// A failed Core ML prediction at window k retires the sidecar process-wide (later
// acquisitions return null) and windows k.. of the same unit run on ggml with
// the plan already chosen; the call then reports "mixed". The next call
// acquires a null route and reverts to the ggml plan.
//
// Portable on purpose (no Core ML or runtime types), so the routing logic and
// its tests build on every platform. Copies share one state: a failure seen
// through any copy is seen through all of them.

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

struct parakeet_coreml_context;

namespace parakeet {

// What the long-form plan reads from the acquired handle: the encoder's I/O
// after load, so no pre-load descriptor is needed.
struct EncoderRouteGeometry {
    int  fixed_mel_frames    = 0;      // 0: flexible or unknown capacity
    int  seam_context_frames = 0;      // 0: the engine default
    bool masks_mel_padding   = false;  // fixed TDT exports with mel_valid_mask
};

class EncoderRoute {
public:
    // Plain function pointers: they cannot capture an engine or a ggml tensor.
    using RetireFn = void (*)(void * hold, const char * reason);
    using LockFn   = std::unique_lock<std::mutex> (*)(void * hold);
    using EncodeFn = int (*)(void * hold, parakeet_coreml_context * ctx,
                             int n_mel_frames, int mel_valid, int n_mels, const float * mel,
                             int n_enc_frames, int d_model, float * encoder_out);
    // Strict routes: raises the runtime's error for a dispatch the handle
    // could not serve. `prediction_failed`: the sidecar ran and failed (it
    // was retired); otherwise it could not take the input at all.
    using RefuseFn = void (*)(void * hold, const std::string & reason, bool prediction_failed);

    struct Ops {
        RetireFn retire = nullptr;  // called once, on the unit's first failure
        LockFn   lock   = nullptr;  // one in-flight prediction per model instance
        EncodeFn encode = nullptr;  // the sidecar forward pass
        RefuseFn refuse = nullptr;  // strict routes; null throws std::runtime_error
    };

    // The ggml route: today's ggml-only plan. `pending` records that the
    // unit runs on ggml only because the sidecar was not ready yet.
    EncoderRoute() = default;
    static EncoderRoute ggml(bool pending = false) {
        EncoderRoute r;
        r.pending_ = pending;
        return r;
    }

    // A route that holds a Core ML encoder for one unit. `hold` keeps the
    // model alive (a runtime handle, or the owner context's lifetime token).
    // `owner`: the sidecar owns the encoder (no ggml weights), so a failure is
    // an error instead of a fallback. `strict` (use = require): a dispatch the
    // handle cannot serve throws instead of running on ggml.
    static EncoderRoute coreml(parakeet_coreml_context * ctx, EncoderRouteGeometry geometry,
                               std::string label, std::shared_ptr<void> hold, Ops ops,
                               bool owner, bool strict = false) {
        EncoderRoute r;
        r.s_ = std::make_shared<State>();
        r.s_->ctx = ctx;
        r.s_->geometry = geometry;
        r.s_->label = std::move(label);
        r.s_->hold = std::move(hold);
        r.s_->ops = ops;
        r.s_->owner = owner;
        r.s_->strict = strict;
        return r;
    }

    // A handle was acquired for this unit; the plan is the Core ML plan even
    // after a failure (windows after the failure run on ggml).
    bool holds() const { return s_ != nullptr; }
    // The handle can still serve the next window of this unit.
    bool usable() const { return s_ != nullptr && !s_->failed; }
    bool failed() const { return s_ != nullptr && s_->failed; }
    bool owner() const { return s_ != nullptr && s_->owner; }
    bool strict() const { return s_ != nullptr && s_->strict; }
    bool pending() const { return pending_; }

    parakeet_coreml_context * context() const { return s_ ? s_->ctx : nullptr; }
    const EncoderRouteGeometry & geometry() const {
        static const EncoderRouteGeometry none;
        return s_ ? s_->geometry : none;
    }
    // The Core ML label of the held handle ("" for the ggml route).
    const std::string & coreml_label() const {
        static const std::string none;
        return s_ ? s_->label : none;
    }

    // Runs the sidecar under the model's prediction lock.
    int encode(int n_mel_frames, int mel_valid, int n_mels, const float * mel,
               int n_enc_frames, int d_model, float * encoder_out) const {
        if (!usable() || s_->ops.encode == nullptr) return -1;
        std::unique_lock<std::mutex> lock;
        if (s_->ops.lock != nullptr) lock = s_->ops.lock(s_->hold.get());
        return s_->ops.encode(s_->hold.get(), s_->ctx, n_mel_frames, mel_valid, n_mels, mel,
                              n_enc_frames, d_model, encoder_out);
    }

    std::unique_lock<std::mutex> lock_for_prediction() const {
        if (s_ == nullptr || s_->ops.lock == nullptr) return {};
        return s_->ops.lock(s_->hold.get());
    }

    // Window k failed: retire the sidecar process-wide and run the rest of
    // this unit on ggml. The owner route never retires (nothing to fall back to).
    void fail(const char * reason) const {
        if (s_ == nullptr || s_->failed) return;
        s_->failed = true;
        if (!s_->owner && s_->ops.retire != nullptr) s_->ops.retire(s_->hold.get(), reason);
    }

    // A strict route refuses a dispatch it could not serve: never returns.
    [[noreturn]] void refuse(const std::string & reason, bool prediction_failed) const {
        if (s_ != nullptr && s_->ops.refuse != nullptr) {
            s_->ops.refuse(s_->hold.get(), reason, prediction_failed);
        }
        throw std::runtime_error(reason);
    }

private:
    struct State {
        parakeet_coreml_context * ctx = nullptr;
        EncoderRouteGeometry geometry;
        std::string label;
        std::shared_ptr<void> hold;
        Ops ops;
        bool owner = false;
        bool strict = false;
        bool failed = false;
    };
    std::shared_ptr<State> s_;
    bool pending_ = false;
};

// The sidecar encode's return codes (coreml/parakeet-encoder.mm):
//   0    success;
//   1-4  the handle cannot serve this call (bad arguments, a shape above its
//        capacity or window, I/O it cannot bind) and no prediction ran;
//   5-7  the prediction failed (an error, a missing output, an output that
//        does not fit).
// Other codes come from the engine's own checks after the sidecar returned
// and are not treated as a failed prediction either, so they never retire it:
//   -10   the sidecar returned non-finite features;
//   other the CTC projection on ggml failed (100 minus its return code).
inline bool encode_prediction_failed(int rc) { return rc >= 5 && rc <= 7; }

// Whether one dispatch can go to the held encoder.
//   ggml          the engine keeps this call on ggml whatever the handle can
//                 do (a per-stage capture harness, a model family or shape the
//                 sidecar does not cover);
//   coreml        the handle can take it;
//   cannot_serve  the call is one the sidecar is meant for, but the handle
//                 cannot take it (an input longer than a fixed capacity).
enum class EncoderEligibility { ggml, coreml, cannot_serve };

// The input fits the handle's fixed capacity (always true for a flexible
// or unknown capacity, or for the ggml route).
inline bool route_fits_capacity(const EncoderRoute & route, int n_mel_frames) {
    const int capacity = route.geometry().fixed_mel_frames;
    return capacity <= 0 || n_mel_frames <= capacity;
}

// One encoder dispatch on a route: Core ML when the route can serve it,
// otherwise ggml. Only a failed prediction retires the sidecar; a call the
// handle cannot serve runs this dispatch on ggml and leaves the route usable
// for the next one. A strict route (use = require) throws instead of running
// on ggml whenever the handle could not serve the dispatch: a cannot_serve
// call, or any Core ML error (a failed prediction still retires first). An
// owner route returns any Core ML error (it has no ggml weights); a strict
// owner route also throws for a cannot_serve call.
// `used_coreml` reports where the dispatch ran.
template <class CoremlFn, class GgmlFn>
int dispatch_on_route(const EncoderRoute & route, EncoderEligibility eligibility,
                      CoremlFn && run_coreml, GgmlFn && run_ggml, bool & used_coreml,
                      int * coreml_rc = nullptr) {
    used_coreml = false;
    if (eligibility == EncoderEligibility::cannot_serve && route.usable() && route.strict()) {
        route.refuse("the Core ML encoder cannot serve this input (longer than its fixed "
                     "capacity of " + std::to_string(route.geometry().fixed_mel_frames) +
                     " mel frames) and Core ML is required", false);
    }
    if (eligibility == EncoderEligibility::coreml && route.usable()) {
        const int rc = run_coreml();
        if (coreml_rc != nullptr) *coreml_rc = rc;
        if (rc == 0) {
            used_coreml = true;
            return 0;
        }
        if (route.owner()) return rc;
        const bool prediction_failed = encode_prediction_failed(rc);
        if (prediction_failed) route.fail("prediction failed");
        if (route.strict()) {
            route.refuse("the Core ML encoder " +
                         std::string(prediction_failed ? "prediction failed" : "cannot serve this call") +
                         " (rc=" + std::to_string(rc) + ") and Core ML is required",
                         prediction_failed);
        }
    }
    return run_ggml();
}

// A push stream whose every full window is one dispatch of
// `window_mel_frames`, with `eligibility` as run_encoder decides it. When no
// full window could run, the stream is refused before it starts: on a strict
// route (use = require) dispatch_on_route would refuse each one, and on an
// owner route there are no ggml encoder weights to run them on, whatever the
// policy. Returns the reason, or "" when the stream can run.
inline std::string strict_stream_window_refusal(const EncoderRoute & route,
                                                EncoderEligibility eligibility,
                                                int window_mel_frames) {
    if (eligibility != EncoderEligibility::cannot_serve || !route.usable() ||
        !(route.strict() || route.owner())) {
        return {};
    }
    const std::string windows =
        "push-stream windows of " + std::to_string(window_mel_frames) +
        " mel frames (left context, chunk and lookahead) are longer than the Core ML "
        "encoder's fixed capacity of " + std::to_string(route.geometry().fixed_mel_frames) +
        " mel frames, so every full window would fail";
    if (route.owner()) {
        return windows + ", and the GGUF holds no ggml encoder weights to run them on; use a "
               "shorter chunk_ms or left_context_ms where the model allows it";
    }
    return windows + ", and Core ML is required; use a shorter chunk_ms or left_context_ms "
           "where the model allows it, or use = automatic";
}

// Per-call backend label: "ggml" when nothing ran on Core ML, "mixed" when
// some dispatches ran on ggml and some on Core ML, otherwise the Core ML
// program of the call's last Core ML dispatch (a model that picks its program
// per dispatch, such as TranscribePsy EU after its language gate, reports the
// program of its main pass, as Engine::encoder_backend() does).
struct EncoderBackendTally {
    int coreml = 0;
    int ggml = 0;
    bool pending = false;
    std::string coreml_label;

    void add(bool used_coreml, const std::string & label) {
        if (used_coreml) {
            ++coreml;
            if (!label.empty()) coreml_label = label;
        } else {
            ++ggml;
        }
    }
    void add_route(const EncoderRoute & route) { pending = pending || route.pending(); }
    // A later part of the same call (a windowed pass after the EU gate).
    void merge(const EncoderBackendTally & later) {
        coreml += later.coreml;
        ggml += later.ggml;
        if (!later.coreml_label.empty()) coreml_label = later.coreml_label;
        pending = pending || later.pending;
    }

    std::string label() const {
        if (coreml == 0) return "ggml";
        if (ggml > 0) return "mixed";
        return coreml_label.empty() ? std::string("coreml") : coreml_label;
    }
    bool all_coreml() const { return coreml > 0 && ggml == 0; }
};

// The label of a call whose every dispatch ran on Core ML. A model that picks
// its program per dispatch (TranscribePsy EU) passes the program that ran as
// `dispatch_label`; otherwise the held route's label is used. Never empty:
// "coreml" when neither names a program, as EncoderBackendTally::label() does.
inline std::string coreml_call_label(const EncoderRoute & route,
                                     const std::string & dispatch_label = {}) {
    if (!dispatch_label.empty()) return dispatch_label;
    if (!route.coreml_label().empty()) return route.coreml_label();
    return "coreml";
}

}  // namespace parakeet
