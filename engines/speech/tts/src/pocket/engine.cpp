#define TTS_CPP_BUILD
#include "tts-cpp/pocket/engine.h"
#include "pocket/flow_lm.h"
#include "pocket/mimi.h"
#include "pocket/frontend.h"
#include "pocket/generation.h"
#include "pocket/reference_audio.h"
#include "pocket/backend.h"
#include "pocket/coreml.h"
#include "voice_features.h"
#include "backend_selection.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace tts_cpp::pocket {
namespace {
using Clock = std::chrono::steady_clock;
[[noreturn]] void fail(const std::string & text) { throw std::runtime_error("Pocket TTS: "+text); }
double milliseconds(Clock::time_point begin) { return std::chrono::duration<double, std::milli>(Clock::now()-begin).count(); }
// Explicit transform avoids implementation-defined std::normal_distribution.
class Noise {
    std::mt19937 random_;
    bool spare_ = false;
    double saved_ = 0;
public:
    explicit Noise(uint32_t seed) : random_(seed) {}
    double normal() {
        if (spare_) { spare_ = false; return saved_; }
        const double u = (double(random_())+0.5)/4294967296.0;
        const double v = (double(random_())+0.5)/4294967296.0;
        const double r = std::sqrt(-2*std::log(u)), angle = 6.2831853071795864769*v;
        saved_ = r*std::sin(angle); spare_ = true; return r*std::cos(angle);
    }
    std::vector<float> sample(int width, float temperature, float clamp) {
        std::vector<float> values(width);
        const double sigma = std::sqrt(double(temperature));
        for (auto & x : values) {
            // Inverse-CDF sampling for very narrow truncation would be more
            // efficient. Bound rejection so hostile options cannot hang.
            int tries = 0;
            do { x = float(normal()*sigma); if (++tries > 100000) fail("noise truncation is too narrow"); }
            while (clamp > 0 && std::abs(x) > clamp);
        }
        return values;
    }
};
}

struct Engine::Impl {
    EngineOptions opts;
    std::unique_ptr<detail::FlowLM> lm;
    std::unique_ptr<detail::Mimi> mimi;
    std::unique_ptr<detail::Frontend> frontend;
    detail::VoiceState voice;
    std::string device_name;
    detail::CoremlPolicy coreml_policy;
    // Used by the constructor and then only by one producer thread at a time.
    std::unique_ptr<detail::CoremlStep> step;
    std::atomic<bool> coreml_active{false};
    std::string coreml_label;
    std::vector<float> bos;
    // Used by the constructor and then only by the synthesizing thread.
    std::unique_ptr<detail::CoremlDecoder> decoder;
    std::atomic<bool> decoder_active{false};
    std::string decoder_label;
    std::vector<float> chunk_latents;  // decoded on Core ML this chunk, for a ggml replay
    // Where this call's stages ran.
    bool flow_coreml = false, flow_ggml = false, mimi_coreml = false, mimi_ggml = false;
    std::atomic<bool> operating{false};
    std::atomic<bool> cancelled{false};
    std::condition_variable changed;

    explicit Impl(const EngineOptions & o) : opts(o) {
        detail::backend_request(o.backend);
        coreml_policy = detail::resolve_coreml_policy(o.coreml_compute_units, o.coreml_strict);
        if (o.output_sample_rate < 8000 || o.output_sample_rate > 192000) fail("output sample rate must be 8000..192000 Hz");
        if (o.steps < 1 || o.steps > 64 || !std::isfinite(o.temperature) || o.temperature < 0 || o.temperature > 10 ||
            !std::isfinite(o.noise_clamp) || o.noise_clamp < 0 || !std::isfinite(o.eos_threshold) ||
            o.frames_after_eos < -1 || o.frames_after_eos > 100 || o.max_tokens < 1 || o.max_tokens > 1024)
            fail("invalid synthesis options");
        if (o.voice_path.empty() == o.reference_audio_path.empty()) fail("supply exactly one prepared voice or reference audio file");
        std::unique_ptr<detail::ReferenceAudio> reference;
        if (!o.reference_audio_path.empty()) reference.reset(new detail::ReferenceAudio(o.reference_audio_path));
        if (!o.backends_dir.empty()) ::tts_cpp::detail::set_backends_directory(o.backends_dir);
        auto backend = detail::select_backend(o.use_gpu, o.backend);
        const auto device = ggml_backend_get_device(backend.get());
        device_name = ggml_backend_dev_description(device);
        frontend.reset(new detail::Frontend(o.frontend_path));
        lm.reset(new detail::FlowLM(o.flow_lm_path, o.context, o.n_threads, device));
        mimi.reset(new detail::Mimi(o.mimi_path, o.n_threads, device));
        if (frontend->source_hash() != lm->source_hash() || mimi->source_hash() != lm->source_hash() ||
            frontend->vocab_size() != lm->config().vocab_size || lm->config().latent_dim != mimi->latent_dim()) fail("model artifacts do not belong to the same checkpoint");
        if (!o.voice_path.empty()) voice = lm->read_voice(o.voice_path);
        else {
            auto pcm = reference->read_mono();
            const int rate = reference->sample_rate();
            if (rate != mimi->sample_rate()) pcm = ::resample_sinc(pcm, rate, mimi->sample_rate());
            lm->prefill(lm->voice_embeddings(mimi->encode(pcm))); voice = lm->capture_voice();
        }
        load_coreml_step();
        load_coreml_decoder();
    }

    [[noreturn]] void fail_strict(const std::string & reason) const {
        fail(reason + " (Core ML strict policy: coreml_strict / " + detail::COREML_STRICT_ENV + ")");
    }
    // The voice prefix never changes and no frame writes below it, so its
    // rows go into the sidecar's state once (again after a strict failure).
    void load_coreml_step() {
        if (!detail::coreml_step_enabled(opts.flow_lm_path, coreml_policy)) return;
        try {
            auto loaded = std::make_unique<detail::CoremlStep>(detail::coreml_step_path(opts.flow_lm_path),
                detail::coreml_units(coreml_policy, detail::COREML_STEP_DEFAULT_UNITS), lm->config(), lm->source_hash());
            // A chunk needs the voice plus at least one token and one frame.
            if (voice.frames+2 > loaded->context())
                throw std::runtime_error("the " + std::to_string(voice.frames) + "-frame voice fills the Core ML step "
                    "sidecar's " + std::to_string(loaded->context()) + " positions; export a longer --context");
            loaded->reset(voice_rows());
            bos = lm->bos_latent();
            coreml_label = loaded->label();
            step = std::move(loaded);
            coreml_active.store(true);
        } catch (const std::exception & e) {
            if (coreml_policy.strict) fail_strict(e.what());
            std::fprintf(stderr, "Pocket TTS: %s; FlowLM stays on ggml\n", e.what());
        }
    }
    void load_coreml_decoder() {
        if (!detail::coreml_decoder_enabled(opts.mimi_path, coreml_policy)) return;
        try {
            decoder = std::make_unique<detail::CoremlDecoder>(detail::coreml_decoder_path(opts.mimi_path),
                detail::coreml_units(coreml_policy, detail::COREML_DECODER_DEFAULT_UNITS), mimi->source_hash());
            decoder_label = decoder->label();
            decoder_active.store(true);
        } catch (const std::exception & e) {
            if (coreml_policy.strict) fail_strict(e.what());
            std::fprintf(stderr, "Pocket TTS: %s; Mimi stays on ggml\n", e.what());
        }
    }
    void retire_decoder(const std::exception & e) {
        // Strict mode keeps the decoder; the next chunk resets its stream.
        if (coreml_policy.strict) fail_strict(e.what());
        std::fprintf(stderr, "Pocket TTS: %s; retiring the Core ML Mimi decoder, Mimi continues on ggml\n", e.what());
        decoder_active.store(false);
        decoder.reset();
    }
    detail::CacheRows voice_rows() const { return {0, voice.frames, voice.keys, voice.values}; }
    // Strict mode keeps the sidecar for later calls: the failed frame's
    // blended write may have left non-finite values in any row, so the state
    // is replaced by one holding only the voice rows.
    void retire_step(const std::exception & e) {
        if (coreml_policy.strict) {
            try { step->reset(voice_rows()); }
            catch (const std::exception &) { coreml_active.store(false); step.reset(); }
            fail_strict(e.what());
        }
        std::fprintf(stderr, "Pocket TTS: %s; retiring the Core ML step sidecar, FlowLM continues on ggml\n", e.what());
        coreml_active.store(false);
        step.reset();
    }

    struct Chunk {
        std::vector<int> tokens;
        detail::FrameBudget budget;
    };
    struct Queue {
        std::mutex mutex;
        std::deque<std::vector<float>> frames;
        bool done = false, stop = false;
        std::exception_ptr error;
    };
    std::vector<Chunk> prepare_chunks(const std::string & text) const {
        std::vector<Chunk> prepared;
        for (const auto & chunk : frontend->split(text, opts.max_tokens)) {
            auto tokens = frontend->encode(chunk.text);
            const int tail = opts.frames_after_eos >= 0 ? opts.frames_after_eos : chunk.tail_frames;
            const auto budget = detail::frame_budget(tokens.size(), tail);
            const size_t positions = voice.frames+tokens.size()+budget.max_frames();
            if (positions > size_t(opts.context))
                fail("text chunk exceeds the context; add sentence punctuation or increase context");
            if (step && coreml_policy.strict && positions > size_t(step->context()))
                fail_strict("text chunk needs " + std::to_string(positions) + " positions; the Core ML step sidecar holds " +
                            std::to_string(step->context()) + " (lower max_tokens or export a longer --context)");
            prepared.push_back({std::move(tokens), budget});
        }
        if (prepared.empty()) fail("no text chunks");
        return prepared;
    }
    bool enqueue(Queue & queue, std::vector<float> latent) {
        std::unique_lock<std::mutex> lock(queue.mutex);
        changed.wait(lock, [&] { return queue.frames.size() < size_t(mimi->max_decode_frames()) || queue.stop || cancelled.load(); });
        if (queue.stop || cancelled.load()) return false;
        queue.frames.push_back(std::move(latent));
        changed.notify_all();
        return true;
    }
    // The Core ML sidecar serves a chunk whose every position fits its state.
    // A failing prediction retires it; the frames it generated are prefilled
    // into the ggml cache and the chunk continues on ggml.
    void generate_frames(const Chunk & chunk, Noise & noise, Queue & queue) {
        const int start = lm->position();
        lm->prefill(lm->text_embeddings(chunk.tokens));
        int position = lm->position();
        bool coreml = step && position+chunk.budget.max_frames() <= step->context();
        if (coreml) {
            try { step->write_rows(lm->cache_rows(start, position-start)); }
            catch (const std::exception & e) { retire_step(e); coreml = false; }
        }
        std::vector<float> previous, fed, hidden, next;
        detail::FrameProgress progress(chunk.budget);
        while (progress.has_next()) {
            if (cancelled.load()) return;
            { std::lock_guard<std::mutex> lock(queue.mutex); if (queue.stop) return; }
            const auto frame_noise = noise.sample(lm->config().latent_dim, opts.temperature, opts.noise_clamp);
            float eos = 0;
            if (coreml) {
                const auto & input = previous.empty() ? bos : previous;
                try { step->step(position, input, frame_noise, hidden, next); }
                catch (const std::exception & e) {
                    retire_step(e); coreml = false;
                    if (!fed.empty()) lm->prefill(lm->audio_embeddings(fed));
                }
                if (coreml) {
                    fed.insert(fed.end(), input.begin(), input.end()); ++position; flow_coreml = true;
                    eos = lm->eos_logit(hidden);
                    if (opts.steps > 1) { next = lm->sample(hidden, frame_noise, opts.steps); flow_ggml = true; }
                }
            }
            if (!coreml) {
                const auto condition = lm->advance(previous);
                eos = condition.eos_logit; flow_ggml = true;
                next = lm->sample(condition.hidden, frame_noise, opts.steps);
            }
            previous.swap(next);
            if (progress.finish_frame(eos > opts.eos_threshold)) return;
            if (!enqueue(queue, lm->denormalize(previous))) return;
        }
        if (!cancelled.load()) fail("generation reached its estimated limit without completing EOS");
    }
    void produce(const Chunk & chunk, Noise & noise, Queue & queue) noexcept {
        try { generate_frames(chunk, noise, queue); }
        catch (...) {
            std::lock_guard<std::mutex> lock(queue.mutex); queue.error = std::current_exception();
        }
        { std::lock_guard<std::mutex> lock(queue.mutex); queue.done = true; }
        changed.notify_all();
    }
    std::vector<float> dequeue(Queue & queue) {
        std::unique_lock<std::mutex> lock(queue.mutex);
        changed.wait(lock, [&] { return !queue.frames.empty() || queue.done || cancelled.load(); });
        if (queue.error) std::rethrow_exception(queue.error);
        if (cancelled.load()) return {};
        std::vector<float> latents;
        const size_t batch_size = size_t(mimi->max_decode_frames())*mimi->latent_dim();
        while (!queue.frames.empty() && latents.size() < batch_size) {
            auto & frame = queue.frames.front();
            latents.insert(latents.end(), frame.begin(), frame.end());
            queue.frames.pop_front();
        }
        changed.notify_all();
        return latents;
    }
    // A failing Core ML decode retires the decoder: the chunk's earlier
    // frames are decoded again on ggml (output discarded) to rebuild Mimi's
    // streaming state, then ggml decodes this batch.
    std::vector<float> decode(const std::vector<float> & latents) {
        if (decoder) {
            try {
                auto pcm = decoder->decode(latents);
                chunk_latents.insert(chunk_latents.end(), latents.begin(), latents.end());
                mimi_coreml = true;
                return pcm;
            } catch (const std::exception & e) {
                retire_decoder(e);
                const size_t batch = size_t(mimi->max_decode_frames())*mimi->latent_dim();
                for (size_t at = 0; at < chunk_latents.size(); at += batch)
                    mimi->decode(std::vector<float>(chunk_latents.begin()+at,
                                                    chunk_latents.begin()+std::min(chunk_latents.size(), at+batch)));
                chunk_latents.clear();
            }
        }
        mimi_ggml = true;
        return mimi->decode(latents);
    }
    using Emit = std::function<void(const std::vector<float> &)>;
    void drain(Queue & queue, OutputResampler & resampler, SynthesisResult & result, const Emit & emit) {
        while (!cancelled.load()) {
            auto latents = dequeue(queue);
            if (latents.empty()) break;
            auto pcm = decode(latents);
            if (cancelled.load()) break;
            result.generated_frames += latents.size()/mimi->latent_dim();
            emit(resampler.process(pcm));
        }
    }
    void stop_producer(Queue & queue, std::thread & producer) {
        { std::lock_guard<std::mutex> lock(queue.mutex); queue.stop = true; }
        changed.notify_all();
        if (producer.joinable()) producer.join();
    }
    void generate_chunk(const Chunk & chunk, Noise & noise, OutputResampler & resampler,
                        SynthesisResult & result, const Emit & emit) {
        lm->restore_voice(voice); mimi->reset_decoder();
        if (decoder) decoder->reset();
        chunk_latents.clear();
        Queue queue;
        std::thread producer([&] { produce(chunk, noise, queue); });
        try { drain(queue, resampler, result, emit); }
        catch (...) { stop_producer(queue, producer); throw; }
        stop_producer(queue, producer);
        if (queue.error && !cancelled.load()) std::rethrow_exception(queue.error);
    }
    void generate_chunks(const std::vector<Chunk> & chunks, Noise & noise, OutputResampler & resampler,
                         SynthesisResult & result, const Emit & emit) {
        for (const auto & chunk : chunks) {
            if (cancelled.load()) break;
            generate_chunk(chunk, noise, resampler, result, emit);
        }
    }
    SynthesisResult generate(const std::string & text, const AudioCallback & callback) {
        bool idle = false;
        if (!operating.compare_exchange_strong(idle, true)) fail("concurrent or reentrant synthesis on one engine is unsupported");
        struct Release { std::atomic<bool> & flag; ~Release() { flag.store(false); } } release{operating};
        cancelled.store(false);
        if (!callback) fail("stream callback is empty");
        const auto begin = Clock::now();
        // Validate every chunk, including its EOS tail, before delivering PCM.
        const auto chunks = prepare_chunks(text);
        SynthesisResult result; result.sample_rate = opts.output_sample_rate;
        Noise noise(opts.seed); OutputResampler resampler(mimi->sample_rate(), opts.output_sample_rate);
        flow_coreml = flow_ggml = mimi_coreml = mimi_ggml = false;
        bool delivered = false;
        const auto emit = [&](const std::vector<float> & pcm) {
            if (pcm.empty() || cancelled.load()) return;
            if (!delivered) { result.first_audio_ms = milliseconds(begin); delivered = true; }
            if (!callback(pcm.data(), pcm.size(), opts.output_sample_rate)) {
                cancelled.store(true); changed.notify_all();
            }
        };
        generate_chunks(chunks, noise, resampler, result, emit);
        if (!cancelled.load()) emit(resampler.finish());
        const auto placement = [](bool coreml, bool ggml, const std::string & label) {
            return !coreml ? detail::GGML_STAGE_BACKEND : ggml ? detail::MIXED_STAGE_BACKEND : label;
        };
        result.flow_lm_backend = placement(flow_coreml, flow_ggml, coreml_label);
        result.mimi_backend = placement(mimi_coreml, mimi_ggml, decoder_label);
        result.cancelled = cancelled.load(); result.generation_ms = milliseconds(begin); return result;
    }
};
Engine::Engine(const EngineOptions & options) : impl_(new Impl(options)) {}
Engine::~Engine() = default;
int Engine::sample_rate() const noexcept { return impl_->opts.output_sample_rate; }
const char * Engine::backend_name() const noexcept { return flow_lm_backend_name(); }
const char * Engine::flow_lm_backend_name() const noexcept { return impl_->lm->backend_name(); }
const char * Engine::mimi_backend_name() const noexcept { return impl_->mimi->backend_name(); }
bool Engine::flow_lm_on_coreml() const noexcept { return impl_->coreml_active.load(); }
bool Engine::mimi_on_coreml() const noexcept { return impl_->decoder_active.load(); }
const char * Engine::device_name() const noexcept { return impl_->device_name.c_str(); }
std::string Engine::model_source_sha256() const { return impl_->lm->source_hash(); }
void Engine::cancel() noexcept { impl_->cancelled.store(true); impl_->changed.notify_all(); }
SynthesisResult Engine::synthesize_stream(const std::string & text, const AudioCallback & callback) { return impl_->generate(text, callback); }
SynthesisResult Engine::synthesize(const std::string & text) {
    std::vector<float> pcm;
    auto result = synthesize_stream(text, [&](const float * p, size_t n, int) { pcm.insert(pcm.end(), p, p+n); return true; });
    result.pcm = std::move(pcm); return result;
}
}
