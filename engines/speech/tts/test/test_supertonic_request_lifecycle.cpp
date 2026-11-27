#include "tts-cpp/supertonic/engine.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using tts_cpp::supertonic::Engine;
using tts_cpp::supertonic::EngineOptions;
using tts_cpp::supertonic::SynthesisResult;

constexpr int synthesis_threads = 4;
constexpr int output_sample_rate = 24000;
constexpr int synthesis_seed = 42;
constexpr int split_chunk_size = 40;
constexpr int unsplit_chunk_size = 10000;
constexpr auto continuation_text = "Each participant checks the record,";
constexpr auto unpunctuated_continuation_text = "Each participant checks the record";
constexpr auto batch_text = "The next request can use the same voice.";
constexpr auto streaming_text =
    "A clear voice reads the first sentence to the listener. "
    "The second sentence must remain available for the next request.";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

EngineOptions make_options(const std::string & model, int chunk_size) {
    EngineOptions options;
    options.model_gguf_path = model;
    options.backend = "cpu";
    options.n_threads = synthesis_threads;
    options.output_sample_rate = output_sample_rate;
    options.seed = synthesis_seed;
    options.stream_chunk_tokens = chunk_size;
    return options;
}

void require_same_audio(const SynthesisResult & expected, const SynthesisResult & actual,
                         const std::string & message) {
    require(!actual.pcm.empty() && actual.sample_rate == output_sample_rate, message + ": invalid PCM");
    require(expected.sample_rate == actual.sample_rate && expected.pcm == actual.pcm, message);
}

void cancel_active_stream(Engine & engine) {
    int callbacks = 0;
    bool cancelled = false;
    try {
        engine.synthesize(streaming_text, [&](const float *, std::size_t count, int index, bool last) {
            require(count > 0 && index == 0 && !last, "Cancellation fixture must contain multiple speech chunks");
            ++callbacks;
            engine.cancel();
        });
    } catch (const std::runtime_error & error) {
        if (std::string(error.what()).find("cancelled") == std::string::npos) throw;
        cancelled = true;
    }
    require(cancelled && callbacks == 1, "In-flight cancellation did not stop the remaining stream chunks");
}

SynthesisResult collect_stream(Engine & engine) {
    std::vector<float> received;
    int callbacks = 0;
    bool finished = false;
    const auto result = engine.synthesize(streaming_text,
        [&](const float * pcm, std::size_t count, int index, bool last) {
            require(index == callbacks && !finished, "Stream callback order changed");
            ++callbacks;
            finished = last;
            received.insert(received.end(), pcm, pcm + count);
        });
    require(callbacks > 1 && finished && !received.empty() && received == result.pcm,
            "Streaming audio did not match the ordered callbacks");
    return result;
}

void test_cancelled_request_reuse(const std::string & model) {
    Engine engine(make_options(model, split_chunk_size));
    const auto batch = engine.synthesize(batch_text);
    const auto stream = collect_stream(engine);
    cancel_active_stream(engine);
    require_same_audio(batch, engine.synthesize(batch_text), "Batch synthesis retained a cancelled stream's state");
    cancel_active_stream(engine);
    require_same_audio(stream, collect_stream(engine), "Streaming synthesis retained a cancelled request's state");
    engine.cancel();
    require_same_audio(batch, engine.synthesize(batch_text), "A new batch request did not clear pre-existing cancellation");
}

SynthesisResult test_preserved_segment(Engine & engine, const std::string & text) {
    const auto reference = engine.synthesize(text,
        [](const float *, std::size_t, int, bool) {});
    int callbacks = 0;
    std::vector<float> received;
    const auto segment = engine.synthesize_segment(text,
        [&](const float * pcm, std::size_t count, int index, bool last) {
            require(index == 0 && last, "A caller-selected segment was split internally");
            ++callbacks;
            received.insert(received.end(), pcm, pcm + count);
        });
    require(callbacks == 1 && received == segment.pcm, "The whole-segment callback changed the returned PCM");
    require_same_audio(reference, segment, "Whole-segment synthesis changed continuation or resampling behavior");
    engine.cancel();
    require_same_audio(reference, engine.synthesize_segment(text),
                       "Optional-callback segment synthesis retained cancellation");
    return reference;
}

void test_segment_bypasses_configured_chunker(const std::string & model) {
    Engine reference_engine(make_options(model, unsplit_chunk_size));
    int reference_callbacks = 0;
    const auto reference = reference_engine.synthesize(streaming_text,
        [&](const float *, std::size_t, int, bool) { ++reference_callbacks; });
    require(reference_callbacks == 1, "Reference synthesis must remain unsplit");

    Engine chunked_engine(make_options(model, split_chunk_size));
    collect_stream(chunked_engine);
    int callbacks = 0;
    std::vector<float> received;
    const auto segment = chunked_engine.synthesize_segment(streaming_text,
        [&](const float * pcm, std::size_t count, int index, bool last) {
            require(index == 0 && last, "Whole-segment synthesis used the configured text chunker");
            ++callbacks;
            received.insert(received.end(), pcm, pcm + count);
        });
    require(callbacks == 1 && received == segment.pcm,
            "Whole-segment synthesis did not deliver exactly one complete PCM callback");
    require_same_audio(reference, segment, "Bypassing the configured chunker changed unsplit synthesis PCM");
}

void test_worker_cache_release(Engine & engine, const SynthesisResult & reference) {
    std::exception_ptr failure;
    std::thread worker([&] {
        try {
            require_same_audio(reference, engine.synthesize_segment(continuation_text),
                               "Worker synthesis differs from calling-thread synthesis");
            engine.release_thread_resources();
            engine.release_thread_resources();
            require_same_audio(reference, engine.synthesize_segment(continuation_text),
                               "Worker cache release changed subsequent synthesis");
        } catch (...) {
            failure = std::current_exception();
        }
        engine.release_thread_resources();
    });
    worker.join();
    if (failure) std::rethrow_exception(failure);
}

void test_segment_and_cache_lifecycle(const std::string & model) {
    Engine engine(make_options(model, unsplit_chunk_size));
    const auto reference = test_preserved_segment(engine, continuation_text);
    const auto unfinished = test_preserved_segment(engine, unpunctuated_continuation_text);
    const auto terminated = engine.synthesize(unpunctuated_continuation_text);
    require(unfinished.pcm != terminated.pcm,
            "The word-ended fixture did not distinguish continuation from an implicitly appended period");
    engine.release_thread_resources();
    engine.release_thread_resources();
    require_same_audio(reference, engine.synthesize_segment(continuation_text),
                       "Repeated cache release changed subsequent synthesis");
    test_worker_cache_release(engine, reference);
}
} // namespace

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "Usage: test-supertonic-request-lifecycle MODEL.gguf");
        test_cancelled_request_reuse(argv[1]);
        test_segment_bypasses_configured_chunker(argv[1]);
        test_segment_and_cache_lifecycle(argv[1]);
        std::printf("Supertonic request cancellation, segment, resampling, and worker-cache regressions passed\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
