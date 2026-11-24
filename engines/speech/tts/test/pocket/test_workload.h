#pragma once
#include "tts-cpp/pocket/engine.h"
#include "pocket/frontend.h"
#include "pocket/mimi.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Compute an independent context boundary from the fixture's actual voice and
// tokens, including the upstream text-duration estimate and explicit EOS tail.
inline int context_with_tail(const tts_cpp::pocket::EngineOptions & opts, const std::string & text) {
    using namespace tts_cpp::pocket::detail;
    auto * voice = gguf_init_from_file(opts.voice_path.c_str(), {true, nullptr});
    if (!voice) throw std::runtime_error("cannot read voice metadata");
    const auto key = gguf_find_key(voice, "pocket.voice_frames");
    if (key < 0) { gguf_free(voice); throw std::runtime_error("missing voice frames"); }
    const int voice_frames = gguf_get_val_u32(voice, key);
    gguf_free(voice);
    Frontend frontend(opts.frontend_path);
    int required = 0;
    for (const auto & chunk : frontend.split(text, opts.max_tokens)) {
        const int tokens = frontend.encode(chunk.text).size();
        const int estimate = std::ceil((tokens/3.0+2)*12.5);
        const int tail = opts.frames_after_eos < 0 ? chunk.tail_frames : opts.frames_after_eos;
        required = std::max(required, voice_frames+tokens+estimate+tail);
    }
    return required;
}
