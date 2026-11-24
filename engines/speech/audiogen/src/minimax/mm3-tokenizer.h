#pragma once

#include "mm3-model.h"

#include "bpe.h"
#include "logic.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct MM3Tokenizer {
    BPETokenizer bpe;
    bool         loaded = false;
    std::string  path;

    std::vector<std::pair<std::string, int32_t>> specials;
};

static bool mm3_tokenizer_load(const MM3Model & m, MM3Tokenizer * t, std::string * err) {
    if (t->loaded && t->path == m.lm_file.path) {
        return true;
    }
    if (!m.lm_file.found) {
        if (err) {
            *err = "the MiniMax-Music3 LM GGUF was not found; nothing to tokenise with";
        }
        return false;
    }

    *t = MM3Tokenizer{};
    if (!load_bpe_from_gguf(&t->bpe, m.lm_file.path.c_str())) {
        if (err) {
            *err = "mm3-lm GGUF carries no tokenizer.ggml.tokens / .merges arrays";
        }
        return false;
    }

    const int32_t hi = (int32_t) m.lm_cfg.semantic_vocab_offset;
    const int32_t lo = hi > 32 ? hi - 32 : 0;
    for (int32_t id = lo; id < hi && id < (int32_t) t->bpe.id_to_str.size(); id++) {
        const std::string & s = t->bpe.id_to_str[(size_t) id];
        if (s.size() >= 4 && s.compare(0, 2, "<|") == 0 && s.compare(s.size() - 2, 2, "|>") == 0) {
            t->specials.emplace_back(s, id);
        }
    }
    std::sort(t->specials.begin(), t->specials.end(),
              [](const std::pair<std::string, int32_t> & a, const std::pair<std::string, int32_t> & b) {
                  return a.first.size() > b.first.size();
              });

    t->loaded = true;
    t->path   = m.lm_file.path;
    fprintf(stderr, "[MM3-Tok] Loaded %d vocab entries, %zu added tokens from %s\n", t->bpe.n_vocab,
            t->specials.size(), m.lm_file.name.c_str());
    return true;
}

static void mm3_tokenizer_encode(const MM3Tokenizer & t, const std::string & text, std::vector<int32_t> * out) {
    out->clear();

    auto flush_text = [&](size_t from, size_t to) {
        if (to <= from) {
            return;
        }
        const std::string segment = text.substr(from, to - from);
        std::vector<int>  ids;
        for (const auto & chunk : gpt2_pre_tokenize(segment)) {
            encode_chunk(&t.bpe, chunk, ids);
        }
        for (int id : ids) {
            out->push_back((int32_t) id);
        }
    };

    size_t seg_start = 0;
    size_t i         = 0;
    while (i < text.size()) {
        if (text[i] != '<') {
            i++;
            continue;
        }
        const std::pair<std::string, int32_t> * hit = nullptr;
        for (const auto & sp : t.specials) {
            if (text.compare(i, sp.first.size(), sp.first) == 0) {
                hit = &sp;
                break;
            }
        }
        if (!hit) {
            i++;
            continue;
        }
        flush_text(seg_start, i);
        out->push_back(hit->second);
        i += hit->first.size();
        seg_start = i;
    }
    flush_text(seg_start, text.size());
}

static void mm3_tokenizer_uncond(const MM3LmConfig & c, const std::vector<int32_t> & cond,
                                 std::vector<int32_t> * uncond) {
    *uncond =
        tts_cpp::minimax::detail::mask_unconditional(cond, (int32_t) c.tok_audio_cfg);
}
