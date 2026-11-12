#include "llama-cparams.h"

#include <numeric>

size_t llama_max_parallel_sequences(void) {
    return LLAMA_MAX_SEQ;
}

bool llama_draft_vocab_is_valid(const std::vector<int32_t> & ranges, int32_t n_vocab, int32_t n_min_ids) {
    if (ranges.size() % 2 != 0) {
        return false;
    }

    int32_t prev_end = 0;
    int32_t n_ids    = 0;
    for (size_t i = 0; i < ranges.size(); i += 2) {
        if (ranges[i] < prev_end || ranges[i] >= ranges[i + 1] || ranges[i + 1] > n_vocab) {
            return false;
        }
        prev_end = ranges[i + 1];
        n_ids   += ranges[i + 1] - ranges[i];
    }
    return ranges.empty() || n_ids >= n_min_ids;
}

std::vector<int32_t> llama_draft_vocab_ids(const std::vector<int32_t> & ranges) {
    std::vector<int32_t> ids;
    for (size_t r = 0; r < ranges.size(); r += 2) {
        const size_t n = ids.size();
        ids.resize(n + ranges[r + 1] - ranges[r]);
        std::iota(ids.begin() + n, ids.end(), ranges[r]);
    }
    return ids;
}
