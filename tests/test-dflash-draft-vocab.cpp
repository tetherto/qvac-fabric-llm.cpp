// checks the DFlash2 draft-vocabulary range validation (including the minimum id count), the token id table built from the ranges,
// and the speculative setup check
#include "llama-cparams.h"
#include "speculative.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

struct range_case {
    std::vector<int32_t> ranges;
    bool                 valid;
};

static const int32_t n_vocab   = 248320;
static const int32_t n_min_ids = 4;

static int32_t reference_id(int32_t col, const std::vector<int32_t> & ranges) {
    for (size_t r = 0; r < ranges.size(); r += 2) {
        const int32_t n = ranges[r + 1] - ranges[r];
        if (col < n) {
            return ranges[r] + col;
        }
        col -= n;
    }
    return -1;
}

static int count_mismatches(const std::vector<int32_t> & ids, const std::vector<int32_t> & ranges) {
    int mismatches = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        mismatches += ids[i] != reference_id((int32_t) i, ranges) ? 1 : 0;
    }
    return reference_id((int32_t) ids.size(), ranges) == -1 ? mismatches : mismatches + 1;
}

static bool test_case(const range_case & c) {
    const bool valid = llama_draft_vocab_is_valid(c.ranges, n_vocab, n_min_ids);

    int mismatches = 0;
    if (valid) {
        mismatches = count_mismatches(llama_draft_vocab_ids(c.ranges), c.ranges);
    }

    const bool ok = valid == c.valid && mismatches == 0;
    printf("%s: %zu values, valid %d (expected %d), %d id mismatches: %s\n", __func__, c.ranges.size(), valid, c.valid, mismatches, ok ? "OK" : "FAIL");
    return ok;
}

static int run_cases(const std::vector<range_case> & cases) {
    int failures = 0;
    for (const auto & c : cases) {
        failures += test_case(c) ? 0 : 1;
    }
    return failures;
}

static bool spec_init_rejects(common_params_speculative & params) {
    try {
        common_speculative_free(common_speculative_init(params, 1));
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// a speculative type other than draft-dflash rejects the ranges and works without them
static bool test_needs_dflash() {
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
    params.draft.vocab_ranges = { 0, 10 };
    const bool rejected = spec_init_rejects(params);

    params.draft.vocab_ranges.clear();
    common_speculative * spec = common_speculative_init(params, 1);
    const bool accepted = spec != nullptr;
    common_speculative_free(spec);

    const bool ok = rejected && accepted;
    printf("%s: rejected with ranges %d, accepted without %d: %s\n", __func__, rejected, accepted, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    const std::vector<range_case> cases = {
        { { 3, 8 },                         true  },
        { { 0, 5, 10, 12, 20, 23 },         true  },
        { { 4, 5, 5, 9 },                   true  },
        { { 0, 98304, 248032, 248320 },     true  },
        { { },                              true  },
        { { 5, 3 },                         false },
        { { 0, 0 },                         false },
        { { 0, 10, 5, 20 },                 false },
        { { -1, 5 },                        false },
        { { 0, n_vocab + 1 },               false },
        { { 0, 10, 20 },                    false },
        { { 0, 4 },                         true  },
        { { 0, 2, 10, 12 },                 true  },
        { { 0, 3 },                         false },
        { { 0, 1, 10, 12 },                 false },
    };

    const int failures = run_cases(cases) + (test_needs_dflash() ? 0 : 1);
    return failures == 0 ? 0 : 1;
}
