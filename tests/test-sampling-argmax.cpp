#include "common.h"
#include "sampling.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct argmax_case {
    std::string name;
    std::function<void(common_params_sampling &)> edit;
    bool is_argmax;
};

static common_params_sampling greedy_params() {
    common_params_sampling params;
    params.temp = 0.0f;
    return params;
}

static const std::vector<argmax_case> ARGMAX_CASES = {
    { "default chain at temp 0",       [](common_params_sampling &) {},                                    true  },
    { "top_p 1 at temp 0",             [](common_params_sampling & p) { p.top_p = 1.0f; },                 true  },
    { "top_k 1 last, temp 1",          [](common_params_sampling & p) { p.temp = 1.0f; p.top_k = 1;
                                           p.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K }; }, true },
    { "top_n_sigma and min_p",         [](common_params_sampling & p) { p.top_n_sigma = 1.0f; p.min_p = 2.0f; }, true },
    { "penalties without history",     [](common_params_sampling & p) { p.penalty_last_n = 0; p.penalty_repeat = 1.3f; }, true },
    { "dry with base below 1",         [](common_params_sampling & p) { p.dry_multiplier = 0.8f; p.dry_base = 0.5f; }, true },
    { "xtc threshold above 0.5",       [](common_params_sampling & p) { p.xtc_probability = 0.5f; p.xtc_threshold = 0.6f; }, true },
    { "temp 0.8",                      [](common_params_sampling & p) { p.temp = 0.8f; },                  false },
    { "negative temp",                 [](common_params_sampling & p) { p.temp = -1.0f; },                 false },
    { "dynamic temp",                  [](common_params_sampling & p) { p.dynatemp_range = 0.5f; },        false },
    { "top_k 1 last, temp 0.7",        [](common_params_sampling & p) { p.temp = 0.7f; p.top_k = 1;
                                           p.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K }; }, false },
    { "n_probs",                       [](common_params_sampling & p) { p.n_probs = 3; },                  false },
    { "logit bias",                    [](common_params_sampling & p) { p.logit_bias = { { 1, -1.0f } }; }, false },
    { "repeat penalty",                [](common_params_sampling & p) { p.penalty_repeat = 1.1f; },        false },
    { "frequency penalty",             [](common_params_sampling & p) { p.penalty_freq = 0.5f; },          false },
    { "presence penalty",              [](common_params_sampling & p) { p.penalty_present = 0.5f; },       false },
    { "dry",                           [](common_params_sampling & p) { p.dry_multiplier = 0.8f; },        false },
    { "typical",                       [](common_params_sampling & p) { p.typ_p = 0.9f; },                 false },
    { "xtc",                           [](common_params_sampling & p) { p.xtc_probability = 0.5f; },       false },
    { "mirostat",                      [](common_params_sampling & p) { p.mirostat = 2; },                 false },
    { "infill",                        [](common_params_sampling & p) { p.samplers.insert(p.samplers.begin(), COMMON_SAMPLER_TYPE_INFILL); }, false },
    { "adaptive-p",                    [](common_params_sampling & p) { p.samplers.push_back(COMMON_SAMPLER_TYPE_ADAPTIVE_P); }, false },
    { "temperature not last",          [](common_params_sampling & p) { p.samplers.push_back(COMMON_SAMPLER_TYPE_MIN_P); }, false },
    { "no samplers",                   [](common_params_sampling & p) { p.samplers.clear(); },             false },
};

static bool check_case(const argmax_case & c) {
    common_params_sampling params = greedy_params();
    c.edit(params);

    const bool is_argmax = common_sampler_params_is_argmax(params);
    printf("%-32s expected %d, got %d\n", c.name.c_str(), c.is_argmax, is_argmax);
    return is_argmax == c.is_argmax;
}

static int count_failures(const std::vector<argmax_case> & cases) {
    int n_failed = 0;
    for (const auto & c : cases) {
        n_failed += check_case(c) ? 0 : 1;
    }
    return n_failed;
}

int main() {
    const int n_failed = count_failures(ARGMAX_CASES);
    if (n_failed > 0) {
        fprintf(stderr, "%d argmax predicate cases failed\n", n_failed);
        return 1;
    }
    printf("all %zu argmax predicate cases passed\n", ARGMAX_CASES.size());
    return 0;
}
