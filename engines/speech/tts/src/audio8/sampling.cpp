#include "audio8/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

constexpr float MIN_TEMPERATURE = 1e-5f;
// Keeps -log(u) finite when the draw lands on zero.
constexpr float MIN_UNIFORM = 1e-20f;
const float REJECTED = -std::numeric_limits<float>::infinity();

// Nothing below rank top_k can survive, so only that many candidates are ever
// ranked. Selecting the leaders instead of sorting the whole vocabulary is what
// keeps the fast head's nine draws per frame off the critical path.
size_t leader_count(const sampling_params & params, size_t vocab) {
    return params.top_k > 0 ? std::min<size_t>(params.top_k, vocab) : vocab;
}

std::vector<int> rank_leaders(const std::vector<float> & logits, size_t count) {
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto by_logit = [&](int left, int right) { return logits[left] > logits[right]; };
    if (count >= order.size()) {
        std::sort(order.begin(), order.end(), by_logit);
        return order;
    }
    std::partial_sort(order.begin(), order.begin() + count, order.end(), by_logit);
    order.resize(count);
    return order;
}

// One exponential per candidate: the whole vector normalises the masses, and
// the leaders read their own entries back out of it.
std::vector<float> exponentiated(const std::vector<float> & logits, float top) {
    std::vector<float> weights(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        weights[index] = std::exp(logits[index] - top);
    }
    return weights;
}

float total_of(const std::vector<float> & weights) {
    float total = 0.0f;
    for (float weight : weights) total += weight;
    return total;
}

std::vector<float> leader_masses(const std::vector<float> & weights,
                                 const std::vector<int> & leaders, float total) {
    std::vector<float> masses(leaders.size());
    for (size_t rank = 0; rank < leaders.size(); ++rank) {
        masses[rank] = weights[leaders[rank]] / total;
    }
    return masses;
}

// How many of the ranked candidates survive top-p. A candidate goes when the
// mass up to and including it passes top_p, so the one that crosses the
// threshold is dropped; the highest-ranked candidate always stays.
size_t surviving_rank_count(const std::vector<float> & masses, float top_p) {
    float mass = 0.0f;
    for (size_t rank = 0; rank < masses.size(); ++rank) {
        mass += masses[rank];
        if (mass > top_p) return std::max<size_t>(rank, 1);
    }
    return std::max<size_t>(masses.size(), 1);
}

// Gumbel-max: the argmax of p / -log(u) is a draw from p. The weights stay
// unnormalised because a common positive factor cannot move that argmax.
int draw(const std::vector<float> & scores, std::mt19937 & rng) {
    const float top = *std::max_element(scores.begin(), scores.end());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    int best = 0;
    float best_score = -1.0f;
    for (size_t index = 0; index < scores.size(); ++index) {
        const float weight = std::exp(scores[index] - top);
        if (weight == 0.0f) continue;
        const float noise = -std::log(std::max(uniform(rng), MIN_UNIFORM));
        if (weight / noise <= best_score) continue;
        best_score = weight / noise;
        best = static_cast<int>(index);
    }
    return best;
}

}  // namespace

std::vector<float> filter_scores(const std::vector<float> & logits,
                                 const sampling_params & params) {
    const std::vector<int> leaders =
        rank_leaders(logits, leader_count(params, logits.size()));
    const std::vector<float> weights = exponentiated(logits, logits[leaders.front()]);
    const std::vector<float> masses = leader_masses(weights, leaders, total_of(weights));
    const size_t kept = surviving_rank_count(masses, params.top_p);
    const float scale = 1.0f / std::max(params.temperature, MIN_TEMPERATURE);
    std::vector<float> scores(logits.size(), REJECTED);
    for (size_t rank = 0; rank < kept; ++rank) {
        scores[leaders[rank]] = logits[leaders[rank]] * scale;
    }
    return scores;
}

// Argmax survives the filter untouched -- it is always kept, and the
// temperature is a positive scale -- so greedy reads the raw logits.
int sample_token(const std::vector<float> & logits, const sampling_params & params,
                 std::mt19937 & rng) {
    if (params.greedy) {
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) -
                                logits.begin());
    }
    return draw(filter_scores(logits, params), rng);
}

RepetitionAwareSampler::RepetitionAwareSampler(int window, int semantic_begin,
                                               int semantic_end,
                                               const sampling_params & narrow)
    : window_(window), semantic_begin_(semantic_begin), semantic_end_(semantic_end),
      narrow_(narrow) {}

bool RepetitionAwareSampler::is_repeat(int token) const {
    if (token < semantic_begin_ || token > semantic_end_) return false;
    return std::find(recent_.begin(), recent_.end(), token) != recent_.end();
}

void RepetitionAwareSampler::record(int token) {
    if (static_cast<int>(recent_.size()) == window_) recent_.pop_front();
    recent_.push_back(token);
}

void RepetitionAwareSampler::remember(int token) {
    if (window_ <= 0) return;
    if (!opened_) {
        opened_ = true;
        return;
    }
    record(token);
}

int RepetitionAwareSampler::pick(const std::vector<float> & logits,
                                 const sampling_params & params, std::mt19937 & rng) {
    int chosen = sample_token(logits, params, rng);
    if (!params.greedy && window_ > 0 && is_repeat(chosen)) {
        sampling_params retry = params;
        retry.temperature = narrow_.temperature;
        retry.top_p = narrow_.top_p;
        chosen = sample_token(logits, retry, rng);
    }
    remember(chosen);
    return chosen;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
