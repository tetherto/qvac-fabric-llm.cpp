// The top-k / top-p / temperature candidate filter (filter_scores), which the
// fixture sampler test only exercises through reference dumps. Synthetic score
// vectors pin what survives and at what value: top-k keeps exactly the k best,
// top-p drops the candidate whose cumulative mass crosses the threshold (the
// leader always stays), and only the survivors are divided by the temperature.
// The nucleus is measured on the RAW logits -- the reference's load-bearing
// order -- so a low temperature must not change which candidates survive.

#include "audio8/sampling.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

constexpr int TRIALS = 200;
constexpr float NO_TEMPERATURE = 1.0f;
constexpr float KEEP_ALL_P = 1.0f;
constexpr int UNLIMITED_K = 0;

int failures = 0;

void check(bool condition, const char * what, int line) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", what, __FILE__, line);
}

#define CHECK(cond, msg) check((cond), (msg), __LINE__)

sampling_params with(int top_k, float top_p, float temperature) {
    sampling_params params;
    params.greedy = false;
    params.top_k = top_k;
    params.top_p = top_p;
    params.temperature = temperature;
    return params;
}

bool kept(const std::vector<float> & scores, size_t index) {
    return std::isfinite(scores[index]);
}

size_t kept_count(const std::vector<float> & scores) {
    size_t n = 0;
    for (float s : scores) {
        if (std::isfinite(s)) ++n;
    }
    return n;
}

// Softmax masses 0.5 / 0.25 / 0.125 / 0.125, in that rank order.
std::vector<float> halving_logits() {
    return { std::log(8.0f), std::log(4.0f), std::log(2.0f), std::log(2.0f) };
}

void test_top_k_keeps_exactly_the_k_best() {
    const std::vector<float> logits = { 1.0f, 5.0f, 3.0f, 4.0f, 2.0f };
    const auto scores = filter_scores(logits, with(3, KEEP_ALL_P, NO_TEMPERATURE));
    CHECK(kept_count(scores) == 3, "top-k 3 keeps exactly 3 candidates");
    CHECK(kept(scores, 1) && kept(scores, 3) && kept(scores, 2),
          "top-k keeps the 3 highest logits");
    CHECK(!kept(scores, 0) && !kept(scores, 4), "top-k rejects the tail");
    CHECK(scores[1] == 5.0f && scores[3] == 4.0f && scores[2] == 3.0f,
          "temperature 1 leaves the surviving logits untouched");
}

void test_top_k_larger_than_the_vocab_keeps_everything() {
    const std::vector<float> logits = { 1.0f, 5.0f, 3.0f, 4.0f, 2.0f };
    const auto scores = filter_scores(logits, with(100, KEEP_ALL_P, NO_TEMPERATURE));
    CHECK(kept_count(scores) == logits.size(), "top-k past the vocab keeps everything");
}

void test_top_k_zero_means_unlimited() {
    const std::vector<float> logits = { 1.0f, 5.0f, 3.0f, 4.0f, 2.0f };
    const auto scores = filter_scores(logits, with(UNLIMITED_K, KEEP_ALL_P, NO_TEMPERATURE));
    CHECK(kept_count(scores) == logits.size(), "top-k 0 disables the k cut");
}

void test_top_p_one_keeps_a_uniform_distribution_whole() {
    // Four equal logits: each mass is exactly 0.25, so the running mass never
    // exceeds 1 and nothing is dropped.
    const std::vector<float> logits(4, 0.0f);
    const auto scores = filter_scores(logits, with(UNLIMITED_K, KEEP_ALL_P, NO_TEMPERATURE));
    CHECK(kept_count(scores) == logits.size(), "top-p 1 keeps every candidate");
}

void test_top_p_drops_the_crossing_candidate() {
    const std::vector<float> logits = halving_logits();
    // 0.5 + 0.25 crosses 0.7, so the rank-1 candidate itself is dropped.
    CHECK(kept_count(filter_scores(logits, with(UNLIMITED_K, 0.7f, NO_TEMPERATURE))) == 1,
          "mass crossing the threshold drops the crossing candidate");
    CHECK(kept_count(filter_scores(logits, with(UNLIMITED_K, 0.76f, NO_TEMPERATURE))) == 2,
          "top-p 0.76 keeps the 0.75 head");
    CHECK(kept_count(filter_scores(logits, with(UNLIMITED_K, 0.9f, NO_TEMPERATURE))) == 3,
          "top-p 0.9 keeps the 0.875 head");
    const auto scores = filter_scores(logits, with(UNLIMITED_K, 0.76f, NO_TEMPERATURE));
    CHECK(kept(scores, 0) && kept(scores, 1), "the kept head is the top-ranked candidates");
}

void test_the_leader_always_survives() {
    const std::vector<float> logits = halving_logits();
    const auto scores = filter_scores(logits, with(UNLIMITED_K, 0.05f, NO_TEMPERATURE));
    CHECK(kept_count(scores) == 1 && kept(scores, 0),
          "a top-p below the leader's own mass still keeps the leader");
}

void test_top_k_caps_before_top_p_finishes() {
    const std::vector<float> logits = halving_logits();
    const auto scores = filter_scores(logits, with(2, KEEP_ALL_P, NO_TEMPERATURE));
    CHECK(kept_count(scores) == 2 && kept(scores, 0) && kept(scores, 1),
          "top-k 2 caps the nucleus even at top-p 1");
}

void test_temperature_scales_only_the_survivors() {
    const std::vector<float> logits = { 1.0f, 5.0f, 3.0f, 4.0f, 2.0f };
    const float temperature = 0.5f;
    const auto scores = filter_scores(logits, with(3, KEEP_ALL_P, temperature));
    const float scale = 1.0f / temperature;
    CHECK(scores[1] == logits[1] * scale && scores[3] == logits[3] * scale &&
          scores[2] == logits[2] * scale,
          "survivors are divided by the temperature");
    CHECK(!kept(scores, 0) && !kept(scores, 4), "rejected candidates stay rejected");
}

void test_temperature_preserves_the_ranking() {
    const std::vector<float> logits = { -2.0f, 3.0f, 0.5f, -0.5f };
    for (float temperature : { 0.25f, 1.0f, 4.0f }) {
        const auto scores = filter_scores(logits, with(UNLIMITED_K, KEEP_ALL_P, temperature));
        CHECK(scores[1] > scores[2] && scores[2] > scores[3] && scores[3] > scores[0],
              "a positive temperature scale keeps the logit order");
    }
}

void test_temperature_does_not_move_the_nucleus() {
    // The nucleus is measured on the raw logits, so a sharpening temperature
    // must not shrink it (applying it first would leave only the leader).
    const std::vector<float> logits = halving_logits();
    CHECK(kept_count(filter_scores(logits, with(UNLIMITED_K, 0.76f, 0.1f))) == 2,
          "temperature 0.1 leaves the top-p survivors unchanged");
}

void test_zero_temperature_clamps_to_the_minimum() {
    // Equal logits keep the softmax masses exact (0.25 each), so the running
    // mass cannot drift past top-p 1 and drop the last candidate.
    const std::vector<float> logits(4, 2.0f);
    const auto at_zero = filter_scores(logits, with(UNLIMITED_K, KEEP_ALL_P, 0.0f));
    const auto at_min = filter_scores(logits, with(UNLIMITED_K, KEEP_ALL_P, 1e-5f));
    CHECK(at_zero == at_min, "temperature 0 behaves as the clamped minimum");
    CHECK(kept_count(at_zero) == logits.size(), "the clamp rejects nothing");
}

void test_single_candidate_survives_any_filter() {
    const std::vector<float> logits = { 2.5f };
    const auto scores = filter_scores(logits, with(UNLIMITED_K, 0.1f, 0.5f));
    CHECK(kept_count(scores) == 1, "a single candidate is always kept");
    CHECK(scores[0] == logits[0] * (1.0f / 0.5f), "and still temperature-scaled");
}

// Ranking stops at top_k, but the masses it is measured against do not: the
// tail the ranking never visits still normalises the nucleus. A leaders-only
// normaliser would inflate every mass and close the nucleus early.
std::vector<float> three_leaders_over_a_flat_tail() {
    std::vector<float> logits = { std::log(8.0f), std::log(4.0f), std::log(2.0f) };
    logits.resize(103, 0.0f);
    return logits;
}

void test_the_tail_normalises_the_nucleus() {
    const std::vector<float> logits = three_leaders_over_a_flat_tail();
    // Masses over the whole 114 of exp-weight: 0.070, 0.035, 0.018 -- the three
    // leaders together hold 0.123, far short of top-p, so top-k is what stops it.
    CHECK(kept_count(filter_scores(logits, with(3, 0.9f, NO_TEMPERATURE))) == 3,
          "the tail's mass keeps the nucleus open across every ranked leader");
    CHECK(kept_count(filter_scores(logits, with(2, 0.9f, NO_TEMPERATURE))) == 2,
          "top-k still caps how many leaders are ranked");
}

void test_a_filter_with_one_survivor_pins_the_draw() {
    const std::vector<float> logits = { 1.0f, 5.0f, 3.0f, 4.0f, 2.0f };
    std::mt19937 rng(7);
    bool always_leader = true;
    for (int trial = 0; trial < TRIALS; ++trial) {
        if (sample_token(logits, with(1, KEEP_ALL_P, NO_TEMPERATURE), rng) != 1) {
            always_leader = false;
        }
    }
    CHECK(always_leader, "top-k 1 makes the draw deterministic on the argmax");
}

}  // namespace

int main() {
    test_top_k_keeps_exactly_the_k_best();
    test_top_k_larger_than_the_vocab_keeps_everything();
    test_top_k_zero_means_unlimited();
    test_top_p_one_keeps_a_uniform_distribution_whole();
    test_top_p_drops_the_crossing_candidate();
    test_the_leader_always_survives();
    test_top_k_caps_before_top_p_finishes();
    test_temperature_scales_only_the_survivors();
    test_temperature_preserves_the_ranking();
    test_temperature_does_not_move_the_nucleus();
    test_zero_temperature_clamps_to_the_minimum();
    test_single_candidate_survives_any_filter();
    test_the_tail_normalises_the_nucleus();
    test_a_filter_with_one_survivor_pins_the_draw();

    if (failures == 0) {
        std::fprintf(stderr, "audio8 sampling filter: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "audio8 sampling filter: %d failure(s)\n", failures);
    return 1;
}
