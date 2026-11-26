// Sinusoidal timestep embedding for the CosyVoice3 flow DiT
// (sinus_time_emb in src/cosyvoice_pipeline.cpp).
//
// No model fixture: the function is pure, and it feeds every CFM step, so a
// silent change to the frequency schedule or the sin/cos layout would shift
// generated audio while every other CosyVoice test still passed.
//
// dim == 2 is not exercised: half - 1 == 0 makes the log spacing divide by
// zero. Callers only ever pass 256.

#include "cosyvoice_pipeline.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int   k_dim = 256;
constexpr int   k_half = k_dim / 2;
constexpr float k_tol = 1e-6f;
constexpr double k_arg_scale = 1000.0;

int g_failures = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        ++g_failures;                                          \
    }                                                          \
} while (0)

bool close_to(float got, double want) {
    return std::fabs((double) got - want) <= k_tol;
}

double frequency(int j) {
    const double logk = std::log(10000.0) / (k_half - 1);
    return std::exp(j * -logk);
}

void test_shape_is_dim_by_batch() {
    const auto emb = sinus_time_emb({0.1f, 0.2f, 0.3f}, k_dim);
    CHECK(emb.size() == (size_t) k_dim * 3, "expected %d values, got %zu",
          k_dim * 3, emb.size());
}

void test_empty_batch_is_empty() {
    CHECK(sinus_time_emb({}, k_dim).empty(),
          "an empty timestep list must produce no embedding");
}

// t == 0 collapses every argument to zero, which is the cheapest way to catch
// a sin/cos swap: the first half must be all zeros, the second all ones.
void test_zero_timestep_splits_sin_and_cos() {
    const auto emb = sinus_time_emb({0.0f}, k_dim);
    for (int j = 0; j < k_half; ++j) {
        CHECK(close_to(emb[j], 0.0), "sin half at j=%d should be 0, got %g",
              j, (double) emb[j]);
        CHECK(close_to(emb[k_half + j], 1.0),
              "cos half at j=%d should be 1, got %g", j,
              (double) emb[k_half + j]);
    }
}

void test_frequency_schedule() {
    const float t = 0.37f;
    const auto emb = sinus_time_emb({t}, k_dim);
    for (int j : {0, 1, 7, 63, k_half - 2, k_half - 1}) {
        const double arg = k_arg_scale * (double) t * frequency(j);
        CHECK(close_to(emb[j], std::sin(arg)),
              "sin at j=%d: got %g want %g", j, (double) emb[j], std::sin(arg));
        CHECK(close_to(emb[k_half + j], std::cos(arg)),
              "cos at j=%d: got %g want %g", j, (double) emb[k_half + j],
              std::cos(arg));
    }
}

// j == 0 has frequency 1, so its argument is the raw scaled timestep. This
// pins the 1000x scale that the schedule is built around.
void test_first_channel_carries_the_scaled_timestep() {
    const float t = 0.25f;
    const auto emb = sinus_time_emb({t}, k_dim);
    CHECK(close_to(emb[0], std::sin(k_arg_scale * (double) t)),
          "j=0 sin should be sin(1000t), got %g", (double) emb[0]);
    CHECK(close_to(emb[k_half], std::cos(k_arg_scale * (double) t)),
          "j=0 cos should be cos(1000t), got %g", (double) emb[k_half]);
}

void test_last_channel_frequency_is_the_decade_floor() {
    CHECK(std::fabs(frequency(k_half - 1) - 1e-4) < 1e-12,
          "the schedule should span four decades down to 1e-4");
}

void test_rows_are_independent() {
    const std::vector<float> batch = {0.0f, 0.4f, 0.9f};
    const auto together = sinus_time_emb(batch, k_dim);
    for (size_t b = 0; b < batch.size(); ++b) {
        const auto alone = sinus_time_emb({batch[b]}, k_dim);
        for (int j = 0; j < k_dim; ++j) {
            CHECK(together[b * k_dim + j] == alone[j],
                  "row %zu channel %d differs when batched", b, j);
        }
    }
}

void test_repeated_calls_are_deterministic() {
    const std::vector<float> batch = {0.05f, 0.5f, 0.95f};
    const auto first = sinus_time_emb(batch, k_dim);
    const auto second = sinus_time_emb(batch, k_dim);
    CHECK(first == second, "sinus_time_emb must be deterministic");
}

void test_values_stay_bounded() {
    const auto emb = sinus_time_emb({0.0f, 0.13f, 0.51f, 1.0f}, k_dim);
    for (size_t i = 0; i < emb.size(); ++i) {
        CHECK(std::isfinite(emb[i]) && emb[i] >= -1.0f && emb[i] <= 1.0f,
              "value %zu out of the sin/cos range: %g", i, (double) emb[i]);
    }
}

}

int main() {
    test_shape_is_dim_by_batch();
    test_empty_batch_is_empty();
    test_zero_timestep_splits_sin_and_cos();
    test_frequency_schedule();
    test_first_channel_carries_the_scaled_timestep();
    test_last_channel_frequency_is_the_decade_floor();
    test_rows_are_independent();
    test_repeated_calls_are_deterministic();
    test_values_stay_bounded();

    if (g_failures) {
        fprintf(stderr, "test-cosyvoice-time-emb: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-cosyvoice-time-emb: all checks passed\n");
    return 0;
}
