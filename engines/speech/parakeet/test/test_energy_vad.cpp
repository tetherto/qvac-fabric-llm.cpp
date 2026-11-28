#include "energy_vad.h"

#include <cstdio>
#include <vector>

namespace {

using parakeet::EnergyVad;

constexpr int   k_sample_rate  = 16000;
constexpr int   k_window_ms    = 30;
constexpr int   k_hangover_ms  = 200;
constexpr float k_threshold_db = -20.0f;

constexpr int k_window_n   = k_sample_rate * k_window_ms / 1000;
constexpr int k_hangover_n = k_sample_rate * k_hangover_ms / 1000;

constexpr float k_loud  = 0.5f;
constexpr float k_quiet = 0.01f;

EnergyVad make_vad() {
    return EnergyVad(k_sample_rate, k_window_ms, k_hangover_ms,
                     k_threshold_db);
}

std::vector<float> constant_block(int n_samples, float amplitude) {
    return std::vector<float>(n_samples, amplitude);
}

bool expect_transition(const EnergyVad::Transition & transition,
                       EnergyVad::State to_state, int64_t at_sample,
                       const char * scenario) {
    if (transition.to_state == to_state && transition.at_sample == at_sample) {
        return true;
    }
    std::fprintf(stderr, "%s: got state %d at sample %lld\n", scenario,
                 (int) transition.to_state,
                 (long long) transition.at_sample);
    return false;
}

bool check_speech_enters_speaking_after_first_window() {
    EnergyVad vad = make_vad();
    const auto block = constant_block(k_window_n * 2, k_loud);
    const auto transition = vad.process(block.data(), (int) block.size(), 0);
    return expect_transition(transition, EnergyVad::State::Speaking,
                             k_window_n - 1, "enter speaking") &&
           vad.current_state() == EnergyVad::State::Speaking;
}

bool check_silence_enters_silent_after_hangover() {
    EnergyVad vad = make_vad();
    const auto block = constant_block(k_window_n + k_hangover_n, k_quiet);
    const auto transition = vad.process(block.data(), (int) block.size(), 0);
    return expect_transition(transition, EnergyVad::State::Silent,
                             k_window_n + k_hangover_n - 2,
                             "enter silent") &&
           vad.current_state() == EnergyVad::State::Silent;
}

bool check_no_transition_before_a_full_window() {
    EnergyVad vad = make_vad();
    const auto block = constant_block(k_window_n - 1, k_loud);
    const auto transition = vad.process(block.data(), (int) block.size(), 0);
    return transition.to_state == EnergyVad::State::Unknown &&
           vad.current_state() == EnergyVad::State::Unknown;
}

bool check_short_dip_does_not_leave_speaking() {
    EnergyVad vad = make_vad();
    const auto speech = constant_block(k_window_n * 2, k_loud);
    vad.process(speech.data(), (int) speech.size(), 0);

    const auto dip = constant_block(k_hangover_n / 2, k_quiet);
    const auto transition =
        vad.process(dip.data(), (int) dip.size(), (int64_t) speech.size());
    return transition.to_state == EnergyVad::State::Unknown &&
           vad.current_state() == EnergyVad::State::Speaking;
}

bool check_within_block_transitions_collapse_to_last() {
    EnergyVad vad = make_vad();
    std::vector<float> block = constant_block(k_window_n * 2, k_loud);
    const auto silence = constant_block(k_window_n + k_hangover_n, k_quiet);
    block.insert(block.end(), silence.begin(), silence.end());

    const auto transition = vad.process(block.data(), (int) block.size(), 0);
    return transition.to_state == EnergyVad::State::Silent &&
           vad.current_state() == EnergyVad::State::Silent;
}

bool check_at_sample_is_absolute_across_calls() {
    EnergyVad vad = make_vad();
    const int64_t start_sample = 100000;
    const auto quiet = constant_block(k_window_n, k_quiet);
    vad.process(quiet.data(), (int) quiet.size(), start_sample);

    const auto speech = constant_block(k_window_n, k_loud);
    const auto transition =
        vad.process(speech.data(), (int) speech.size(),
                    start_sample + quiet.size());
    return transition.to_state == EnergyVad::State::Speaking &&
           transition.at_sample > start_sample &&
           transition.at_sample <
               start_sample + (int64_t) (quiet.size() + speech.size());
}

bool check_reset_returns_to_unknown() {
    EnergyVad vad = make_vad();
    const auto speech = constant_block(k_window_n * 2, k_loud);
    vad.process(speech.data(), (int) speech.size(), 0);

    vad.reset();
    if (vad.current_state() != EnergyVad::State::Unknown) return false;

    const auto transition = vad.process(speech.data(), (int) speech.size(), 0);
    return expect_transition(transition, EnergyVad::State::Speaking,
                             k_window_n - 1, "re-detect after reset");
}

bool check_empty_block_reports_no_transition() {
    EnergyVad vad = make_vad();
    const auto transition = vad.process(nullptr, 0, 0);
    return transition.to_state == EnergyVad::State::Unknown;
}

bool check_threshold_boundary() {
    constexpr float k_threshold_rms = 0.1f;
    constexpr float k_just_below    = k_threshold_rms * 0.9f;
    constexpr float k_just_above    = k_threshold_rms * 1.1f;

    EnergyVad below = make_vad();
    const auto quiet = constant_block(k_window_n + k_hangover_n, k_just_below);
    below.process(quiet.data(), (int) quiet.size(), 0);
    if (below.current_state() != EnergyVad::State::Silent) {
        std::fprintf(stderr, "threshold boundary: just-below reached %d\n",
                     (int) below.current_state());
        return false;
    }

    EnergyVad above = make_vad();
    const auto speech = constant_block(k_window_n, k_just_above);
    above.process(speech.data(), (int) speech.size(), 0);
    return above.current_state() == EnergyVad::State::Speaking;
}

}

int main() {
    int failures = 0;

    failures += check_speech_enters_speaking_after_first_window() ? 0 : 1;
    failures += check_silence_enters_silent_after_hangover() ? 0 : 1;
    failures += check_no_transition_before_a_full_window() ? 0 : 1;
    failures += check_short_dip_does_not_leave_speaking() ? 0 : 1;
    failures += check_within_block_transitions_collapse_to_last() ? 0 : 1;
    failures += check_at_sample_is_absolute_across_calls() ? 0 : 1;
    failures += check_reset_returns_to_unknown() ? 0 : 1;
    failures += check_empty_block_reports_no_transition() ? 0 : 1;
    failures += check_threshold_boundary() ? 0 : 1;

    if (failures > 0) {
        std::fprintf(stderr, "test_energy_vad: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_energy_vad: all checks passed\n");
    return 0;
}
