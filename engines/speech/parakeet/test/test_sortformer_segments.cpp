#include "sortformer_segments.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double k_stride_s  = 0.08;
constexpr float  k_threshold = 0.5f;

std::vector<parakeet::SortformerSegment> threshold(
        const std::vector<float> & probs, int T_enc, int num_spks) {
    std::vector<parakeet::SortformerSegment> segments;
    parakeet::sf_threshold_segments(probs, T_enc, num_spks,
                                    k_stride_s, k_threshold, segments);
    return segments;
}

bool near(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

bool expect_segment(const parakeet::SortformerSegment & seg,
                    int speaker_id, double start_s, double end_s,
                    const char * scenario) {
    if (seg.speaker_id == speaker_id && near(seg.start_s, start_s) &&
        near(seg.end_s, end_s)) {
        return true;
    }
    std::fprintf(stderr, "%s: got speaker %d [%f, %f)\n",
                 scenario, seg.speaker_id, seg.start_s, seg.end_s);
    return false;
}

bool check_single_run() {
    const auto segments = threshold({0.1f, 0.9f, 0.9f, 0.2f}, 4, 1);
    return segments.size() == 1 &&
           expect_segment(segments[0], 0, 1 * k_stride_s, 3 * k_stride_s,
                          "single run");
}

bool check_run_open_at_end_is_closed_at_t_enc() {
    const auto segments = threshold({0.1f, 0.9f, 0.9f}, 3, 1);
    return segments.size() == 1 &&
           expect_segment(segments[0], 0, 1 * k_stride_s, 3 * k_stride_s,
                          "open at end");
}

bool check_threshold_is_strict() {
    const auto segments = threshold({k_threshold, k_threshold}, 2, 1);
    if (!segments.empty()) {
        std::fprintf(stderr, "strict threshold: prob == threshold activated\n");
        return false;
    }
    return true;
}

bool check_single_frame_blip() {
    const auto segments = threshold({0.0f, 0.9f, 0.0f}, 3, 1);
    return segments.size() == 1 &&
           expect_segment(segments[0], 0, 1 * k_stride_s, 2 * k_stride_s,
                          "single-frame blip");
}

bool check_silence_yields_no_segments() {
    const auto segments = threshold(std::vector<float>(8, 0.0f), 4, 2);
    if (!segments.empty()) {
        std::fprintf(stderr, "silence: produced %zu segments\n",
                     segments.size());
        return false;
    }
    return true;
}

bool check_gap_splits_runs() {
    const auto segments = threshold({0.9f, 0.1f, 0.9f, 0.9f}, 4, 1);
    return segments.size() == 2 &&
           expect_segment(segments[0], 0, 0.0, 1 * k_stride_s, "gap first") &&
           expect_segment(segments[1], 0, 2 * k_stride_s, 4 * k_stride_s,
                          "gap second");
}

bool check_speakers_sorted_by_start_then_id() {
    const std::vector<float> probs = {
        0.0f, 0.9f,
        0.9f, 0.9f,
        0.9f, 0.0f,
    };
    const auto segments = threshold(probs, 3, 2);
    return segments.size() == 2 &&
           expect_segment(segments[0], 1, 0.0, 2 * k_stride_s,
                          "sorted first") &&
           expect_segment(segments[1], 0, 1 * k_stride_s, 3 * k_stride_s,
                          "sorted second");
}

bool check_overlap_start_ties_break_on_speaker_id() {
    const std::vector<float> probs = {
        0.9f, 0.9f,
        0.9f, 0.9f,
    };
    const auto segments = threshold(probs, 2, 2);
    return segments.size() == 2 &&
           expect_segment(segments[0], 0, 0.0, 2 * k_stride_s, "tie first") &&
           expect_segment(segments[1], 1, 0.0, 2 * k_stride_s, "tie second");
}

bool check_output_vector_is_replaced() {
    std::vector<parakeet::SortformerSegment> segments(3);
    parakeet::sf_threshold_segments({0.9f}, 1, 1, k_stride_s, k_threshold,
                                    segments);
    return segments.size() == 1 &&
           expect_segment(segments[0], 0, 0.0, 1 * k_stride_s,
                          "output replaced");
}

}

int main() {
    int failures = 0;

    failures += check_single_run() ? 0 : 1;
    failures += check_run_open_at_end_is_closed_at_t_enc() ? 0 : 1;
    failures += check_threshold_is_strict() ? 0 : 1;
    failures += check_single_frame_blip() ? 0 : 1;
    failures += check_silence_yields_no_segments() ? 0 : 1;
    failures += check_gap_splits_runs() ? 0 : 1;
    failures += check_speakers_sorted_by_start_then_id() ? 0 : 1;
    failures += check_overlap_start_ties_break_on_speaker_id() ? 0 : 1;
    failures += check_output_vector_is_replaced() ? 0 : 1;

    if (failures > 0) {
        std::fprintf(stderr, "test_sortformer_segments: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::printf("test_sortformer_segments: all checks passed\n");
    return 0;
}
