#pragma once

// Model-independent thresholding of Sortformer speaker probabilities into
// time-sorted segments. Header-only so both the diarization paths and the
// unit test consume the same definition.

#include <algorithm>
#include <vector>

namespace parakeet {

struct SortformerSegment {
    int    speaker_id = 0;
    double start_s    = 0.0;
    double end_s      = 0.0;
};

inline void sf_threshold_segments(const std::vector<float> & speaker_probs,
                                  int T_enc, int num_spks,
                                  double frame_stride_s, float threshold,
                                  std::vector<SortformerSegment> & segments) {
    segments.clear();
    for (int s = 0; s < num_spks; ++s) {
        bool active = false;
        int  start_frame = 0;
        for (int t = 0; t < T_enc; ++t) {
            const bool a = speaker_probs[(size_t)t * num_spks + s] > threshold;
            if (a && !active)  { start_frame = t; active = true; }
            if (!a && active) {
                SortformerSegment seg;
                seg.speaker_id = s;
                seg.start_s = start_frame * frame_stride_s;
                seg.end_s   = t           * frame_stride_s;
                segments.push_back(seg);
                active = false;
            }
        }
        if (active) {
            SortformerSegment seg;
            seg.speaker_id = s;
            seg.start_s = start_frame * frame_stride_s;
            seg.end_s   = T_enc       * frame_stride_s;
            segments.push_back(seg);
        }
    }
    std::sort(segments.begin(), segments.end(),
              [](const SortformerSegment & a, const SortformerSegment & b) {
                  if (a.start_s != b.start_s) return a.start_s < b.start_s;
                  return a.speaker_id < b.speaker_id;
              });
}

}
