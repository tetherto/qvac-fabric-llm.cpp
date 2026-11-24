#pragma once
// lyrics-alignment.h: dependency-free lyric alignment and scoring math.
//
// Ported from ACE-Step 1.5's _dtw.py, dit_alignment.py, and dit_score.py.
// The API begins after attention-head selection: each input head is a
// row-major [tokens, frames] matrix, independent of any inference backend.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tts_cpp::acestep::lyrics {

struct Matrix {
    size_t             rows = 0;
    size_t             cols = 0;
    std::vector<float> values;

    Matrix() = default;
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, std::vector<float> values);

    float &       operator()(size_t row, size_t col);
    const float & operator()(size_t row, size_t col) const;
};

// Convert a ggml-style 2D buffer, where rows are the contiguous ne[0]
// dimension, into the row-major [rows, cols] layout used by alignment math.
Matrix matrix_from_column_major(const float * values, size_t rows, size_t cols);

struct DtwPath {
    std::vector<int32_t> token_indices;
    std::vector<int32_t> frame_indices;
};

struct AlignmentPreprocessResult {
    Matrix calc_matrix;
    Matrix energy_matrix;
    Matrix visual_matrix;
};

struct ScoringPreprocessResult {
    Matrix calc_matrix;
    Matrix energy_matrix;
    Matrix average_matrix;
};

struct TokenTimestamp {
    int         token_id   = 0;
    std::string text;
    double      start      = 0.0;
    double      end        = 0.0;
    double      probability = 0.0;
};

struct SentenceTimestamp {
    std::string                 text;
    double                      start      = 0.0;
    double                      end        = 0.0;
    std::vector<TokenTimestamp> tokens;
    double                      confidence = 0.0;
};

struct AlignmentMetrics {
    double coverage        = 0.0;
    double monotonicity    = 0.0;
    double path_confidence = 0.0;
};

using TokenDecoder = std::function<std::string(const std::vector<int> &)>;

// Dynamic time warping with the reference's exact comparison chain
// (acestep/core/scoring/_dtw.py): a direction is taken only when it is
// STRICTLY smaller than both others, so horizontal absorbs every tie —
// including diagonal == vertical < horizontal, where the strictly larger
// horizontal cost is accumulated. Kept bug-for-bug for timestamp parity.
DtwPath dtw(const Matrix & costs);

// Reflect-padded median filtering along the frame axis.
Matrix median_filter(const Matrix & input, int filter_width);

// Bidirectional-consensus preprocessing used by dit_alignment.py.
AlignmentPreprocessResult preprocess_alignment(
    const std::vector<Matrix> & selected_heads,
    float                       violence_level = 2.0f,
    int                         median_filter_width = 1);

// Head averaging, median filtering, min-max normalization, and square
// sharpening used by dit_score.py.
ScoringPreprocessResult preprocess_scoring(
    const std::vector<Matrix> & selected_heads,
    int                         median_filter_width = 1);

// Mark decoded tokens inside bracketed structural tags as 0 and lyrics as 1.
std::vector<int32_t> token_type_mask(const std::vector<std::string> & decoded_tokens);

// Reproduce the reference's prefix-decoding behavior for byte-level tokenizers.
std::vector<std::string> decode_tokens_incrementally(
    const std::vector<int> & token_ids,
    const TokenDecoder &     decoder);

std::vector<TokenTimestamp> token_timestamps(
    const Matrix &                   calc_matrix,
    const std::vector<int> &         token_ids,
    const std::vector<std::string> & decoded_tokens,
    double                           total_duration_seconds);

std::vector<SentenceTimestamp> sentence_timestamps(
    const std::vector<TokenTimestamp> & token_alignment,
    const TokenDecoder &                 decoder);

std::string format_lrc(
    const std::vector<SentenceTimestamp> & sentences,
    bool                                   include_end_time = false);

AlignmentMetrics compute_alignment_metrics(
    const Matrix &              energy_matrix,
    const DtwPath &             path,
    const std::vector<int32_t> & type_mask,
    double                      time_weight = 0.01,
    double                      overlap_frames = 9.0,
    double                      instrumental_weight = 1.0);

// Returns the Python-compatible four-decimal score:
// coverage^2 * monotonicity^2 * path_confidence, clipped to [0, 1].
double calculate_lyrics_score(const AlignmentMetrics & metrics);

}  // namespace tts_cpp::acestep::lyrics
