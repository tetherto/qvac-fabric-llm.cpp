#include "lyrics_alignment.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tts_cpp::acestep::lyrics {
namespace {

void validate_matrix(const Matrix & matrix, const char * name) {
    if (matrix.rows == 0 || matrix.cols == 0) {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    if (matrix.values.size() != matrix.rows * matrix.cols) {
        throw std::invalid_argument(std::string(name) + " has inconsistent dimensions");
    }
}

void validate_heads(const std::vector<Matrix> & heads) {
    if (heads.empty()) {
        throw std::invalid_argument("selected_heads must not be empty");
    }
    validate_matrix(heads.front(), "attention head");
    for (const Matrix & head : heads) {
        validate_matrix(head, "attention head");
        if (head.rows != heads.front().rows || head.cols != heads.front().cols) {
            throw std::invalid_argument("all attention heads must have the same dimensions");
        }
    }
}

float quantile_half(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0U) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) * 0.5f;
}

double round_half_even(double value, int decimals) {
    double scale  = std::pow(10.0, decimals);
    double scaled = value * scale;
    double lower  = std::floor(scaled);
    double delta  = scaled - lower;
    double result = lower;
    if (delta > 0.5 || (delta == 0.5 && std::fmod(std::fabs(lower), 2.0) == 1.0)) {
        result = lower + 1.0;
    }
    return result / scale;
}

bool is_valid_utf8(const std::string & value) {
    size_t i = 0;
    while (i < value.size()) {
        unsigned char lead = static_cast<unsigned char>(value[i]);
        size_t        count;
        uint32_t      codepoint;
        if (lead <= 0x7fU) {
            count     = 1;
            codepoint = lead;
        } else if ((lead & 0xe0U) == 0xc0U) {
            count     = 2;
            codepoint = lead & 0x1fU;
        } else if ((lead & 0xf0U) == 0xe0U) {
            count     = 3;
            codepoint = lead & 0x0fU;
        } else if ((lead & 0xf8U) == 0xf0U) {
            count     = 4;
            codepoint = lead & 0x07U;
        } else {
            return false;
        }
        if (i + count > value.size()) {
            return false;
        }
        for (size_t j = 1; j < count; ++j) {
            unsigned char continuation = static_cast<unsigned char>(value[i + j]);
            if ((continuation & 0xc0U) != 0x80U) {
                return false;
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3fU);
        }
        if ((count == 2 && codepoint < 0x80U) ||
            (count == 3 && codepoint < 0x800U) ||
            (count == 4 && codepoint < 0x10000U) ||
            codepoint > 0x10ffffU ||
            (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
            return false;
        }
        i += count;
    }
    return true;
}

bool is_ascii_space(unsigned char value) {
    return value == ' ' || value == '\t' || value == '\n' ||
           value == '\r' || value == '\f' || value == '\v';
}

std::string trim_ascii(const std::string & value) {
    size_t first = 0;
    while (first < value.size() && is_ascii_space(static_cast<unsigned char>(value[first]))) {
        ++first;
    }
    size_t last = value.size();
    while (last > first && is_ascii_space(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }
    return value.substr(first, last - first);
}

Matrix average_heads(const std::vector<Matrix> & heads) {
    Matrix average(heads.front().rows, heads.front().cols);
    const float inverse_count = 1.0f / static_cast<float>(heads.size());
    for (const Matrix & head : heads) {
        for (size_t i = 0; i < head.values.size(); ++i) {
            average.values[i] += head.values[i] * inverse_count;
        }
    }
    return average;
}

std::string lrc_time(double seconds) {
    long long minutes = static_cast<long long>(std::floor(seconds / 60.0));
    double    remainder = seconds - static_cast<double>(minutes) * 60.0;
    char      buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%02lld:%05.2f", minutes, remainder);
    return buffer;
}

}  // namespace

Matrix::Matrix(size_t rows_value, size_t cols_value) :
    rows(rows_value),
    cols(cols_value),
    values(rows_value * cols_value, 0.0f) {
}

Matrix::Matrix(size_t rows_value, size_t cols_value, std::vector<float> data) :
    rows(rows_value),
    cols(cols_value),
    values(std::move(data)) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument("matrix data has inconsistent dimensions");
    }
}

float & Matrix::operator()(size_t row, size_t col) {
    return values.at(row * cols + col);
}

const float & Matrix::operator()(size_t row, size_t col) const {
    return values.at(row * cols + col);
}

Matrix matrix_from_column_major(const float * values, size_t rows, size_t cols) {
    if (!values && rows * cols != 0) {
        throw std::invalid_argument("column-major matrix data is null");
    }
    Matrix matrix(rows, cols);
    for (size_t col = 0; col < cols; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            matrix(row, col) = values[row + col * rows];
        }
    }
    return matrix;
}

DtwPath dtw(const Matrix & costs) {
    validate_matrix(costs, "cost matrix");
    const size_t rows = costs.rows;
    const size_t cols = costs.cols;
    const float  infinity = std::numeric_limits<float>::infinity();

    std::vector<float>  accumulated((rows + 1) * (cols + 1), infinity);
    std::vector<int8_t> trace((rows + 1) * (cols + 1), -1);
    accumulated[0] = 0.0f;

    auto index = [rows](size_t row, size_t col) {
        return row + col * (rows + 1);
    };
    for (size_t col = 1; col <= cols; ++col) {
        for (size_t row = 1; row <= rows; ++row) {
            float  diagonal = accumulated[index(row - 1, col - 1)];
            float  vertical = accumulated[index(row - 1, col)];
            float  horizontal = accumulated[index(row, col - 1)];
            float  prior;
            int8_t direction;
            if (diagonal < vertical && diagonal < horizontal) {
                prior     = diagonal;
                direction = 0;
            } else if (vertical < diagonal && vertical < horizontal) {
                prior     = vertical;
                direction = 1;
            } else {
                prior     = horizontal;
                direction = 2;
            }
            accumulated[index(row, col)] = costs(row - 1, col - 1) + prior;
            trace[index(row, col)]       = direction;
        }
    }

    DtwPath path;
    path.token_indices.reserve(rows + cols);
    path.frame_indices.reserve(rows + cols);
    size_t row = rows;
    size_t col = cols;
    while (row > 0 || col > 0) {
        path.token_indices.push_back(static_cast<int32_t>(row) - 1);
        path.frame_indices.push_back(static_cast<int32_t>(col) - 1);
        int8_t direction;
        if (row == 0) {
            direction = 2;
        } else if (col == 0) {
            direction = 1;
        } else {
            direction = trace[index(row, col)];
        }
        if (direction == 0) {
            --row;
            --col;
        } else if (direction == 1) {
            --row;
        } else if (direction == 2) {
            --col;
        } else {
            break;
        }
    }
    std::reverse(path.token_indices.begin(), path.token_indices.end());
    std::reverse(path.frame_indices.begin(), path.frame_indices.end());
    return path;
}

Matrix median_filter(const Matrix & input, int filter_width) {
    validate_matrix(input, "median filter input");
    if (filter_width <= 0) {
        throw std::invalid_argument("median filter width must be positive");
    }
    const size_t padding = static_cast<size_t>(filter_width / 2);
    if (input.cols <= padding) {
        return input;
    }
    const size_t output_cols = input.cols + 2 * padding -
                               static_cast<size_t>(filter_width) + 1;
    Matrix output(input.rows, output_cols);
    std::vector<float> window(static_cast<size_t>(filter_width));
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t col = 0; col < output_cols; ++col) {
            for (int offset = 0; offset < filter_width; ++offset) {
                int64_t source = static_cast<int64_t>(col) + offset -
                                 static_cast<int64_t>(padding);
                if (source < 0) {
                    source = -source;
                } else if (source >= static_cast<int64_t>(input.cols)) {
                    source = 2 * static_cast<int64_t>(input.cols) - 2 - source;
                }
                window[static_cast<size_t>(offset)] =
                    input(row, static_cast<size_t>(source));
            }
            std::sort(window.begin(), window.end());
            output(row, col) = window[padding];
        }
    }
    return output;
}

AlignmentPreprocessResult preprocess_alignment(
    const std::vector<Matrix> & selected_heads,
    float                       violence_level,
    int                         median_filter_width) {
    validate_heads(selected_heads);
    const size_t head_count = selected_heads.size();
    const size_t rows       = selected_heads.front().rows;
    const size_t cols       = selected_heads.front().cols;
    const size_t plane      = rows * cols;

    AlignmentPreprocessResult result;
    result.visual_matrix = average_heads(selected_heads);
    std::vector<float> processed(head_count * plane);

    for (size_t head_index = 0; head_index < head_count; ++head_index) {
        const Matrix & head = selected_heads[head_index];
        for (size_t row = 0; row < rows; ++row) {
            float max_value = -std::numeric_limits<float>::infinity();
            for (size_t col = 0; col < cols; ++col) {
                max_value = std::max(max_value, head(row, col));
            }
            float sum = 0.0f;
            for (size_t col = 0; col < cols; ++col) {
                float value = std::exp(head(row, col) - max_value);
                processed[head_index * plane + row * cols + col] = value;
                sum += value;
            }
            for (size_t col = 0; col < cols; ++col) {
                processed[head_index * plane + row * cols + col] /= sum;
            }
        }
        for (size_t col = 0; col < cols; ++col) {
            float max_value = -std::numeric_limits<float>::infinity();
            for (size_t row = 0; row < rows; ++row) {
                max_value = std::max(max_value, head(row, col));
            }
            float sum = 0.0f;
            std::vector<float> column(rows);
            for (size_t row = 0; row < rows; ++row) {
                column[row] = std::exp(head(row, col) - max_value);
                sum += column[row];
            }
            for (size_t row = 0; row < rows; ++row) {
                processed[head_index * plane + row * cols + col] *= column[row] / sum;
            }
        }
    }

    std::vector<float> values(std::max(rows, cols));
    for (size_t head_index = 0; head_index < head_count; ++head_index) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                values[col] = processed[head_index * plane + row * cols + col];
            }
            float median = quantile_half(std::vector<float>(values.begin(), values.begin() + cols));
            for (size_t col = 0; col < cols; ++col) {
                float & value = processed[head_index * plane + row * cols + col];
                value = std::max(0.0f, value - violence_level * median);
            }
        }
        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                values[row] = processed[head_index * plane + row * cols + col];
            }
            float median = quantile_half(std::vector<float>(values.begin(), values.begin() + rows));
            for (size_t row = 0; row < rows; ++row) {
                float & value = processed[head_index * plane + row * cols + col];
                value = std::max(0.0f, value - violence_level * median);
                value *= value;
            }
        }
    }

    result.energy_matrix = Matrix(rows, cols);
    const float inverse_heads = 1.0f / static_cast<float>(head_count);
    for (size_t i = 0; i < plane; ++i) {
        for (size_t head_index = 0; head_index < head_count; ++head_index) {
            result.energy_matrix.values[i] += processed[head_index * plane + i] * inverse_heads;
        }
    }

    double sum = 0.0;
    for (float value : processed) {
        sum += value;
    }
    float mean = static_cast<float>(sum / static_cast<double>(processed.size()));
    double squared_difference = 0.0;
    for (float value : processed) {
        double difference = static_cast<double>(value) - mean;
        squared_difference += difference * difference;
    }
    float standard_deviation = static_cast<float>(
        std::sqrt(squared_difference / static_cast<double>(processed.size())));

    std::vector<Matrix> filtered_heads;
    filtered_heads.reserve(head_count);
    for (size_t head_index = 0; head_index < head_count; ++head_index) {
        Matrix normalized(rows, cols);
        for (size_t i = 0; i < plane; ++i) {
            normalized.values[i] =
                (processed[head_index * plane + i] - mean) / (standard_deviation + 1e-9f);
        }
        filtered_heads.push_back(median_filter(normalized, median_filter_width));
    }
    result.calc_matrix = average_heads(filtered_heads);
    return result;
}

ScoringPreprocessResult preprocess_scoring(
    const std::vector<Matrix> & selected_heads,
    int                         median_filter_width) {
    validate_heads(selected_heads);
    ScoringPreprocessResult result;
    result.average_matrix = average_heads(selected_heads);
    result.energy_matrix  = median_filter(result.average_matrix, median_filter_width);

    auto minimum_maximum = std::minmax_element(
        result.energy_matrix.values.begin(), result.energy_matrix.values.end());
    float minimum = *minimum_maximum.first;
    float maximum = *minimum_maximum.second;
    float range   = maximum - minimum;
    if (range > 1e-9f) {
        for (float & value : result.energy_matrix.values) {
            value = (value - minimum) / range;
        }
    } else {
        std::fill(result.energy_matrix.values.begin(), result.energy_matrix.values.end(), 0.0f);
    }
    result.calc_matrix = result.energy_matrix;
    for (float & value : result.calc_matrix.values) {
        value *= value;
    }
    return result;
}

std::vector<int32_t> token_type_mask(const std::vector<std::string> & decoded_tokens) {
    std::vector<int32_t> mask(decoded_tokens.size(), 1);
    bool                 in_bracket = false;
    for (size_t i = 0; i < decoded_tokens.size(); ++i) {
        if (decoded_tokens[i].find('[') != std::string::npos) {
            in_bracket = true;
        }
        if (in_bracket) {
            mask[i] = 0;
        }
        if (decoded_tokens[i].find(']') != std::string::npos) {
            in_bracket = false;
            mask[i]    = 0;
        }
    }
    return mask;
}

std::vector<std::string> decode_tokens_incrementally(
    const std::vector<int> & token_ids,
    const TokenDecoder &     decoder) {
    if (!decoder) {
        throw std::invalid_argument("token decoder must be provided");
    }
    std::vector<std::string> decoded;
    decoded.reserve(token_ids.size());
    std::vector<int> prefix;
    prefix.reserve(token_ids.size());
    std::string previous;
    for (int token_id : token_ids) {
        prefix.push_back(token_id);
        std::string current = decoder(prefix);
        std::string contribution;
        if (current.size() >= previous.size()) {
            contribution = current.substr(previous.size());
            if (!is_valid_utf8(contribution)) {
                contribution.clear();
            }
        }
        decoded.push_back(std::move(contribution));
        previous = std::move(current);
    }
    return decoded;
}

std::vector<TokenTimestamp> token_timestamps(
    const Matrix &                   calc_matrix,
    const std::vector<int> &         token_ids,
    const std::vector<std::string> & decoded_tokens,
    double                           total_duration_seconds) {
    validate_matrix(calc_matrix, "calculation matrix");
    if (token_ids.size() != calc_matrix.rows || decoded_tokens.size() != token_ids.size()) {
        throw std::invalid_argument("token data must match calculation matrix rows");
    }
    Matrix costs = calc_matrix;
    for (float & value : costs.values) {
        value = -value;
    }
    DtwPath path = dtw(costs);
    double seconds_per_frame = total_duration_seconds / static_cast<double>(calc_matrix.cols);

    std::vector<TokenTimestamp> result;
    result.reserve(token_ids.size());
    for (size_t token = 0; token < token_ids.size(); ++token) {
        bool   found = false;
        double start = result.empty() ? 0.0 : result.back().end;
        double end   = start;
        for (size_t step = 0; step < path.token_indices.size(); ++step) {
            if (path.token_indices[step] == static_cast<int32_t>(token)) {
                double time = static_cast<double>(path.frame_indices[step]) * seconds_per_frame;
                if (!found) {
                    start = time;
                    found = true;
                }
                end = time;
            }
        }
        if (end < start) {
            end = start;
        }
        result.push_back({token_ids[token], decoded_tokens[token], start, end, 0.0});
    }
    return result;
}

std::vector<SentenceTimestamp> sentence_timestamps(
    const std::vector<TokenTimestamp> & token_alignment,
    const TokenDecoder &                 decoder) {
    if (!decoder) {
        throw std::invalid_argument("token decoder must be provided");
    }
    std::vector<SentenceTimestamp> result;
    std::vector<TokenTimestamp>    current;

    auto append_sentence = [&]() {
        if (current.empty()) {
            return;
        }
        std::vector<int> token_ids;
        token_ids.reserve(current.size());
        for (const TokenTimestamp & token : current) {
            token_ids.push_back(token.token_id);
        }
        std::string text = trim_ascii(decoder(token_ids));
        if (!text.empty()) {
            double confidence = 0.0;
            size_t count      = 0;
            for (const TokenTimestamp & token : current) {
                if (token.probability > 0.0) {
                    confidence += token.probability;
                    ++count;
                }
            }
            if (count > 0) {
                confidence /= static_cast<double>(count);
            }
            result.push_back({
                text,
                round_half_even(current.front().start, 3),
                round_half_even(current.back().end, 3),
                current,
                confidence,
            });
        }
        current.clear();
    };

    for (const TokenTimestamp & token : token_alignment) {
        current.push_back(token);
        if (token.text.find('\n') != std::string::npos) {
            append_sentence();
        }
    }
    append_sentence();

    if (!result.empty()) {
        auto minimum_maximum = std::minmax_element(
            result.begin(),
            result.end(),
            [](const SentenceTimestamp & left, const SentenceTimestamp & right) {
                return left.confidence < right.confidence;
            });
        double minimum = minimum_maximum.first->confidence;
        double maximum = minimum_maximum.second->confidence;
        double range   = maximum - minimum;
        for (SentenceTimestamp & sentence : result) {
            if (range > 1e-9) {
                sentence.confidence = (sentence.confidence - minimum) / range;
            }
            sentence.confidence = round_half_even(sentence.confidence, 2);
        }
    }
    return result;
}

std::string format_lrc(
    const std::vector<SentenceTimestamp> & sentences,
    bool                                   include_end_time) {
    std::string result;
    for (size_t i = 0; i < sentences.size(); ++i) {
        if (i > 0) {
            result.push_back('\n');
        }
        result += '[' + lrc_time(sentences[i].start) + ']';
        if (include_end_time) {
            result += '[' + lrc_time(sentences[i].end) + ']';
        }
        result += sentences[i].text;
    }
    return result;
}

AlignmentMetrics compute_alignment_metrics(
    const Matrix &               energy_matrix,
    const DtwPath &              path,
    const std::vector<int32_t> & type_mask,
    double                       time_weight,
    double                       overlap_frames,
    double                       instrumental_weight) {
    validate_matrix(energy_matrix, "energy matrix");
    if (type_mask.size() != energy_matrix.rows) {
        throw std::invalid_argument("type mask must match energy matrix rows");
    }
    if (path.token_indices.size() != path.frame_indices.size()) {
        throw std::invalid_argument("DTW path coordinate arrays must have the same size");
    }

    AlignmentMetrics metrics;
    size_t           lyric_rows = 0;
    size_t           covered_rows = 0;
    std::vector<double> centroids;
    for (size_t row = 0; row < energy_matrix.rows; ++row) {
        if (type_mask[row] != 1) {
            continue;
        }
        ++lyric_rows;
        float row_maximum = -std::numeric_limits<float>::infinity();
        double weight_sum = 0.0;
        double time_sum   = 0.0;
        for (size_t col = 0; col < energy_matrix.cols; ++col) {
            double energy = energy_matrix(row, col);
            row_maximum = std::max(row_maximum, static_cast<float>(energy));
            if (energy > time_weight) {
                weight_sum += energy;
                time_sum += energy * static_cast<double>(col);
            }
        }
        if (row_maximum > 0.1f) {
            ++covered_rows;
        }
        if (weight_sum > 1e-9) {
            centroids.push_back(time_sum / weight_sum);
        }
    }
    metrics.coverage = lyric_rows > 0 ?
        static_cast<double>(covered_rows) / static_cast<double>(lyric_rows) : 1.0;

    if (centroids.size() <= 1) {
        metrics.monotonicity = 1.0;
    } else {
        size_t non_decreasing = 0;
        for (size_t i = 1; i < centroids.size(); ++i) {
            if (centroids[i] >= centroids[i - 1] - overlap_frames) {
                ++non_decreasing;
            }
        }
        metrics.monotonicity =
            static_cast<double>(non_decreasing) / static_cast<double>(centroids.size() - 1);
    }

    double weighted_energy = 0.0;
    double total_weight    = 0.0;
    for (size_t i = 0; i < path.token_indices.size(); ++i) {
        int32_t row = path.token_indices[i];
        int32_t col = path.frame_indices[i];
        if (row < 0 || col < 0 ||
            row >= static_cast<int32_t>(energy_matrix.rows) ||
            col >= static_cast<int32_t>(energy_matrix.cols)) {
            throw std::out_of_range("DTW path coordinate is outside the energy matrix");
        }
        double weight = type_mask[static_cast<size_t>(row)] == 0 ?
            instrumental_weight : 1.0;
        weighted_energy += energy_matrix(static_cast<size_t>(row), static_cast<size_t>(col)) * weight;
        total_weight += weight;
    }
    metrics.path_confidence = total_weight > 0.0 ? weighted_energy / total_weight : 0.0;
    return metrics;
}

double calculate_lyrics_score(const AlignmentMetrics & metrics) {
    double score = metrics.coverage * metrics.coverage *
                   metrics.monotonicity * metrics.monotonicity *
                   metrics.path_confidence;
    score = std::max(0.0, std::min(1.0, score));
    return round_half_even(score, 4);
}

}  // namespace tts_cpp::acestep::lyrics
