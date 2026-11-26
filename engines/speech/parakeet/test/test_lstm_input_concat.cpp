// The [w_ih | w_hh] row stack the TDT decoder builds at load time must be a
// pure byte copy: every packed row has to dequantize to the two source rows
// concatenated, with no requantisation drift.
//   test-lstm-input-concat
#include "parakeet_tdt.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t kInputWidth = 128;  // whole q8_0 blocks on both halves
constexpr int64_t kRows       = 8;
constexpr ggml_type kType     = GGML_TYPE_Q8_0;

std::vector<float> make_matrix(int64_t rows, int64_t width, float seed) {
    std::vector<float> m((size_t) (rows * width));
    for (int64_t i = 0; i < rows * width; ++i) {
        m[(size_t) i] = std::sin(seed + 0.017f * (float) i);
    }
    return m;
}

std::vector<uint8_t> quantize_matrix(const std::vector<float> & src, int64_t rows, int64_t width) {
    std::vector<uint8_t> q((size_t) rows * ggml_row_size(kType, width));
    ggml_quantize_chunk(kType, src.data(), q.data(), 0, rows, width, nullptr);
    return q;
}

std::vector<float> dequantize_rows(const std::vector<uint8_t> & q, int64_t rows, int64_t width) {
    std::vector<float> out((size_t) (rows * width));
    ggml_get_type_traits(kType)->to_float(q.data(), out.data(), rows * width);
    return out;
}

int report_mismatch(int64_t row, int64_t col, float got, float want) {
    std::fprintf(stderr, "[lstm-input-concat] FAIL: row %lld col %lld got %.9g want %.9g\n",
                 (long long) row, (long long) col, got, want);
    return 1;
}

int compare_packed_rows(const std::vector<float> & packed,
                        const std::vector<float> & ih,
                        const std::vector<float> & hh) {
    for (int64_t r = 0; r < kRows; ++r) {
        for (int64_t c = 0; c < 2 * kInputWidth; ++c) {
            const float got  = packed[(size_t) (r * 2 * kInputWidth + c)];
            const float want = c < kInputWidth
                                 ? ih[(size_t) (r * kInputWidth + c)]
                                 : hh[(size_t) (r * kInputWidth + c - kInputWidth)];
            if (got != want) return report_mismatch(r, c, got, want);
        }
    }
    return 0;
}

} // namespace

int main() {
    const std::vector<float> ih_f32 = make_matrix(kRows, kInputWidth, 0.25f);
    const std::vector<float> hh_f32 = make_matrix(kRows, kInputWidth, 3.5f);

    const std::vector<uint8_t> ih_q = quantize_matrix(ih_f32, kRows, kInputWidth);
    const std::vector<uint8_t> hh_q = quantize_matrix(hh_f32, kRows, kInputWidth);

    const size_t row_bytes = ggml_row_size(kType, kInputWidth);
    std::vector<uint8_t> packed_q((size_t) kRows * ggml_row_size(kType, 2 * kInputWidth));
    parakeet::pack_lstm_input_rows(kRows, row_bytes, row_bytes,
                                   ih_q.data(), hh_q.data(), packed_q.data());

    if (packed_q.size() != ih_q.size() + hh_q.size()) {
        std::fprintf(stderr, "[lstm-input-concat] FAIL: packed %zu bytes, sources %zu\n",
                     packed_q.size(), ih_q.size() + hh_q.size());
        return 1;
    }

    const std::vector<float> packed = dequantize_rows(packed_q, kRows, 2 * kInputWidth);
    const std::vector<float> ih     = dequantize_rows(ih_q, kRows, kInputWidth);
    const std::vector<float> hh     = dequantize_rows(hh_q, kRows, kInputWidth);
    if (int rc = compare_packed_rows(packed, ih, hh); rc != 0) return rc;

    std::printf("[lstm-input-concat] PASS: %lld packed rows dequantize to [w_ih | w_hh]\n",
                (long long) kRows);
    return 0;
}
