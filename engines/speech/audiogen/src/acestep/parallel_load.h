#pragma once

// Host-side parallel weight-loading helpers, header-only so the row and chunk
// decomposition is unit-testable without model files.

#include "ggml.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr int    LOAD_MAX_THREADS     = 16;
inline constexpr int    LOAD_SERIAL_MIN_ROWS = 128;
inline constexpr size_t F16_CONVERT_CHUNK    = 4096;

// Splits [0, n) across joined threads; rows are independent and per-row
// arithmetic order is unchanged, so results match the single-threaded pass.
template <typename F>
void parallel_rows(int n, F && fn) {
    const unsigned hw        = std::thread::hardware_concurrency();
    const int      n_threads = (int) std::min<unsigned>(hw ? hw : 1, LOAD_MAX_THREADS);
    if (n < LOAD_SERIAL_MIN_ROWS || n_threads <= 1) {
        fn(0, n);
        return;
    }
    const int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> workers;
    int started_end = 0;
    try {
        for (int t = 0; t < n_threads; t++) {
            const int begin = t * chunk;
            const int end   = std::min(n, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back([&fn, begin, end] { fn(begin, end); });
            started_end = end;
        }
    } catch (...) {
        // Thread creation failed (thread cap, allocation): finish inline below.
    }
    for (auto & w : workers) w.join();
    if (started_end < n) fn(started_end, n);
}

// Chunked f32 -> f16 conversion; same per-element order as a single
// ggml_fp32_to_fp16_row call, parallelized across F16_CONVERT_CHUNK blocks.
inline void convert_f32_to_f16_rows(const float * src, ggml_fp16_t * dst, size_t count) {
    const int n_chunks = (int) (count / F16_CONVERT_CHUNK) + 1;
    parallel_rows(n_chunks, [&](int begin, int end) {
        const size_t lo = (size_t) begin * F16_CONVERT_CHUNK;
        const size_t hi = std::min(count, (size_t) end * F16_CONVERT_CHUNK);
        if (lo < hi) ggml_fp32_to_fp16_row(src + lo, dst + lo, (int) (hi - lo));
    });
}

}  // namespace tts_cpp::acestep
