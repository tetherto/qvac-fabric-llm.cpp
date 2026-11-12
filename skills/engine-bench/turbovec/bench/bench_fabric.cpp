// Fabric TurboVec bench: one cold add, then the 1,000-query search timed
// `iters` times. Prints one JSON line. Run it in a fresh process per sample
// so the cold add includes rotation and codebook setup.
//
// usage: bench_fabric <base.f32> <queries.f32> <dim> <bits> <k> <threads> <iters>
//
// BENCH_ONE=1 instead searches one query per call, like RAG does: a warm-up
// pass over all queries, then `iters` timed passes; prints the median and p95
// of every per-call latency in microseconds.

#include "ggml-vector-index.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static std::vector<float> read_f32(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    const std::streamsize size = f.tellg();
    std::vector<float> out(static_cast<size_t>(size) / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(out.data()), size);
    return out;
}

// FNV-1a over every returned id and score bit pattern, so two builds can be
// checked for identical results over the whole query set.
static uint64_t fnv_add(uint64_t h, const void * data, size_t n) {
    const unsigned char * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s base.f32 queries.f32 dim bits k threads iters\n", argv[0]);
        return 1;
    }
    const std::vector<float> base = read_f32(argv[1]);
    const std::vector<float> queries = read_f32(argv[2]);
    const int dim = std::atoi(argv[3]);
    const int bits = std::atoi(argv[4]);
    const int k = std::atoi(argv[5]);
    const int threads = std::atoi(argv[6]);
    const int iters = std::atoi(argv[7]);
    const int n = static_cast<int>(base.size() / dim);
    const int n_q = static_cast<int>(queries.size() / dim);

    std::vector<uint64_t> ids(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        ids[i] = static_cast<uint64_t>(i);
    }

    const double t0 = now_ms();
    ggml_vec_index_t * idx = bits == 2 ? ggml_vec_index_create_turbovec_q2(dim) : ggml_vec_index_create_turbovec_q4(dim);
    if (idx == nullptr) {
        std::fprintf(stderr, "create failed\n");
        return 1;
    }
    int rc = GGML_VEC_INDEX_OK;
#ifdef BENCH_HAS_CALIBRATE
    // BENCH_CALIBRATE=<rows> fits TQ+ on the first rows before the add, as
    // the addon does on a first add.
    const char * calib = std::getenv("BENCH_CALIBRATE");
    if (calib != nullptr && std::atoi(calib) > 0) {
        rc = ggml_vec_index_turbovec_calibrate(idx, base.data(), std::min(std::atoi(calib), n));
        if (rc != GGML_VEC_INDEX_OK) {
            std::fprintf(stderr, "calibrate failed: %s\n", ggml_vec_index_error_to_string(rc));
            return 1;
        }
    }
#endif
#ifdef BENCH_HAS_THREADS
    // Threads set before the add, so the add itself runs on them (the
    // search threads are set again below either way).
    rc = ggml_vec_index_set_threads(idx, threads);
    if (rc != GGML_VEC_INDEX_OK) {
        std::fprintf(stderr, "set_threads failed\n");
        return 1;
    }
#endif
    rc = ggml_vec_index_add(idx, base.data(), n, ids.data());
    const double add_ms = now_ms() - t0;
    if (rc != GGML_VEC_INDEX_OK) {
        std::fprintf(stderr, "add failed: %s\n", ggml_vec_index_error_to_string(rc));
        return 1;
    }
#ifdef BENCH_HAS_THREADS
    rc = ggml_vec_index_set_threads(idx, threads);
    if (rc != GGML_VEC_INDEX_OK) {
        std::fprintf(stderr, "set_threads failed\n");
        return 1;
    }
    // BENCH_STAGED=0 turns staged search off, to compare with the full scan.
    const char * staged = std::getenv("BENCH_STAGED");
    if (staged != nullptr && ggml_vec_index_turbovec_set_staged(idx, std::atoi(staged)) != GGML_VEC_INDEX_OK) {
        std::fprintf(stderr, "set_staged failed\n");
        return 1;
    }
#else
    if (threads != 1) {
        std::fprintf(stderr, "this build has no thread API\n");
        return 1;
    }
#endif

    const char * one = std::getenv("BENCH_ONE");
    if (one != nullptr && std::atoi(one) != 0) {
        std::vector<float> scores1(static_cast<size_t>(k));
        std::vector<uint64_t> ids1(static_cast<size_t>(k));
        std::vector<double> call_us;
        uint64_t top1 = 0;
        uint64_t hash = 1469598103934665603ull;
        for (int pass = 0; pass <= iters; ++pass) {
            for (int q = 0; q < n_q; ++q) {
                const double s0 = now_ms();
                rc = ggml_vec_index_search(idx, queries.data() + static_cast<size_t>(q) * dim, 1, k, scores1.data(), ids1.data());
                const double us = (now_ms() - s0) * 1e3;
                if (rc != GGML_VEC_INDEX_OK) {
                    std::fprintf(stderr, "search failed: %s\n", ggml_vec_index_error_to_string(rc));
                    return 1;
                }
                if (pass > 0) {
                    call_us.push_back(us);
                }
                if (pass == iters) {
                    hash = fnv_add(hash, ids1.data(), ids1.size() * sizeof(uint64_t));
                    hash = fnv_add(hash, scores1.data(), scores1.size() * sizeof(float));
                }
                if (q == 0) {
                    top1 = ids1[0];
                }
            }
        }
        std::sort(call_us.begin(), call_us.end());
        std::printf("{\"add_ms\": %.3f, \"p50_us\": %.1f, \"p95_us\": %.1f, \"top1_q0\": %llu, \"hash\": \"%016llx\"}\n", add_ms,
                    call_us[call_us.size() / 2], call_us[call_us.size() * 95 / 100], static_cast<unsigned long long>(top1),
                    static_cast<unsigned long long>(hash));
        ggml_vec_index_free(idx);
        return 0;
    }

    std::vector<float> scores(static_cast<size_t>(n_q) * k);
    std::vector<uint64_t> out_ids(static_cast<size_t>(n_q) * k);
    std::vector<double> search_ms;
    for (int it = 0; it < iters; ++it) {
        const double s0 = now_ms();
        rc = ggml_vec_index_search(idx, queries.data(), n_q, k, scores.data(), out_ids.data());
        search_ms.push_back(now_ms() - s0);
        if (rc != GGML_VEC_INDEX_OK) {
            std::fprintf(stderr, "search failed: %s\n", ggml_vec_index_error_to_string(rc));
            return 1;
        }
    }
    std::sort(search_ms.begin(), search_ms.end());
    // BENCH_IDS=<path> writes the last search's ids, n_q x k uint64, for recall.py.
    const char * ids_path = std::getenv("BENCH_IDS");
    if (ids_path != nullptr && ids_path[0] != '\0') {
        std::ofstream f(ids_path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(out_ids.data()), static_cast<std::streamsize>(out_ids.size() * sizeof(uint64_t)));
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", ids_path);
            return 1;
        }
    }
    // Top-1 id of query 0, so implementations can be sanity-checked against
    // each other.
    uint64_t hash = fnv_add(1469598103934665603ull, out_ids.data(), out_ids.size() * sizeof(uint64_t));
    hash = fnv_add(hash, scores.data(), scores.size() * sizeof(float));
    std::printf("{\"add_ms\": %.3f, \"search_ms\": %.3f, \"top1_q0\": %llu, \"hash\": \"%016llx\"}\n",
                add_ms, search_ms[search_ms.size() / 2], static_cast<unsigned long long>(out_ids[0]),
                static_cast<unsigned long long>(hash));
    ggml_vec_index_free(idx);
    return 0;
}
