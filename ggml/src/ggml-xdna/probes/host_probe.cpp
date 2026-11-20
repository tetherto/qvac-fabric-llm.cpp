// What ggml-cpu takes for the MUL_MATs of a prefill, on the shapes of the model and the thread count of a chunk: the host side
// of the routing cost model (xdna-host-cost.h). Each case is a graph of many independent MUL_MATs of different weights run in order,
// like the projections of a pass: every weight is cold when it is read, and the per-node barrier and dispatch cost is paid, so
// total / count is the cost of one projection in a pass.
//
//   host_probe [--types q8_0,bf16] [--rows 1,15,41,67,132] [--shapes K:N,K:N,...] [--threads 16,4] [--mb 400] [--reps 5]
//
// Build (from the build directory, next to the libraries):
//   c++ -O2 -std=gnu++17 -fopenmp -I<src>/ggml/include -I<src>/ggml/src probes/host_probe.cpp -o bin/host_probe -Lbin -lggml-cpu
//       -lggml-base -Wl,-rpath,'$ORIGIN'
// Run with the wait policy of the run line under test (GOMP_SPINCOUNT=20000 or OMP_WAIT_POLICY=PASSIVE).
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct probe_case {
    enum ggml_type type;
    int64_t        K, N, M;
    int            threads;
};

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::stringstream        ss(s);
    for (std::string part; std::getline(ss, part, sep);) {
        out.push_back(part);
    }
    return out;
}

static std::vector<uint8_t> random_weight(enum ggml_type type, int64_t K, int64_t N, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float>                    f((size_t) K * N);
    for (float & x : f) {
        x = u(rng);
    }
    std::vector<uint8_t> q(ggml_row_size(type, K) * N);
    if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) q.data(), (int64_t) f.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) q.data(), (int64_t) f.size());
    } else {
        ggml_quantize_chunk(type, f.data(), q.data(), 0, N, K, nullptr);
    }
    return q;
}

static int copies_for(const probe_case & c, int mb) {
    const size_t bytes = ggml_row_size(c.type, c.K) * c.N;
    return (int) std::clamp<size_t>((size_t) mb * 1024 * 1024 / bytes, 4, 512);
}

// ms of one MUL_MAT of the case, as the median over `reps` of a graph of `copies` of them
static double time_case(ggml_backend_t cpu, const probe_case & c, int mb, int reps) {
    const int            copies = copies_for(c, mb);
    ggml_init_params     ip     = { ggml_tensor_overhead() * (size_t) (copies * 2 + 4) + ggml_graph_overhead_custom(copies + 4, false), nullptr, true };
    ggml_context *       ctx    = ggml_init(ip);
    ggml_tensor *        act    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.M);
    ggml_cgraph *        gf     = ggml_new_graph_custom(ctx, copies + 4, false);
    std::vector<ggml_tensor *> ws;
    for (int i = 0; i < copies; i++) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, c.type, c.K, c.N);
        ws.push_back(w);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, act));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::mt19937          rng(1);
    {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float>                    a((size_t) c.K * c.M);
        for (float & x : a) {
            x = u(rng);
        }
        ggml_backend_tensor_set(act, a.data(), 0, ggml_nbytes(act));
    }
    const std::vector<uint8_t> w0 = random_weight(c.type, c.K, c.N, rng);
    for (ggml_tensor * w : ws) {
        ggml_backend_tensor_set(w, w0.data(), 0, w0.size());
    }
    ggml_backend_cpu_set_n_threads(cpu, c.threads);
    std::vector<double> ms;
    for (int r = 0; r < reps + 1; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(cpu, gf);
        const auto t1 = std::chrono::steady_clock::now();
        if (r > 0) {
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / copies);
        }
    }
    std::sort(ms.begin(), ms.end());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ms[ms.size() / 2];
}

static enum ggml_type type_of(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((enum ggml_type) t);
        if (n && name == n) {
            return (enum ggml_type) t;
        }
    }
    std::fprintf(stderr, "host_probe: unknown type %s\n", name.c_str());
    std::exit(2);
}

int main(int argc, char ** argv) {
    std::string types   = "q8_0,bf16";
    std::string rows    = "1,15,41,67,132";
    std::string shapes  = "512:2048,512:512,512:256,256:512,1024:512,2048:512,512:4096,512:12288";
    std::string threads = "16";
    int         mb = 400, reps = 5;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--types") {
            types = v;
        } else if (k == "--rows") {
            rows = v;
        } else if (k == "--shapes") {
            shapes = v;
        } else if (k == "--threads") {
            threads = v;
        } else if (k == "--mb") {
            mb = std::atoi(v.c_str());
        } else if (k == "--reps") {
            reps = std::atoi(v.c_str());
        }
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    std::printf("type,threads,rows,K,N,weight_bytes,flops,ms\n");
    for (const std::string & tn : split(types, ',')) {
        for (const std::string & th : split(threads, ',')) {
            for (const std::string & sh : split(shapes, ',')) {
                const std::vector<std::string> kn = split(sh, ':');
                for (const std::string & r : split(rows, ',')) {
                    const probe_case c = { type_of(tn), std::atoll(kn[0].c_str()), std::atoll(kn[1].c_str()), std::atoll(r.c_str()),
                                           std::atoi(th.c_str()) };
                    const double ms = time_case(cpu, c, mb, reps);
                    std::printf("%s,%d,%lld,%lld,%lld,%zu,%.0f,%.5f\n", tn.c_str(), c.threads, (long long) c.M, (long long) c.K,
                                (long long) c.N, ggml_row_size(c.type, c.K) * (size_t) c.N, 2.0 * c.M * c.K * c.N, ms);
                    std::fflush(stdout);
                }
            }
        }
    }
    ggml_backend_free(cpu);
    return 0;
}
