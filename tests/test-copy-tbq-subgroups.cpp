// Sweeps GGML_VK_TBQ_COPY_SG_SIZE = {0, 8, 16, 32, 64} and, for each value,
// runs the f32 -> {TBQ3_0, TBQ4_0, PQ3_0, PQ4_0, *_64} copy-to-quantize
// kernel on the Vulkan backend and compares the resulting quantized bytes
// against the CPU ggml_quantize_chunk reference.
//
// Why this test exists:
//   copy_to_quant.comp's cooperative TBQ/PQ path is a 32-thread workgroup
//   that uses subgroupAdd / subgroupBallot. On hardware with
//   gl_SubgroupSize < 32 (Intel Xe/Arc at 8/16, ARM Mali, Qualcomm Adreno,
//   some AMD configurations) those ops reduce within a subgroup, not the
//   whole workgroup, so the original shader silently produced wrong bytes.
//   The shader is now parameterized on the SG_SIZE spec constant and takes
//   a shared-memory "stitch" path for SG_SIZE < 32. This test exercises
//   the stitch path on devices that only have one native subgroup size,
//   by forcing the pipeline's requiredSubgroupSize + SG_SIZE spec const to
//   8/16/etc.
//
// How it works:
//   Since GGML_VK_TBQ_COPY_SG_SIZE is consumed at Vulkan device init (once
//   per process), we need a separate process per SG value. This binary
//   self-spawns: in "child" mode it runs exactly one (SG, type) combination
//   and prints a machine-readable summary line. In "parent" mode it forks
//   itself for every combination and aggregates the results.
//
// Accuracy metric:
//   The GPU and CPU quantizers compute the same math but in different float
//   orders (horizontal reductions across subgroups vs. scalar sums), so
//   byte-exact equality is not guaranteed -- we report both "bytes match"
//   and a dequantize NMSE against the f32 input, and we also compare GPU
//   dequantize vs. CPU dequantize so SG==32 vs SG==8 is a direct numerical
//   comparison. The fast-path (SG>=32) and stitch-path (SG<32) are required
//   to land within a tight NMSE tolerance of each other: if the stitch
//   implementation is wrong, NMSE blows up to O(1).
//
// Performance:
//   For each (SG, type) we time N repetitions of the copy on the device and
//   report ms/iter and GB/s (input bytes). This is not a rigorous benchmark
//   -- the intent is to catch order-of-magnitude regressions (e.g. a SG=8
//   stitch that barriers too aggressively) rather than to tune performance.

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <process.h>
#    define POPEN  _popen
#    define PCLOSE _pclose
#else
#    include <unistd.h>
#    define POPEN  popen
#    define PCLOSE pclose
#endif

namespace {

// ---------------------------------------------------------------------------
// Shared config
// ---------------------------------------------------------------------------

// Workgroup is 32 threads, block is 128 (or 64 for _64 variants). Use a
// shape that is large enough to amortize launch overhead and to dispatch
// many workgroups, but small enough that the test runs in < 1 s.
struct Shape {
    int64_t      ne0;  // fastest-moving (row length, multiple of block size)
    int64_t      ne1;  // number of rows
    const char * label;
};

static const std::array<Shape, 2> kShapes = {
    {
     // Small: ~0.5 MB f32 input. Good for timing overhead checks.
        { 512, 256, "small" },
     // Medium: ~8 MB f32 input. Dispatches enough workgroups to keep a
        // small iGPU busy, and touches enough blocks to catch cross-subgroup
        // stitch bugs that don't repro on just a couple of blocks.
        { 2048, 1024, "medium" },
     }
};

struct QType {
    ggml_type    t;
    const char * name;
    int          blck;
};

// block size here is the SHADER's BK, not ggml_blck_size. For TBQ/PQ
// the shader always processes BK elements per workgroup (BK=128 for
// the regular variants, BK=64 for the _64 variants). ne0 must be a
// multiple of this.
static const std::array<QType, 8> kTypes = {
    {
     { GGML_TYPE_TBQ3_0, "tbq3_0", 128 },
     { GGML_TYPE_TBQ4_0, "tbq4_0", 128 },
     { GGML_TYPE_PQ3_0, "pq3_0", 128 },
     { GGML_TYPE_PQ4_0, "pq4_0", 128 },
     { GGML_TYPE_TBQ3_0_64, "tbq3_0_64", 64 },
     { GGML_TYPE_TBQ4_0_64, "tbq4_0_64", 64 },
     { GGML_TYPE_PQ3_0_64, "pq3_0_64", 64 },
     { GGML_TYPE_PQ4_0_64, "pq4_0_64", 64 },
     }
};

// SG sizes to sweep. 0 means "leave GGML_VK_TBQ_COPY_SG_SIZE unset", i.e.
// let the backend pick its default (the hardware's native SG size on
// size-control devices, or the old SG_SIZE=32 hardcoded path otherwise).
// 4/8/16 exercise the stitch path; 32/64 exercise the fast path. 4 is the
// smallest value the shader's tq_sh_red scratch (sized TQ_WG/4 = 8) can
// accommodate (NSG = 32/4 = 8). Values that the current device does not
// expose in [subgroup_min_size, subgroup_max_size] are rejected host-side
// and the test records them as "skipped".
static const std::array<uint32_t, 6> kSgSizes = {
    { 0, 4, 8, 16, 32, 64 }
};

// Number of warm-up + timed iterations for the perf number.
static constexpr int kWarmupIters = 2;
static constexpr int kTimedIters  = 10;

// ---------------------------------------------------------------------------
// Helpers (shared between parent and child)
// ---------------------------------------------------------------------------

static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0, denom = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        denom += (double) a[i] * (double) a[i];
    }
    if (denom == 0.0) {
        return 0.0;
    }
    return num / denom;
}

static double max_abs_diff(const float * a, const float * b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static size_t byte_mismatch_count(const uint8_t * a, const uint8_t * b, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            ++c;
        }
    }
    return c;
}

// Deterministic Gaussian fill (matches the distribution of post-Hadamard
// rotated attention activations, which is what cpy_f32_tbq* actually sees
// in llama-perplexity).
static void fill_normal(std::vector<float> & v, uint32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (auto & x : v) {
        x = d(rng);
    }
}

// Pick a Vulkan backend; return nullptr if none.
static ggml_backend_t pick_vulkan_backend(std::string & name_out) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (std::strcmp(ggml_backend_reg_name(reg), "Vulkan") != 0) {
            continue;
        }
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (b) {
            name_out = ggml_backend_dev_name(dev);
            return b;
        }
    }
    return nullptr;
}

// Dequantize via the CPU type traits so we can compare f32 distributions.
static void dequantize(ggml_type t, const void * src, float * dst, int64_t nrows, int64_t ne0) {
    const auto * tt = ggml_get_type_traits(t);
    if (!tt || !tt->to_float) {
        std::fprintf(stderr, "dequantize: no to_float for %s\n", ggml_type_name(t));
        std::abort();
    }
    // to_float takes a count in f32 elements.
    for (int64_t r = 0; r < nrows; ++r) {
        const size_t row_bytes = ggml_row_size(t, ne0);
        tt->to_float((const char *) src + r * row_bytes, dst + r * ne0, ne0);
    }
}

// ---------------------------------------------------------------------------
// Child-process mode: run one (SG, type, shape) and print a result line.
// ---------------------------------------------------------------------------

struct ChildResult {
    bool   ok_run          = false;  // kernel executed without error
    bool   supported       = false;  // backend supports the op for this type
    size_t n_bytes         = 0;      // quantized output size in bytes
    size_t mismatch_bytes  = 0;      // bytes that differ from CPU reference
    double nmse_gpu_vs_cpu = 0.0;    // NMSE(dequant(gpu), dequant(cpu))
    double nmse_gpu_vs_src = 0.0;    // NMSE(dequant(gpu), src_f32)
    double nmse_cpu_vs_src = 0.0;    // NMSE(dequant(cpu), src_f32)  -- sanity
    double max_abs_vs_cpu  = 0.0;
    double ms_per_iter     = 0.0;
    double gb_per_s        = 0.0;
};

static ChildResult run_one(ggml_backend_t backend, ggml_type qtype, int64_t ne0, int64_t ne1) {
    ChildResult r{};

    const int64_t nrows = ne1;
    const int64_t nels  = ne0 * nrows;

    // ---- Build a minimal graph: f32 input -> ggml_cpy -> qtype output.
    // Using ggml_cpy (not ggml_set_rows) because it's the common path and
    // it's what the host-side pipeline wiring in ggml-vulkan.cpp hits for
    // the regression workload (llama-perplexity KV-cache fill).
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(ip));

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, nrows);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), qtype, ne0, nrows);
    ggml_set_name(src, "src_f32");
    ggml_set_name(dst, "dst_q");
    ggml_tensor * cpy = ggml_cpy(ctx.get(), src, dst);
    ggml_set_name(cpy, "cpy");

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        return r;
    }

    if (!ggml_backend_supports_op(backend, cpy)) {
        r.supported = false;
        return r;
    }
    r.supported = true;

    // ---- Generate input.
    std::vector<float> input(nels);
    fill_normal(input, 0xC0FFEEu ^ (uint32_t) qtype ^ (uint32_t) ne0 ^ (uint32_t) ne1);
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));

    // ---- CPU reference: row-by-row quantize_chunk.
    const size_t         row_bytes_q = ggml_row_size(qtype, ne0);
    std::vector<uint8_t> cpu_q(row_bytes_q * nrows);
    const size_t         blck = ggml_blck_size(qtype);
    for (int64_t r_i = 0; r_i < nrows; ++r_i) {
        ggml_quantize_chunk(qtype, input.data() + r_i * ne0, cpu_q.data() + r_i * row_bytes_q, 0, ne0 / blck, blck,
                            nullptr);
    }

    // ---- GPU quantize (warmup + timed).
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, cpy);

    for (int i = 0; i < kWarmupIters; ++i) {
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            return r;
        }
    }
    ggml_backend_synchronize(backend);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimedIters; ++i) {
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            return r;
        }
    }
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::high_resolution_clock::now();

    r.ok_run          = true;
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    r.ms_per_iter     = 1e3 * secs / kTimedIters;
    r.gb_per_s        = (double) (nels * sizeof(float)) / (secs / kTimedIters) / 1e9;

    // ---- Read back GPU output.
    std::vector<uint8_t> gpu_q(row_bytes_q * nrows);
    ggml_backend_tensor_get(dst, gpu_q.data(), 0, gpu_q.size());

    // ---- Accuracy metrics.
    r.n_bytes        = gpu_q.size();
    r.mismatch_bytes = byte_mismatch_count(gpu_q.data(), cpu_q.data(), gpu_q.size());

    std::vector<float> gpu_f32(nels), cpu_f32(nels);
    dequantize(qtype, gpu_q.data(), gpu_f32.data(), nrows, ne0);
    dequantize(qtype, cpu_q.data(), cpu_f32.data(), nrows, ne0);

    r.nmse_gpu_vs_cpu = nmse(cpu_f32.data(), gpu_f32.data(), nels);
    r.nmse_gpu_vs_src = nmse(input.data(), gpu_f32.data(), nels);
    r.nmse_cpu_vs_src = nmse(input.data(), cpu_f32.data(), nels);
    r.max_abs_vs_cpu  = max_abs_diff(cpu_f32.data(), gpu_f32.data(), nels);

    return r;
}

// Probe mode: initialize the backend only, so ggml-vulkan emits the
// `tbq_copy_sg_size_status` line for the current GGML_VK_TBQ_COPY_SG_SIZE,
// then exit. This lets the parent decide whether to actually spawn a full
// workload child for this SG value, avoiding wasted compute on rows that
// would land back on the default path anyway.
static int probe_main() {
    ggml_backend_load_all();
    std::string    backend_name;
    ggml_backend_t backend = pick_vulkan_backend(backend_name);
    if (!backend) {
        std::printf("PROBE ok=0\n");
        return 0;
    }
    std::printf("PROBE ok=1 backend=%s\n", backend_name.c_str());
    ggml_backend_free(backend);
    return 0;
}

// Child mode entrypoint. Args: <qtype_index> <shape_index>.
// Prints a single line "RESULT ok=... supp=... mism=... nmse_gvsc=... ms=... gbps=..."
// so the parent can parse it trivially.
static int child_main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "child: expected 2 args (qtype_idx, shape_idx)\n");
        return 2;
    }
    const int qi = std::atoi(argv[1]);
    const int si = std::atoi(argv[2]);
    if (qi < 0 || qi >= (int) kTypes.size() || si < 0 || si >= (int) kShapes.size()) {
        std::fprintf(stderr, "child: bad indices qi=%d si=%d\n", qi, si);
        return 2;
    }

    ggml_backend_load_all();
    std::string    backend_name;
    ggml_backend_t backend = pick_vulkan_backend(backend_name);
    if (!backend) {
        std::printf("RESULT ok=0 supp=0 skip=no_gpu\n");
        return 0;
    }

    const QType & qt          = kTypes[qi];
    const Shape & sh          = kShapes[si];
    // ne0 must be a multiple of the shader's BK (=ggml_blck_size).
    const int64_t ne0_aligned = (sh.ne0 / qt.blck) * qt.blck;

    ChildResult r = run_one(backend, qt.t, ne0_aligned, sh.ne1);
    std::printf("RESULT backend=%s type=%s shape=%s ne0=%" PRId64 " ne1=%" PRId64
                " ok=%d supp=%d bytes=%zu mism=%zu nmse_gvsc=%.3e nmse_gvss=%.3e "
                "nmse_cvss=%.3e maxabs=%.3e ms=%.3f gbps=%.2f\n",
                backend_name.c_str(), qt.name, sh.label, ne0_aligned, sh.ne1, r.ok_run ? 1 : 0, r.supported ? 1 : 0,
                r.n_bytes, r.mismatch_bytes, r.nmse_gpu_vs_cpu, r.nmse_gpu_vs_src, r.nmse_cpu_vs_src, r.max_abs_vs_cpu,
                r.ms_per_iter, r.gb_per_s);

    ggml_backend_free(backend);
    return 0;
}

// ---------------------------------------------------------------------------
// Parent-process mode: spawn children with different SG env values.
// ---------------------------------------------------------------------------

struct ParsedLine {
    bool        present = false;
    bool        ok = false, supp = false;
    size_t      bytes = 0, mism = 0;
    double      nmse_gvsc = 0, nmse_gvss = 0, nmse_cvss = 0, maxabs = 0;
    double      ms = 0, gbps = 0;
    std::string backend;
    // Parsed from "ggml_vulkan: tbq_copy_sg_size_status requested=R applied=A reason=X"
    // in child stderr. override_rejected is true when the child's requested SG
    // differs from what the backend actually applied (applied=0 with a non-zero
    // request, or applied != requested for any other reason). Used by the parent
    // to label these rows SKIPPED instead of OK, since they effectively ran at
    // the default SG and are duplicates of the sg=0 case.
    bool        status_seen         = false;
    uint32_t    status_requested    = 0;
    uint32_t    status_applied      = 0;
    std::string status_reason;
    bool        override_rejected() const {
        return status_seen && status_requested != 0 && status_applied != status_requested;
    }
};

static bool parse_key_double(const std::string & line, const std::string & key, double & out) {
    auto p = line.find(key + "=");
    if (p == std::string::npos) {
        return false;
    }
    out = std::atof(line.c_str() + p + key.size() + 1);
    return true;
}

static bool parse_key_size(const std::string & line, const std::string & key, size_t & out) {
    double d = 0;
    if (!parse_key_double(line, key, d)) {
        return false;
    }
    out = (size_t) d;
    return true;
}

static bool parse_key_int(const std::string & line, const std::string & key, int & out) {
    double d = 0;
    if (!parse_key_double(line, key, d)) {
        return false;
    }
    out = (int) d;
    return true;
}

static bool parse_key_str(const std::string & line, const std::string & key, std::string & out) {
    auto p = line.find(key + "=");
    if (p == std::string::npos) {
        return false;
    }
    p += key.size() + 1;
    auto q = line.find(' ', p);
    out    = line.substr(p, q == std::string::npos ? std::string::npos : (q - p));
    return true;
}

// Minimal result from a --probe child: did the backend apply the requested
// SG override. Used to prune the (SG, type, shape) sweep before the expensive
// workload children are spawned. When `seen` is false the probe did not emit
// a status line (env var unset, or backend init failed), which we treat as
// "applied" for sg=0 and "rejected-unknown" for sg!=0.
struct ProbeResult {
    bool        seen      = false;
    bool        applied   = false;  // requested == applied && seen
    uint32_t    requested = 0;
    uint32_t    applied_v = 0;
    std::string reason;
};

static ProbeResult probe_sg(const std::string & self_path, uint32_t sg) {
    ProbeResult pr{};
    pr.requested = sg;

    char cmd[2048];
#if defined(_WIN32)
    if (sg == 0) {
        std::snprintf(cmd, sizeof(cmd), "set \"GGML_VK_TBQ_COPY_SG_SIZE=\" && \"%s\" --probe 2>&1", self_path.c_str());
    } else {
        std::snprintf(cmd, sizeof(cmd), "set \"GGML_VK_TBQ_COPY_SG_SIZE=%u\" && \"%s\" --probe 2>&1", sg, self_path.c_str());
    }
#else
    if (sg == 0) {
        std::snprintf(cmd, sizeof(cmd), "unset GGML_VK_TBQ_COPY_SG_SIZE; \"%s\" --probe 2>&1", self_path.c_str());
    } else {
        std::snprintf(cmd, sizeof(cmd), "GGML_VK_TBQ_COPY_SG_SIZE=%u \"%s\" --probe 2>&1", sg, self_path.c_str());
    }
#endif

    FILE * f = POPEN(cmd, "r");
    if (!f) {
        return pr;
    }
    char        buf[4096];
    std::string status_line;
    while (std::fgets(buf, sizeof(buf), f)) {
        std::string s = buf;
        if (s.find("tbq_copy_sg_size_status") != std::string::npos) {
            status_line = s;
        }
    }
    PCLOSE(f);

    if (sg == 0) {
        // No env set, no status line expected -- treat as applied (default).
        pr.seen      = true;
        pr.applied   = true;
        pr.applied_v = 0;
        return pr;
    }
    if (status_line.empty()) {
        return pr;
    }
    pr.seen = true;
    int req = 0, app = 0;
    parse_key_int(status_line, "requested", req);
    parse_key_int(status_line, "applied", app);
    pr.requested = (uint32_t) req;
    pr.applied_v = (uint32_t) app;
    parse_key_str(status_line, "reason", pr.reason);
    while (!pr.reason.empty() && (pr.reason.back() == '\n' || pr.reason.back() == '\r')) {
        pr.reason.pop_back();
    }
    pr.applied = pr.requested != 0 && pr.applied_v == pr.requested;
    return pr;
}

static ParsedLine run_child(const std::string & self_path, uint32_t sg, int qi, int si) {
    ParsedLine pl;

    // Build the command. We re-exec ourselves with --child to force child_main.
    // Env var sg==0 means "don't set the var at all".
    char cmd[2048];
#if defined(_WIN32)
    if (sg == 0) {
        std::snprintf(cmd, sizeof(cmd), "set \"GGML_VK_TBQ_COPY_SG_SIZE=\" && \"%s\" --child %d %d 2>&1", self_path.c_str(),
                      qi, si);
    } else {
        std::snprintf(cmd, sizeof(cmd), "set \"GGML_VK_TBQ_COPY_SG_SIZE=%u\" && \"%s\" --child %d %d 2>&1", sg, self_path.c_str(),
                      qi, si);
    }
#else
    if (sg == 0) {
        std::snprintf(cmd, sizeof(cmd), "unset GGML_VK_TBQ_COPY_SG_SIZE; \"%s\" --child %d %d 2>&1", self_path.c_str(),
                      qi, si);
    } else {
        std::snprintf(cmd, sizeof(cmd), "GGML_VK_TBQ_COPY_SG_SIZE=%u \"%s\" --child %d %d 2>&1", sg, self_path.c_str(),
                      qi, si);
    }
#endif

    FILE * f = POPEN(cmd, "r");
    if (!f) {
        std::fprintf(stderr, "run_child: popen failed for cmd=%s\n", cmd);
        return pl;
    }
    char        buf[4096];
    std::string result_line;
    std::string status_line;  // ggml_vulkan: tbq_copy_sg_size_status ...
    while (std::fgets(buf, sizeof(buf), f)) {
        std::string s = buf;
        // forward child output to parent stderr for debugging; keep only
        // the last RESULT line for parsing.
        std::fprintf(stderr, "[sg=%u %s] %s", sg, kTypes[qi].name, s.c_str());
        if (s.rfind("RESULT", 0) == 0) {
            result_line = s;
        }
        // The backend emits this once per device init when the env var is set.
        // We grep it out of the combined stdout+stderr stream popen() gave us.
        if (s.find("tbq_copy_sg_size_status") != std::string::npos) {
            status_line = s;
        }
    }
    PCLOSE(f);

    if (result_line.empty()) {
        return pl;
    }
    pl.present = true;
    int ok_i = 0, supp_i = 0;
    parse_key_int(result_line, "ok", ok_i);
    parse_key_int(result_line, "supp", supp_i);
    pl.ok   = ok_i != 0;
    pl.supp = supp_i != 0;
    parse_key_size(result_line, "bytes", pl.bytes);
    parse_key_size(result_line, "mism", pl.mism);
    parse_key_double(result_line, "nmse_gvsc", pl.nmse_gvsc);
    parse_key_double(result_line, "nmse_gvss", pl.nmse_gvss);
    parse_key_double(result_line, "nmse_cvss", pl.nmse_cvss);
    parse_key_double(result_line, "maxabs", pl.maxabs);
    parse_key_double(result_line, "ms", pl.ms);
    parse_key_double(result_line, "gbps", pl.gbps);
    parse_key_str(result_line, "backend", pl.backend);

    if (!status_line.empty()) {
        pl.status_seen = true;
        int req = 0, app = 0;
        parse_key_int(status_line, "requested", req);
        parse_key_int(status_line, "applied", app);
        pl.status_requested = (uint32_t) req;
        pl.status_applied   = (uint32_t) app;
        parse_key_str(status_line, "reason", pl.status_reason);
        // parse_key_str takes everything up to the next whitespace, but the
        // key is at end-of-line so it swallows the trailing '\n'. Strip any
        // trailing CR/LF before we print it into a table.
        while (!pl.status_reason.empty() &&
               (pl.status_reason.back() == '\n' || pl.status_reason.back() == '\r')) {
            pl.status_reason.pop_back();
        }
    }
    return pl;
}

// Resolve a comma-separated list of type names (e.g. "tbq3_0,pq3_0") against
// kTypes, returning the matching indices in their original kTypes order.
// Returns empty on no match (caller decides whether that's a hard error).
// Unknown names are reported on stderr and skipped -- we don't want a typo in
// a test script to silently run zero cases, but we also don't want a mismatch
// between a script and a newly renamed type to fail the whole leg.
static std::vector<size_t> resolve_type_filter(const std::string & csv) {
    std::vector<size_t> out;
    size_t              start = 0;
    while (start <= csv.size()) {
        size_t            end   = csv.find(',', start);
        const std::string token = csv.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!token.empty()) {
            bool found = false;
            for (size_t i = 0; i < kTypes.size(); ++i) {
                if (token == kTypes[i].name) {
                    out.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::fprintf(stderr, "warning: --types token '%s' does not match any known type; ignoring\n",
                             token.c_str());
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

static int parent_main(const std::string & self_path, const std::vector<size_t> & type_filter) {
    // Sanity: make sure a Vulkan backend is actually available at all.
    {
        ggml_backend_load_all();
        std::string    nm;
        ggml_backend_t b = pick_vulkan_backend(nm);
        if (!b) {
            std::fprintf(stdout, "SKIPPED: no Vulkan backend available\n");
            return 0;
        }
        std::fprintf(stdout, "using backend: %s\n", nm.c_str());
        ggml_backend_free(b);
    }

    // Tolerance for NMSE(gpu_sgN_dequant, cpu_dequant). The quantization math
    // is IEEE-float non-associative, so we don't require bit-identity between
    // GPU orderings. 1e-6 is tight enough to catch a broken stitch (which
    // produces O(1) NMSE) and loose enough to tolerate float reordering.
    const double kNmseTol = 1e-6;

    int n_fail    = 0;
    int n_run     = 0;
    int n_skipped = 0;

    // Build the list of type indices to iterate. Empty filter == all types.
    std::vector<size_t> type_indices;
    if (type_filter.empty()) {
        for (size_t i = 0; i < kTypes.size(); ++i) {
            type_indices.push_back(i);
        }
    } else {
        type_indices = type_filter;
    }
    if (!type_filter.empty()) {
        std::fprintf(stdout, "type filter: ");
        for (size_t i = 0; i < type_indices.size(); ++i) {
            std::fprintf(stdout, "%s%s", i ? "," : "", kTypes[type_indices[i]].name);
        }
        std::fprintf(stdout, "\n");
    }

    // Probe each SG value once on this device to find out which overrides are
    // actually honored. Rejections are independent of (type, shape), so doing
    // this up front lets us skip spawning a full workload child for every
    // (SG, type, shape) combination where the SG would be rejected. Without
    // this, a rejected SG would still run the shader at the default path and
    // produce duplicate OK numbers -- wasting compute and confusing the table.
    std::vector<ProbeResult> probe(kSgSizes.size());
    std::fprintf(stdout, "probing SG overrides on device:\n");
    for (size_t k = 0; k < kSgSizes.size(); ++k) {
        probe[k] = probe_sg(self_path, kSgSizes[k]);
        const char * tag;
        if (kSgSizes[k] == 0) {
            tag = "default";
        } else if (!probe[k].seen) {
            tag = "unknown";
        } else if (probe[k].applied) {
            tag = "applied";
        } else {
            tag = probe[k].reason.empty() ? "rejected" : probe[k].reason.c_str();
        }
        std::fprintf(stdout, "  sg=%-2u -> %s\n", kSgSizes[k], tag);
    }

    for (size_t si = 0; si < kShapes.size(); ++si) {
        for (size_t qi : type_indices) {
            std::fprintf(stdout, "\n=== %s %s ===\n", kTypes[qi].name, kShapes[si].label);
            // Reference = cpu dequantize: implicit via child's nmse_gvsc.
            // Additionally, within-GPU consistency: nmse_gvsc should be
            // (a) small for every SG, and (b) the same for all SGs modulo
            // reordering. If SG=8 disagrees with SG=32 by more than kNmseTol
            // the stitch path is broken.
            std::vector<ParsedLine> by_sg(kSgSizes.size());
            for (size_t k = 0; k < kSgSizes.size(); ++k) {
                // If the up-front probe said this SG won't be honored, don't
                // even spawn the workload child: it would just run the
                // default path and produce duplicate numbers. Fill in a
                // synthetic ParsedLine that the report/accounting code below
                // treats as a rejected-override row.
                if (kSgSizes[k] != 0 && probe[k].seen && !probe[k].applied) {
                    ParsedLine pl;
                    pl.present          = true;
                    pl.status_seen      = true;
                    pl.status_requested = probe[k].requested;
                    pl.status_applied   = probe[k].applied_v;
                    pl.status_reason    = probe[k].reason;
                    by_sg[k]            = pl;
                    continue;
                }
                by_sg[k] = run_child(self_path, kSgSizes[k], (int) qi, (int) si);
            }

            // Report. Rows where the child requested a non-default SG but the
            // backend rejected it (e.g. SG=8 on gfx1150 which only exposes
            // [32,64]) are labelled SKIPPED-<reason>: the workload child is
            // not spawned for those SGs at all, so the metric columns are
            // empty. Showing metrics for them would duplicate the sg=0 row
            // and misleadingly suggest the requested SG was exercised.
            std::fprintf(stdout, "  %-8s | %-18s | %12s | %12s | %9s | %9s\n", "sg", "status", "nmse(g v c)",
                         "nmse(g v s)", "ms/iter", "GB/s");
            for (size_t k = 0; k < kSgSizes.size(); ++k) {
                const auto & p = by_sg[k];
                char tag[32];
                if (!p.present) {
                    std::snprintf(tag, sizeof(tag), "NOPROC");
                } else if (p.override_rejected()) {
                    // e.g. "SKIPPED-out_of_range", "SKIPPED-unsupported_by_shader",
                    // "SKIPPED-no_size_control". kSgSizes[k] == 0 never triggers
                    // this branch because status_requested == 0 -> not rejected.
                    std::snprintf(tag, sizeof(tag), "SKIP-%s",
                                  p.status_reason.empty() ? "rejected" : p.status_reason.c_str());
                } else if (!p.supp) {
                    std::snprintf(tag, sizeof(tag), "NOSUPP");
                } else if (!p.ok) {
                    std::snprintf(tag, sizeof(tag), "FAIL");
                } else {
                    std::snprintf(tag, sizeof(tag), "OK");
                }
                // For SKIP-* rows we don't spawn the workload child, so the
                // metric fields are meaningless -- print dashes instead of
                // zeros (or worse, stale duplicated numbers from the old
                // default-path fallback) so the reader can tell at a glance
                // that the row carries no measurement.
                const bool no_metrics = p.override_rejected() || !p.present || !p.supp;
                if (no_metrics) {
                    std::fprintf(stdout, "  sg=%-5u | %-18s | %12s | %12s | %9s | %9s\n", kSgSizes[k], tag, "-", "-",
                                 "-", "-");
                } else {
                    std::fprintf(stdout, "  sg=%-5u | %-18s | %12.3e | %12.3e | %9.3f | %9.2f\n", kSgSizes[k], tag,
                                 p.nmse_gvsc, p.nmse_gvss, p.ms, p.gbps);
                }
            }

            // Decide pass/fail. Pass criteria:
            //   1. Every SG that ran, is supported, and had its override
            //      actually applied must have nmse_gvsc <= kNmseTol.
            //   2. nmse_gvsc across all such SGs must agree to within
            //      kNmseTol*10 (differences come from float reduction order only).
            //   3. At least one SG must have produced a valid result.
            // Rows whose override was rejected are excluded from the pass/fail
            // math (they are effectively duplicates of sg=0) but we DO count
            // them in the skipped tally so the summary is transparent.
            bool   any_applied = false, any_ok = false;
            double min_nmse = 1e300, max_nmse = -1e300;
            int    n_skipped_local = 0;
            for (size_t k = 0; k < kSgSizes.size(); ++k) {
                const auto & p = by_sg[k];
                if (!p.present) {
                    continue;
                }
                // Rejected-override rows are synthetic (no workload child
                // was spawned) so they won't have `supp` set; check for them
                // before we drop `!p.supp` rows.
                if (p.override_rejected()) {
                    ++n_skipped_local;
                    continue;
                }
                if (!p.supp) {
                    continue;
                }
                any_applied = true;
                if (!p.ok) {
                    std::fprintf(stderr, "  FAIL: sg=%u did not execute\n", kSgSizes[k]);
                    ++n_fail;
                    continue;
                }
                any_ok = true;
                ++n_run;
                if (p.nmse_gvsc > kNmseTol) {
                    std::fprintf(stderr, "  FAIL: sg=%u nmse(gpu vs cpu)=%.3e > %.1e\n", kSgSizes[k], p.nmse_gvsc,
                                 kNmseTol);
                    ++n_fail;
                }
                min_nmse = std::min(min_nmse, p.nmse_gvsc);
                max_nmse = std::max(max_nmse, p.nmse_gvsc);
            }
            n_skipped += n_skipped_local;
            if (!any_applied || !any_ok) {
                std::fprintf(stderr, "  FAIL: no SG produced a valid result\n");
                ++n_fail;
            }
            if (any_ok && max_nmse - min_nmse > kNmseTol * 10.0) {
                std::fprintf(stderr, "  FAIL: nmse spread across SG sizes %.3e > %.1e\n", max_nmse - min_nmse,
                             kNmseTol * 10.0);
                ++n_fail;
            }
        }
    }

    // "ran" counts rows whose requested SG was applied and produced an NMSE we
    // actually checked; "skipped" counts rows the backend rejected. On an
    // AMD-RDNA box with [min=32, max=64], you'll typically see sg=4/8/16
    // appearing under skipped and sg=0/32/64 under ran. If everything is
    // skipped except sg=0 the test passes with ran > 0 but gives no
    // stitch-path coverage -- that's honest reporting, not a failure.
    std::fprintf(stdout, "\n%s: ran=%d skipped=%d failed=%d\n", n_fail ? "FAILED" : "PASSED", n_run, n_skipped,
                 n_fail);
    return n_fail ? 1 : 0;
}

}  // namespace

// Print parent-mode usage. Child mode is an internal ABI and intentionally
// undocumented -- users should never invoke it directly.
static void print_usage(const char * prog) {
    std::fprintf(stderr,
                 "usage: %s [--types t1,t2,...]\n"
                 "\n"
                 "  --types LIST   Comma-separated list of quant type names to test.\n"
                 "                 Known: tbq3_0, tbq4_0, pq3_0, pq4_0,\n"
                 "                        tbq3_0_64, tbq4_0_64, pq3_0_64, pq4_0_64.\n"
                 "                 If omitted, every known type is tested (slow).\n"
                 "\n"
                 "The test self-spawns one child process per (SG, type, shape) triple,\n"
                 "so restricting the type list linearly reduces runtime.\n",
                 prog);
}

int main(int argc, char ** argv) {
    // Self-spawning: the parent invocation executes parent_main which
    // popen()s this same binary with --child for each (SG, type, shape).
    if (argc >= 2 && std::strcmp(argv[1], "--child") == 0) {
        return child_main(argc - 1, argv + 1);
    }
    // --probe: backend-init-only path used by the parent to decide whether
    // a given GGML_VK_TBQ_COPY_SG_SIZE is actually honored on this device.
    // No workload is dispatched, so a rejection costs only process startup.
    if (argc >= 2 && std::strcmp(argv[1], "--probe") == 0) {
        return probe_main();
    }
    const std::string self_path = argv[0];

    // Parse parent-mode CLI. Keep this tiny and hand-rolled: no library
    // dependency needed, and the surface is intentionally small.
    std::vector<size_t> type_filter;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--types") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --types requires a comma-separated argument\n");
                print_usage(argv[0]);
                return 2;
            }
            type_filter = resolve_type_filter(argv[++i]);
            if (type_filter.empty()) {
                std::fprintf(stderr, "error: --types resolved to zero known types\n");
                return 2;
            }
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    return parent_main(self_path, type_filter);
}
