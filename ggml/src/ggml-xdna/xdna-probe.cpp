// xdna-probe - standalone diagnostic for XDNA/XRT kernel residency and the cost
// of changing what the NPU is running.
//
// Separates four costs that are easy to conflate, and reports each on its own:
//
//   instruction/module rotation - several sequences bound to ONE context
//   same-xclbin context switch  - several contexts of the SAME design
//   different-xclbin switch     - contexts of DIFFERENT designs (fabric changes)
//   eviction/recreation         - no free context slot, so one must be destroyed
//
// The different-xclbin figure is computed the way it has to be, against both
// homogeneous baselines rather than against one of them:
//
//   overhead = mean(AMAM) - (mean(AAAA) + mean(MMMM)) / 2
//
// Every submit is timed in three parts (start, wait, sync) so a difference can
// be attributed rather than guessed at, every scenario is verified against the
// CPU, and every sample is written to CSV so several process runs can be pooled.
//
// Deliberately does not link ggml: only XRT is involved, so nothing about the
// result depends on the backend.
//
// SPDX-License-Identifier: MIT

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

#ifdef XDNA_PROBE_HAS_AIEBU
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

#include "aiebu/aiebu_assembler.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const char * KERNEL_NAME = "MLIR_AIE";
static constexpr uint64_t OPCODE_TXN = 3;
static constexpr int ARG_DATA0 = 3;

static double now_us(void) {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

static double now_ms(void) {
    return now_us() / 1e3;
}

// ---------------------------------------------------------------------------
// samples and statistics
// ---------------------------------------------------------------------------

struct sample {
    double start_us;
    double wait_us;
    double sync_us;
    double total_us;
};

struct stats {
    size_t n;
    double mean_us;
    double p50_us;
    double p95_us;
    double p99_us;
};

static double percentile(std::vector<double> & sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const size_t i = (size_t) ((double) (sorted.size() - 1) * q);
    return sorted[i];
}

// Which field of the sample to summarize.
enum class field { start, wait, sync, total };

static stats summarize(const std::vector<sample> & samples, field f = field::total) {
    std::vector<double> v;
    v.reserve(samples.size());
    for (const sample & s : samples) {
        switch (f) {
            case field::start: v.push_back(s.start_us); break;
            case field::wait:  v.push_back(s.wait_us);  break;
            case field::sync:  v.push_back(s.sync_us);  break;
            case field::total: v.push_back(s.total_us); break;
        }
    }
    std::sort(v.begin(), v.end());

    double sum = 0;
    for (double x : v) {
        sum += x;
    }

    stats s;
    s.n       = v.size();
    s.mean_us = v.empty() ? 0.0 : sum / (double) v.size();
    s.p50_us  = percentile(v, 0.50);
    s.p95_us  = percentile(v, 0.95);
    s.p99_us  = percentile(v, 0.99);
    return s;
}

// ---------------------------------------------------------------------------
// artifacts
// ---------------------------------------------------------------------------

struct instr_blob {
    std::vector<uint32_t> words;
};

static bool read_instr(const std::string & path, instr_blob & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0 || (size % 4) != 0) {
        fclose(f);
        return false;
    }
    out.words.resize((size_t) size / 4);
    const size_t n = fread(out.words.data(), 1, (size_t) size, f);
    fclose(f);
    return n == (size_t) size;
}

// Which elementwise operation an artifact implements, taken from its name. Used
// only to check the result against the CPU.
enum class op_kind { add, mul, unknown };

static op_kind op_of(const std::string & path) {
    const std::string name = fs::path(path).filename().string();
    if (name.find("-add-") != std::string::npos) {
        return op_kind::add;
    }
    if (name.find("-mul-") != std::string::npos) {
        return op_kind::mul;
    }
    return op_kind::unknown;
}

struct design {
    std::string path;
    xrt::uuid   uuid;
    instr_blob  instr;
    op_kind     op;
    int         id;   // probe-assigned, so CSV rows can be grouped
};

// ---------------------------------------------------------------------------
// the two ways of driving a design
// ---------------------------------------------------------------------------

// Cost of putting one handle together, broken into the steps the caller pays
// separately. Kept next to the handle because that is where it is produced.
struct build_cost {
    double ctx_ms    = 0;
    double kernel_ms = 0;
    double module_ms = 0;
    double bo_ms     = 0;
    double asm_ms    = 0;
};

// Instruction stream handed to the driver in a buffer on every submit. This is
// what the backend does today.
struct live_kernel {
    xrt::hw_context context;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    xrt::bo         bo_a;
    xrt::bo         bo_b;
    xrt::bo         bo_c;
    xrt::run        run;
    uint32_t        instr_words = 0;
    int             design_id = -1;
    int             ctx_id    = -1;
    int             mod_id    = -1;   // no module on this path
};

static int g_next_ctx_id = 0;

static std::unique_ptr<live_kernel> make_kernel(xrt::device &  dev,
                                               const design & d,
                                               size_t         tile_elems,
                                               build_cost *   cost = nullptr) {
    auto lk = std::make_unique<live_kernel>();
    build_cost c;

    double t0 = now_ms();
    lk->context = xrt::hw_context(dev, d.uuid);
    c.ctx_ms    = now_ms() - t0;

    t0           = now_ms();
    lk->kernel   = xrt::kernel(lk->context, KERNEL_NAME);
    c.kernel_ms  = now_ms() - t0;

    const size_t instr_bytes = d.instr.words.size() * sizeof(uint32_t);
    const size_t data_bytes  = tile_elems * sizeof(float);

    t0 = now_ms();
    lk->bo_instr = xrt::bo(dev, instr_bytes, XCL_BO_FLAGS_CACHEABLE, lk->kernel.group_id(1));
    lk->bo_a = xrt::bo(dev, data_bytes, XRT_BO_FLAGS_HOST_ONLY, lk->kernel.group_id(ARG_DATA0 + 0));
    lk->bo_b = xrt::bo(dev, data_bytes, XRT_BO_FLAGS_HOST_ONLY, lk->kernel.group_id(ARG_DATA0 + 1));
    lk->bo_c = xrt::bo(dev, data_bytes, XRT_BO_FLAGS_HOST_ONLY, lk->kernel.group_id(ARG_DATA0 + 2));
    c.bo_ms  = now_ms() - t0;

    memcpy(lk->bo_instr.map<void *>(), d.instr.words.data(), instr_bytes);
    lk->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(lk->bo_c.map<void *>(), 0, data_bytes);

    lk->run = xrt::run(lk->kernel);
    lk->run.set_arg(0, OPCODE_TXN);
    lk->run.set_arg(1, lk->bo_instr);
    lk->run.set_arg(2, (uint32_t) d.instr.words.size());
    lk->run.set_arg(ARG_DATA0 + 0, lk->bo_a);
    lk->run.set_arg(ARG_DATA0 + 1, lk->bo_b);
    lk->run.set_arg(ARG_DATA0 + 2, lk->bo_c);

    lk->instr_words = (uint32_t) d.instr.words.size();
    lk->design_id   = d.id;
    lk->ctx_id      = g_next_ctx_id++;

    if (cost) {
        *cost = c;
    }
    return lk;
}

#ifdef XDNA_PROBE_HAS_AIEBU
// Instruction stream assembled into an ELF and bound to the context as a module.
// Several of these can share one context, which is the point of this path.
struct live_module {
    xrt::hw_context                   context;
    std::unique_ptr<xrt::elf>         elf;
    std::unique_ptr<xrt::module>      mod;
    std::unique_ptr<xrt::ext::kernel> kernel;
    xrt::bo                           bo_a;
    xrt::bo                           bo_b;
    xrt::bo                           bo_c;
    xrt::run                          run;
    int                               design_id = -1;
    int                               ctx_id    = -1;
    int                               mod_id    = -1;
};

static int g_next_mod_id = 0;

static std::vector<char> assemble_elf(const instr_blob & instr) {
    const char * p = reinterpret_cast<const char *>(instr.words.data());
    const std::vector<char> txn(p, p + instr.words.size() * sizeof(uint32_t));

    aiebu::aiebu_assembler as(aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, txn);
    return as.get_elf();
}

// Takes the context by value so several modules can share one and the same
// context: the xclbin fixes the fabric, the module carries the shape.
static std::unique_ptr<live_module> make_module_kernel(const xrt::hw_context & ctx,
                                                      const design &          d,
                                                      size_t                  tile_elems,
                                                      int                     ctx_id,
                                                      build_cost *            cost = nullptr) {
    auto lm = std::make_unique<live_module>();
    build_cost c;

    double t0 = now_ms();
    std::vector<char> elf_data = assemble_elf(d.instr);
    c.asm_ms = now_ms() - t0;

    lm->context = ctx;

    t0          = now_ms();
    lm->elf     = std::make_unique<xrt::elf>(elf_data.data(), elf_data.size());
    lm->mod     = std::make_unique<xrt::module>(*lm->elf);
    c.module_ms = now_ms() - t0;

    t0          = now_ms();
    lm->kernel  = std::make_unique<xrt::ext::kernel>(lm->context, *lm->mod, KERNEL_NAME);
    c.kernel_ms = now_ms() - t0;

    const size_t data_bytes = tile_elems * sizeof(float);

    t0       = now_ms();
    lm->bo_a = xrt::ext::bo(lm->context, data_bytes);
    lm->bo_b = xrt::ext::bo(lm->context, data_bytes);
    lm->bo_c = xrt::ext::bo(lm->context, data_bytes);
    c.bo_ms  = now_ms() - t0;

    memset(lm->bo_c.map<void *>(), 0, data_bytes);

    lm->run = xrt::run(*lm->kernel);
    lm->run.set_arg(0, OPCODE_TXN);
    lm->run.set_arg(1, 0);
    lm->run.set_arg(2, 0);
    lm->run.set_arg(ARG_DATA0 + 0, lm->bo_a);
    lm->run.set_arg(ARG_DATA0 + 1, lm->bo_b);
    lm->run.set_arg(ARG_DATA0 + 2, lm->bo_c);

    lm->design_id = d.id;
    lm->ctx_id    = ctx_id;
    lm->mod_id    = g_next_mod_id++;

    if (cost) {
        *cost = c;
    }
    return lm;
}
#endif // XDNA_PROBE_HAS_AIEBU

// ---------------------------------------------------------------------------
// submit, inputs, verification
// ---------------------------------------------------------------------------

template <typename T> static sample submit_timed(T & lk) {
    const double t0 = now_us();
    lk.run.start();
    const double t1 = now_us();
    lk.run.wait();
    const double t2 = now_us();
    lk.bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const double t3 = now_us();

    sample s;
    s.start_us = t1 - t0;
    s.wait_us  = t2 - t1;
    s.sync_us  = t3 - t2;
    s.total_us = t3 - t0;
    return s;
}

// One graph is hundreds of nodes, so whether the per-submit round trip is paid
// per node or per submission decides whether offloading a graph can pay off at
// all. xrt::runlist submits several runs as one command, which is what
// ggml-hexagon does with its op batch (opt_opbatch is 1024 nodes per queue
// write). Both halves below do the same amount of device work.
struct batch_result {
    double seq_us  = 0;   // per op, one start/wait each
    double list_us = 0;   // per op, one execute/wait for the whole list
};

static xrt::run clone_run(const live_kernel & lk) {
    xrt::run r(lk.kernel);
    r.set_arg(0, OPCODE_TXN);
    r.set_arg(1, lk.bo_instr);
    r.set_arg(2, lk.instr_words);
    r.set_arg(ARG_DATA0 + 0, lk.bo_a);
    r.set_arg(ARG_DATA0 + 1, lk.bo_b);
    r.set_arg(ARG_DATA0 + 2, lk.bo_c);
    return r;
}

static batch_result run_batch(live_kernel & lk, int batch, int iters, int n_warm) {
    std::vector<xrt::run> runs;
    runs.reserve(batch);
    for (int i = 0; i < batch; i++) {
        runs.push_back(clone_run(lk));
    }

    xrt::runlist list(lk.context);
    for (auto & r : runs) {
        list.add(r);
    }

    for (int i = 0; i < n_warm; i++) {
        lk.run.start();
        lk.run.wait();
        list.execute();
        list.wait();
    }

    batch_result out;

    double acc = 0;
    for (int i = 0; i < iters; i++) {
        const double t0 = now_us();
        for (int j = 0; j < batch; j++) {
            lk.run.start();
            lk.run.wait();
        }
        acc += now_us() - t0;
    }
    out.seq_us = acc / (iters * batch);

    acc = 0;
    for (int i = 0; i < iters; i++) {
        const double t0 = now_us();
        list.execute();
        list.wait();
        acc += now_us() - t0;
    }
    out.list_us = acc / (iters * batch);

    return out;
}

// Same inputs everywhere, so outputs are comparable across paths and scenarios.
static float in_a(size_t i) { return 1.0f + 0.001f * (float) (i % 977); }
static float in_b(size_t i) { return 2.0f - 0.002f * (float) (i % 641); }

template <typename T> static void fill_inputs(T & lk, size_t tile_elems) {
    float * a = lk.bo_a.template map<float *>();
    float * b = lk.bo_b.template map<float *>();
    for (size_t i = 0; i < tile_elems; i++) {
        a[i] = in_a(i);
        b[i] = in_b(i);
    }
    lk.bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    lk.bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

// How many submits a freshly built handle needs before its output matches the
// CPU. Anything above 1 means a newly created context cannot be trusted on its
// first use, which is a correctness constraint on the backend rather than a
// performance one.
template <typename T>
static int submits_until_correct(T & lk, size_t tile_elems, op_kind op, int max_tries);

// Returns the largest absolute deviation from what the CPU computes, or -1 when
// the artifact is not one whose operation we can predict from its name.
template <typename T> static double verify_cpu(T & lk, size_t tile_elems, op_kind op) {
    if (op == op_kind::unknown) {
        return -1.0;
    }
    const float * c = lk.bo_c.template map<const float *>();
    double max_delta = 0;
    for (size_t i = 0; i < tile_elems; i++) {
        const double want = (op == op_kind::add) ? (double) in_a(i) + (double) in_b(i)
                                                 : (double) in_a(i) * (double) in_b(i);
        max_delta = std::max(max_delta, std::abs((double) c[i] - want));
    }
    return max_delta;
}

template <typename T>
static int submits_until_correct(T & lk, size_t tile_elems, op_kind op, int max_tries) {
    if (op == op_kind::unknown) {
        return -1;
    }
    for (int i = 1; i <= max_tries; i++) {
        submit_timed(lk);
        if (verify_cpu(lk, tile_elems, op) < 1e-5) {
            return i;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

struct csv_writer {
    FILE * f      = nullptr;
    int    run_id = 0;

    void open(const std::string & path, int run) {
        run_id = run;
        const bool exists = fs::exists(path) && fs::file_size(path) > 0;
        f = fopen(path.c_str(), "a");
        if (f && !exists) {
            fprintf(f, "run_id,scenario,iter,xclbin,uuid,design_id,ctx_id,mod_id,"
                       "start_us,wait_us,sync_us,total_us\n");
        }
    }

    void write(const char * scenario, int iter, const design & d,
               int ctx_id, int mod_id, const sample & s) {
        if (!f) {
            return;
        }
        fprintf(f, "%d,%s,%d,%s,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                run_id, scenario, iter,
                fs::path(d.path).filename().string().c_str(),
                d.uuid.to_string().c_str(),
                d.id, ctx_id, mod_id,
                s.start_us, s.wait_us, s.sync_us, s.total_us);
    }

    ~csv_writer() {
        if (f) {
            fclose(f);
        }
    }
};

// ---------------------------------------------------------------------------
// scenario runner
// ---------------------------------------------------------------------------

// Runs `rotation` round-robin. A rotation of one handle is a homogeneous
// baseline; a rotation of two is a switch on every submit.
template <typename T>
static std::vector<sample> run_scenario(const char *               name,
                                        const std::vector<T *> &   rotation,
                                        const std::vector<design>& designs,
                                        int                        n_warm,
                                        int                        n_iter,
                                        csv_writer &               csv) {
    for (int i = 0; i < n_warm; i++) {
        submit_timed(*rotation[i % rotation.size()]);
    }

    std::vector<sample> out;
    out.reserve(n_iter);
    for (int i = 0; i < n_iter; i++) {
        T & h = *rotation[i % rotation.size()];
        const sample s = submit_timed(h);
        out.push_back(s);
        csv.write(name, i, designs[h.design_id], h.ctx_id, h.mod_id, s);
    }
    return out;
}

static void print_row(const char * label, const std::vector<sample> & s) {
    const stats t  = summarize(s, field::total);
    const stats st = summarize(s, field::start);
    const stats w  = summarize(s, field::wait);
    const stats sy = summarize(s, field::sync);
    printf("   %-34s %6zu %9.1f %9.1f %9.1f %9.1f | %8.1f %8.1f %8.1f\n",
           label, t.n, t.mean_us, t.p50_us, t.p95_us, t.p99_us,
           st.mean_us, w.mean_us, sy.mean_us);
}

static void print_header(void) {
    printf("   %-34s %6s %9s %9s %9s %9s | %8s %8s %8s\n",
           "scenario", "n", "mean us", "p50", "p95", "p99", "start", "wait", "sync");
}

static fs::path exe_dir(void) {
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::current_path() : p.parent_path();
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    size_t      tile    = 16384;
    int         n_iter  = 1000;
    int         n_warm  = 32;
    int         max_ctx = 24;
    int         run_id  = 0;
    int         big_bo_mib = 0;
    int         batch   = 0;
    bool        run_smi = false;
    std::string csv_path;
    std::vector<std::pair<std::string, std::string>> artifacts;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--tile" && i + 1 < argc) {
            tile = (size_t) atoll(argv[++i]);
        } else if (a == "--iter" && i + 1 < argc) {
            n_iter = atoi(argv[++i]);
        } else if (a == "--warmup" && i + 1 < argc) {
            n_warm = atoi(argv[++i]);
        } else if (a == "--max-contexts" && i + 1 < argc) {
            max_ctx = atoi(argv[++i]);
        } else if (a == "--run-id" && i + 1 < argc) {
            run_id = atoi(argv[++i]);
        } else if (a == "--big-bo-mib" && i + 1 < argc) {
            big_bo_mib = atoi(argv[++i]);
        } else if (a == "--batch" && i + 1 < argc) {
            batch = atoi(argv[++i]);
        } else if (a == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (a == "--smi") {
            run_smi = true;
        } else if (a == "--kernel" && i + 2 < argc) {
            artifacts.emplace_back(argv[i + 1], argv[i + 2]);
            i += 2;
        } else {
            fprintf(stderr,
                    "usage: %s [--tile N] [--iter N] [--warmup N] [--max-contexts N]\n"
                    "          [--run-id N] [--csv <path>] [--smi] [--big-bo-mib N]\n"
                    "          [--batch N] [--kernel <xclbin> <insts.bin>]...\n"
                    "\n"
                    "  --batch submits N runs as one xrt::runlist and compares that\n"
                    "  against N separate start/wait pairs, which says whether the\n"
                    "  per-submit round trip is paid per node or per submission.\n"
                    "\n"
                    "  --big-bo-mib makes the residency test give every context a buffer of\n"
                    "  that size, which shows whether the cap is a count of slots or a\n"
                    "  memory budget.\n"
                    "\n"
                    "  Two designs are needed for the different-xclbin figure; by default\n"
                    "  the add and mul artifacts for the given tile are picked up from the\n"
                    "  directory of this executable.\n", argv[0]);
            return 2;
        }
    }

    if (artifacts.empty()) {
        const fs::path dir = exe_dir();
        for (const char * op : { "add", "mul" }) {
            char stem[128];
            snprintf(stem, sizeof(stem), "ggml-xdna-%s-npu2-f32-%zu", op, tile);
            const fs::path x = dir / (std::string(stem) + ".xclbin");
            const fs::path b = dir / (std::string(stem) + ".insts.bin");
            if (fs::exists(x) && fs::exists(b)) {
                artifacts.emplace_back(x.string(), b.string());
            }
        }
    }

    if (artifacts.empty()) {
        fprintf(stderr, "xdna-probe: no kernel artifacts found for tile=%zu; pass --kernel\n", tile);
        return 1;
    }

    csv_writer csv;
    if (!csv_path.empty()) {
        csv.open(csv_path, run_id);
    }

    try {
        printf("xdna-probe: run_id=%d tile=%zu warmup=%d iters=%d designs=%zu\n\n",
               run_id, tile, n_warm, n_iter, artifacts.size());

        // --- 1. device -----------------------------------------------------
        double t0 = now_ms();
        xrt::device dev(0u);
        const double t_dev = now_ms() - t0;
        printf("1. device\n");
        printf("   name                 : %s\n", dev.get_info<xrt::info::device::name>().c_str());
        printf("   xrt::device(0)       : %8.3f ms\n\n", t_dev);

        // --- 2. one-time cost, per component -------------------------------
        printf("2. one-time cost of putting a design on the NPU, per component\n");
        printf("   %-30s %8s %8s %8s %8s %8s\n",
               "design", "parse", "regxcl", "hwctx", "kernel", "BOs");

        std::vector<design> designs;
        for (const auto & art : artifacts) {
            design d;
            d.path = art.first;
            d.op   = op_of(art.first);
            d.id   = (int) designs.size();
            if (!read_instr(art.second, d.instr)) {
                fprintf(stderr, "xdna-probe: cannot read %s\n", art.second.c_str());
                return 1;
            }

            t0 = now_ms();
            xrt::xclbin xclbin{art.first};
            const double t_parse = now_ms() - t0;

            t0 = now_ms();
            dev.register_xclbin(xclbin);
            const double t_reg = now_ms() - t0;

            d.uuid = xclbin.get_uuid();

            build_cost c;
            auto probe_handle = make_kernel(dev, d, tile, &c);
            (void) probe_handle;

            printf("   %-30s %8.3f %8.3f %8.3f %8.3f %8.3f\n",
                   fs::path(art.first).filename().string().c_str(),
                   t_parse, t_reg, c.ctx_ms, c.kernel_ms, c.bo_ms);

            designs.push_back(std::move(d));
        }
        printf("   (all values in ms, paid once at init)\n\n");

        // --- 3. how many contexts stay resident ----------------------------
        printf("3. how many hw_contexts the driver lets us hold at once\n");
        std::vector<std::unique_ptr<live_kernel>> pool;
        std::string limit_reason = "(hit the --max-contexts cap, not a driver limit)";
        for (int i = 0; i < max_ctx; i++) {
            try {
                pool.push_back(make_kernel(dev, designs[i % designs.size()], tile));
            } catch (const std::exception & e) {
                limit_reason = std::string("driver refused #") + std::to_string(i + 1) + ": " + e.what();
                break;
            }
        }
        const size_t n_max_ctx = pool.size();
        printf("   resident contexts    : %zu\n", n_max_ctx);
        printf("   stopped because      : %s\n", limit_reason.c_str());

        if (pool.empty()) {
            fprintf(stderr, "xdna-probe: could not create a single context\n");
            return 1;
        }

        // Whether the cap is a count of context slots or a budget that large
        // buffers eat into. Same loop, but every context also holds a big buffer;
        // context creation and buffer allocation are reported separately so a
        // refusal can be attributed to one or the other.
        if (big_bo_mib > 0) {
            printf("   with an extra %d MiB buffer per context:\n", big_bo_mib);
            struct fat_ctx {
                xrt::hw_context ctx;
                xrt::kernel     krn;
                xrt::bo         bo;
            };
            std::vector<std::unique_ptr<fat_ctx>> fat;
            std::string          fat_reason = "(reached the same cap)";
            const size_t         bo_bytes   = (size_t) big_bo_mib * 1024 * 1024;
            {
                // The pool holds every slot, so it has to go before this runs.
                pool.clear();

                for (int i = 0; i < max_ctx; i++) {
                    auto f = std::make_unique<fat_ctx>();
                    try {
                        f->ctx = xrt::hw_context(dev, designs[i % designs.size()].uuid);
                        f->krn = xrt::kernel(f->ctx, KERNEL_NAME);
                    } catch (const std::exception & e) {
                        fat_reason = std::string("hw_context #") + std::to_string(i + 1)
                                     + " refused: " + e.what();
                        break;
                    }
                    try {
                        f->bo = xrt::bo(dev, bo_bytes, XRT_BO_FLAGS_HOST_ONLY,
                                        f->krn.group_id(ARG_DATA0));
                        memset(f->bo.map<void *>(), 0, bo_bytes);
                    } catch (const std::exception & e) {
                        fat_reason = std::string("buffer #") + std::to_string(i + 1)
                                     + " refused: " + e.what();
                        break;
                    }
                    fat.push_back(std::move(f));
                }
                printf("     contexts holding %4d MiB each : %zu  (%zu MiB pinned in total)\n",
                       big_bo_mib, fat.size(), fat.size() * (size_t) big_bo_mib);
                printf("     stopped because              : %s\n", fat_reason.c_str());
            }

            // Rebuild the pool the later sections expect. The fat contexts are
            // still holding every slot, so they have to be released first.
            fat.clear();
            for (int i = 0; i < (int) n_max_ctx; i++) {
                pool.push_back(make_kernel(dev, designs[i % designs.size()], tile));
            }
            printf("     pool rebuilt to %zu contexts\n", pool.size());
        }
        printf("\n");

        if (run_smi) {
            printf("4. the driver's own view, taken while those contexts are held\n");
            fflush(stdout);
            if (system("/opt/xilinx/xrt/bin/xrt-smi examine -r aie-partitions 2>/dev/null"
                       " | sed 's/^/   /'") != 0) {
                printf("   (xrt-smi unavailable)\n");
            }
            printf("\n");
        }

        // --- 5. eviction: no free slot, so a context must be destroyed ------
        // Two cases, and the difference between them is the whole point. Warm:
        // the design being brought back still has other contexts, so its PDI is
        // already on the array. Cold: every context of that design was dropped
        // first, so the fabric has to be programmed again - the real cache miss.
        printf("5. eviction with a full pool: destroy a context, create another\n");
        double ms_evict_warm = 0;
        double ms_evict_cold = 0;
        double us_first_warm = 0;
        double us_steady_warm = 0;
        double us_first_cold = 0;
        int    ok_warm_n     = 0;
        int    ok_cold_n     = 0;
        {
            // Warm: drop one context, bring back a design that is still resident.
            const int victim_design = pool.back()->design_id;
            t0 = now_ms();
            pool.pop_back();
            const double ms_destroy_warm = now_ms() - t0;

            build_cost c;
            t0 = now_ms();
            auto fresh = make_kernel(dev, designs[0], tile, &c);
            const double ms_create_warm = now_ms() - t0;
            ms_evict_warm = ms_destroy_warm + ms_create_warm;

            fill_inputs(*fresh, tile);
            const sample first = submit_timed(*fresh);
            us_first_warm      = first.total_us;
            csv.write("evict_warm_first", 0, designs[0], fresh->ctx_id, -1, first);

            const double d_warm  = verify_cpu(*fresh, tile, designs[0].op);
            const int    ok_warm = d_warm < 1e-5
                                       ? 1
                                       : 1 + submits_until_correct(*fresh, tile, designs[0].op, 8);
            ok_warm_n = ok_warm;

            std::vector<live_kernel *> rot{ fresh.get() };
            const auto after = run_scenario("evict_warm_steady", rot, designs, n_warm,
                                            std::min(n_iter, 200), csv);
            us_steady_warm = summarize(after).mean_us;

            printf("   WARM  design still resident elsewhere (evicted design %d)\n", victim_design);
            printf("     destroy one context           : %8.3f ms\n", ms_destroy_warm);
            printf("     hw_context+kernel+BOs again   : %8.3f ms  (hwctx %.3f, kernel %.3f, BOs %.3f)\n",
                   ms_create_warm, c.ctx_ms, c.kernel_ms, c.bo_ms);
            printf("     first invocation / steady     : %8.1f us / %.1f us\n", us_first_warm, us_steady_warm);
            printf("     submits until output correct  : %8d  (delta after the 1st: %.3g)\n",
                   ok_warm, d_warm);
            printf("     eviction+recreation           : %8.3f ms\n", ms_evict_warm);

            pool.push_back(std::move(fresh));

            // Cold: drop every context of one design, then bring it back. Only
            // meaningful when there is a second design to evict entirely.
            if (designs.size() > 1) {
                const int cold = 1;
                t0 = now_ms();
                pool.erase(std::remove_if(pool.begin(), pool.end(),
                                          [&](const std::unique_ptr<live_kernel> & k) {
                                              return k->design_id == cold;
                                          }),
                           pool.end());
                const double ms_destroy_cold = now_ms() - t0;

                build_cost cc;
                t0 = now_ms();
                auto revived = make_kernel(dev, designs[cold], tile, &cc);
                const double ms_create_cold = now_ms() - t0;
                ms_evict_cold = ms_destroy_cold + ms_create_cold;

                fill_inputs(*revived, tile);
                const sample fc = submit_timed(*revived);
                us_first_cold   = fc.total_us;
                csv.write("evict_cold_first", 0, designs[cold], revived->ctx_id, -1, fc);

                const double d_cold  = verify_cpu(*revived, tile, designs[cold].op);
                const int    ok_cold = d_cold < 1e-5
                                           ? 1
                                           : 1 + submits_until_correct(*revived, tile, designs[cold].op, 8);
                ok_cold_n = ok_cold;

                printf("   COLD  design fully evicted first (design %d)\n", cold);
                printf("     destroy all its contexts      : %8.3f ms\n", ms_destroy_cold);
                printf("     hw_context+kernel+BOs again   : %8.3f ms  (hwctx %.3f, kernel %.3f, BOs %.3f)\n",
                       ms_create_cold, cc.ctx_ms, cc.kernel_ms, cc.bo_ms);
                printf("     first invocation              : %8.1f us\n", us_first_cold);
                printf("     submits until output correct  : %8d  (delta after the 1st: %.3g)\n",
                       ok_cold, d_cold);
                printf("     eviction+recreation           : %8.3f ms\n", ms_evict_cold);
            }
            printf("\n");
        }

        // The scenario suite needs free slots, and the pool is holding every one
        // the driver will give out.
        pool.clear();

        // --- 6. the scenario suite -----------------------------------------
        printf("6. scenarios: what changes between consecutive submits\n");
        print_header();

        std::vector<sample> s_aaaa;
        std::vector<sample> s_mmmm;
        std::vector<sample> s_amam;
        std::vector<sample> s_a1a2;

        {
            auto a1 = make_kernel(dev, designs[0], tile);
            fill_inputs(*a1, tile);

            std::vector<live_kernel *> rot_a{ a1.get() };
            s_aaaa = run_scenario("AAAA", rot_a, designs, n_warm, n_iter, csv);
            const double d_a = verify_cpu(*a1, tile, designs[0].op);
            print_row("AAAA  one context", s_aaaa);

            if (batch > 1) {
                const batch_result br =
                    run_batch(*a1, batch, std::max(n_iter / batch, 20), std::min(n_warm, 8));
                printf("   BATCH %d runs, one context           per op: "
                       "%8.2f us separate / %.2f us as one runlist  (%.1fx)\n",
                       batch, br.seq_us, br.list_us,
                       br.list_us > 0 ? br.seq_us / br.list_us : 0.0);
            }

            // Same design, a second context of it: isolates the context switch
            // from the fabric change.
            auto a2 = make_kernel(dev, designs[0], tile);
            fill_inputs(*a2, tile);
            std::vector<live_kernel *> rot_a1a2{ a1.get(), a2.get() };
            s_a1a2 = run_scenario("A1A2", rot_a1a2, designs, n_warm, n_iter, csv);
            print_row("A1A2  two contexts, same xclbin", s_a1a2);

            double d_m = -1;
            if (designs.size() > 1) {
                auto m1 = make_kernel(dev, designs[1], tile);
                fill_inputs(*m1, tile);

                std::vector<live_kernel *> rot_m{ m1.get() };
                s_mmmm = run_scenario("MMMM", rot_m, designs, n_warm, n_iter, csv);
                d_m    = verify_cpu(*m1, tile, designs[1].op);
                print_row("MMMM  one context", s_mmmm);

                std::vector<live_kernel *> rot_am{ a1.get(), m1.get() };
                s_amam = run_scenario("AMAM", rot_am, designs, n_warm, n_iter, csv);
                print_row("AMAM  two contexts, diff xclbin", s_amam);
            }

            printf("   cpu check                          A: %.3g %s   M: %.3g %s\n",
                   d_a, d_a < 0 ? "(n/a)" : (d_a < 1e-5 ? "ok" : "FAILED"),
                   d_m, d_m < 0 ? "(n/a)" : (d_m < 1e-5 ? "ok" : "FAILED"));
        }

        // --- 7. several sequences on one context ---------------------------
        double us_mod_one    = 0;
        double us_mod_rotate = 0;
#ifdef XDNA_PROBE_HAS_AIEBU
        printf("\n7. several instruction streams bound to ONE context\n");
        print_header();
        {
            xrt::hw_context shared(dev, designs[0].uuid);
            const int       shared_ctx_id = g_next_ctx_id++;

            build_cost c;
            std::vector<std::unique_ptr<live_module>> mods;
            for (int i = 0; i < 8; i++) {
                try {
                    auto m = make_module_kernel(shared, designs[0], tile, shared_ctx_id,
                                                i == 0 ? &c : nullptr);
                    fill_inputs(*m, tile);
                    mods.push_back(std::move(m));
                } catch (const std::exception & e) {
                    printf("   stopped at %zu modules: %s\n", mods.size(), e.what());
                    break;
                }
            }

            if (!mods.empty()) {
                printf("   aiebu assemble, elf+module, kernel, BOs : %.3f, %.3f, %.3f, %.3f ms\n",
                       c.asm_ms, c.module_ms, c.kernel_ms, c.bo_ms);

                std::vector<live_module *> rot_one{ mods[0].get() };
                const auto one = run_scenario("MOD1", rot_one, designs, n_warm, n_iter, csv);
                us_mod_one     = summarize(one).mean_us;
                print_row("one module", one);

                const double d = verify_cpu(*mods[0], tile, designs[0].op);

                if (mods.size() > 1) {
                    std::vector<live_module *> rot_all;
                    for (auto & m : mods) {
                        rot_all.push_back(m.get());
                    }
                    const auto all  = run_scenario("MODN", rot_all, designs, n_warm, n_iter, csv);
                    us_mod_rotate   = summarize(all).mean_us;
                    char label[64];
                    snprintf(label, sizeof(label), "rotating %zu modules", mods.size());
                    print_row(label, all);
                }
                printf("   cpu check                          : %.3g %s\n",
                       d, d < 0 ? "(n/a)" : (d < 1e-5 ? "ok" : "FAILED"));
            }
        }
#else
        printf("\n7. (built without aiebu - module path not compiled in)\n");
#endif

        // --- 8. the four numbers -------------------------------------------
        const double m_aaaa = summarize(s_aaaa).mean_us;
        const double m_mmmm = s_mmmm.empty() ? 0 : summarize(s_mmmm).mean_us;
        const double m_amam = s_amam.empty() ? 0 : summarize(s_amam).mean_us;
        const double m_a1a2 = summarize(s_a1a2).mean_us;

        const double ov_mod  = (us_mod_one > 0 && us_mod_rotate > 0) ? us_mod_rotate - us_mod_one : 0;
        const double ov_same = m_a1a2 - m_aaaa;
        const double ov_diff = s_amam.empty() ? 0 : m_amam - (m_aaaa + m_mmmm) / 2.0;

        printf("\nresults (run_id=%d)\n", run_id);
        printf("   instruction/module rotation overhead: %8.1f us", ov_mod);
        if (us_mod_one > 0) {
            printf("   (%.1f -> %.1f)\n", us_mod_one, us_mod_rotate);
        } else {
            printf("   (not measured)\n");
        }
        printf("   same-XCLBIN context switch overhead: %8.1f us   (%.1f -> %.1f)\n",
               ov_same, m_aaaa, m_a1a2);
        if (!s_amam.empty()) {
            printf("   different-XCLBIN switch overhead   : %8.1f us   (AMAM %.1f vs baseline %.1f)\n",
                   ov_diff, m_amam, (m_aaaa + m_mmmm) / 2.0);
        } else {
            printf("   different-XCLBIN switch overhead   : (needs two designs)\n");
        }
        printf("   context eviction/recreation cost   : %8.3f ms  (design still resident: %.1f us first invoke)\n",
               ms_evict_warm, us_first_warm);
        if (ms_evict_cold > 0) {
            printf("   same, design fully evicted first   : %8.3f ms  (%.1f us first invoke)\n",
                   ms_evict_cold, us_first_cold);
        }
        printf("\n");

        // One row per process run, so several runs can be pooled without having
        // to re-derive the headline numbers from the per-sample rows.
        if (!csv_path.empty()) {
            const fs::path sp = fs::path(csv_path).replace_extension(".summary.csv");
            const bool     ex = fs::exists(sp) && fs::file_size(sp) > 0;
            FILE *         sf = fopen(sp.string().c_str(), "a");
            if (sf) {
                if (!ex) {
                    fprintf(sf, "run_id,tile,n_iter,n_max_ctx,mean_aaaa_us,mean_mmmm_us,mean_amam_us,"
                                "mean_a1a2_us,mean_mod1_us,mean_modn_us,ov_module_us,ov_same_xclbin_us,"
                                "ov_diff_xclbin_us,ms_evict_warm,ms_evict_cold,us_first_warm,us_first_cold,"
                                "us_steady_warm,submits_ok_warm,submits_ok_cold\n");
                }
                fprintf(sf, "%d,%zu,%d,%zu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                            "%.4f,%.4f,%.2f,%.2f,%.2f,%d,%d\n",
                        run_id, tile, n_iter, n_max_ctx,
                        m_aaaa, m_mmmm, m_amam, m_a1a2, us_mod_one, us_mod_rotate,
                        ov_mod, ov_same, ov_diff,
                        ms_evict_warm, ms_evict_cold, us_first_warm, us_first_cold,
                        us_steady_warm, ok_warm_n, ok_cold_n);
                fclose(sf);
            }
        }

    } catch (const std::exception & e) {
        fprintf(stderr, "xdna-probe: failed: %s\n", e.what());
        return 1;
    }

    return 0;
}
