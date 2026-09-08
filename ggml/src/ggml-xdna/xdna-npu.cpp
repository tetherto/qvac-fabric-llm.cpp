#include "xdna-npu.h"

#include "ggml-impl.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

const char * ggml_xdna_op_name(enum ggml_xdna_op op) {
    switch (op) {
        case GGML_XDNA_OP_GEMM:         return "gemm";
#if defined(GGML_XDNA_EXPERIMENTAL)
        case GGML_XDNA_OP_ADD:          return "add";
        case GGML_XDNA_OP_MUL:          return "mul";
        case GGML_XDNA_OP_RMS_NORM:     return "rms-norm";
        case GGML_XDNA_OP_ADD_RMS_MUL:  return "add-rms-mul";
        case GGML_XDNA_OP_GDN:          return "gdn";
#endif
    }
    return "?";
}

#if defined(GGML_XDNA_HAS_XRT)

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

// Kernel ABI emitted by mlir-aie: MLIR_AIE(opcode, instr, n_instr, bo0, bo1, bo2).
// Data buffers start at group_id(3); opcode 3 selects the transaction flow.
static const char * XDNA_KERNEL_NAME = "MLIR_AIE";
static constexpr uint64_t XDNA_OPCODE_TXN = 3;
static constexpr int XDNA_ARG_DATA0 = 3;

// One xrt::device for the whole process. Every kernel builds its own
// hw_context on top of it, which is what lets several xclbins be resident at
// the same time; opening the device once also avoids handing the driver two
// unrelated clients for the same NPU.
struct ggml_xdna_device {
    xrt::device device;
    std::string name;
    bool        ok = false;
};

static ggml_xdna_device & ggml_xdna_shared_device(void) {
    static ggml_xdna_device dev;
    static std::once_flag   once;

    std::call_once(once, [&]() {
        try {
            dev.device = xrt::device(0u);
            dev.name   = dev.device.get_info<xrt::info::device::name>();
            dev.ok     = true;
        } catch (const std::exception & e) {
            GGML_LOG_DEBUG("%s: no XDNA device: %s\n", "ggml-xdna", e.what());
        } catch (...) {
            GGML_LOG_DEBUG("%s: no XDNA device (unknown error)\n", "ggml-xdna");
        }
    });

    return dev;
}

struct ggml_xdna_npu {
    xrt::hw_context  context;
    xrt::kernel      kernel;
    xrt::bo          bo_instr;
    xrt::bo          bo_a;
    xrt::bo          bo_b;
    xrt::bo          bo_c;
    // Optional second A/C bank for pack/submit overlap. Empty when n_banks==1.
    xrt::bo          bo_a1;
    xrt::bo          bo_c1;
    xrt::run         run;
    // Pre-built runs for runlist batching of K-slices (one per a_views entry).
    std::vector<xrt::run> runs;
    xrt::runlist *   runlist = nullptr;
    int              pending_kb_n = 0;
    size_t           pending_c_bytes = 0;
    int              pending_bank = 0;
    bool             pending_async = false;

    ggml_xdna_op     op = GGML_XDNA_OP_GEMM;

    ggml_xdna_gemm_shape shape{};
    // One sub-buffer per (weight buffer, offset). Building an xrt::bo view is
    // not free, and the same weight tensor is submitted once per prefill chunk
    // for the whole run.
    std::unordered_map<std::string, xrt::bo> w_views;

    // GEMM operands stay in device-visible memory: the host packs A and reads C
    // in place, so a submit copies nothing. One view per K-slice, so all slices
    // of a row block can be packed once and replayed for every column block.
    std::vector<xrt::bo> a_views;
    std::vector<xrt::bo> b_views;
    std::vector<xrt::bo> c_views;
    std::vector<xrt::bo> a_views1;
    std::vector<xrt::bo> c_views1;
    int                  kb_max = 1;
    int                  n_banks = 1;
    int                  bank = 0;

    uint32_t         n_instr    = 0;
    // Manual cache invalidation after bo.sync(FROM_DEVICE). Kept on by default
    // because the amdxdna path has shown stale CPU reads without it; can be
    // disabled with GGML_XDNA_CLFLUSH=0 to test whether sync() alone suffices.
    bool             invalidate = true;

#if defined(GGML_XDNA_EXPERIMENTAL)
    ggml_xdna_gdn_shape gdn{};
    size_t              tile_elems = 0;
    size_t              b_elems    = 0;
    bool                verify     = false;
    float               verify_atol = 1e-5f;
    float               verify_rtol = 1e-5f;
#endif

    ggml_xdna_npu_stats stats{};
    std::mutex          mutex;
};

static int ggml_xdna_env_int(const char * name, int def) {
    const char * val = getenv(name);
    return val ? atoi(val) : def;
}

static uint64_t ggml_xdna_now_ns(void) {
    using clock = std::chrono::steady_clock;
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        clock::now().time_since_epoch()).count();
}

static void ggml_xdna_invalidate(const void * addr, size_t bytes) {
#if defined(__x86_64__) || defined(__i386__)
    const char * end = (const char *) addr + bytes;
    const char * p   = (const char *) ((uintptr_t) addr & ~(uintptr_t) 63);

    _mm_mfence();
    for (; p < end; p += 64) {
        _mm_clflush(p);
    }
    _mm_mfence();
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(bytes);
#endif
}

static bool ggml_xdna_read_instr(const char * path, std::vector<uint32_t> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }

    const std::streamsize size = f.tellg();
    if (size <= 0 || (size % sizeof(uint32_t)) != 0) {
        return false;
    }

    out.resize((size_t) size / sizeof(uint32_t));
    f.seekg(0);
    return (bool) f.read((char *) out.data(), size);
}

bool ggml_xdna_npu_probe(char * name, size_t name_size) {
    ggml_xdna_device & dev = ggml_xdna_shared_device();
    if (name && name_size > 0) {
        snprintf(name, name_size, "%s", dev.ok ? dev.name.c_str() : "none");
    }
    return dev.ok;
}


void ggml_xdna_npu_free(ggml_xdna_npu * npu) {
    if (npu) {
        delete npu->runlist;
        npu->runlist = nullptr;
    }
    delete npu;
}

void ggml_xdna_npu_get_stats(const ggml_xdna_npu * npu, ggml_xdna_npu_stats * stats) {
    if (!stats) {
        return;
    }
    *stats = npu ? npu->stats : ggml_xdna_npu_stats{};
}


// ** GEMM **

struct ggml_xdna_buf {
    xrt::bo bo;
};

static size_t gemm_a_bytes(const ggml_xdna_gemm_shape & s) {
    return (size_t) s.grid_r * s.mb * s.kt * s.tile_m * s.tile_k * sizeof(uint16_t);
}

static size_t gemm_b_bytes(const ggml_xdna_gemm_shape & s) {
    // One column block per submit (nb is always 1 in the artifact).
    return (size_t) s.grid_c * s.nb * s.kt * s.tile_k * s.tile_n * sizeof(uint16_t);
}

static size_t gemm_c_bytes(const ggml_xdna_gemm_shape & s) {
    return (size_t) s.grid_c * s.mb * s.grid_r * s.tile_m * s.tile_n * sizeof(float);
}

int ggml_xdna_npu_gemm_m(const ggml_xdna_npu * npu) {
    return npu ? npu->shape.mb * npu->shape.grid_r * npu->shape.tile_m : 0;
}

int ggml_xdna_npu_gemm_k(const ggml_xdna_npu * npu) {
    return npu ? npu->shape.kt * npu->shape.tile_k : 0;
}

int ggml_xdna_npu_gemm_n(const ggml_xdna_npu * npu) {
    return npu ? npu->shape.nb * npu->shape.grid_c * npu->shape.tile_n : 0;
}

size_t ggml_xdna_npu_gemm_a_bytes(const ggml_xdna_npu * npu) {
    return npu ? gemm_a_bytes(npu->shape) : 0;
}

size_t ggml_xdna_npu_gemm_b_bytes(const ggml_xdna_npu * npu) {
    return npu ? gemm_b_bytes(npu->shape) : 0;
}

size_t ggml_xdna_npu_gemm_c_bytes(const ggml_xdna_npu * npu) {
    return npu ? gemm_c_bytes(npu->shape) : 0;
}

ggml_xdna_npu * ggml_xdna_npu_gemm_init(const char *                        xclbin_path,
                                        const char *                        insts_path,
                                        const struct ggml_xdna_gemm_shape * shape) {
    if (!xclbin_path || !insts_path || !shape) {
        return nullptr;
    }
    if (shape->tile_m <= 0 || shape->tile_k <= 0 || shape->tile_n <= 0 ||
        shape->grid_r <= 0 || shape->grid_c <= 0 || shape->kt <= 0 ||
        shape->nb <= 0 || shape->mb <= 0) {
        return nullptr;
    }

    ggml_xdna_device & dev = ggml_xdna_shared_device();
    if (!dev.ok) {
        return nullptr;
    }

    std::vector<uint32_t> instr;
    if (!ggml_xdna_read_instr(insts_path, instr)) {
        GGML_LOG_WARN("%s: failed to read instruction stream %s\n", __func__, insts_path);
        return nullptr;
    }

    try {
        std::unique_ptr<ggml_xdna_npu> npu(new ggml_xdna_npu);

        npu->op    = GGML_XDNA_OP_GEMM;
        npu->shape = *shape;

        xrt::xclbin xclbin_obj{std::string(xclbin_path)};
        dev.device.register_xclbin(xclbin_obj);
        npu->context = xrt::hw_context(dev.device, xclbin_obj.get_uuid());
        npu->kernel  = xrt::kernel(npu->context, XDNA_KERNEL_NAME);

        const size_t instr_bytes = instr.size() * sizeof(uint32_t);
        const size_t a_bytes     = gemm_a_bytes(*shape);
        const size_t b_bytes     = gemm_b_bytes(*shape);
        const size_t c_bytes     = gemm_c_bytes(*shape);

        // Room for every K-slice of one row block, so the host packs A once and
        // the K-loop is just a submit per slice with no data movement.
        npu->kb_max = std::max(1, ggml_xdna_env_int("GGML_XDNA_GEMM_KB_MAX", 16));
        // Production: one B buffer, every K-slice view aliases offset 0 (weights
        // come from ggml_xdna_buf, not this BO). Experimental act-x-act packs
        // a distinct B slice per K-block.
#if defined(GGML_XDNA_EXPERIMENTAL)
        const size_t b_alloc_bytes = b_bytes * (size_t) npu->kb_max;
        const bool   b_per_slice   = true;
#else
        const size_t b_alloc_bytes = b_bytes;
        const bool   b_per_slice   = false;
#endif
        // Second A/C bank lets the host pack the next M-block while the NPU
        // runs the current one. Disable with GGML_XDNA_GEMM_BANKS=1.
        npu->n_banks = std::max(1, std::min(2, ggml_xdna_env_int("GGML_XDNA_GEMM_BANKS", 2)));
        npu->bank = 0;

        npu->bo_instr = xrt::bo(dev.device, instr_bytes, XCL_BO_FLAGS_CACHEABLE, npu->kernel.group_id(1));
        npu->bo_a     = xrt::bo(dev.device, a_bytes * npu->kb_max, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 0));
        npu->bo_b     = xrt::bo(dev.device, b_alloc_bytes, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 1));
        npu->bo_c     = xrt::bo(dev.device, c_bytes * npu->kb_max, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 2));

        for (int kb = 0; kb < npu->kb_max; kb++) {
            npu->a_views.emplace_back(npu->bo_a, a_bytes, a_bytes * (size_t) kb);
            npu->b_views.emplace_back(npu->bo_b, b_bytes, b_per_slice ? b_bytes * (size_t) kb : 0);
            npu->c_views.emplace_back(npu->bo_c, c_bytes, c_bytes * (size_t) kb);
        }
        if (npu->n_banks > 1) {
            npu->bo_a1 = xrt::bo(dev.device, a_bytes * npu->kb_max, XRT_BO_FLAGS_HOST_ONLY,
                                 npu->kernel.group_id(XDNA_ARG_DATA0 + 0));
            npu->bo_c1 = xrt::bo(dev.device, c_bytes * npu->kb_max, XRT_BO_FLAGS_HOST_ONLY,
                                 npu->kernel.group_id(XDNA_ARG_DATA0 + 2));
            for (int kb = 0; kb < npu->kb_max; kb++) {
                npu->a_views1.emplace_back(npu->bo_a1, a_bytes, a_bytes * (size_t) kb);
                npu->c_views1.emplace_back(npu->bo_c1, c_bytes, c_bytes * (size_t) kb);
            }
            memset(npu->bo_a1.map<void *>(), 0, a_bytes * npu->kb_max);
            memset(npu->bo_c1.map<void *>(), 0, c_bytes * npu->kb_max);
            npu->bo_a1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        memcpy(npu->bo_instr.map<void *>(), instr.data(), instr_bytes);
        npu->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        memset(npu->bo_a.map<void *>(), 0, a_bytes * npu->kb_max);
        memset(npu->bo_b.map<void *>(), 0, b_alloc_bytes);
        memset(npu->bo_c.map<void *>(), 0, c_bytes * npu->kb_max);
        npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        npu->n_instr    = (uint32_t) instr.size();
        npu->invalidate = ggml_xdna_env_int("GGML_XDNA_CLFLUSH", 1) != 0;

        npu->run = xrt::run(npu->kernel);
        npu->run.set_arg(0, XDNA_OPCODE_TXN);
        npu->run.set_arg(1, npu->bo_instr);
        npu->run.set_arg(2, npu->n_instr);
        npu->run.set_arg(XDNA_ARG_DATA0 + 0, npu->a_views[0]);
        npu->run.set_arg(XDNA_ARG_DATA0 + 1, npu->bo_b);
        npu->run.set_arg(XDNA_ARG_DATA0 + 2, npu->c_views[0]);

        npu->runs.clear();
        npu->runs.reserve(npu->kb_max);
        for (int kb = 0; kb < npu->kb_max; kb++) {
            xrt::run r(npu->kernel);
            r.set_arg(0, XDNA_OPCODE_TXN);
            r.set_arg(1, npu->bo_instr);
            r.set_arg(2, npu->n_instr);
            r.set_arg(XDNA_ARG_DATA0 + 0, npu->a_views[kb]);
            r.set_arg(XDNA_ARG_DATA0 + 1, npu->b_views[kb]);
            r.set_arg(XDNA_ARG_DATA0 + 2, npu->c_views[kb]);
            npu->runs.push_back(std::move(r));
        }
        npu->runlist = new xrt::runlist(npu->context);

        // while_true workers need the ObjectFifo pipeline primed; zeroed
        // operands are enough for that.
        const int n_warmup = ggml_xdna_env_int("GGML_XDNA_WARMUP", 4);
        for (int i = 0; i < n_warmup; ++i) {
            npu->run.start();
            if (npu->run.wait() != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_WARN("%s: NPU gemm warm-up run %d did not complete\n", "ggml-xdna", i);
                break;
            }
            npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }

        GGML_LOG_INFO("%s: NPU gemm ready: %s, C[%d,%d] = A[%d,%d] @ B, grid %dx%d, "
                      "%.2f MiB per submit, instr=%u words\n",
                      "ggml-xdna", dev.name.c_str(),
                      ggml_xdna_npu_gemm_m(npu.get()), ggml_xdna_npu_gemm_n(npu.get()),
                      ggml_xdna_npu_gemm_m(npu.get()), ggml_xdna_npu_gemm_k(npu.get()),
                      shape->grid_r, shape->grid_c,
                      (double) (a_bytes + b_bytes + c_bytes) / (1024.0 * 1024.0),
                      npu->n_instr);

        return npu.release();
    } catch (const std::exception & e) {
        GGML_LOG_WARN("%s: NPU gemm init failed: %s\n", "ggml-xdna", e.what());
    } catch (...) {
        GGML_LOG_WARN("%s: NPU gemm init failed (unknown error)\n", "ggml-xdna");
    }

    return nullptr;
}

ggml_xdna_buf * ggml_xdna_buf_alloc(ggml_xdna_npu * npu, size_t nbytes) {
    if (!npu || nbytes == 0) {
        return nullptr;
    }

    ggml_xdna_device & dev = ggml_xdna_shared_device();
    if (!dev.ok) {
        return nullptr;
    }

    try {
        std::unique_ptr<ggml_xdna_buf> buf(new ggml_xdna_buf);
        buf->bo = xrt::bo(dev.device, nbytes, XRT_BO_FLAGS_HOST_ONLY,
                          npu->kernel.group_id(XDNA_ARG_DATA0 + 1));
        memset(buf->bo.map<void *>(), 0, nbytes);
        return buf.release();
    } catch (const std::exception & e) {
        GGML_LOG_WARN("%s: weight buffer of %zu bytes failed: %s\n", "ggml-xdna", nbytes, e.what());
    } catch (...) {
        GGML_LOG_WARN("%s: weight buffer of %zu bytes failed\n", "ggml-xdna", nbytes);
    }

    return nullptr;
}

void ggml_xdna_buf_free(ggml_xdna_buf * buf) {
    delete buf;
}

void * ggml_xdna_buf_host(ggml_xdna_buf * buf) {
    return buf ? buf->bo.map<void *>() : nullptr;
}

void ggml_xdna_buf_sync_to_device(ggml_xdna_buf * buf, size_t offset, size_t nbytes) {
    if (buf && nbytes) {
        buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, nbytes, offset);
    }
}

int ggml_xdna_npu_gemm_kb_max(const ggml_xdna_npu * npu) {
    return npu ? npu->kb_max : 0;
}

int ggml_xdna_npu_gemm_n_banks(const ggml_xdna_npu * npu) {
    return npu ? npu->n_banks : 0;
}

void ggml_xdna_npu_gemm_set_bank(ggml_xdna_npu * npu, int bank) {
    if (npu && bank >= 0 && bank < npu->n_banks) {
        npu->bank = bank;
    }
}

static xrt::bo & ggml_xdna_gemm_a_bo(ggml_xdna_npu * npu, int bank) {
    return (bank == 1 && npu->n_banks > 1) ? npu->bo_a1 : npu->bo_a;
}

static xrt::bo & ggml_xdna_gemm_c_bo(ggml_xdna_npu * npu, int bank) {
    return (bank == 1 && npu->n_banks > 1) ? npu->bo_c1 : npu->bo_c;
}

static std::vector<xrt::bo> & ggml_xdna_gemm_a_views(ggml_xdna_npu * npu, int bank) {
    return (bank == 1 && npu->n_banks > 1) ? npu->a_views1 : npu->a_views;
}

static std::vector<xrt::bo> & ggml_xdna_gemm_c_views(ggml_xdna_npu * npu, int bank) {
    return (bank == 1 && npu->n_banks > 1) ? npu->c_views1 : npu->c_views;
}

uint16_t * ggml_xdna_npu_gemm_a_slice_bank(ggml_xdna_npu * npu, int bank, int k_block) {
    if (!npu || k_block < 0 || k_block >= npu->kb_max || bank < 0 || bank >= npu->n_banks) {
        return nullptr;
    }
    return (uint16_t *) (ggml_xdna_gemm_a_bo(npu, bank).map<char *>() +
                         gemm_a_bytes(npu->shape) * (size_t) k_block);
}

const float * ggml_xdna_npu_gemm_c_slice_bank(ggml_xdna_npu * npu, int bank, int k_block) {
    if (!npu || k_block < 0 || k_block >= npu->kb_max || bank < 0 || bank >= npu->n_banks) {
        return nullptr;
    }
    return (const float *) (ggml_xdna_gemm_c_bo(npu, bank).map<char *>() +
                            gemm_c_bytes(npu->shape) * (size_t) k_block);
}

uint16_t * ggml_xdna_npu_gemm_a_slice(ggml_xdna_npu * npu, int k_block) {
    return npu ? ggml_xdna_npu_gemm_a_slice_bank(npu, npu->bank, k_block) : nullptr;
}

const float * ggml_xdna_npu_gemm_c_slice(ggml_xdna_npu * npu, int k_block) {
    return npu ? ggml_xdna_npu_gemm_c_slice_bank(npu, npu->bank, k_block) : nullptr;
}

void ggml_xdna_npu_gemm_acquire(ggml_xdna_npu * npu) {
    if (npu) {
        npu->mutex.lock();
    }
}

void ggml_xdna_npu_gemm_release(ggml_xdna_npu * npu) {
    if (npu) {
        npu->mutex.unlock();
    }
}

static bool ggml_xdna_npu_gemm_submit(ggml_xdna_npu * npu,
                                      ggml_xdna_buf * buf,
                                      size_t          w_off,
                                      int             k_block) {
    if (!npu || !buf || k_block < 0 || k_block >= npu->kb_max) {
        return false;
    }

    const size_t b_bytes = gemm_b_bytes(npu->shape);
    const size_t c_bytes = gemm_c_bytes(npu->shape);
    const int bank = npu->bank;
    auto & a_views = ggml_xdna_gemm_a_views(npu, bank);
    auto & c_views = ggml_xdna_gemm_c_views(npu, bank);

    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        char key[64];
        snprintf(key, sizeof(key), "%p+%zu", (const void *) buf, w_off);

        auto it = npu->w_views.find(key);
        if (it == npu->w_views.end()) {
            it = npu->w_views.emplace(key, xrt::bo(buf->bo, b_bytes, w_off)).first;
        }

        npu->run.set_arg(XDNA_ARG_DATA0 + 0, a_views[k_block]);
        npu->run.set_arg(XDNA_ARG_DATA0 + 1, it->second);
        npu->run.set_arg(XDNA_ARG_DATA0 + 2, c_views[k_block]);

        a_views[k_block].sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const uint64_t t_submit = ggml_xdna_now_ns();
        npu->run.start();
        const ert_cmd_state state = npu->run.wait();
        npu->stats.ns_submit += ggml_xdna_now_ns() - t_submit;
        npu->stats.n_runs++;

        if (state != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("%s: NPU gemm run did not complete\n", "ggml-xdna");
            return false;
        }

        c_views[k_block].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (npu->invalidate) {
            ggml_xdna_invalidate(ggml_xdna_gemm_c_bo(npu, bank).map<char *>() +
                                 c_bytes * (size_t) k_block, c_bytes);
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU gemm failed: %s\n", "ggml-xdna", e.what());
        return false;
    }

    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;

    return true;
}

bool ggml_xdna_npu_gemm_submit_list_async(ggml_xdna_npu * npu,
                                          ggml_xdna_buf * buf,
                                          const size_t *  w_offs,
                                          int             kb_n) {
    if (!npu || !buf || !w_offs || kb_n < 1 || kb_n > npu->kb_max) {
        return false;
    }
    if (npu->pending_async) {
        if (!ggml_xdna_npu_gemm_runlist_wait(npu)) {
            return false;
        }
    }
    if (kb_n == 1 || !npu->runlist || (int) npu->runs.size() < kb_n) {
        // Single-slice path is still synchronous.
        const bool ok = ggml_xdna_npu_gemm_submit(npu, buf, w_offs[0], 0);
        return ok;
    }

    const size_t b_bytes = gemm_b_bytes(npu->shape);
    const size_t c_bytes = gemm_c_bytes(npu->shape);
    const int bank = npu->bank;
    auto & a_views = ggml_xdna_gemm_a_views(npu, bank);
    auto & c_views = ggml_xdna_gemm_c_views(npu, bank);
    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        npu->runlist->reset();
        for (int kb = 0; kb < kb_n; kb++) {
            char key[64];
            snprintf(key, sizeof(key), "%p+%zu", (const void *) buf, w_offs[kb]);
            auto it = npu->w_views.find(key);
            if (it == npu->w_views.end()) {
                it = npu->w_views.emplace(key, xrt::bo(buf->bo, b_bytes, w_offs[kb])).first;
            }
            npu->runs[kb].set_arg(XDNA_ARG_DATA0 + 0, a_views[kb]);
            npu->runs[kb].set_arg(XDNA_ARG_DATA0 + 1, it->second);
            npu->runs[kb].set_arg(XDNA_ARG_DATA0 + 2, c_views[kb]);
            a_views[kb].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            npu->runlist->add(npu->runs[kb]);
        }

        const uint64_t t_submit = ggml_xdna_now_ns();
        npu->runlist->execute();
        // Wait time is attributed in runlist_wait; execute() itself is cheap.
        npu->stats.n_runs += (uint64_t) kb_n;
        npu->pending_kb_n = kb_n;
        npu->pending_c_bytes = c_bytes;
        npu->pending_bank = bank;
        npu->pending_async = true;
        npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
        npu->stats.n_calls++;
        GGML_UNUSED(t_submit);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU gemm runlist async failed: %s\n", "ggml-xdna", e.what());
        return false;
    }
}

bool ggml_xdna_npu_gemm_runlist_wait(ggml_xdna_npu * npu) {
    if (!npu || !npu->pending_async) {
        return true;
    }
    const uint64_t t0 = ggml_xdna_now_ns();
    try {
        npu->runlist->wait();
        const uint64_t t_waited = ggml_xdna_now_ns();
        npu->stats.ns_submit += t_waited - t0;
        auto & c_views = ggml_xdna_gemm_c_views(npu, npu->pending_bank);
        auto & c_bo = ggml_xdna_gemm_c_bo(npu, npu->pending_bank);
        for (int kb = 0; kb < npu->pending_kb_n; kb++) {
            c_views[kb].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (npu->invalidate) {
                ggml_xdna_invalidate(c_bo.map<char *>() +
                                     npu->pending_c_bytes * (size_t) kb,
                                     npu->pending_c_bytes);
            }
        }
        const uint64_t t_end = ggml_xdna_now_ns();
        npu->stats.ns_sync  += t_end - t_waited;
        npu->stats.ns_total += t_end - t0;
        npu->pending_async = false;
        npu->pending_kb_n = 0;
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU gemm runlist wait failed: %s\n", "ggml-xdna", e.what());
        npu->pending_async = false;
        return false;
    }
}

#if defined(GGML_XDNA_EXPERIMENTAL)
static float ggml_xdna_env_float(const char * name, float def) {
    const char * val = getenv(name);
    return val ? (float) atof(val) : def;
}

static float ggml_xdna_apply(ggml_xdna_op op, float a, float b) {
    return op == GGML_XDNA_OP_MUL ? a * b : a + b;
}

ggml_xdna_npu * ggml_xdna_npu_init(enum ggml_xdna_op op,
                                   const char *      xclbin_path,
                                   const char *      insts_path,
                                   size_t            tile_elems) {
    if (!xclbin_path || !insts_path || tile_elems == 0) {
        return nullptr;
    }

    ggml_xdna_device & dev = ggml_xdna_shared_device();
    if (!dev.ok) {
        return nullptr;
    }

    const size_t b_elems    = (op == GGML_XDNA_OP_RMS_NORM) ? 1 : tile_elems;
    const size_t data_bytes = tile_elems * sizeof(float);
    const size_t b_bytes    = b_elems * sizeof(float);
    const bool   is_fused   = op == GGML_XDNA_OP_ADD_RMS_MUL;
    const size_t ab_bytes   = is_fused ? (2 * tile_elems + 1) * sizeof(float) : 0;
    const size_t out_bytes  = is_fused ? 2 * data_bytes : data_bytes;

    std::vector<uint32_t> instr;
    if (!ggml_xdna_read_instr(insts_path, instr)) {
        GGML_LOG_WARN("%s: failed to read instruction stream %s\n", __func__, insts_path);
        return nullptr;
    }

    try {
        // RAII: the unique_ptr frees the partially constructed NPU (and its
        // XRT resources) if any step below throws. Ownership is released to
        // the caller only on the success path.
        std::unique_ptr<ggml_xdna_npu> npu(new ggml_xdna_npu);

        npu->op = op;

        xrt::xclbin xclbin_obj{std::string(xclbin_path)};
        dev.device.register_xclbin(xclbin_obj);
        npu->context = xrt::hw_context(dev.device, xclbin_obj.get_uuid());
        npu->kernel  = xrt::kernel(npu->context, XDNA_KERNEL_NAME);

        const size_t instr_bytes = instr.size() * sizeof(uint32_t);

        npu->bo_instr = xrt::bo(dev.device, instr_bytes, XCL_BO_FLAGS_CACHEABLE, npu->kernel.group_id(1));
        if (is_fused) {
            npu->bo_a = xrt::bo(dev.device, ab_bytes,   XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 0));
            npu->bo_b = xrt::bo(dev.device, data_bytes, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 1));
            npu->bo_c = xrt::bo(dev.device, out_bytes,  XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 2));
        } else {
            npu->bo_a = xrt::bo(dev.device, data_bytes, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 0));
            npu->bo_b = xrt::bo(dev.device, b_bytes,    XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 1));
            npu->bo_c = xrt::bo(dev.device, data_bytes, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 2));
        }

        memcpy(npu->bo_instr.map<void *>(), instr.data(), instr_bytes);
        npu->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        memset(npu->bo_a.map<void *>(), 0, is_fused ? ab_bytes : data_bytes);
        if (!is_fused) {
            memset(npu->bo_b.map<void *>(), 0, b_bytes);
        } else {
            memset(npu->bo_b.map<void *>(), 0, data_bytes);
        }
        memset(npu->bo_c.map<void *>(), 0, is_fused ? out_bytes : data_bytes);

        npu->n_instr    = (uint32_t) instr.size();
        npu->tile_elems = tile_elems;
        npu->b_elems    = b_elems;

        const char * verify_env = getenv("GGML_XDNA_VERIFY");
        npu->verify      = verify_env && atoi(verify_env) != 0;
        npu->verify_atol = ggml_xdna_env_float("GGML_XDNA_VERIFY_ATOL", 1e-5f);
        npu->verify_rtol = ggml_xdna_env_float("GGML_XDNA_VERIFY_RTOL", 1e-5f);
        npu->invalidate  = ggml_xdna_env_int("GGML_XDNA_CLFLUSH", 1) != 0;

        npu->run = xrt::run(npu->kernel);
        npu->run.set_arg(0, XDNA_OPCODE_TXN);
        npu->run.set_arg(1, npu->bo_instr);
        npu->run.set_arg(2, npu->n_instr);
        npu->run.set_arg(XDNA_ARG_DATA0 + 0, npu->bo_a);
        npu->run.set_arg(XDNA_ARG_DATA0 + 1, npu->bo_b);
        npu->run.set_arg(XDNA_ARG_DATA0 + 2, npu->bo_c);

        // The design keeps its worker alive across submissions (while_true),
        // so the ObjectFifo pipeline has to be primed.
        const int n_warmup = ggml_xdna_env_int("GGML_XDNA_WARMUP", 4);
        if (op == GGML_XDNA_OP_RMS_NORM) {
            npu->bo_b.map<float *>()[0] = 1.0f;
        } else if (is_fused) {
            npu->bo_a.map<float *>()[2 * tile_elems] = 1.0f;
        }
        npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int i = 0; i < n_warmup; ++i) {
            npu->run.start();
            if (npu->run.wait() != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_WARN("%s: NPU %s warm-up run %d did not complete\n",
                              "ggml-xdna", ggml_xdna_op_name(op), i);
                break;
            }
            npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }

        GGML_LOG_INFO("%s: NPU %s ready: %s, kernel=%s, tile=%zu f32, instr=%u words\n",
                      "ggml-xdna", ggml_xdna_op_name(op), dev.name.c_str(),
                      XDNA_KERNEL_NAME, tile_elems, npu->n_instr);
        return npu.release();
    } catch (const std::exception & e) {
        GGML_LOG_WARN("%s: NPU %s init failed: %s\n", "ggml-xdna", ggml_xdna_op_name(op), e.what());
        return nullptr;
    } catch (...) {
        GGML_LOG_WARN("%s: NPU %s init failed (unknown error)\n", "ggml-xdna", ggml_xdna_op_name(op));
        return nullptr;
    }
}

bool ggml_xdna_npu_binary_f32(ggml_xdna_npu * npu, const float * a, const float * b, float * dst, size_t n) {
    if (!npu || !a || !b || !dst) {
        return false;
    }

    const size_t tile = npu->tile_elems;

    std::lock_guard<std::mutex> lock(npu->mutex);

    float * ha = npu->bo_a.map<float *>();
    float * hb = npu->bo_b.map<float *>();
    float * hc = npu->bo_c.map<float *>();

    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        for (size_t off = 0; off < n; off += tile) {
            const size_t cur = std::min(tile, n - off);

            memcpy(ha, a + off, cur * sizeof(float));
            memcpy(hb, b + off, cur * sizeof(float));
            if (cur < tile) {
                memset(ha + cur, 0, (tile - cur) * sizeof(float));
                memset(hb + cur, 0, (tile - cur) * sizeof(float));
            }

            npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            const uint64_t t_submit = ggml_xdna_now_ns();
            npu->run.start();
            const ert_cmd_state state = npu->run.wait();
            npu->stats.ns_submit += ggml_xdna_now_ns() - t_submit;

            if (state != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_ERROR("%s: NPU %s run did not complete\n",
                               "ggml-xdna", ggml_xdna_op_name(npu->op));
                return false;
            }

            npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (npu->invalidate) {
                ggml_xdna_invalidate(hc, cur * sizeof(float));
            }
            memcpy(dst + off, hc, cur * sizeof(float));

            npu->stats.n_runs++;

            if (npu->verify) {
                size_t n_bad     = 0;
                size_t first_bad = 0;
                for (size_t i = 0; i < cur; i++) {
                    const float want = ggml_xdna_apply(npu->op, ha[i], hb[i]);
                    // Combined absolute + relative tolerance (atol + rtol*|want|),
                    // matching ggml's test-backend-ops convention.
                    const float tol = npu->verify_atol + npu->verify_rtol * std::fabs(want);
                    if (std::fabs(dst[off + i] - want) > tol) {
                        if (n_bad == 0) {
                            first_bad = i;
                        }
                        n_bad++;
                    }
                }
                if (n_bad) {
                    GGML_LOG_ERROR("%s: verify %s: offset %zu: %zu/%zu differ, first %zu (got %.9g, want %.9g)\n",
                                   "ggml-xdna", ggml_xdna_op_name(npu->op), off, n_bad, cur, first_bad,
                                   dst[off + first_bad],
                                   ggml_xdna_apply(npu->op, ha[first_bad], hb[first_bad]));
                }
            }
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU %s failed: %s\n", "ggml-xdna", ggml_xdna_op_name(npu->op), e.what());
        return false;
    }

    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;

    return true;
}

bool ggml_xdna_npu_rms_norm_f32(ggml_xdna_npu * npu, const float * x, float * dst, int64_t n_rows, float eps) {
    if (!npu || !x || !dst || n_rows <= 0) {
        return false;
    }

    const size_t ne0 = npu->tile_elems;

    std::lock_guard<std::mutex> lock(npu->mutex);

    float * hx = npu->bo_a.map<float *>();
    float * he = npu->bo_b.map<float *>();
    float * hy = npu->bo_c.map<float *>();

    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        if (he[0] != eps) {
            he[0] = eps;
            npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        for (int64_t r = 0; r < n_rows; r++) {
            const float * row_in  = x   + (size_t) r * ne0;
            float *       row_out = dst + (size_t) r * ne0;

            memcpy(hx, row_in, ne0 * sizeof(float));
            npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            const uint64_t t_submit = ggml_xdna_now_ns();
            npu->run.start();
            const ert_cmd_state state = npu->run.wait();
            npu->stats.ns_submit += ggml_xdna_now_ns() - t_submit;

            if (state != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_ERROR("%s: NPU %s run did not complete\n",
                               "ggml-xdna", ggml_xdna_op_name(npu->op));
                return false;
            }

            npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (npu->invalidate) {
                ggml_xdna_invalidate(hy, ne0 * sizeof(float));
            }
            memcpy(row_out, hy, ne0 * sizeof(float));

            npu->stats.n_runs++;

            if (npu->verify) {
                double sum = 0.0;
                for (size_t i = 0; i < ne0; i++) {
                    sum += (double) hx[i] * (double) hx[i];
                }
                const float scale = 1.0f / std::sqrt((float) (sum / (double) ne0) + eps);

                size_t n_bad     = 0;
                size_t first_bad = 0;
                for (size_t i = 0; i < ne0; i++) {
                    const float want = hx[i] * scale;
                    const float tol  = npu->verify_atol + npu->verify_rtol * std::fabs(want);
                    if (std::fabs(row_out[i] - want) > tol) {
                        if (n_bad == 0) {
                            first_bad = i;
                        }
                        n_bad++;
                    }
                }
                if (n_bad) {
                    GGML_LOG_ERROR("%s: verify %s: row %lld: %zu/%zu differ, first %zu (got %.9g, want %.9g)\n",
                                   "ggml-xdna", ggml_xdna_op_name(npu->op), (long long) r, n_bad, ne0, first_bad,
                                   row_out[first_bad], hx[first_bad] * scale);
                }
            }
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU %s failed: %s\n", "ggml-xdna", ggml_xdna_op_name(npu->op), e.what());
        return false;
    }

    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;

    return true;
}

bool ggml_xdna_npu_add_rms_mul_f32(ggml_xdna_npu * npu,
                                   const float *  a,
                                   const float *  b,
                                   const float *  weight,
                                   float *        add_dst,
                                   float *        mul_dst,
                                   int64_t        n_rows,
                                   int64_t        weight_n_rows,
                                   float          eps) {
    if (!npu || !a || !b || !weight || !add_dst || !mul_dst || n_rows <= 0) {
        return false;
    }

    const size_t ne0 = npu->tile_elems;
    const bool   w_broadcast = weight_n_rows <= 1;

    std::lock_guard<std::mutex> lock(npu->mutex);

    float * hab  = npu->bo_a.map<float *>();
    float * hw   = npu->bo_b.map<float *>();
    float * hout = npu->bo_c.map<float *>();

    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        float cached_eps = std::numeric_limits<float>::quiet_NaN();

        for (int64_t r = 0; r < n_rows; r++) {
            const float * row_a  = a        + (size_t) r * ne0;
            const float * row_b  = b        + (size_t) r * ne0;
            const float * row_w  = w_broadcast ? weight : weight + (size_t) r * ne0;
            float *       row_add = add_dst + (size_t) r * ne0;
            float *       row_mul = mul_dst + (size_t) r * ne0;

            memcpy(hab, row_a, ne0 * sizeof(float));
            memcpy(hab + ne0, row_b, ne0 * sizeof(float));
            if (cached_eps != eps) {
                hab[2 * ne0] = eps;
                cached_eps   = eps;
            }
            memcpy(hw, row_w, ne0 * sizeof(float));
            npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            const uint64_t t_submit = ggml_xdna_now_ns();
            npu->run.start();
            const ert_cmd_state state = npu->run.wait();
            npu->stats.ns_submit += ggml_xdna_now_ns() - t_submit;

            if (state != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_ERROR("%s: NPU %s run did not complete\n",
                               "ggml-xdna", ggml_xdna_op_name(npu->op));
                return false;
            }

            npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (npu->invalidate) {
                ggml_xdna_invalidate(hout, 2 * ne0 * sizeof(float));
            }
            memcpy(row_add, hout, ne0 * sizeof(float));
            memcpy(row_mul, hout + ne0, ne0 * sizeof(float));

            npu->stats.n_runs++;

            if (npu->verify) {
                double sum = 0.0;
                for (size_t i = 0; i < ne0; i++) {
                    const float s = row_a[i] + row_b[i];
                    sum += (double) s * (double) s;
                }
                const float scale = 1.0f / std::sqrt((float) (sum / (double) ne0) + eps);

                size_t n_bad_add = 0;
                size_t n_bad_mul = 0;
                for (size_t i = 0; i < ne0; i++) {
                    const float want_add = row_a[i] + row_b[i];
                    const float want_mul = want_add * scale * row_w[i];
                    const float tol_add  = npu->verify_atol + npu->verify_rtol * std::fabs(want_add);
                    const float tol_mul  = npu->verify_atol + npu->verify_rtol * std::fabs(want_mul);
                    if (std::fabs(row_add[i] - want_add) > tol_add) {
                        n_bad_add++;
                    }
                    if (std::fabs(row_mul[i] - want_mul) > tol_mul) {
                        n_bad_mul++;
                    }
                }
                if (n_bad_add) {
                    GGML_LOG_ERROR("%s: verify %s: row %lld add: %zu/%zu differ\n",
                                   "ggml-xdna", ggml_xdna_op_name(npu->op),
                                   (long long) r, n_bad_add, ne0);
                }
                if (n_bad_mul) {
                    GGML_LOG_ERROR("%s: verify %s: row %lld mul: %zu/%zu differ\n",
                                   "ggml-xdna", ggml_xdna_op_name(npu->op),
                                   (long long) r, n_bad_mul, ne0);
                }
            }
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU %s failed: %s\n", "ggml-xdna", ggml_xdna_op_name(npu->op), e.what());
        return false;
    }

    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;

    return true;
}

bool ggml_xdna_npu_gemm_submit_ab_list(ggml_xdna_npu * npu, int kb_n) {
    if (!npu || kb_n < 1 || kb_n > npu->kb_max) {
        return false;
    }
    if (kb_n == 1 || !npu->runlist || (int) npu->runs.size() < kb_n ||
        (int) npu->b_views.size() < kb_n) {
        for (int kb = 0; kb < kb_n; kb++) {
            if (!ggml_xdna_npu_gemm_submit_ab(npu, kb)) {
                return false;
            }
        }
        return true;
    }

    const size_t c_bytes = gemm_c_bytes(npu->shape);
    const int bank = npu->bank;
    auto & a_views = ggml_xdna_gemm_a_views(npu, bank);
    auto & c_views = ggml_xdna_gemm_c_views(npu, bank);
    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        npu->runlist->reset();
        for (int kb = 0; kb < kb_n; kb++) {
            npu->runs[kb].set_arg(XDNA_ARG_DATA0 + 0, a_views[kb]);
            npu->runs[kb].set_arg(XDNA_ARG_DATA0 + 1, npu->b_views[kb]);
            npu->runs[kb].set_arg(XDNA_ARG_DATA0 + 2, c_views[kb]);
            a_views[kb].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            npu->b_views[kb].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            npu->runlist->add(npu->runs[kb]);
        }
        const uint64_t t_submit = ggml_xdna_now_ns();
        npu->runlist->execute();
        npu->runlist->wait();
        const uint64_t t_waited = ggml_xdna_now_ns();
        npu->stats.ns_submit += t_waited - t_submit;
        npu->stats.n_runs += (uint64_t) kb_n;
        for (int kb = 0; kb < kb_n; kb++) {
            c_views[kb].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (npu->invalidate) {
                ggml_xdna_invalidate(ggml_xdna_gemm_c_bo(npu, bank).map<char *>() +
                                     c_bytes * (size_t) kb, c_bytes);
            }
        }
        npu->stats.ns_sync += ggml_xdna_now_ns() - t_waited;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU gemm_ab runlist failed: %s\n", "ggml-xdna", e.what());
        return false;
    }
    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;
    return true;
}

uint16_t * ggml_xdna_npu_gemm_b_slice(ggml_xdna_npu * npu, int k_block) {
    if (!npu || k_block < 0 || k_block >= npu->kb_max ||
        k_block >= (int) npu->b_views.size()) {
        return nullptr;
    }
    return (uint16_t *) npu->b_views[k_block].map<void *>();
}

uint16_t * ggml_xdna_npu_gemm_b_host(ggml_xdna_npu * npu) {
    return ggml_xdna_npu_gemm_b_slice(npu, 0);
}

bool ggml_xdna_npu_gemm_submit_ab(ggml_xdna_npu * npu, int k_block) {
    if (!npu || k_block < 0 || k_block >= npu->kb_max ||
        k_block >= (int) npu->b_views.size()) {
        return false;
    }

    const size_t c_bytes = gemm_c_bytes(npu->shape);
    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        npu->run.set_arg(XDNA_ARG_DATA0 + 0, npu->a_views[k_block]);
        npu->run.set_arg(XDNA_ARG_DATA0 + 1, npu->b_views[k_block]);
        npu->run.set_arg(XDNA_ARG_DATA0 + 2, npu->c_views[k_block]);

        npu->a_views[k_block].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        npu->b_views[k_block].sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const uint64_t t_submit = ggml_xdna_now_ns();
        npu->run.start();
        const ert_cmd_state state = npu->run.wait();
        npu->stats.ns_submit += ggml_xdna_now_ns() - t_submit;
        npu->stats.n_runs++;

        if (state != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("%s: NPU gemm_ab run did not complete\n", "ggml-xdna");
            return false;
        }

        npu->c_views[k_block].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (npu->invalidate) {
            ggml_xdna_invalidate(npu->bo_c.map<char *>() + c_bytes * (size_t) k_block, c_bytes);
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU gemm_ab failed: %s\n", "ggml-xdna", e.what());
        return false;
    }

    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;

    return true;
}

ggml_xdna_npu * ggml_xdna_npu_gdn_init(const char * xclbin_path,
                                       const char * insts_path,
                                       const struct ggml_xdna_gdn_shape * shape) {
    if (!xclbin_path || !insts_path || !shape ||
        shape->S <= 0 || shape->ROWS <= 0 || shape->CS <= 0) {
        return nullptr;
    }

    ggml_xdna_device & dev = ggml_xdna_shared_device();
    if (!dev.ok) {
        return nullptr;
    }

    const int S = shape->S;
    const int ROWS = shape->ROWS;
    const int CS = shape->CS;
    const int workers = shape->workers > 0 ? shape->workers : 1;
    const size_t tok_elems = (size_t) CS * (size_t) (3 * S + 2);
    const size_t strip_elems = (size_t) ROWS * (size_t) S;
    const size_t state_elems = (size_t) workers * strip_elems;
    const size_t attn_elems = (size_t) CS * (size_t) ROWS * (size_t) workers;

    std::vector<uint32_t> instr;
    if (!ggml_xdna_read_instr(insts_path, instr)) {
        GGML_LOG_WARN("%s: failed to read instruction stream %s\n", __func__, insts_path);
        return nullptr;
    }

    try {
        std::unique_ptr<ggml_xdna_npu> npu(new ggml_xdna_npu);
        npu->op = GGML_XDNA_OP_GDN;
        npu->gdn = *shape;
        if (npu->gdn.workers <= 0) {
            npu->gdn.workers = workers;
        }

        xrt::xclbin xclbin_obj{std::string(xclbin_path)};
        dev.device.register_xclbin(xclbin_obj);
        npu->context = xrt::hw_context(dev.device, xclbin_obj.get_uuid());
        npu->kernel  = xrt::kernel(npu->context, XDNA_KERNEL_NAME);

        const size_t instr_bytes = instr.size() * sizeof(uint32_t);
        const size_t tok_bytes = tok_elems * sizeof(float);
        const size_t strip_bytes = state_elems * sizeof(float);
        const size_t out_bytes = attn_elems * sizeof(float);

        npu->bo_instr = xrt::bo(dev.device, instr_bytes, XCL_BO_FLAGS_CACHEABLE, npu->kernel.group_id(1));
        npu->bo_a = xrt::bo(dev.device, tok_bytes, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 0));
        npu->bo_b = xrt::bo(dev.device, strip_bytes, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 1));
        npu->bo_c = xrt::bo(dev.device, out_bytes, XRT_BO_FLAGS_HOST_ONLY, npu->kernel.group_id(XDNA_ARG_DATA0 + 2));

        memcpy(npu->bo_instr.map<void *>(), instr.data(), instr_bytes);
        npu->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memset(npu->bo_a.map<void *>(), 0, tok_bytes);
        memset(npu->bo_b.map<void *>(), 0, strip_bytes);
        memset(npu->bo_c.map<void *>(), 0, out_bytes);

        npu->n_instr = (uint32_t) instr.size();
        npu->tile_elems = strip_elems;
        npu->invalidate = ggml_xdna_env_int("GGML_XDNA_CLFLUSH", 1) != 0;

        npu->run = xrt::run(npu->kernel);
        npu->run.set_arg(0, XDNA_OPCODE_TXN);
        npu->run.set_arg(1, npu->bo_instr);
        npu->run.set_arg(2, npu->n_instr);
        npu->run.set_arg(XDNA_ARG_DATA0 + 0, npu->bo_a);
        npu->run.set_arg(XDNA_ARG_DATA0 + 1, npu->bo_b);
        npu->run.set_arg(XDNA_ARG_DATA0 + 2, npu->bo_c);

        const int n_warmup = ggml_xdna_env_int("GGML_XDNA_WARMUP", 2);
        npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int i = 0; i < n_warmup; ++i) {
            npu->run.start();
            if (npu->run.wait() != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_WARN("%s: NPU gdn warm-up run %d did not complete\n", "ggml-xdna", i);
                break;
            }
            npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }

        GGML_LOG_INFO("%s: NPU gdn ready: %s, S=%d ROWS=%d CS=%d workers=%d, instr=%u words\n",
                      "ggml-xdna", dev.name.c_str(), S, ROWS, CS, npu->gdn.workers, npu->n_instr);
        return npu.release();
    } catch (const std::exception & e) {
        GGML_LOG_WARN("%s: NPU gdn init failed: %s\n", "ggml-xdna", e.what());
        return nullptr;
    } catch (...) {
        GGML_LOG_WARN("%s: NPU gdn init failed (unknown error)\n", "ggml-xdna");
        return nullptr;
    }
}

bool ggml_xdna_npu_gdn_submit(ggml_xdna_npu * npu,
                              const float *   tok,
                              const float *   state_strip,
                              float *         attn_strip,
                              float *         new_state_strip) {
    if (!npu || npu->op != GGML_XDNA_OP_GDN || !tok || !state_strip ||
        !attn_strip || !new_state_strip) {
        return false;
    }
    if (npu->gdn.workers > 1) {
        return false;
    }

    const int S = npu->gdn.S;
    const int ROWS = npu->gdn.ROWS;
    const int CS = npu->gdn.CS;
    const size_t tok_elems = (size_t) CS * (size_t) (3 * S + 2);
    const size_t strip_elems = (size_t) ROWS * (size_t) S;
    const size_t attn_elems = (size_t) CS * (size_t) ROWS;

    std::lock_guard<std::mutex> lock(npu->mutex);
    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        float * ha = npu->bo_a.map<float *>();
        float * hb = npu->bo_b.map<float *>();
        float * hc = npu->bo_c.map<float *>();

        memcpy(ha, tok, tok_elems * sizeof(float));
        memcpy(hb, state_strip, strip_elems * sizeof(float));

        npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const uint64_t t_submit = ggml_xdna_now_ns();
        npu->run.start();
        const ert_cmd_state state = npu->run.wait();
        npu->stats.ns_submit += ggml_xdna_now_ns() - t_submit;
        npu->stats.n_runs++;

        if (state != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("%s: NPU gdn run did not complete\n", "ggml-xdna");
            return false;
        }

        npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        npu->bo_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (npu->invalidate) {
            ggml_xdna_invalidate(hc, attn_elems * sizeof(float));
            ggml_xdna_invalidate(hb, strip_elems * sizeof(float));
        }

        memcpy(attn_strip, hc, attn_elems * sizeof(float));
        memcpy(new_state_strip, hb, strip_elems * sizeof(float));
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU gdn failed: %s\n", "ggml-xdna", e.what());
        return false;
    }

    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;
    return true;
}

bool ggml_xdna_npu_gdn_submit_full(ggml_xdna_npu * npu,
                                   const float *   tok,
                                   const float *   state,
                                   float *         attn,
                                   float *         new_state) {
    return ggml_xdna_npu_gdn_submit_full_chunks(npu, tok, 1, state, attn, new_state);
}

bool ggml_xdna_npu_gdn_submit_full_chunks(ggml_xdna_npu * npu,
                                          const float *   tok_chunks,
                                          int             n_chunks,
                                          const float *   state,
                                          float *         attn_chunks,
                                          float *         new_state) {
    if (!npu || npu->op != GGML_XDNA_OP_GDN || !tok_chunks || n_chunks < 1 ||
        !state || !attn_chunks || !new_state) {
        return false;
    }

    const int S = npu->gdn.S;
    const int ROWS = npu->gdn.ROWS;
    const int CS = npu->gdn.CS;
    const int workers = npu->gdn.workers > 0 ? npu->gdn.workers : 1;
    if (workers <= 1) {
        return false;
    }
    const size_t tok_elems = (size_t) CS * (size_t) (3 * S + 2);
    const size_t strip_elems = (size_t) ROWS * (size_t) S;
    const size_t state_elems = (size_t) workers * strip_elems;
    const size_t attn_elems = (size_t) CS * (size_t) ROWS * (size_t) workers;

    std::lock_guard<std::mutex> lock(npu->mutex);
    const uint64_t t_begin = ggml_xdna_now_ns();

    try {
        float * ha = npu->bo_a.map<float *>();
        float * hb = npu->bo_b.map<float *>();
        float * hc = npu->bo_c.map<float *>();

        memcpy(hb, state, state_elems * sizeof(float));
        npu->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        for (int chunk = 0; chunk < n_chunks; chunk++) {
            memcpy(ha, tok_chunks + (size_t) chunk * tok_elems, tok_elems * sizeof(float));
            npu->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            const uint64_t t_submit = ggml_xdna_now_ns();
            npu->run.start();
            const ert_cmd_state state_cmd = npu->run.wait();
            npu->stats.ns_submit += ggml_xdna_now_ns() - t_submit;
            npu->stats.n_runs++;

            if (state_cmd != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_ERROR("%s: NPU gdn full run did not complete\n", "ggml-xdna");
                return false;
            }

            npu->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (npu->invalidate) {
                ggml_xdna_invalidate(hc, attn_elems * sizeof(float));
            }
            memcpy(attn_chunks + (size_t) chunk * attn_elems,
                   hc, attn_elems * sizeof(float));
        }

        npu->bo_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (npu->invalidate) {
            ggml_xdna_invalidate(hb, state_elems * sizeof(float));
        }
        memcpy(new_state, hb, state_elems * sizeof(float));
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: NPU gdn full failed: %s\n", "ggml-xdna", e.what());
        return false;
    }

    npu->stats.ns_total += ggml_xdna_now_ns() - t_begin;
    npu->stats.n_calls++;
    return true;
}
#endif

#else // !GGML_XDNA_HAS_XRT

bool ggml_xdna_npu_probe(char * name, size_t name_size) {
    if (name && name_size > 0) {
        snprintf(name, name_size, "none");
    }
    return false;
}

void ggml_xdna_npu_free(ggml_xdna_npu * npu) {
    GGML_UNUSED(npu);
}

void ggml_xdna_npu_get_stats(const ggml_xdna_npu * npu, ggml_xdna_npu_stats * stats) {
    GGML_UNUSED(npu);
    if (stats) {
        *stats = ggml_xdna_npu_stats{};
    }
}

ggml_xdna_npu * ggml_xdna_npu_gemm_init(const char *                        xclbin_path,
                                        const char *                        insts_path,
                                        const struct ggml_xdna_gemm_shape * shape) {
    GGML_UNUSED(xclbin_path);
    GGML_UNUSED(insts_path);
    GGML_UNUSED(shape);
    return nullptr;
}

int ggml_xdna_npu_gemm_m(const ggml_xdna_npu * npu) { GGML_UNUSED(npu); return 0; }
int ggml_xdna_npu_gemm_k(const ggml_xdna_npu * npu) { GGML_UNUSED(npu); return 0; }
int ggml_xdna_npu_gemm_n(const ggml_xdna_npu * npu) { GGML_UNUSED(npu); return 0; }

size_t ggml_xdna_npu_gemm_a_bytes(const ggml_xdna_npu * npu) { GGML_UNUSED(npu); return 0; }
size_t ggml_xdna_npu_gemm_b_bytes(const ggml_xdna_npu * npu) { GGML_UNUSED(npu); return 0; }
size_t ggml_xdna_npu_gemm_c_bytes(const ggml_xdna_npu * npu) { GGML_UNUSED(npu); return 0; }

ggml_xdna_buf * ggml_xdna_buf_alloc(ggml_xdna_npu * npu, size_t nbytes) {
    GGML_UNUSED(npu);
    GGML_UNUSED(nbytes);
    return nullptr;
}

void   ggml_xdna_buf_free(ggml_xdna_buf * buf)            { GGML_UNUSED(buf); }
void * ggml_xdna_buf_host(ggml_xdna_buf * buf)            { GGML_UNUSED(buf); return nullptr; }
void   ggml_xdna_buf_sync_to_device(ggml_xdna_buf * buf, size_t offset, size_t nbytes) {
    GGML_UNUSED(buf);
    GGML_UNUSED(offset);
    GGML_UNUSED(nbytes);
}

int ggml_xdna_npu_gemm_kb_max(const ggml_xdna_npu * npu) {
    GGML_UNUSED(npu);
    return 0;
}

uint16_t * ggml_xdna_npu_gemm_a_slice(ggml_xdna_npu * npu, int k_block) {
    GGML_UNUSED(npu);
    GGML_UNUSED(k_block);
    return nullptr;
}

const float * ggml_xdna_npu_gemm_c_slice(ggml_xdna_npu * npu, int k_block) {
    GGML_UNUSED(npu);
    GGML_UNUSED(k_block);
    return nullptr;
}

void ggml_xdna_npu_gemm_acquire(ggml_xdna_npu * npu) {
    GGML_UNUSED(npu);
}

void ggml_xdna_npu_gemm_release(ggml_xdna_npu * npu) {
    GGML_UNUSED(npu);
}

bool ggml_xdna_npu_gemm_runlist_wait(ggml_xdna_npu * npu) {
    GGML_UNUSED(npu);
    return false;
}

bool ggml_xdna_npu_gemm_submit_list_async(ggml_xdna_npu * npu,
                                          ggml_xdna_buf * buf,
                                          const size_t *  w_offs,
                                          int             kb_n) {
    GGML_UNUSED(npu);
    GGML_UNUSED(buf);
    GGML_UNUSED(w_offs);
    GGML_UNUSED(kb_n);
    return false;
}

int ggml_xdna_npu_gemm_n_banks(const ggml_xdna_npu * npu) {
    GGML_UNUSED(npu);
    return 0;
}

void ggml_xdna_npu_gemm_set_bank(ggml_xdna_npu * npu, int bank) {
    GGML_UNUSED(npu);
    GGML_UNUSED(bank);
}

uint16_t * ggml_xdna_npu_gemm_a_slice_bank(ggml_xdna_npu * npu, int bank, int k_block) {
    GGML_UNUSED(npu);
    GGML_UNUSED(bank);
    GGML_UNUSED(k_block);
    return nullptr;
}

const float * ggml_xdna_npu_gemm_c_slice_bank(ggml_xdna_npu * npu, int bank, int k_block) {
    GGML_UNUSED(npu);
    GGML_UNUSED(bank);
    GGML_UNUSED(k_block);
    return nullptr;
}

#if defined(GGML_XDNA_EXPERIMENTAL)
ggml_xdna_npu * ggml_xdna_npu_init(enum ggml_xdna_op op,
                                   const char *      xclbin_path,
                                   const char *      insts_path,
                                   size_t            tile_elems) {
    GGML_UNUSED(op);
    GGML_UNUSED(xclbin_path);
    GGML_UNUSED(insts_path);
    GGML_UNUSED(tile_elems);
    return nullptr;
}

bool ggml_xdna_npu_binary_f32(ggml_xdna_npu * npu, const float * a, const float * b, float * dst, size_t n) {
    GGML_UNUSED(npu);
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(dst);
    GGML_UNUSED(n);
    return false;
}

bool ggml_xdna_npu_rms_norm_f32(ggml_xdna_npu * npu, const float * x, float * dst, int64_t n_rows, float eps) {
    GGML_UNUSED(npu);
    GGML_UNUSED(x);
    GGML_UNUSED(dst);
    GGML_UNUSED(n_rows);
    GGML_UNUSED(eps);
    return false;
}

bool ggml_xdna_npu_add_rms_mul_f32(ggml_xdna_npu * npu,
                                   const float *  a,
                                   const float *  b,
                                   const float *  weight,
                                   float *        add_dst,
                                   float *        mul_dst,
                                   int64_t        n_rows,
                                   int64_t        weight_n_rows,
                                   float          eps) {
    GGML_UNUSED(npu);
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(weight);
    GGML_UNUSED(add_dst);
    GGML_UNUSED(mul_dst);
    GGML_UNUSED(n_rows);
    GGML_UNUSED(weight_n_rows);
    GGML_UNUSED(eps);
    return false;
}

bool ggml_xdna_npu_gemm_submit_ab_list(ggml_xdna_npu * npu, int kb_n) {
    GGML_UNUSED(npu);
    GGML_UNUSED(kb_n);
    return false;
}

uint16_t * ggml_xdna_npu_gemm_b_host(ggml_xdna_npu * npu) {
    GGML_UNUSED(npu);
    return nullptr;
}

uint16_t * ggml_xdna_npu_gemm_b_slice(ggml_xdna_npu * npu, int k_block) {
    GGML_UNUSED(npu);
    GGML_UNUSED(k_block);
    return nullptr;
}

bool ggml_xdna_npu_gemm_submit_ab(ggml_xdna_npu * npu, int k_block) {
    GGML_UNUSED(npu);
    GGML_UNUSED(k_block);
    return false;
}

ggml_xdna_npu * ggml_xdna_npu_gdn_init(const char * xclbin_path,
                                       const char * insts_path,
                                       const struct ggml_xdna_gdn_shape * shape) {
    GGML_UNUSED(xclbin_path);
    GGML_UNUSED(insts_path);
    GGML_UNUSED(shape);
    return nullptr;
}

bool ggml_xdna_npu_gdn_submit(ggml_xdna_npu * npu,
                              const float *   tok,
                              const float *   state_strip,
                              float *         attn_strip,
                              float *         new_state_strip) {
    GGML_UNUSED(npu);
    GGML_UNUSED(tok);
    GGML_UNUSED(state_strip);
    GGML_UNUSED(attn_strip);
    GGML_UNUSED(new_state_strip);
    return false;
}

bool ggml_xdna_npu_gdn_submit_full(ggml_xdna_npu * npu,
                                   const float *   tok,
                                   const float *   state,
                                   float *         attn,
                                   float *         new_state) {
    GGML_UNUSED(npu);
    GGML_UNUSED(tok);
    GGML_UNUSED(state);
    GGML_UNUSED(attn);
    GGML_UNUSED(new_state);
    return false;
}

bool ggml_xdna_npu_gdn_submit_full_chunks(ggml_xdna_npu * npu,
                                          const float *   tok_chunks,
                                          int             n_chunks,
                                          const float *   state,
                                          float *         attn_chunks,
                                          float *         new_state) {
    GGML_UNUSED(npu);
    GGML_UNUSED(tok_chunks);
    GGML_UNUSED(n_chunks);
    GGML_UNUSED(state);
    GGML_UNUSED(attn_chunks);
    GGML_UNUSED(new_state);
    return false;
}
#endif

#endif // GGML_XDNA_HAS_XRT
