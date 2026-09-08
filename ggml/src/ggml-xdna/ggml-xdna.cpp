#include "ggml-impl.h"
#include "ggml-xdna.h"
#include "ggml-backend-impl.h"

#include "xdna-npu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef GGML_XDNA_TILE
#define GGML_XDNA_TILE 1024
#endif


namespace fs = std::filesystem;

static int ggml_xdna_env_int(const char * name, int def) {
    const char * val = getenv(name);
    return val ? atoi(val) : def;
}

#if defined(GGML_XDNA_EXPERIMENTAL)
static bool ggml_xdna_experimental_enabled(void) {
    const char * val = getenv("GGML_XDNA_ENABLE_EXPERIMENTAL");
    return val && strcmp(val, "1") == 0;
}
#endif

static int ggml_xdna_debug_level(void) {
    static int level = -1;
    if (level < 0) {
        level = ggml_xdna_env_int("GGML_XDNA_DEBUG", 0);
    }
    return level;
}

static fs::path ggml_xdna_executable_dir(void) {
#if defined(__linux__)
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        return exe.parent_path();
    }
#endif
    return fs::path();
}

static std::vector<fs::path> ggml_xdna_kernel_dirs(void) {
    std::vector<fs::path> dirs;
    if (const char * env_dir = getenv("GGML_XDNA_KERNELS_DIR")) {
        dirs.emplace_back(env_dir);
    }
#ifdef GGML_BACKEND_DIR
    dirs.emplace_back(GGML_BACKEND_DIR);
#endif
    if (fs::path exe_dir = ggml_xdna_executable_dir(); !exe_dir.empty()) {
        dirs.push_back(exe_dir);
    }
    dirs.emplace_back(fs::current_path());
    return dirs;
}

static bool ggml_xdna_find_stem(const char * stem, std::string & xclbin_out, std::string & insts_out) {
    for (const fs::path & dir : ggml_xdna_kernel_dirs()) {
        const fs::path xclbin = dir / (std::string(stem) + ".xclbin");
        const fs::path insts  = dir / (std::string(stem) + ".insts.bin");
        if (fs::exists(xclbin) && fs::exists(insts)) {
            xclbin_out = xclbin.string();
            insts_out  = insts.string();
            return true;
        }
    }
    return false;
}

#if defined(GGML_XDNA_EXPERIMENTAL)
static bool ggml_xdna_find_kernel(const char * op_name, size_t tile_elems,
                                  std::string & xclbin_out, std::string & insts_out) {
    char stem[96];
    snprintf(stem, sizeof(stem), "ggml-xdna-%s-npu2-f32-%zu", op_name, tile_elems);
    return ggml_xdna_find_stem(stem, xclbin_out, insts_out);
}

// One loaded xclbin. Each slot owns its own hw_context, so every kernel stays
// resident on the NPU for the lifetime of the process and dispatching between
// them is a submission to a different context, never an xclbin reload.
struct ggml_xdna_kernel_slot {
    const char *  env_name;   // GGML_XDNA_<env_name> toggles this kernel
    ggml_xdna_op  op;

    bool        enabled = true;
    bool        found   = false;
    size_t      tile    = 0;  // elements per submission, baked into the artifact
    std::string xclbin_path;
    std::string insts_path;

    std::once_flag  init_once;
    ggml_xdna_npu * npu = nullptr;
};

enum ggml_xdna_slot_idx {
    GGML_XDNA_SLOT_ADD = 0,
    GGML_XDNA_SLOT_MUL,
    GGML_XDNA_SLOT_RMS_NORM,
    GGML_XDNA_SLOT_ADD_RMS_MUL,
    GGML_XDNA_N_SLOTS,
};
#endif

// Per-worker tile and worker grid, both baked into the GEMM xclbin. Must match
// iron/ggml-xdna-gemm.py.
static constexpr int GGML_XDNA_GEMM_TILE_M = 32;
static constexpr int GGML_XDNA_GEMM_TILE_K = 64;
static constexpr int GGML_XDNA_GEMM_TILE_N = 64;
static constexpr int GGML_XDNA_GEMM_GRID_R = 4;
static constexpr int GGML_XDNA_GEMM_GRID_C = 8;

// MMUL micro-tile the operands are swizzled into (bfp16 on aie2p)
static constexpr int GGML_XDNA_GEMM_MAC_R = 8;
static constexpr int GGML_XDNA_GEMM_MAC_S = 8;
static constexpr int GGML_XDNA_GEMM_MAC_T = 8;

// One GEMM artifact covers K=1024 (kt=16) per submit. Larger K is split on the
// host with C accumulation; N is blocked into columns of grid_c*tile_n.
// MB (row blocks) is baked into the xclbin: MB=16 -> M_block=2048, MB=32 -> 4096.
struct ggml_xdna_gemm_slot {
    const char * env_name;
    int          kt = 16; // baked into the artifact; K_slice = kt * TILE_K
    int          mb = 16; // row blocks per submit; M_block = mb*grid_r*tile_m

    bool enabled = true;
    bool found   = false;

    ggml_xdna_gemm_shape shape{};
    std::string          xclbin_path;
    std::string          insts_path;

    std::once_flag  init_once;
    ggml_xdna_npu * npu = nullptr;
};

enum ggml_xdna_gemm_idx {
    GGML_XDNA_GEMM_MB16 = 0,
    GGML_XDNA_GEMM_MB32 = 1,
    GGML_XDNA_N_GEMM    = 2,
};

// Skip lm_head (vocab-width) while still covering FFN like Qwen3.5-9B
// (N=12288). The kernel already loops N in grid_c*tile_n=512 columns; this
// is only a host cap on what lands in XDNA_REPACK.
static constexpr int64_t GGML_XDNA_GEMM_N_MAX = 16384;
static constexpr int     GGML_XDNA_GEMM_KT    = 16;  // K_slice = 1024
// Upper bound on the K-slices the design keeps resident at once, so the slice
// pointers can live on the stack.
static constexpr int     GGML_XDNA_GEMM_KB_LIMIT = 64;

struct ggml_backend_xdna_device_context {
    bool        npu_present = false;
    bool        npu_enabled = true;
    std::string npu_name    = "none";
    std::string description = "AMD XDNA (no NPU device)";

    bool gemm_enabled = true;
    // Below this many activation rows the NPU submit is bandwidth-wasteful;
    // compute stays in this backend (weights are repacked) but runs on the host.
    int64_t gemm_m_min = 128;
    int     gemm_mb    = 16;
    // Prefer the MB=32 (M_block=4096) artifact once M no longer fits one MB=16
    // pass. At exactly M_block the smaller artifact covers it without padding,
    // so the switch has to be strictly above the threshold.
    int64_t gemm_mb32_min = 2048;
    // Keep a host-DRAM copy of the GGUF bytes so decode (M < gemm_m_min) can
    // use the nested CPU kernels on the same memory as `-ngl 0`. The XRT BO
    // holds only the swizzled BF16 the NPU reads.
    bool    keep_src   = true;

#if defined(GGML_XDNA_EXPERIMENTAL)
    bool        experimental = false;
    size_t      tile_elems = GGML_XDNA_TILE;
    size_t      rms_row    = 1024;
    int64_t     op_min     = 1;
    bool        fusion     = true;
    bool        act_gemm   = false;
    ggml_xdna_kernel_slot slots[GGML_XDNA_N_SLOTS] = {
        { "ADD",         GGML_XDNA_OP_ADD         },
        { "MUL",         GGML_XDNA_OP_MUL         },
        { "RMS_NORM",    GGML_XDNA_OP_RMS_NORM    },
        { "ADD_RMS_MUL", GGML_XDNA_OP_ADD_RMS_MUL },
    };
    bool                 gdn_enabled = false;
    bool                 gdn_found   = false;
    ggml_xdna_gdn_shape  gdn_shape{128, 32, 64, 1};
    std::string          gdn_xclbin;
    std::string          gdn_insts;
    std::once_flag       gdn_init_once;
    ggml_xdna_npu *      gdn_npu = nullptr;
    bool                 gdn_fuse_pre = false;
    bool                 chain_bf16 = false;
#endif
    // Claim every op so the scheduler keeps the whole prefill graph on this
    // backend (zero splits). Ops without an NPU kernel run on a nested CPU
    // backend; decode still works but is not the optimization target.
    bool    own_graph    = true;
    // 1: per-op timings, 2: also split by tensor shape.
    int     host_profile = 0;
    // Threads for the swizzle passes. Deliberately not the OpenMP default: with
    // a thread per core the spinning workers delay the thread waiting on the
    // NPU, which more than doubles the observed submit latency.
    int     host_threads = 8;

    ggml_xdna_gemm_slot gemm[GGML_XDNA_N_GEMM] = {
        { "GEMM_MB16", GGML_XDNA_GEMM_KT, 16 },
        { "GEMM_MB32", GGML_XDNA_GEMM_KT, 32 },
    };

    ggml_backend_buffer_type repack_buft{};
};

// Opens one NPU kernel exactly once and caches the result. Returns the shared
// handle or nullptr when the NPU is absent/disabled, the artifact is missing,
// or XRT init failed.
//
// The heavy work (load xclbin, create hw_context, allocate BOs, warm up) is
// triggered eagerly from ggml_backend_xdna_init (backend init), so device
// registration / enumeration stays cheap and side-effect free. By the time the
// scheduler calls supports_op the call_once has already run, so this is a fast
// idempotent guard there, not a per-query init.
//
// Semantics on failure: if this first init fails, the result is sticky for the
// lifetime of the process - the kernel stays unavailable until restart. We do
// NOT retry (here or in supports_op): the failure causes are deterministic
// within a process, and retrying under supports_op would make scheduling both
// expensive and unpredictable.
#if defined(GGML_XDNA_EXPERIMENTAL)
static ggml_xdna_npu * ggml_xdna_device_npu(ggml_backend_xdna_device_context * dctx, ggml_xdna_slot_idx idx) {
    ggml_xdna_kernel_slot & slot = dctx->slots[idx];

    if (!dctx->npu_present || !dctx->npu_enabled || !slot.enabled || !slot.found) {
        return nullptr;
    }

    std::call_once(slot.init_once, [dctx, &slot]() {
        slot.npu = ggml_xdna_npu_init(slot.op,
                                      slot.xclbin_path.c_str(),
                                      slot.insts_path.c_str(),
                                      slot.tile);
        if (!slot.npu) {
            GGML_LOG_WARN("%s: NPU present (%s) but %s kernel init failed; "
                          "XDNA will not claim it\n", "ggml-xdna",
                          dctx->npu_name.c_str(), slot.env_name);
        }
    });

    return slot.npu;
}
#endif

static ggml_xdna_npu * ggml_xdna_device_gemm(ggml_backend_xdna_device_context * dctx, ggml_xdna_gemm_idx idx) {
    ggml_xdna_gemm_slot & slot = dctx->gemm[idx];

    if (!dctx->npu_present || !dctx->npu_enabled || !dctx->gemm_enabled ||
        !slot.enabled || !slot.found) {
        return nullptr;
    }

    std::call_once(slot.init_once, [dctx, &slot]() {
        slot.npu = ggml_xdna_npu_gemm_init(slot.xclbin_path.c_str(),
                                           slot.insts_path.c_str(),
                                           &slot.shape);
        if (!slot.npu) {
            GGML_LOG_WARN("%s: NPU present (%s) but %s kernel init failed; "
                          "XDNA will not claim MUL_MAT\n", "ggml-xdna",
                          dctx->npu_name.c_str(), slot.env_name);
        }
    });

    return slot.npu;
}

#if defined(GGML_XDNA_EXPERIMENTAL)
static ggml_xdna_npu * ggml_xdna_device_gdn(ggml_backend_xdna_device_context * dctx) {
    if (!dctx->npu_present || !dctx->npu_enabled || !dctx->gdn_enabled || !dctx->gdn_found) {
        return nullptr;
    }

    std::call_once(dctx->gdn_init_once, [dctx]() {
        dctx->gdn_npu = ggml_xdna_npu_gdn_init(dctx->gdn_xclbin.c_str(),
                                               dctx->gdn_insts.c_str(),
                                               &dctx->gdn_shape);
        if (!dctx->gdn_npu) {
            GGML_LOG_WARN("%s: NPU present (%s) but GDN kernel init failed\n",
                          "ggml-xdna", dctx->npu_name.c_str());
        }
    });

    return dctx->gdn_npu;
}
#endif

// Shape gate shared by weight-buffer selection and runtime dispatch: K must be
// a multiple of the tile depth, N a multiple of the column block. A K that does
// not fill the last slice is padded with zeros by both packers, which costs
// MACs on the tail but keeps wide-K projections on the NPU.
static bool ggml_xdna_gemm_shape_ok(int64_t k_total, int64_t n_total) {
    const int n_cols = GGML_XDNA_GEMM_GRID_C * GGML_XDNA_GEMM_TILE_N;

    return k_total > 0 && n_total > 0 &&
           n_total <= GGML_XDNA_GEMM_N_MAX &&
           (k_total % GGML_XDNA_GEMM_TILE_K) == 0 &&
           (n_total % n_cols) == 0;
}

// Returns the GEMM slot when K and N are compatible with host-side blocking
// (K multiple of the tile depth, N multiple of the column width). When m_hint
// is set, prefer the MB=32 artifact for prefills too long for one MB=16 pass.
static ggml_xdna_gemm_slot * ggml_xdna_gemm_for_shape(ggml_backend_xdna_device_context * dctx,
                                                      int64_t k_total, int64_t n_total,
                                                      int64_t m_hint = 0) {
    if (!ggml_xdna_gemm_shape_ok(k_total, n_total)) {
        return nullptr;
    }

    if (m_hint > dctx->gemm_mb32_min &&
        ggml_xdna_device_gemm(dctx, GGML_XDNA_GEMM_MB32)) {
        return &dctx->gemm[GGML_XDNA_GEMM_MB32];
    }
    if (ggml_xdna_device_gemm(dctx, GGML_XDNA_GEMM_MB16)) {
        return &dctx->gemm[GGML_XDNA_GEMM_MB16];
    }
    // Fall back to whatever slot actually initialised (e.g. only MB=32 built).
    if (ggml_xdna_device_gemm(dctx, GGML_XDNA_GEMM_MB32)) {
        return &dctx->gemm[GGML_XDNA_GEMM_MB32];
    }
    return nullptr;
}

static void ggml_xdna_gemm_init_slots(ggml_backend_xdna_device_context & ctx) {
    ctx.gemm_mb       = ggml_xdna_env_int("GGML_XDNA_GEMM_MB", 16);
    ctx.gemm_mb32_min = ggml_xdna_env_int("GGML_XDNA_GEMM_MB32_MIN", 2048);
    ctx.gemm_m_min    = ggml_xdna_env_int("GGML_XDNA_GEMM_M_MIN", 128);
    ctx.keep_src      = ggml_xdna_env_int("GGML_XDNA_KEEP_SRC", 1) != 0;

    if (ctx.gemm_mb < 1) {
        ctx.gemm_mb = 1;
    }

    for (ggml_xdna_gemm_slot & slot : ctx.gemm) {
        char var[48];
        snprintf(var, sizeof(var), "GGML_XDNA_%s", slot.env_name);
        slot.enabled = ggml_xdna_env_int(var, 1) != 0;
        slot.kt      = GGML_XDNA_GEMM_KT;
        // Slot 0 tracks GGML_XDNA_GEMM_MB so custom single-geometry builds keep
        // working; slot 1 is always the MB=32 sibling.
        if (&slot == &ctx.gemm[GGML_XDNA_GEMM_MB16]) {
            slot.mb = ctx.gemm_mb;
        }

        slot.shape = ggml_xdna_gemm_shape{
            /* tile_m */ GGML_XDNA_GEMM_TILE_M,
            /* tile_k */ GGML_XDNA_GEMM_TILE_K,
            /* tile_n */ GGML_XDNA_GEMM_TILE_N,
            /* grid_r */ GGML_XDNA_GEMM_GRID_R,
            /* grid_c */ GGML_XDNA_GEMM_GRID_C,
            /* kt     */ slot.kt,
            /* nb     */ 1,
            /* mb     */ slot.mb,
        };

        char stem[96];
        snprintf(stem, sizeof(stem), "ggml-xdna-gemm-npu2-bf16-%d-%d",
                 slot.shape.kt, slot.shape.mb);
        slot.found = ggml_xdna_find_stem(stem, slot.xclbin_path, slot.insts_path);
        if (!slot.found) {
            GGML_LOG_WARN("%s: no %s artifact (%s); that geometry stays unavailable\n",
                          "ggml-xdna", slot.env_name, stem);
        }
    }
}

static ggml_backend_xdna_device_context * ggml_xdna_device_context(void) {
    static ggml_backend_xdna_device_context ctx;
    static std::once_flag once;

    std::call_once(once, [&]() {
        ctx.own_graph    = ggml_xdna_env_int("GGML_XDNA_OWN_GRAPH", 1) != 0;
        ctx.host_profile = (int) ggml_xdna_env_int("GGML_XDNA_HOST_PROFILE", 0);
        ctx.host_threads = (int) ggml_xdna_env_int("GGML_XDNA_HOST_THREADS", 8);
        if (ctx.host_threads < 1) {
            ctx.host_threads = 1;
        }
        ctx.npu_enabled = ggml_xdna_env_int("GGML_XDNA_NPU", 1) != 0;
        ctx.gemm_enabled = ggml_xdna_env_int("GGML_XDNA_GEMM", 1) != 0;
#if defined(GGML_XDNA_EXPERIMENTAL)
        ctx.experimental = ggml_xdna_experimental_enabled();
        ctx.tile_elems  = (size_t) ggml_xdna_env_int("GGML_XDNA_TILE", GGML_XDNA_TILE);
        ctx.op_min      = ggml_xdna_env_int("GGML_XDNA_OP_MIN", 1000000000);
        ctx.rms_row     = (size_t) ggml_xdna_env_int("GGML_XDNA_RMS_ROW", 1024);
        ctx.fusion      = ctx.experimental &&
                          ggml_xdna_env_int("GGML_XDNA_DISABLE_FUSION", 0) == 0;
        ctx.act_gemm     = ctx.experimental &&
                           ggml_xdna_env_int("GGML_XDNA_ACT_GEMM", 0) != 0;
        // Stage 4 GDN: off by default until MemTile-full-state cuts submit count.
        ctx.gdn_enabled = ctx.experimental &&
                          ggml_xdna_env_int("GGML_XDNA_GDN", 0) != 0;
        ctx.gdn_fuse_pre = ctx.experimental &&
                           ggml_xdna_env_int("GGML_XDNA_GDN_FUSE_PRE", 0) != 0;
        ctx.chain_bf16   = ctx.gdn_enabled &&
                           ggml_xdna_env_int("GGML_XDNA_CHAIN_BF16", 1) != 0;
        if (ctx.experimental) {
            const int S = 128;
            const int ROWS = 32;
            const int CS = 64;
            const int NS = S / ROWS;
            ctx.gdn_shape = ggml_xdna_gdn_shape{S, ROWS, CS, NS};
            char stem_w4[96];
            char stem_w1[96];
            snprintf(stem_w4, sizeof(stem_w4), "ggml-xdna-gdn-npu2-s%d-r%d-cs%d-w%d",
                     S, ROWS, CS, NS);
            snprintf(stem_w1, sizeof(stem_w1), "ggml-xdna-gdn-npu2-s%d-r%d-cs%d",
                     S, ROWS, CS);
            if (ggml_xdna_find_stem(stem_w4, ctx.gdn_xclbin, ctx.gdn_insts)) {
                ctx.gdn_shape.workers = NS;
                ctx.gdn_found = true;
            } else if (ggml_xdna_find_stem(stem_w1, ctx.gdn_xclbin, ctx.gdn_insts)) {
                ctx.gdn_shape.workers = 1;
                ctx.gdn_found = true;
            } else {
                ctx.gdn_found = false;
            }
            if (ctx.gdn_enabled && !ctx.gdn_found) {
                GGML_LOG_WARN("%s: GGML_XDNA_GDN=1 but no artifact (%s or %s)\n",
                              "ggml-xdna", stem_w4, stem_w1);
            }
        }
#endif

        char name[128] = "none";
        ctx.npu_present = ggml_xdna_npu_probe(name, sizeof(name));
        ctx.npu_name    = name;

        if (!ctx.npu_present) {
            ctx.description = "AMD XDNA (no NPU device)";
            return;
        }

        std::string claimed;
#if defined(GGML_XDNA_EXPERIMENTAL)
        for (ggml_xdna_kernel_slot & slot : ctx.slots) {
            char var[48];
            snprintf(var, sizeof(var), "GGML_XDNA_%s", slot.env_name);
            slot.enabled = ctx.experimental && ggml_xdna_env_int(var, 1) != 0;
            if (!slot.enabled) {
                continue;
            }

            slot.tile  = (slot.op == GGML_XDNA_OP_RMS_NORM || slot.op == GGML_XDNA_OP_ADD_RMS_MUL)
                         ? ctx.rms_row : ctx.tile_elems;
            slot.found = ggml_xdna_find_kernel(ggml_xdna_op_name(slot.op),
                                               slot.tile, slot.xclbin_path, slot.insts_path);
            if (!slot.found) {
                GGML_LOG_WARN("%s: NPU present (%s) but no %s kernel for tile=%zu; "
                              "build with -DGGML_XDNA_BUILD_KERNELS=ON\n",
                              "ggml-xdna", ctx.npu_name.c_str(), slot.env_name, slot.tile);
                continue;
            }
            if (ctx.npu_enabled && slot.enabled) {
                claimed += claimed.empty() ? "" : ", ";
                claimed += slot.env_name;
            }
        }
#endif

        ggml_xdna_gemm_init_slots(ctx);
        for (const ggml_xdna_gemm_slot & slot : ctx.gemm) {
            if (ctx.gemm_enabled && slot.enabled && slot.found) {
                claimed += claimed.empty() ? "" : ", ";
                claimed += slot.env_name;
            }
        }

        ctx.description = "AMD XDNA " + ctx.npu_name +
                          (claimed.empty() ? " (no NPU kernels)" : " (NPU " + claimed + ")") +
                          (ctx.own_graph ? " [own-graph]" : "");
    });

    return &ctx;
}

#if defined(GGML_XDNA_EXPERIMENTAL)
static ggml_xdna_slot_idx ggml_xdna_slot_for_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_MUL:      return GGML_XDNA_SLOT_MUL;
        case GGML_OP_RMS_NORM: return GGML_XDNA_SLOT_RMS_NORM;
        default:               return GGML_XDNA_SLOT_ADD;
    }
}
#endif

struct ggml_backend_xdna_context {
    ggml_backend_xdna_device_context * dev = nullptr;

    // Nested CPU backend for ops we claim but have no NPU kernel for. Keeps the
    // scheduler from splitting the graph between XDNA and CPU.
    ggml_backend_t host = nullptr;

    // Threads the caller asked for. Used by the host GEMM (decode), which runs
    // without a pending NPU submit and so can take every core, unlike the
    // operand packing that dev->host_threads deliberately keeps narrow.
    int n_threads = 0;

#if defined(GGML_XDNA_EXPERIMENTAL)
    uint64_t n_fused_add_rms_mul      = 0;
    uint64_t n_fused_add_rms_mul_npu  = 0;
#endif
    uint64_t n_host_graph_runs        = 0;
    uint64_t n_host_nodes             = 0;
    uint64_t ns_host                  = 0;

    // Per-op nested CPU time, only filled when host_profile is set: the slices
    // are then computed one node at a time, which is slower but attributable.
    std::unordered_map<std::string, uint64_t> host_op_ns;
    std::unordered_map<std::string, uint64_t> host_op_calls;
    std::unordered_map<std::string, uint64_t> host_op_bytes;

    // Time spent rearranging operands on the host, which is not part of the
    // kernel's own accounting. Kept apart because they scale differently: the
    // A pack is O(M*K) per call and skippable, the C scatter is O(M*N*k_blocks)
    // per call and is not.
    uint64_t ns_gemm_pack    = 0;
    uint64_t ns_gemm_scatter = 0;
    uint64_t n_gemm_pack     = 0;
    uint64_t n_gemm_pack_reused = 0;

    // Last packed activation identity so consecutive weight GEMMs that share
    // src1 can skip the A pack (gate/up style projections).
    const ggml_tensor * gemm_pack_src = nullptr;
    int64_t             gemm_pack_m0  = -1;
    int                 gemm_pack_kb0 = -1;
    int                 gemm_pack_kb_n = 0;
    int                 gemm_pack_bank = 0;
    ggml_xdna_npu *     gemm_pack_npu = nullptr;

#if defined(GGML_XDNA_EXPERIMENTAL)
    // BF16 side copies of recent GEMM destinations for GDN packing.
    struct chain_entry {
        const void * key = nullptr;
        std::vector<ggml_bf16_t> data;
        int64_t ne0 = 0;
        int64_t ne1 = 0;
    };
    static constexpr int CHAIN_SLOTS = 8;
    chain_entry chain[CHAIN_SLOTS];
    int chain_next = 0;
#endif
};

#if defined(GGML_XDNA_EXPERIMENTAL)
static void ggml_xdna_chain_store(ggml_backend_xdna_context * ctx,
                                  const void * key,
                                  const float * src,
                                  int64_t ne0,
                                  int64_t ne1) {
    if (!ctx || !ctx->dev->chain_bf16 || !key || !src || ne0 <= 0 || ne1 <= 0) {
        return;
    }
    const size_t n = (size_t) ne0 * (size_t) ne1;
    ggml_backend_xdna_context::chain_entry & e = ctx->chain[ctx->chain_next % ctx->CHAIN_SLOTS];
    ctx->chain_next++;
    e.key = key;
    e.ne0 = ne0;
    e.ne1 = ne1;
    e.data.resize(n);
#ifdef GGML_XDNA_OPENMP
#   pragma omp parallel for schedule(static) num_threads(ctx->dev->host_threads)
#endif
    for (int64_t i = 0; i < (int64_t) n; i++) {
        e.data[(size_t) i] = GGML_FP32_TO_BF16(src[(size_t) i]);
    }
}

static const ggml_bf16_t * ggml_xdna_chain_find(ggml_backend_xdna_context * ctx,
                                                const void * key,
                                                int64_t ne0,
                                                int64_t ne1) {
    if (!ctx || !ctx->dev->chain_bf16 || !key) {
        return nullptr;
    }
    for (int i = 0; i < ctx->CHAIN_SLOTS; i++) {
        const auto & e = ctx->chain[i];
        if (e.key == key && e.ne0 == ne0 && e.ne1 == ne1 && !e.data.empty()) {
            return e.data.data();
        }
    }
    return nullptr;
}
#endif

static uint64_t ggml_xdna_now_ns(void) {
    using clock = std::chrono::steady_clock;
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        clock::now().time_since_epoch()).count();
}

#if defined(GGML_XDNA_EXPERIMENTAL)
static void ggml_xdna_log_op(ggml_backend_xdna_context * ctx, const struct ggml_tensor * node,
                             const ggml_xdna_kernel_slot & slot) {
    const int level = ggml_xdna_debug_level();

    ggml_xdna_npu_stats stats;
    ggml_xdna_npu_get_stats(slot.npu, &stats);
    const uint64_t n = stats.n_calls;

    const bool log_now = (level >= 2) || (level >= 1 && (n == 1 || (n % 64) == 0)) || (level <= 0 && n == 1);
    if (!log_now) {
        return;
    }

    GGML_LOG_INFO("%s: graph_compute %s name=%s nelements=%lld on=NPU npu=%s calls=%llu runs=%llu "
                  "avg_submit=%.1fus avg_total=%.1fus\n",
                  "ggml-xdna",
                  slot.env_name,
                  node->name ? node->name : "?",
                  (long long) ggml_nelements(node),
                  ctx->dev->npu_name.c_str(),
                  (unsigned long long) stats.n_calls,
                  (unsigned long long) stats.n_runs,
                  stats.n_runs  ? (double) stats.ns_submit / stats.n_runs  / 1e3 : 0.0,
                  stats.n_calls ? (double) stats.ns_total  / stats.n_calls / 1e3 : 0.0);
}

// Runs an elementwise op on the NPU. There is no CPU fallback: supports_op only
// claims an op when its kernel is live, so a submit failure here is a hard error.
static bool ggml_xdna_compute_binary(ggml_backend_xdna_context * ctx, const struct ggml_tensor * node) {
    ggml_xdna_kernel_slot & slot = ctx->dev->slots[ggml_xdna_slot_for_op(node->op)];

    if (!slot.npu) {
        GGML_LOG_ERROR("%s: graph_compute %s but its NPU kernel is not initialized\n",
                       "ggml-xdna", slot.env_name);
        return false;
    }

    const bool ok = ggml_xdna_npu_binary_f32(slot.npu,
                                             (const float *) node->src[0]->data,
                                             (const float *) node->src[1]->data,
                                             (float *) node->data,
                                             (size_t) ggml_nelements(node));
    if (!ok) {
        GGML_LOG_ERROR("%s: NPU %s submit failed for %s\n", "ggml-xdna", slot.env_name,
                       node->name ? node->name : "?");
        return false;
    }

    ggml_xdna_log_op(ctx, node, slot);
    return true;
}

static bool ggml_xdna_compute_rms_norm(ggml_backend_xdna_context * ctx, const struct ggml_tensor * node) {
    ggml_xdna_kernel_slot & slot = ctx->dev->slots[GGML_XDNA_SLOT_RMS_NORM];

    if (!slot.npu) {
        GGML_LOG_ERROR("%s: graph_compute %s but its NPU kernel is not initialized\n",
                       "ggml-xdna", slot.env_name);
        return false;
    }

    float eps;
    memcpy(&eps, node->op_params, sizeof(float));

    const bool ok = ggml_xdna_npu_rms_norm_f32(slot.npu,
                                               (const float *) node->src[0]->data,
                                               (float *) node->data,
                                               ggml_nrows(node),
                                               eps);
    if (!ok) {
        GGML_LOG_ERROR("%s: NPU %s submit failed for %s\n", "ggml-xdna", slot.env_name,
                       node->name ? node->name : "?");
        return false;
    }

    ggml_xdna_log_op(ctx, node, slot);
    return true;
}

// ADD -> RMS_NORM -> MUL, the pre-norm block of every transformer layer.
//
// ggml_can_fuse is the wrong predicate here: it wants every node but the last
// to have a single consumer, and the ADD is the residual, so it has two - the
// RMS_NORM and the next layer's residual ADD. ggml_can_fuse_subgraph is the one
// that models this, by letting both the ADD and the MUL be declared outputs of
// the region. A single NPU design covering the pattern therefore has to write
// two buffers, not one.
static bool ggml_xdna_can_fuse_add_rms_mul(const ggml_backend_xdna_context * ctx,
                                           const struct ggml_cgraph * cgraph, int i) {
    if (!ctx->dev->fusion) {
        return false;
    }

    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 2 })) {
        return false;
    }

    // rms_norm consumes the add, mul consumes the rms_norm
    if (!ggml_check_edges(cgraph, i, { { 1, 0, 0 }, { 2, 0, 1 } })) {
        return false;
    }

    const struct ggml_tensor * add = cgraph->nodes[i];
    const struct ggml_tensor * mul = cgraph->nodes[i + 2];
    const struct ggml_tensor * w   = mul->src[1];

    if (!w || w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous_rows(w) || w->ne[0] != mul->ne[0]) {
        return false;
    }

    // Artifact is baked to rms_row. The fused kernel still submits one run per
    // row, so only use it for decode-sized batches; prefill residuals go through
    // the nested CPU path (own_graph) or separate kernels. op_min gates it the
    // same way as the plain elementwise ops: a submit costs ~330us, which no
    // decode-sized residual can earn back.
    return (size_t) add->ne[0] == ctx->dev->rms_row &&
           ggml_nrows(add) <= 32 &&
           ggml_nelements(add) >= ctx->dev->op_min &&
           ggml_xdna_device_npu(ctx->dev, GGML_XDNA_SLOT_ADD_RMS_MUL) != nullptr;
}

// Executes the fused region. Uses the single-submission ADD_RMS_MUL design when
// its artifact is loaded; otherwise falls back to three separate kernels.
static bool ggml_xdna_compute_add_rms_mul(ggml_backend_xdna_context * ctx,
                                          const struct ggml_cgraph * cgraph, int i) {
    struct ggml_tensor * add = cgraph->nodes[i];
    struct ggml_tensor * rms = cgraph->nodes[i + 1];
    struct ggml_tensor * mul = cgraph->nodes[i + 2];

    ctx->n_fused_add_rms_mul++;

    ggml_xdna_kernel_slot & fused = ctx->dev->slots[GGML_XDNA_SLOT_ADD_RMS_MUL];
    if (fused.npu) {
        float eps;
        memcpy(&eps, rms->op_params, sizeof(float));

        const bool ok = ggml_xdna_npu_add_rms_mul_f32(fused.npu,
                                                      (const float *) add->src[0]->data,
                                                      (const float *) add->src[1]->data,
                                                      (const float *) mul->src[1]->data,
                                                      (float *) add->data,
                                                      (float *) mul->data,
                                                      ggml_nrows(add),
                                                      /*weight_n_rows=*/ ggml_nrows(mul->src[1]),
                                                      eps);
        if (!ok) {
            GGML_LOG_ERROR("%s: NPU %s submit failed for fused ADD+RMS_NORM+MUL\n",
                           "ggml-xdna", fused.env_name);
            return false;
        }

        ctx->n_fused_add_rms_mul_npu++;
        ggml_xdna_log_op(ctx, add, fused);
        return true;
    }

    return ggml_xdna_compute_binary(ctx, add) &&
           ggml_xdna_compute_rms_norm(ctx, rms) &&
           ggml_xdna_compute_binary(ctx, mul);
}
#endif

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XDNA";
}

// Per-kernel totals, so an A/B over GGML_XDNA_ADD / GGML_XDNA_MUL shows how
// much of a run went into each hw_context and what a submission costs.
static void ggml_xdna_log_slot_stats(const char * name, const ggml_xdna_npu * npu) {
    ggml_xdna_npu_stats stats;
    ggml_xdna_npu_get_stats(npu, &stats);
    if (stats.n_calls == 0) {
        return;
    }
    GGML_LOG_INFO("%s: summary %s: calls=%llu runs=%llu total=%.1fms submit=%.1fms "
                  "(%.1f%% of total) sync=%.1fms avg_submit=%.1fus\n",
                  "ggml-xdna",
                  name,
                  (unsigned long long) stats.n_calls,
                  (unsigned long long) stats.n_runs,
                  (double) stats.ns_total  / 1e6,
                  (double) stats.ns_submit / 1e6,
                  stats.ns_total ? 100.0 * (double) stats.ns_submit / (double) stats.ns_total : 0.0,
                  (double) stats.ns_sync   / 1e6,
                  stats.n_runs ? (double) stats.ns_submit / stats.n_runs / 1e3 : 0.0);
}

static void ggml_xdna_log_summary(ggml_backend_xdna_device_context * dev) {
    if (ggml_xdna_debug_level() <= 0) {
        return;
    }

    uint64_t n_calls = 0;
#if defined(GGML_XDNA_EXPERIMENTAL)
    for (const ggml_xdna_kernel_slot & slot : dev->slots) {
        ggml_xdna_npu_stats stats;
        ggml_xdna_npu_get_stats(slot.npu, &stats);
        n_calls += stats.n_calls;
    }
#endif
    for (const ggml_xdna_gemm_slot & slot : dev->gemm) {
        ggml_xdna_npu_stats stats;
        ggml_xdna_npu_get_stats(slot.npu, &stats);
        n_calls += stats.n_calls;
    }
    // A backend can be created and freed before any work reaches the NPU, so
    // that free must not consume the one-shot report.
    if (n_calls == 0) {
        return;
    }

    static std::once_flag once;
    std::call_once(once, [dev]() {
#if defined(GGML_XDNA_EXPERIMENTAL)
        for (const ggml_xdna_kernel_slot & slot : dev->slots) {
            ggml_xdna_log_slot_stats(slot.env_name, slot.npu);
        }
#endif
        for (const ggml_xdna_gemm_slot & slot : dev->gemm) {
            ggml_xdna_log_slot_stats(slot.env_name, slot.npu);
        }
    });
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *)backend->context;
    if (ggml_xdna_debug_level() > 0 && (ctx->ns_gemm_pack || ctx->ns_gemm_scatter)) {
        GGML_LOG_INFO("%s: summary GEMM host: A pack %.1fms (%llu packs, %llu reused), "
                      "C scatter %.1fms\n",
                      "ggml-xdna",
                      (double) ctx->ns_gemm_pack / 1e6,
                      (unsigned long long) ctx->n_gemm_pack,
                      (unsigned long long) ctx->n_gemm_pack_reused,
                      (double) ctx->ns_gemm_scatter / 1e6);
    }

#if defined(GGML_XDNA_EXPERIMENTAL)
    if (ggml_xdna_debug_level() > 0 && ctx->n_fused_add_rms_mul) {
        GGML_LOG_INFO("%s: summary FUSION: ADD+RMS_NORM+MUL matched %llu times (%llu on fused kernel)\n",
                      "ggml-xdna",
                      (unsigned long long) ctx->n_fused_add_rms_mul,
                      (unsigned long long) ctx->n_fused_add_rms_mul_npu);
    }
#endif
    if (ggml_xdna_debug_level() > 0 && ctx->n_host_graph_runs) {
        GGML_LOG_INFO("%s: summary HOST: nested CPU runs=%llu nodes=%llu total=%.1fms\n",
                      "ggml-xdna",
                      (unsigned long long) ctx->n_host_graph_runs,
                      (unsigned long long) ctx->n_host_nodes,
                      (double) ctx->ns_host / 1e6);

        uint64_t all_ns = 0, all_bytes = 0;
        for (const auto & kv : ctx->host_op_ns) {
            if (kv.first.compare(0, 8, "[local] ") == 0) {
                continue;
            }
            all_ns    += kv.second;
            all_bytes += ctx->host_op_bytes[kv.first];
        }
        if (all_ns) {
            GGML_LOG_INFO("%s: summary HOST profiled ops: %.1fms, %.1f GiB touched, %.1f GB/s\n",
                          "ggml-xdna",
                          (double) all_ns / 1e6,
                          (double) all_bytes / (1024.0 * 1024.0 * 1024.0),
                          (double) all_bytes / (double) all_ns);
        }

        std::vector<std::pair<std::string, uint64_t>> by_op(ctx->host_op_ns.begin(),
                                                            ctx->host_op_ns.end());
        std::sort(by_op.begin(), by_op.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });
        for (size_t i = 0; i < by_op.size(); i++) {
            const uint64_t ns    = by_op[i].second;
            const uint64_t bytes = ctx->host_op_bytes[by_op[i].first];
            GGML_LOG_INFO("%s: summary HOST op %-28s %8.1fms  calls=%-6llu %6.1f GB/s\n",
                          "ggml-xdna",
                          by_op[i].first.c_str(),
                          (double) ns / 1e6,
                          (unsigned long long) ctx->host_op_calls[by_op[i].first],
                          ns ? (double) bytes / (double) ns : 0.0);
        }
    }
    ggml_xdna_log_summary(ctx->dev);
    // The kernels are owned by the device singleton (shared across backends),
    // so they are intentionally not freed here; they live until process exit.
    if (ctx->host) {
        ggml_backend_free(ctx->host);
    }
    delete ctx;
    delete backend;
}

// Defined with the repack buffer it reads its weights from.
static ggml_xdna_gemm_slot * ggml_xdna_gemm_slot_for_op(ggml_backend_xdna_device_context * dctx,
                                                       const struct ggml_tensor *                op);
static bool ggml_xdna_gemm_runs_here(ggml_backend_xdna_device_context * dctx,
                                     const struct ggml_tensor *         op);
static bool ggml_xdna_uses_repack_buft(const ggml_backend_xdna_device_context * dctx,
                                      const ggml_tensor *                      op);
#if defined(GGML_XDNA_EXPERIMENTAL)
static bool ggml_xdna_act_gemm_ok(ggml_backend_xdna_device_context * dctx,
                                  const ggml_tensor *                op);
static bool ggml_xdna_binary_shape_ok(const struct ggml_tensor * op, int64_t op_min);
#endif
static bool ggml_xdna_compute_mul_mat(ggml_backend_xdna_context * ctx,
                                      const struct ggml_tensor * node,
                                      float * dst_override);

#if defined(GGML_XDNA_EXPERIMENTAL)
// Stage 4: GATED_DELTA_NET on NPU (scalar gate, S=128, K=1). State is walked
// as ROWS=32 strips; tok packs slide v so v[0:ROWS] match the active strip.
static bool ggml_xdna_gdn_shape_ok(const struct ggml_tensor * node,
                                   const ggml_xdna_gdn_shape & sh) {
    if (!node || node->op != GGML_OP_GATED_DELTA_NET) {
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * g = node->src[3];
    const ggml_tensor * beta = node->src[4];
    const ggml_tensor * state = node->src[5];
    if (!q || !k || !v || !g || !beta || !state) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 ||
        beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t S = v->ne[0];
    if (S != sh.S || (S % sh.ROWS) != 0) {
        return false;
    }
    if (g->ne[0] != 1 || beta->ne[0] != 1) {
        return false; // KDA / non-scalar gate not in this artifact
    }
    if (ggml_get_op_params_i32(node, 0) != 1) {
        return false; // only K=1 (final state)
    }
    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) ||
        !ggml_is_contiguous_rows(v) || !ggml_is_contiguous(g) ||
        !ggml_is_contiguous(beta) || !ggml_is_contiguous(state)) {
        return false;
    }
    return true;
}

static void ggml_xdna_l2_norm_row(float * dst, const float * src, int64_t n, float eps) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += src[i] * src[i];
    }
    const float inv = 1.0f / sqrtf(sum / (float) n + eps);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = src[i] * inv;
    }
}

// Stage 5: L2_NORM(q/k) that only feed a GDN we will run can be absorbed.
static bool ggml_xdna_l2_absorbed_by_gdn(ggml_backend_xdna_device_context * dctx,
                                         const struct ggml_tensor * l2) {
    if (!dctx->gdn_fuse_pre || !ggml_xdna_device_gdn(dctx) || !l2 ||
        l2->op != GGML_OP_L2_NORM || !l2->src[0]) {
        return false;
    }
    // Shape must match a GDN head row: ne0 == S.
    return l2->ne[0] == dctx->gdn_shape.S && l2->type == GGML_TYPE_F32;
}

static bool ggml_xdna_compute_gated_delta_net(ggml_backend_xdna_context * ctx,
                                              const struct ggml_tensor * node) {
    ggml_backend_xdna_device_context * dctx = ctx->dev;
    ggml_xdna_npu * npu = ggml_xdna_device_gdn(dctx);
    if (!npu || !ggml_xdna_gdn_shape_ok(node, dctx->gdn_shape)) {
        return false;
    }

    const int S = dctx->gdn_shape.S;
    const int ROWS = dctx->gdn_shape.ROWS;
    const int CS = dctx->gdn_shape.CS;
    const int workers = dctx->gdn_shape.workers > 0 ? dctx->gdn_shape.workers : 1;
    const int NS = S / ROWS;
    const bool full = workers > 1;

    const ggml_tensor * src_q = node->src[0];
    const ggml_tensor * src_k = node->src[1];
    const ggml_tensor * src_v = node->src[2];
    const ggml_tensor * src_g = node->src[3];
    const ggml_tensor * src_beta = node->src[4];
    const ggml_tensor * src_state = node->src[5];

    float eps_q = 1e-6f;
    float eps_k = 1e-6f;
    bool fuse_q = false;
    bool fuse_k = false;
    if (dctx->gdn_fuse_pre) {
        if (src_q->op == GGML_OP_L2_NORM && src_q->src[0]) {
            eps_q = ggml_get_op_params_f32(src_q, 0);
            src_q = src_q->src[0];
            fuse_q = true;
        }
        if (src_k->op == GGML_OP_L2_NORM && src_k->src[0]) {
            eps_k = ggml_get_op_params_f32(src_k, 0);
            src_k = src_k->src[0];
            fuse_k = true;
        }
    }

    const int64_t H = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs = src_v->ne[3];
    const int64_t H_k = src_k->ne[1];
    const int64_t H_q = src_q->ne[1];

    const int64_t attn_elems = (int64_t) S * H * n_tokens * n_seqs;
    float * attn_out = (float *) node->data;
    float * state_out = attn_out + attn_elems;

    const size_t tok_elems = (size_t) CS * (size_t) (3 * S + 2);
    const size_t strip_elems = (size_t) ROWS * (size_t) S;
    const size_t state_elems = full ? (size_t) S * (size_t) S : strip_elems;
    const size_t attn_strip_elems = full ? (size_t) CS * (size_t) S
                                         : (size_t) CS * (size_t) ROWS;
    const int64_t n_chunks = (n_tokens + CS - 1) / CS;

    std::vector<float> tok(tok_elems * (full ? (size_t) n_chunks : 1));
    std::vector<float> state_buf(state_elems);
    std::vector<float> attn_strip(attn_strip_elems * (full ? (size_t) n_chunks : 1));
    std::vector<float> new_state(state_elems);
    std::vector<float> q_tmp((size_t) S);
    std::vector<float> k_tmp((size_t) S);

    const ggml_bf16_t * q_bf = ggml_xdna_chain_find(ctx, src_q->data, src_q->ne[0], ggml_nrows(src_q));
    const ggml_bf16_t * k_bf = ggml_xdna_chain_find(ctx, src_k->data, src_k->ne[0], ggml_nrows(src_k));
    const ggml_bf16_t * v_bf = ggml_xdna_chain_find(ctx, src_v->data, src_v->ne[0], ggml_nrows(src_v));

    auto load_row = [](float * dst, const float * src_f, const ggml_bf16_t * src_bf,
                       size_t off, int64_t n, bool fuse, float eps, float * tmp) {
        if (src_bf) {
            for (int64_t i = 0; i < n; ++i) {
                tmp[i] = GGML_BF16_TO_FP32(src_bf[off + (size_t) i]);
            }
            if (fuse) {
                ggml_xdna_l2_norm_row(dst, tmp, n, eps);
            } else {
                memcpy(dst, tmp, (size_t) n * sizeof(float));
            }
        } else if (fuse) {
            ggml_xdna_l2_norm_row(dst, src_f, n, eps);
        } else {
            memcpy(dst, src_f, (size_t) n * sizeof(float));
        }
    };

    for (int64_t is = 0; is < n_seqs; ++is) {
        for (int64_t ih = 0; ih < H; ++ih) {
            const int64_t iq1 = ih % H_q;
            const int64_t ik1 = ih % H_k;

            float * s_dst = state_out + (is * H + ih) * S * S;
            const float * s_src = (const float *) src_state->data +
                is * (src_state->nb[3] / sizeof(float)) + ih * S * S;
            memcpy(s_dst, s_src, (size_t) S * S * sizeof(float));
            if (full) {
                memcpy(state_buf.data(), s_dst, state_elems * sizeof(float));
            }

            for (int64_t t0 = 0; t0 < n_tokens; t0 += CS) {
                const int64_t n_chunk = std::min((int64_t) CS, n_tokens - t0);
                const int64_t chunk = t0 / CS;
                float * tok_chunk = tok.data() + (full ? (size_t) chunk * tok_elems : 0);

                memset(tok_chunk, 0, tok_elems * sizeof(float));
                for (int64_t t = 0; t < n_chunk; ++t) {
                    const int64_t tt = t0 + t;
                    const size_t q_off = (size_t) ((is * n_tokens + tt) * H_q + iq1) * (size_t) S;
                    const size_t k_off = (size_t) ((is * n_tokens + tt) * H_k + ik1) * (size_t) S;
                    const size_t v_off = (size_t) ((is * n_tokens + tt) * H + ih) * (size_t) S;

                    const float * q = (const float *) ((const char *) src_q->data +
                        is * src_q->nb[3] + tt * src_q->nb[2] + iq1 * src_q->nb[1]);
                    const float * k = (const float *) ((const char *) src_k->data +
                        is * src_k->nb[3] + tt * src_k->nb[2] + ik1 * src_k->nb[1]);
                    const float * v = (const float *) ((const char *) src_v->data +
                        is * src_v->nb[3] + tt * src_v->nb[2] + ih * src_v->nb[1]);
                    const float g = *(const float *) ((const char *) src_g->data +
                        is * src_g->nb[3] + tt * src_g->nb[2] + ih * src_g->nb[1]);
                    const float beta = *(const float *) ((const char *) src_beta->data +
                        is * src_beta->nb[3] + tt * src_beta->nb[2] + ih * src_beta->nb[1]);

                    float * pack = tok_chunk + t * (3 * S + 2);
                    load_row(pack, q, q_bf, q_off, S, fuse_q, eps_q, q_tmp.data());
                    load_row(pack + S, k, k_bf, k_off, S, fuse_k, eps_k, k_tmp.data());
                    if (v_bf) {
                        for (int64_t i = 0; i < S; ++i) {
                            pack[2 * S + i] = GGML_BF16_TO_FP32(v_bf[v_off + (size_t) i]);
                        }
                    } else {
                        memcpy(pack + 2 * S, v, (size_t) S * sizeof(float));
                    }
                    pack[3 * S] = expf(g);
                    pack[3 * S + 1] = beta;
                }
                for (int64_t t = n_chunk; t < CS; ++t) {
                    tok_chunk[t * (3 * S + 2) + 3 * S] = 1.0f;
                }

                if (full) {
                    continue;
                } else {
                    for (int ns = 0; ns < NS; ++ns) {
                        const int j0 = ns * ROWS;
                        memcpy(state_buf.data(), s_dst + j0 * S, strip_elems * sizeof(float));

                        std::vector<float> tok_slid = tok;
                        for (int64_t t = 0; t < n_chunk; ++t) {
                            float * pack = tok_slid.data() + t * (3 * S + 2);
                            memmove(pack + 2 * S, pack + 2 * S + j0, (size_t) ROWS * sizeof(float));
                        }

                        if (!ggml_xdna_npu_gdn_submit(npu, tok_slid.data(), state_buf.data(),
                                                      attn_strip.data(), new_state.data())) {
                            return false;
                        }

                        memcpy(s_dst + j0 * S, new_state.data(), strip_elems * sizeof(float));
                        for (int64_t t = 0; t < n_chunk; ++t) {
                            float * a = attn_out + ((is * n_tokens + t0 + t) * H + ih) * S + j0;
                            memcpy(a, attn_strip.data() + t * ROWS, (size_t) ROWS * sizeof(float));
                        }
                    }
                }
            }

            if (full) {
                if (!ggml_xdna_npu_gdn_submit_full_chunks(
                        npu, tok.data(), (int) n_chunks, state_buf.data(),
                        attn_strip.data(), new_state.data())) {
                    return false;
                }
                memcpy(s_dst, new_state.data(), state_elems * sizeof(float));
                for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
                    const int64_t t0 = chunk * CS;
                    const int64_t n_chunk = std::min((int64_t) CS, n_tokens - t0);
                    const float * chunk_attn =
                        attn_strip.data() + (size_t) chunk * attn_strip_elems;
                    for (int64_t t = 0; t < n_chunk; ++t) {
                        float * a = attn_out + ((is * n_tokens + t0 + t) * H + ih) * S;
                        for (int ns = 0; ns < NS; ++ns) {
                            memcpy(a + ns * ROWS,
                                   chunk_attn + ((size_t) ns * CS + (size_t) t) * ROWS,
                                   (size_t) ROWS * sizeof(float));
                        }
                    }
                }
            }
        }
    }

    return true;
}
#endif

// Ops we execute inside this backend.
// Everything else goes to the nested CPU backend when own_graph is on.
static bool ggml_xdna_is_local_op(ggml_backend_xdna_context * ctx, const struct ggml_tensor * node) {
    ggml_backend_xdna_device_context * dctx = ctx->dev;

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            // With own_graph, absorb into nested-CPU slices. In split mode the
            // scheduler only assigns these here as no-ops on this backend.
            return ctx->host == nullptr;

        case GGML_OP_MUL_MAT:
            // Prefill (large M): NPU weight GEMM. Decode
            // falls through to the nested CPU like every other op.
            return ggml_xdna_gemm_runs_here(dctx, node)
#if defined(GGML_XDNA_EXPERIMENTAL)
                   || ggml_xdna_act_gemm_ok(dctx, node)
#endif
                   ;

#if defined(GGML_XDNA_EXPERIMENTAL)
        case GGML_OP_GATED_DELTA_NET:
            return ggml_xdna_device_gdn(dctx) != nullptr &&
                   ggml_xdna_gdn_shape_ok(node, dctx->gdn_shape);

        case GGML_OP_L2_NORM:
            return ggml_xdna_l2_absorbed_by_gdn(dctx, node);

        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return ggml_xdna_device_npu(dctx, ggml_xdna_slot_for_op(node->op)) != nullptr &&
                   !ggml_xdna_uses_repack_buft(dctx, node) &&
                   ggml_xdna_binary_shape_ok(node, dctx->op_min);

        case GGML_OP_RMS_NORM:
            return ggml_xdna_device_npu(dctx, GGML_XDNA_SLOT_RMS_NORM) != nullptr &&
                   node->src[0] &&
                   node->type == GGML_TYPE_F32 && node->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(node) && ggml_is_contiguous(node->src[0]) &&
                   ggml_are_same_shape(node, node->src[0]) &&
                   (size_t) node->ne[0] == dctx->rms_row &&
                   ggml_nelements(node) >= dctx->op_min &&
                   !ggml_xdna_uses_repack_buft(dctx, node);
#endif

        default:
            return false;
    }
}

static enum ggml_status ggml_xdna_compute_host_slice(ggml_backend_xdna_context * ctx,
                                                     struct ggml_cgraph * cgraph,
                                                     int i0, int i1) {
    if (!ctx->host) {
        GGML_LOG_ERROR("%s: own-graph host CPU backend is not initialized\n", "ggml-xdna");
        return GGML_STATUS_FAILED;
    }
    if (i1 <= i0) {
        return GGML_STATUS_SUCCESS;
    }

    const uint64_t t0 = ggml_xdna_now_ns();

    if (ctx->dev->host_profile) {
        // One node per submission so the time lands on a single op. Slower than
        // the batched path, so this is a diagnostic mode only.
        for (int i = i0; i < i1; i++) {
            struct ggml_cgraph one = ggml_graph_view(cgraph, i, i + 1);
            const uint64_t t_node = ggml_xdna_now_ns();
            const enum ggml_status st = ggml_backend_graph_compute(ctx->host, &one);
            const uint64_t ns = ggml_xdna_now_ns() - t_node;
            if (st != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("%s: nested CPU compute failed for node %d (%s)\n",
                               "ggml-xdna", i, ggml_op_desc(cgraph->nodes[i]));
                return st;
            }
            const struct ggml_tensor * n = cgraph->nodes[i];
            const char * name = ggml_op_desc(n);

            std::string key = name ? name : "?";
            if (ctx->dev->host_profile > 1) {
                char buf[128];
                snprintf(buf, sizeof(buf), " d[%lld,%lld,%lld,%lld]",
                         (long long) n->ne[0], (long long) n->ne[1],
                         (long long) n->ne[2], (long long) n->ne[3]);
                key += buf;
                if (n->src[0]) {
                    snprintf(buf, sizeof(buf), " s0[%lld,%lld,%lld,%lld]%s",
                             (long long) n->src[0]->ne[0], (long long) n->src[0]->ne[1],
                             (long long) n->src[0]->ne[2], (long long) n->src[0]->ne[3],
                             ggml_is_contiguous(n->src[0]) ? "c" : "-");
                    key += buf;
                }
            } else if (n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[1]) {
                // Shape decides whether an NPU submit can ever pay for itself,
                // so keep matmuls apart by their dimensions.
                char buf[64];
                snprintf(buf, sizeof(buf), " K=%lld M=%lld N=%lld x%lld",
                         (long long) n->src[1]->ne[0],
                         (long long) n->src[1]->ne[1],
                         (long long) n->ne[0],
                         (long long) (n->ne[2] * n->ne[3]));
                key += buf;
            }

            uint64_t bytes = ggml_nbytes(n);
            for (int s = 0; s < GGML_MAX_SRC && n->src[s]; s++) {
                bytes += ggml_nbytes(n->src[s]);
            }

            ctx->host_op_ns[key]    += ns;
            ctx->host_op_calls[key] += 1;
            ctx->host_op_bytes[key] += bytes;
        }
    } else {
        struct ggml_cgraph view = ggml_graph_view(cgraph, i0, i1);
        const enum ggml_status st = ggml_backend_graph_compute(ctx->host, &view);
        if (st != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: nested CPU compute failed for nodes [%d, %d)\n",
                           "ggml-xdna", i0, i1);
            return st;
        }
    }

    const uint64_t nested_ns = ggml_xdna_now_ns() - t0;
    ctx->ns_host += nested_ns;
    ctx->n_host_graph_runs++;
    ctx->n_host_nodes += (uint64_t) (i1 - i0);
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

#if defined(GGML_XDNA_EXPERIMENTAL)
        if (ggml_xdna_can_fuse_add_rms_mul(ctx, cgraph, i)) {
            if (!ggml_xdna_compute_add_rms_mul(ctx, cgraph, i)) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
#endif

        if (!ggml_xdna_is_local_op(ctx, node)) {
            int j = i + 1;
            while (j < cgraph->n_nodes) {
                struct ggml_tensor * next = cgraph->nodes[j];
                if ((next->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    j++;
                    continue;
                }
                if (
#if defined(GGML_XDNA_EXPERIMENTAL)
                    ggml_xdna_can_fuse_add_rms_mul(ctx, cgraph, j) ||
#endif
                    ggml_xdna_is_local_op(ctx, next)) {
                    break;
                }
                j++;
            }
            const enum ggml_status st = ggml_xdna_compute_host_slice(ctx, cgraph, i, j);
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
            i = j - 1;
            continue;
        }

        const uint64_t t_local = ctx->dev->host_profile ? ggml_xdna_now_ns() : 0;

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (!ggml_xdna_compute_mul_mat(ctx, node, nullptr)) {
                    return GGML_STATUS_FAILED;
                }
                break;

#if defined(GGML_XDNA_EXPERIMENTAL)
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                if (!ggml_xdna_compute_binary(ctx, node)) {
                    return GGML_STATUS_FAILED;
                }
                break;

            case GGML_OP_RMS_NORM:
                if (!ggml_xdna_compute_rms_norm(ctx, node)) {
                    return GGML_STATUS_FAILED;
                }
                break;

            case GGML_OP_GATED_DELTA_NET:
                if (!ggml_xdna_compute_gated_delta_net(ctx, node)) {
                    return GGML_STATUS_FAILED;
                }
                break;

            case GGML_OP_L2_NORM:
                // Absorbed into GDN packing (stage 5). No device write.
                break;
#endif

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: local-op table out of sync for %s\n",
                           __func__, ggml_op_desc(node));
        }

        if (t_local) {
            const std::string key = std::string("[local] ") + ggml_op_desc(node);
            uint64_t bytes = ggml_nbytes(node);
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                bytes += ggml_nbytes(node->src[s]);
            }
            ctx->host_op_ns[key]    += ggml_xdna_now_ns() - t_local;
            ctx->host_op_calls[key] += 1;
            ctx->host_op_bytes[key] += bytes;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i xdna_backend_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0xd1, 0xa5, 0x0d, 0xda, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb };
    return &guid;
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_backend_xdna_device_context * dev = ggml_xdna_device_context();

    ggml_backend_xdna_context * ctx = new ggml_backend_xdna_context;
    ctx->dev = dev;

    if (dev->own_graph) {
        ctx->host = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!ctx->host) {
            GGML_LOG_WARN("%s: OWN_GRAPH enabled but CPU backend init failed; "
                          "falling back to split scheduling\n", __func__);
            dev->own_graph = false;
        }
    }

    std::string live;
    for (const ggml_xdna_gemm_slot & slot : dev->gemm) {
        if (slot.enabled && slot.found) {
            live += live.empty() ? "" : ", ";
            live += slot.env_name;
        }
    }
#if defined(GGML_XDNA_EXPERIMENTAL)
    // Load every available experimental kernel up front.
    for (int i = 0; i < GGML_XDNA_N_SLOTS; i++) {
        if (ggml_xdna_device_npu(dev, (ggml_xdna_slot_idx) i)) {
            live += live.empty() ? "" : ", ";
            live += dev->slots[i].env_name;
        }
    }
    if (ggml_xdna_device_gdn(dev)) {
        live += live.empty() ? "" : ", ";
        live += "GDN";
    }
#endif

    GGML_LOG_INFO("%s: XDNA backend init: device=%s, kernels=%s, own_graph=%s\n",
                  __func__,
                  dev->npu_name.c_str(),
                  live.empty() ? "none" : live.c_str(),
                  dev->own_graph ? "on" : "off");

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "XDNA";
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    return ((ggml_backend_xdna_device_context *)dev->context)->description.c_str();
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_xdna_init();
}

// ** MUL_MAT **
//
// The design takes bf16 operands swizzled into the MMUL micro-tile order and
// returns f32 in the same order, so every operand has to be laid out for it.
// Weights are converted once at model load into device-visible memory; only the
// activations and the result are rearranged per submit.
//
// Streams a submit consumes, in the order the runtime pushes them:
//   A  [grid_r][nb][kt]  swizzled (tile_m x tile_k)
//   B  [grid_c][nb][kt]  swizzled (tile_k x tile_n)
//   C  [grid_c][nb][grid_r] swizzled (tile_m x tile_n)
// with the column block a worker sees at n0 = (nb*grid_c + j)*tile_n.

static inline size_t ggml_xdna_swz_a(int mm, int kk, int tile_k) {
    const int r = GGML_XDNA_GEMM_MAC_R, s = GGML_XDNA_GEMM_MAC_S;
    return ((size_t) (mm / r) * (tile_k / s) + kk / s) * r * s + (size_t) (mm % r) * s + (kk % s);
}

static inline size_t ggml_xdna_swz_b(int kk, int nn, int tile_n) {
    const int s = GGML_XDNA_GEMM_MAC_S, t = GGML_XDNA_GEMM_MAC_T;
    return ((size_t) (kk / s) * (tile_n / t) + nn / t) * s * t + (size_t) (kk % s) * t + (nn % t);
}

static inline size_t ggml_xdna_swz_c(int mm, int nn, int tile_n) {
    const int r = GGML_XDNA_GEMM_MAC_R, t = GGML_XDNA_GEMM_MAC_T;
    return ((size_t) (mm / r) * (tile_n / t) + nn / t) * r * t + (size_t) (mm % r) * t + (nn % t);
}

// Bytes a full weight tensor occupies after repack.
// Layout: [n_block][k_block][grid_c][kt][swizzled tile]
static size_t ggml_xdna_gemm_weight_bytes(int64_t k_src, int64_t n_src,
                                          const ggml_xdna_gemm_shape & s) {
    const int n_cols  = s.grid_c * s.tile_n;
    const int k_slice = s.kt * s.tile_k;
    const int n_blocks = (int) ((n_src + n_cols - 1) / n_cols);
    const int k_blocks = (int) ((k_src + k_slice - 1) / k_slice);
    return (size_t) n_blocks * k_blocks * s.grid_c * s.kt * s.tile_k * s.tile_n * sizeof(ggml_bf16_t);
}

static size_t ggml_xdna_gemm_b_block_bytes(const ggml_xdna_gemm_shape & s) {
    return (size_t) s.grid_c * s.kt * s.tile_k * s.tile_n * sizeof(ggml_bf16_t);
}

// Pack weights as [n_block][k_block][grid_c][kt][tile].
static void ggml_xdna_gemm_pack_weights(const ggml_tensor *          tensor,
                                        const void *                 data,
                                        ggml_bf16_t *                out,
                                        const ggml_xdna_gemm_shape & s) {
    const int64_t k_src = tensor->ne[0];
    const int64_t n_src = tensor->ne[1];
    const int     k_slice = s.kt * s.tile_k;
    const int     n_cols  = s.grid_c * s.tile_n;
    const int     k_blocks = (int) ((k_src + k_slice - 1) / k_slice);
    const int     n_blocks = (int) ((n_src + n_cols - 1) / n_cols);

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    const size_t row_bytes = ggml_row_size(tensor->type, k_src);

    std::vector<float> row(k_blocks * k_slice, 0.0f);
    const size_t b_block = ggml_xdna_gemm_b_block_bytes(s) / sizeof(ggml_bf16_t);

    memset(out, 0, (size_t) n_blocks * k_blocks * b_block * sizeof(ggml_bf16_t));

    for (int64_t n = 0; n < n_src; n++) {
        traits->to_float((const char *) data + (size_t) n * row_bytes, row.data(), k_src);

        const int n_blk = (int) (n / s.tile_n);
        const int nn    = (int) (n % s.tile_n);
        const int nb    = n_blk / s.grid_c;
        const int j     = n_blk % s.grid_c;

        for (int kb = 0; kb < k_blocks; kb++) {
            ggml_bf16_t * base = out + ((size_t) nb * k_blocks + kb) * b_block;
            for (int kt = 0; kt < s.kt; kt++) {
                const int k0 = kb * k_slice + kt * s.tile_k;
                ggml_bf16_t * tile = base + ((size_t) j * s.kt + kt) * s.tile_k * s.tile_n;
                for (int kk = 0; kk < s.tile_k; kk++) {
                    tile[ggml_xdna_swz_b(kk, nn, s.tile_n)] =
                        GGML_FP32_TO_BF16(row[k0 + kk]);
                }
            }
        }
    }
}

// Pack activations for one K-slice starting at k0 (length s.kt*tile_k).
static void ggml_xdna_gemm_pack_activations(const float *                src,
                                            int64_t                      k_src,
                                            int64_t                      m_src,
                                            int64_t                      m0,
                                            int64_t                      k0,
                                            ggml_bf16_t *                out,
                                            const ggml_xdna_gemm_shape & s,
                                            int                          n_threads) {
    const size_t per_mb  = (size_t) s.kt * s.tile_m * s.tile_k;
    const int    k_slice = s.kt * s.tile_k;
    const bool   k_full  = (k0 + k_slice) <= k_src;
    const int    n_units = s.grid_r * s.mb;

    // Each unit owns one contiguous per_mb region, so the zero fill for padded
    // rows/columns can be done per unit and the loop parallelised.
#ifdef GGML_XDNA_OPENMP
#   pragma omp parallel for schedule(static) num_threads(n_threads)
#else
    GGML_UNUSED(n_threads);
#endif
    for (int unit = 0; unit < n_units; unit++) {
        const int i  = unit / s.mb;
        const int mb = unit % s.mb;

        ggml_bf16_t * base = out + (size_t) unit * per_mb;
        const int64_t m_base = m0 + (int64_t) (mb * s.grid_r + i) * s.tile_m;

        if (m_base >= m_src || !k_full || m_base + s.tile_m > m_src) {
            memset(base, 0, per_mb * sizeof(ggml_bf16_t));
        }
        if (m_base >= m_src) {
            continue;
        }

        for (int kt = 0; kt < s.kt; kt++) {
            ggml_bf16_t * tile = base + (size_t) kt * s.tile_m * s.tile_k;
            const int64_t k_base = k0 + (int64_t) kt * s.tile_k;

            for (int mm = 0; mm < s.tile_m; mm++) {
                const int64_t m = m_base + mm;
                if (m >= m_src) {
                    break;
                }
                const float * row = src + (size_t) m * k_src + k_base;
                if (k_full) {
                    for (int kk = 0; kk < s.tile_k; kk++) {
                        tile[ggml_xdna_swz_a(mm, kk, s.tile_k)] =
                            GGML_FP32_TO_BF16(row[kk]);
                    }
                } else {
                    for (int kk = 0; kk < s.tile_k; kk++) {
                        if (k_base + kk < k_src) {
                            tile[ggml_xdna_swz_a(mm, kk, s.tile_k)] =
                                GGML_FP32_TO_BF16(row[kk]);
                        }
                    }
                }
            }
        }
    }
}

#if defined(GGML_XDNA_EXPERIMENTAL)
// Pack activations for one K-slice. Supports dim-0-contiguous tensors with
// arbitrary row/head strides (permuted K/Q after attention reshape).
static void ggml_xdna_gemm_pack_activations_strided(const char *                src,
                                                    size_t                      nb1,
                                                    int64_t                     k_src,
                                                    int64_t                     m_src,
                                                    int64_t                     m0,
                                                    int64_t                     k0,
                                                    ggml_bf16_t *               out,
                                                    const ggml_xdna_gemm_shape & s,
                                                    int                         n_threads) {
    const size_t per_mb  = (size_t) s.kt * s.tile_m * s.tile_k;
    const int    k_slice = s.kt * s.tile_k;
    const bool   k_full  = (k0 + k_slice) <= k_src;
    const int    n_units = s.grid_r * s.mb;

#ifdef GGML_XDNA_OPENMP
#   pragma omp parallel for schedule(static) num_threads(n_threads)
#else
    GGML_UNUSED(n_threads);
#endif
    for (int unit = 0; unit < n_units; unit++) {
        const int i  = unit / s.mb;
        const int mb = unit % s.mb;

        ggml_bf16_t * base = out + (size_t) unit * per_mb;
        const int64_t m_base = m0 + (int64_t) (mb * s.grid_r + i) * s.tile_m;

        if (m_base >= m_src || !k_full || m_base + s.tile_m > m_src) {
            memset(base, 0, per_mb * sizeof(ggml_bf16_t));
        }
        if (m_base >= m_src) {
            continue;
        }

        for (int kt = 0; kt < s.kt; kt++) {
            ggml_bf16_t * tile = base + (size_t) kt * s.tile_m * s.tile_k;
            const int64_t k_base = k0 + (int64_t) kt * s.tile_k;

            for (int mm = 0; mm < s.tile_m; mm++) {
                const int64_t m = m_base + mm;
                if (m >= m_src) {
                    break;
                }
                const float * row = (const float *) (src + (size_t) m * nb1) + k_base;
                if (k_full) {
                    for (int kk = 0; kk < s.tile_k; kk++) {
                        tile[ggml_xdna_swz_a(mm, kk, s.tile_k)] =
                            GGML_FP32_TO_BF16(row[kk]);
                    }
                } else {
                    for (int kk = 0; kk < s.tile_k; kk++) {
                        if (k_base + kk < k_src) {
                            tile[ggml_xdna_swz_a(mm, kk, s.tile_k)] =
                                GGML_FP32_TO_BF16(row[kk]);
                        }
                    }
                }
            }
        }
    }
}

// Pack one column block of an F32/F16 activation matrix as the B operand.
// src is [K, N] with dim-0 contiguous; nb1 is the byte stride of dimension 1.
static void ggml_xdna_gemm_pack_b_activations(const char *                 src,
                                              size_t                       nb1,
                                              enum ggml_type               type,
                                              int64_t                      k_src,
                                              int64_t                      n_src,
                                              int64_t                      n0,
                                              int64_t                      k0,
                                              ggml_bf16_t *                out,
                                              const ggml_xdna_gemm_shape & s,
                                              int                          n_threads) {
    const size_t b_elems = (size_t) s.grid_c * s.kt * s.tile_k * s.tile_n;
    const size_t ts      = ggml_type_size(type);

    memset(out, 0, b_elems * sizeof(ggml_bf16_t));

#ifdef GGML_XDNA_OPENMP
#   pragma omp parallel for schedule(static) num_threads(n_threads) collapse(2)
#else
    GGML_UNUSED(n_threads);
#endif
    for (int j = 0; j < s.grid_c; j++) {
        for (int nn = 0; nn < s.tile_n; nn++) {
            const int64_t n = n0 + (int64_t) j * s.tile_n + nn;
            if (n >= n_src) {
                continue;
            }
            const char * col = src + (size_t) n * nb1 + (size_t) k0 * ts;
            for (int kt = 0; kt < s.kt; kt++) {
                ggml_bf16_t * tile = out + ((size_t) j * s.kt + kt) * s.tile_k * s.tile_n;
                const int64_t k_base = (int64_t) kt * s.tile_k;
                for (int kk = 0; kk < s.tile_k; kk++) {
                    if (k0 + k_base + kk >= k_src) {
                        continue;
                    }
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = GGML_FP16_TO_FP32(((const ggml_fp16_t *) col)[k_base + kk]);
                    } else {
                        v = ((const float *) col)[k_base + kk];
                    }
                    tile[ggml_xdna_swz_b(kk, nn, s.tile_n)] = GGML_FP32_TO_BF16(v);
                }
            }
        }
    }
}
#endif

// Sum the K-slices of one column block straight out of the device buffers into
// dst. Units write disjoint (row, column) regions, so the flattened loop is
// safe to run in parallel, and dst is touched exactly once per column block.
static void ggml_xdna_gemm_scatter_result(const float * const *        c_slices,
                                          int                          n_slices,
                                          float *                      dst,
                                          int64_t                      n_dst,
                                          int64_t                      n0,
                                          int64_t                      m_src,
                                          int64_t                      m0,
                                          const ggml_xdna_gemm_shape & s,
                                          bool                         accumulate,
                                          int                          n_threads) {
    const int n_units = s.grid_c * s.mb * s.grid_r;

#ifdef GGML_XDNA_OPENMP
#   pragma omp parallel for schedule(static) num_threads(n_threads)
#else
    GGML_UNUSED(n_threads);
#endif
    for (int unit = 0; unit < n_units; unit++) {
        const int i  = unit % s.grid_r;
        const int mb = (unit / s.grid_r) % s.mb;
        const int j  = unit / (s.grid_r * s.mb);

        const size_t  tile_off = (size_t) unit * s.tile_m * s.tile_n;
        const int64_t m_base   = m0 + (int64_t) (mb * s.grid_r + i) * s.tile_m;
        if (m_base >= m_src) {
            continue;
        }

        for (int mm = 0; mm < s.tile_m; mm++) {
            const int64_t m = m_base + mm;
            if (m >= m_src) {
                break;
            }
            float * row = dst + (size_t) m * n_dst;

            for (int nn = 0; nn < s.tile_n; nn++) {
                const int64_t n = n0 + (int64_t) j * s.tile_n + nn;
                if (n >= n_dst) {
                    continue;
                }
                const size_t e = tile_off + ggml_xdna_swz_c(mm, nn, s.tile_n);

                float v = c_slices[0][e];
                for (int k = 1; k < n_slices; k++) {
                    v += c_slices[k][e];
                }
                row[n] = accumulate ? row[n] + v : v;
            }
        }
    }
}

// ** repack buffer **
//
// The XRT BO holds the swizzled BF16 the GEMM design reads, so a submit never
// copies weights. With keep_src, tensor->data is a separate host allocation of
// the original GGUF bytes (same DRAM as `-ngl 0`); packed[] maps each weight
// back to its BO address. Without keep_src, tensor->data is the swizzle and
// the nested CPU cannot read it.

struct ggml_xdna_host_src {
    void * ptr  = nullptr;
    size_t size = 0;
};

struct ggml_backend_xdna_buffer_context {
    ggml_xdna_buf * buf  = nullptr;
    void *          base = nullptr;
    size_t          size = 0;
    std::vector<ggml_xdna_host_src> host_src;
    std::unordered_map<const ggml_tensor *, char *> packed;
};

static char * ggml_xdna_weight_packed(ggml_backend_xdna_buffer_context * ctx,
                                      const ggml_tensor * tensor) {
    auto it = ctx->packed.find(tensor);
    if (it != ctx->packed.end()) {
        return it->second;
    }
    return (char *) tensor->data;
}

static ggml_xdna_gemm_slot * ggml_xdna_gemm_for_weight(ggml_backend_xdna_device_context * dctx,
                                                       const ggml_tensor * tensor) {
    if (!tensor || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return nullptr;
    }
    return ggml_xdna_gemm_for_shape(dctx, tensor->ne[0], tensor->ne[1]);
}

// Weight-buffer probe: return false on the host buft so select_weight_buft
// lands on XDNA_REPACK. Quantized was used as a proxy for "model weight";
// BF16/F16 GGUF weights also pack (to_float) and must take the same path.
// F32 src0 with matching K/N is usually an activation (QK/AV has ne[2]!=1).
static bool ggml_xdna_gemm_should_repack_weight(ggml_backend_xdna_device_context * dctx,
                                                const ggml_tensor * w) {
    if (!ggml_xdna_gemm_for_weight(dctx, w)) {
        return false;
    }
    if (w->type == GGML_TYPE_F32) {
        return false;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(w->type);
    return traits && traits->to_float != nullptr;
}

static void ggml_backend_xdna_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_backend_xdna_buffer_context * ctx = (ggml_backend_xdna_buffer_context *) buffer->context;
    for (ggml_xdna_host_src & src : ctx->host_src) {
        ggml_aligned_free(src.ptr, src.size);
    }
    ggml_xdna_buf_free(ctx->buf);
    delete ctx;
}

static enum ggml_status ggml_backend_xdna_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor) {
    ggml_backend_xdna_buffer_context * ctx = (ggml_backend_xdna_buffer_context *) buffer->context;
    ggml_backend_xdna_device_context * dctx =
        (ggml_backend_xdna_device_context *) buffer->buft->device->context;

    if (!dctx->keep_src || !ggml_xdna_gemm_for_weight(dctx, tensor)) {
        return GGML_STATUS_SUCCESS;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    void * host = ggml_aligned_malloc(nbytes);
    if (!host) {
        GGML_LOG_ERROR("%s: host copy of '%s' (%zu bytes) failed\n",
                       "ggml-xdna", tensor->name, nbytes);
        return GGML_STATUS_FAILED;
    }

    ctx->packed[tensor] = (char *) tensor->data;
    ctx->host_src.push_back({ host, nbytes });
    tensor->data = host;
    return GGML_STATUS_SUCCESS;
}

static void * ggml_backend_xdna_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_xdna_buffer_context *) buffer->context)->base;
}

static void ggml_backend_xdna_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *         tensor,
                                                const void *          data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_xdna_buffer_context * ctx = (ggml_backend_xdna_buffer_context *) buffer->context;
    ggml_backend_xdna_device_context * dctx =
        (ggml_backend_xdna_device_context *) buffer->buft->device->context;

    ggml_xdna_gemm_slot * slot = ggml_xdna_gemm_for_weight(dctx, tensor);
    if (!slot) {
        // Not a tensor we repack: it was placed here only because it shares a
        // buffer with ones we do, so keep the plain bytes.
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));

    if (dctx->keep_src) {
        memcpy(tensor->data, data, size);
    }

    char * packed = ggml_xdna_weight_packed(ctx, tensor);
    ggml_xdna_gemm_pack_weights(tensor, data, (ggml_bf16_t *) packed, slot->shape);

    const size_t w_off = (size_t) (packed - (char *) ctx->base);
    ggml_xdna_buf_sync_to_device(ctx->buf, w_off,
        ggml_xdna_gemm_weight_bytes(tensor->ne[0], tensor->ne[1], slot->shape));
}

static void ggml_backend_xdna_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *   tensor,
                                                void *                data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_xdna_device_context * dctx =
        (ggml_backend_xdna_device_context *) buffer->buft->device->context;

    if (dctx->keep_src || !ggml_xdna_gemm_for_weight(dctx, tensor)) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    // Without the source copy only the swizzled bf16 is left, and reading it
    // back would mean requantizing into the source format.
    GGML_ABORT("ggml-xdna: reading back repacked weight '%s' (%s [%lld,%lld]) is not supported\n",
               tensor->name, ggml_type_name(tensor->type),
               (long long) tensor->ne[0], (long long) tensor->ne[1]);
}

static void ggml_backend_xdna_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_xdna_buffer_context * ctx = (ggml_backend_xdna_buffer_context *) buffer->context;
    memset(ctx->base, value, ctx->size);
    ggml_xdna_buf_sync_to_device(ctx->buf, 0, ctx->size);
}

static const ggml_backend_buffer_i ggml_backend_xdna_buffer_i = {
    /* .free_buffer     = */ ggml_backend_xdna_buffer_free,
    /* .get_base        = */ ggml_backend_xdna_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_xdna_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_xdna_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_xdna_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_xdna_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_xdna_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "XDNA_REPACK";
}

static ggml_backend_buffer_t ggml_backend_xdna_buffer_type_alloc(ggml_backend_buffer_type_t buft,
                                                                 size_t                     size) {
    ggml_backend_xdna_device_context * dctx =
        (ggml_backend_xdna_device_context *) buft->device->context;

    // Any GEMM context can own the allocation: the BOs are host-only and all
    // artifacts sit on the same device with the same memory topology.
    ggml_xdna_npu * npu = nullptr;
    for (int i = 0; i < GGML_XDNA_N_GEMM && !npu; i++) {
        npu = ggml_xdna_device_gemm(dctx, (ggml_xdna_gemm_idx) i);
    }
    if (!npu) {
        return nullptr;
    }

    ggml_xdna_buf * buf = ggml_xdna_buf_alloc(npu, size);
    if (!buf) {
        return nullptr;
    }

    ggml_backend_xdna_buffer_context * ctx = new ggml_backend_xdna_buffer_context;
    ctx->buf  = buf;
    ctx->base = ggml_xdna_buf_host(buf);
    ctx->size = size;

    return ggml_backend_buffer_init(buft, ggml_backend_xdna_buffer_i, ctx, size);
}

static size_t ggml_backend_xdna_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

static size_t ggml_backend_xdna_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                           const ggml_tensor *        tensor) {
    ggml_backend_xdna_device_context * dctx =
        (ggml_backend_xdna_device_context *) buft->device->context;

    if (tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return ggml_nbytes(tensor);
    }

    for (const ggml_xdna_gemm_slot & slot : dctx->gemm) {
        if (!slot.found) {
            continue;
        }
        if (ggml_xdna_gemm_shape_ok(tensor->ne[0], tensor->ne[1])) {
            return ggml_xdna_gemm_weight_bytes(tensor->ne[0], tensor->ne[1], slot.shape);
        }
    }

    return ggml_nbytes(tensor);
}

static bool ggml_backend_xdna_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_t ggml_backend_xdna_repack_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_xdna_device_context * dctx = (ggml_backend_xdna_device_context *) dev->context;

    if (dctx->repack_buft.device == nullptr) {
        dctx->repack_buft = {
            /* .iface = */ {
                /* .get_name       = */ ggml_backend_xdna_buffer_type_name,
                /* .alloc_buffer   = */ ggml_backend_xdna_buffer_type_alloc,
                /* .get_alignment  = */ ggml_backend_xdna_buffer_type_get_alignment,
                /* .get_max_size   = */ NULL,
                /* .get_alloc_size = */ ggml_backend_xdna_buffer_type_get_alloc_size,
                /* .is_host        = */ ggml_backend_xdna_buffer_type_is_host,
            },
            /* .device  = */ dev,
            /* .context = */ nullptr,
        };
    }

    return &dctx->repack_buft;
}

// A MUL_MAT is ours only when src0 was repacked into our buffer: no other layout
// matches the design, and equally nothing else can read ours.
static ggml_xdna_gemm_slot * ggml_xdna_gemm_slot_for_op(ggml_backend_xdna_device_context * dctx,
                                                       const ggml_tensor *                op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (!src0 || !src1 || op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return nullptr;
    }
    if (!src0->buffer || src0->buffer->buft != &dctx->repack_buft) {
        return nullptr;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return nullptr;
    }
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return nullptr;
    }
    if (src1->ne[0] != src0->ne[0] || op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1]) {
        return nullptr;
    }

    return ggml_xdna_gemm_for_shape(dctx, src0->ne[0], src0->ne[1], src1->ne[1]);
}

static bool ggml_xdna_gemm_supports(ggml_backend_xdna_device_context * dctx,
                                    const ggml_tensor *                op) {
    return ggml_xdna_gemm_slot_for_op(dctx, op) != nullptr;
}

// A repacked MUL_MAT the NPU would not take (decode-sized M) is better off on
// the nested CPU: with keep_src the original GGUF bytes live in host DRAM, so
// the quantized kernels beat a bf16 host GEMM on twice the weight traffic.
static bool ggml_xdna_gemm_runs_here(ggml_backend_xdna_device_context * dctx,
                                     const ggml_tensor *                op) {
    if (!ggml_xdna_gemm_slot_for_op(dctx, op)) {
        return false;
    }
    return !dctx->keep_src || op->src[1]->ne[1] >= dctx->gemm_m_min;
}

#if defined(GGML_XDNA_EXPERIMENTAL)
// Act x act MUL_MAT (attention QK/AV): pad K/N up to the artifact quantum and
// run on the same GEMM design with a runtime-packed B. src0 may be F16 (KV
// cache); src1/dst are F32.
static bool ggml_xdna_act_gemm_ok(ggml_backend_xdna_device_context * dctx,
                                  const ggml_tensor *                op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (!dctx->gemm_enabled || !dctx->npu_enabled || !src0 || !src1) {
        return false;
    }
    if (!dctx->act_gemm) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }
    // Permuted K/V are common; require only dim-0 contiguous (element stride).
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(op)) {
        return false;
    }
    if (ggml_xdna_uses_repack_buft(dctx, op)) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0] || op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1]) {
        return false;
    }
    if (op->ne[3] != 1 || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (m < dctx->gemm_m_min || k <= 0 || n <= 0) {
        return false;
    }

    const int     k_slice = GGML_XDNA_GEMM_KT * GGML_XDNA_GEMM_TILE_K;
    const int     n_cols  = GGML_XDNA_GEMM_GRID_C * GGML_XDNA_GEMM_TILE_N;
    const int64_t k_pad   = ((k + k_slice - 1) / k_slice) * k_slice;
    const int64_t n_pad   = ((n + n_cols  - 1) / n_cols)  * n_cols;

    // Skip GDN-scale mats (64/128): submit overhead dominates there.
    if ((int64_t) k * m * n < (int64_t) 256 * 512 * 256) {
        return false;
    }

    return ggml_xdna_gemm_for_shape(dctx, k_pad, n_pad, m) != nullptr;
}

static bool ggml_xdna_compute_mul_mat_act(ggml_backend_xdna_context * ctx,
                                          const struct ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    const int64_t k_src = src0->ne[0];
    const int64_t n_src = src0->ne[1];
    const int64_t m_src = src1->ne[1];

    const int     k_slice = GGML_XDNA_GEMM_KT * GGML_XDNA_GEMM_TILE_K;
    const int     n_cols  = GGML_XDNA_GEMM_GRID_C * GGML_XDNA_GEMM_TILE_N;
    const int64_t k_pad   = ((k_src + k_slice - 1) / k_slice) * k_slice;
    const int64_t n_pad   = ((n_src + n_cols  - 1) / n_cols)  * n_cols;

    ggml_xdna_gemm_slot * slot = ggml_xdna_gemm_for_shape(ctx->dev, k_pad, n_pad, m_src);
    if (!slot || !slot->npu) {
        return false;
    }

    const ggml_xdna_gemm_shape & s = slot->shape;
    const int64_t m_block = (int64_t) s.mb * s.grid_r * s.tile_m;
    const int     k_blocks = (int) (k_pad / k_slice);
    const int     nth      = ctx->dev->host_threads;
    const int     kb_max   = std::min(ggml_xdna_npu_gemm_kb_max(slot->npu), GGML_XDNA_GEMM_KB_LIMIT);
    if (kb_max < 1) {
        return false;
    }

    const int64_t n_heads = node->ne[2];
    const int64_t n_h0    = src0->ne[2];
    const int64_t n_h1    = src1->ne[2];

    ggml_xdna_npu_gemm_acquire(slot->npu);

    const float * c_slices[GGML_XDNA_GEMM_KB_LIMIT];

    for (int64_t h = 0; h < n_heads; h++) {
        const int64_t h0 = h * n_h0 / n_heads;
        const int64_t h1 = h * n_h1 / n_heads;

        const char * a_base = (const char *) src1->data + (size_t) h1 * src1->nb[2];
        const char * b_base = (const char *) src0->data + (size_t) h0 * src0->nb[2];
        float *      dst    = (float *) ((char *) node->data + (size_t) h * node->nb[2]);

        for (int64_t m0 = 0; m0 < m_src; m0 += m_block) {
            for (int kb0 = 0; kb0 < k_blocks; kb0 += kb_max) {
                const int kb_n = std::min(kb_max, k_blocks - kb0);

                const uint64_t t_pack = ggml_xdna_now_ns();
                for (int kb = 0; kb < kb_n; kb++) {
                    ggml_xdna_gemm_pack_activations_strided(
                        a_base, src1->nb[1], k_src, m_src, m0,
                        (int64_t) (kb0 + kb) * k_slice,
                        (ggml_bf16_t *) ggml_xdna_npu_gemm_a_slice(slot->npu, kb),
                        s, nth);
                    c_slices[kb] = ggml_xdna_npu_gemm_c_slice(slot->npu, kb);
                }
                ctx->ns_gemm_pack += ggml_xdna_now_ns() - t_pack;

                for (int64_t n0 = 0; n0 < n_pad; n0 += n_cols) {
                    const uint64_t t_b = ggml_xdna_now_ns();
                    for (int kb = 0; kb < kb_n; kb++) {
                        ggml_xdna_gemm_pack_b_activations(
                            b_base, src0->nb[1], src0->type, k_src, n_src, n0,
                            (int64_t) (kb0 + kb) * k_slice,
                            (ggml_bf16_t *) ggml_xdna_npu_gemm_b_slice(slot->npu, kb),
                            s, nth);
                    }
                    ctx->ns_gemm_pack += ggml_xdna_now_ns() - t_b;

                    if (!ggml_xdna_npu_gemm_submit_ab_list(slot->npu, kb_n)) {
                        ggml_xdna_npu_gemm_release(slot->npu);
                        return false;
                    }

                    const uint64_t t_scatter = ggml_xdna_now_ns();
                    ggml_xdna_gemm_scatter_result(c_slices, kb_n, dst, n_src, n0, m_src, m0, s,
                                                  /*accumulate=*/ kb0 > 0, nth);
                    ctx->ns_gemm_pack += ggml_xdna_now_ns() - t_scatter;
                }
            }
        }
    }

    ggml_xdna_npu_gemm_release(slot->npu);
    return true;
}
#endif

// The repack buffer is offered for every weight, so it also gets tried for the
// norm weights that feed MUL. Only MUL_MAT can read that layout, and the
// elementwise kernels take plain host pointers, so anything else has to refuse
// operands that landed there - otherwise the weight is placed in a buffer whose
// contents nothing can read back.
static bool ggml_xdna_uses_repack_buft(const ggml_backend_xdna_device_context * dctx,
                                       const ggml_tensor *                      op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * src = op->src[i];
        if (src && src->buffer && src->buffer->buft == &dctx->repack_buft) {
            return true;
        }
    }
    return false;
}

// Host-side GEMM from the swizzled bf16 weight layout. Used when M is below
// gemm_m_min: an NPU submit would still move a full B block for almost no MACs.
//
// The swizzle keeps MAC_S x MAC_T (k x n) contiguous, so the weights are walked
// in storage order and accumulated into a tile_n-wide column block. Decode then
// streams each weight tensor once instead of gathering one K column per output.
static void ggml_xdna_gemm_host(const ggml_bf16_t *          w,
                                const float *                a,
                                float *                      dst,
                                int64_t                      k_src,
                                int64_t                      m_src,
                                int64_t                      n_dst,
                                const ggml_xdna_gemm_shape & s,
                                int                          n_threads) {
    const int    ks       = GGML_XDNA_GEMM_MAC_S;
    const int    ns       = GGML_XDNA_GEMM_MAC_T;
    const int    k_slice  = s.kt * s.tile_k;
    const int    k_blocks = (int) ((k_src + k_slice - 1) / k_slice);
    const size_t b_block  = ggml_xdna_gemm_b_block_bytes(s) / sizeof(ggml_bf16_t);
    const int    n_groups = (int) ((n_dst + s.tile_n - 1) / s.tile_n);

#ifdef GGML_XDNA_OPENMP
#   pragma omp parallel for schedule(static) num_threads(n_threads)
#else
    GGML_UNUSED(n_threads);
#endif
    for (int g = 0; g < n_groups; g++) {
        const int nb = g / s.grid_c;
        const int j  = g % s.grid_c;

        std::vector<float> acc((size_t) m_src * s.tile_n, 0.0f);

        for (int kb = 0; kb < k_blocks; kb++) {
            const ggml_bf16_t * base = w + ((size_t) nb * k_blocks + kb) * b_block;
            for (int kt = 0; kt < s.kt; kt++) {
                const ggml_bf16_t * tile = base + ((size_t) j * s.kt + kt) * s.tile_k * s.tile_n;
                const int64_t k0 = (int64_t) kb * k_slice + (int64_t) kt * s.tile_k;
                for (int ki = 0; ki < s.tile_k / ks; ki++) {
                    for (int kk = 0; kk < ks; kk++) {
                        const int64_t k = k0 + (int64_t) ki * ks + kk;
                        if (k >= k_src) {
                            break;
                        }
                        // Column groups are independent accumulators, so keeping
                        // them in the inner loop leaves the FMAs free to pipeline.
                        for (int ni = 0; ni < s.tile_n / ns; ni++) {
                            const ggml_bf16_t * blk =
                                tile + ((size_t) ki * (s.tile_n / ns) + ni) * ks * ns + kk * ns;
                            float wv[GGML_XDNA_GEMM_MAC_T];
                            for (int nn = 0; nn < ns; nn++) {
                                wv[nn] = GGML_BF16_TO_FP32(blk[nn]);
                            }
                            for (int64_t m = 0; m < m_src; m++) {
                                const float av = a[(size_t) m * k_src + k];
                                float * ac = acc.data() + (size_t) m * s.tile_n + ni * ns;
                                for (int nn = 0; nn < ns; nn++) {
                                    ac[nn] += av * wv[nn];
                                }
                            }
                        }
                    }
                }
            }
        }

        const int64_t n0 = (int64_t) g * s.tile_n;
        const int64_t nn_max = std::min<int64_t>(s.tile_n, n_dst - n0);
        for (int64_t m = 0; m < m_src; m++) {
            memcpy(dst + (size_t) m * n_dst + n0,
                   acc.data() + (size_t) m * s.tile_n,
                   (size_t) nn_max * sizeof(float));
        }
    }
}

static bool ggml_xdna_compute_mul_mat(ggml_backend_xdna_context * ctx,
                                      const struct ggml_tensor * node,
                                      float * dst_override) {
#if defined(GGML_XDNA_EXPERIMENTAL)
    if (dst_override == nullptr && ggml_xdna_act_gemm_ok(ctx->dev, node)) {
        return ggml_xdna_compute_mul_mat_act(ctx, node);
    }
#endif

    ggml_xdna_gemm_slot * slot = ggml_xdna_gemm_slot_for_op(ctx->dev, node);
    if (!slot || !slot->npu) {
        return false;
    }

    const ggml_xdna_gemm_shape & s = slot->shape;

    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * src0 = node->src[0];

    const int64_t k_src = src1->ne[0];
    const int64_t m_src = src1->ne[1];
    const int64_t n_dst = node->ne[0];

    ggml_backend_xdna_buffer_context * wctx =
        (ggml_backend_xdna_buffer_context *) src0->buffer->context;
    const char * w = ggml_xdna_weight_packed(wctx, src0);
    const size_t w_base = (size_t) (w - (char *) wctx->base);

    const float * src = (const float *) src1->data;
    float *       dst = dst_override ? dst_override : (float *) node->data;

    if (m_src < ctx->dev->gemm_m_min) {
        const uint64_t t0 = ggml_xdna_now_ns();
        ggml_xdna_gemm_host((const ggml_bf16_t *) w, src, dst,
                            k_src, m_src, n_dst, s,
                            ctx->n_threads > 0 ? ctx->n_threads : ctx->dev->host_threads);
        ctx->ns_gemm_pack += ggml_xdna_now_ns() - t0;
#if defined(GGML_XDNA_EXPERIMENTAL)
        if (dst_override == nullptr) {
            ggml_xdna_chain_store(ctx, node->data, dst, n_dst, m_src);
        }
#endif
        return true;
    }

    const int64_t m_block = (int64_t) s.mb * s.grid_r * s.tile_m;
    const int64_t n_cols  = (int64_t) s.grid_c * s.tile_n;
    const int64_t k_slice = (int64_t) s.kt * s.tile_k;
    const int     k_blocks = (int) ((k_src + k_slice - 1) / k_slice);
    const size_t  b_bytes = ggml_xdna_gemm_b_block_bytes(s);

    const int nth    = ctx->dev->host_threads;
    const int kb_max = std::min(ggml_xdna_npu_gemm_kb_max(slot->npu), GGML_XDNA_GEMM_KB_LIMIT);
    const int n_banks = std::max(1, ggml_xdna_npu_gemm_n_banks(slot->npu));
    if (kb_max < 1) {
        return false;
    }

    ggml_xdna_npu_gemm_acquire(slot->npu);

    // Pending scatter from the previous async submit (other bank).
    bool pending = false;
    int  pend_bank = 0;
    int  pend_kb_n = 0;
    int64_t pend_n0 = 0;
    int64_t pend_m0 = 0;
    int  pend_kb0 = 0;
    const float * pend_c[GGML_XDNA_GEMM_KB_LIMIT];

    auto flush_pending = [&]() -> bool {
        if (!pending) {
            return true;
        }
        if (!ggml_xdna_npu_gemm_runlist_wait(slot->npu)) {
            return false;
        }
        const uint64_t t_scatter = ggml_xdna_now_ns();
        ggml_xdna_gemm_scatter_result(pend_c, pend_kb_n, dst, n_dst, pend_n0, m_src, pend_m0, s,
                                      /*accumulate=*/ pend_kb0 > 0, nth);
        ctx->ns_gemm_scatter += ggml_xdna_now_ns() - t_scatter;
        pending = false;
        return true;
    };

    int bank = 0;
    for (int64_t m0 = 0; m0 < m_src; m0 += m_block) {
        for (int kb0 = 0; kb0 < k_blocks; kb0 += kb_max) {
            const int kb_n = std::min(kb_max, k_blocks - kb0);

            const bool reuse_a =
                ctx->gemm_pack_npu == slot->npu &&
                ctx->gemm_pack_src == src1 &&
                ctx->gemm_pack_m0 == m0 &&
                ctx->gemm_pack_kb0 == kb0 &&
                ctx->gemm_pack_kb_n == kb_n &&
                ctx->gemm_pack_bank == bank;

            const float * c_slices[GGML_XDNA_GEMM_KB_LIMIT];
            if (!reuse_a) {
                // Overlap next A pack with the previous NPU run when double-buffered.
                if (pending && n_banks > 1) {
                    bank = 1 - pend_bank;
                }
                ggml_xdna_npu_gemm_set_bank(slot->npu, bank);
                const uint64_t t_pack = ggml_xdna_now_ns();
                for (int kb = 0; kb < kb_n; kb++) {
                    ggml_xdna_gemm_pack_activations(
                        src, k_src, m_src, m0,
                        (int64_t) (kb0 + kb) * k_slice,
                        (ggml_bf16_t *) ggml_xdna_npu_gemm_a_slice_bank(slot->npu, bank, kb),
                        s, nth);
                    c_slices[kb] = ggml_xdna_npu_gemm_c_slice_bank(slot->npu, bank, kb);
                }
                ctx->ns_gemm_pack += ggml_xdna_now_ns() - t_pack;
                ctx->n_gemm_pack++;
                ctx->gemm_pack_src = src1;
                ctx->gemm_pack_m0 = m0;
                ctx->gemm_pack_kb0 = kb0;
                ctx->gemm_pack_kb_n = kb_n;
                ctx->gemm_pack_bank = bank;
                ctx->gemm_pack_npu = slot->npu;
            } else {
                ctx->n_gemm_pack_reused++;
                ggml_xdna_npu_gemm_set_bank(slot->npu, bank);
                for (int kb = 0; kb < kb_n; kb++) {
                    c_slices[kb] = ggml_xdna_npu_gemm_c_slice_bank(slot->npu, bank, kb);
                }
            }

            for (int64_t n0 = 0; n0 < n_dst; n0 += n_cols) {
                if (!flush_pending()) {
                    ggml_xdna_npu_gemm_release(slot->npu);
                    return false;
                }

                const int nb = (int) (n0 / n_cols);
                size_t w_offs[GGML_XDNA_GEMM_KB_LIMIT];
                for (int kb = 0; kb < kb_n; kb++) {
                    w_offs[kb] = w_base + ((size_t) nb * k_blocks + kb0 + kb) * b_bytes;
                }

                ggml_xdna_npu_gemm_set_bank(slot->npu, bank);
                if (!ggml_xdna_npu_gemm_submit_list_async(slot->npu, wctx->buf, w_offs, kb_n)) {
                    ggml_xdna_npu_gemm_release(slot->npu);
                    return false;
                }

                pending = true;
                pend_bank = bank;
                pend_kb_n = kb_n;
                pend_n0 = n0;
                pend_m0 = m0;
                pend_kb0 = kb0;
                for (int kb = 0; kb < kb_n; kb++) {
                    pend_c[kb] = c_slices[kb];
                }

                // Next column reuses the same A; next M-block packs into the
                // other bank while this submit runs.
                if (n_banks > 1 && n0 + n_cols >= n_dst) {
                    bank = 1 - bank;
                }
            }
        }
    }

    if (!flush_pending()) {
        ggml_xdna_npu_gemm_release(slot->npu);
        return false;
    }

    ggml_xdna_npu_gemm_release(slot->npu);

#if defined(GGML_XDNA_EXPERIMENTAL)
    if (dst_override == nullptr) {
        ggml_xdna_chain_store(ctx, node->data, dst, n_dst, m_src);
    }
#endif

    const int level = ggml_xdna_debug_level();
    if (level > 0) {
        ggml_xdna_npu_stats stats;
        ggml_xdna_npu_get_stats(slot->npu, &stats);
        const uint64_t n = stats.n_calls;
        if (level >= 2 || n == 1 || (n % 64) == 0) {
            GGML_LOG_INFO("%s: graph_compute %s name=%s M=%lld K=%lld N=%lld on=NPU npu=%s "
                          "calls=%llu runs=%llu total=%.1fms submit=%.1fms sync=%.1fms "
                          "A_pack=%.1fms C_scatter=%.1fms\n",
                          "ggml-xdna", slot->env_name,
                          node->name[0] ? node->name : "?",
                          (long long) m_src, (long long) k_src, (long long) n_dst,
                          ctx->dev->npu_name.c_str(),
                          (unsigned long long) stats.n_calls,
                          (unsigned long long) stats.n_runs,
                          (double) stats.ns_total / 1e6,
                          (double) stats.ns_submit / 1e6,
                          (double) stats.ns_sync / 1e6,
                          (double) ctx->ns_gemm_pack / 1e6,
                          (double) ctx->ns_gemm_scatter / 1e6);
        }
    }

    return true;
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static void ggml_xdna_log_supports_op(const struct ggml_tensor * op, bool supported) {
    const int level = ggml_xdna_debug_level();
    if (level <= 0 || op == nullptr) {
        return;
    }

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW ||
        op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return;
    }

    static std::mutex mu;
    static std::unordered_map<std::string, int> seen_result;
    static std::unordered_map<std::string, uint64_t> reject_counts;
    static uint64_t total_queries = 0;
    static uint64_t total_rejects = 0;
    static bool intro_logged = false;

    const char * op_name = ggml_op_desc(op);
    std::string key = op_name ? op_name : "?";

    std::lock_guard<std::mutex> lock(mu);
    total_queries++;
    if (!supported) {
        total_rejects++;
        reject_counts[key]++;
    }

    if (!intro_logged) {
        GGML_LOG_INFO("%s: graph scheduling debug enabled (GGML_XDNA_DEBUG=%d). "
                      "own_graph claims the full graph; non-local ops -> nested CPU.\n",
                      "ggml-xdna", level);
        intro_logged = true;
    }

    auto it = seen_result.find(key);
    if (it == seen_result.end()) {
        GGML_LOG_INFO("%s: supports_op(%s) = %s -> %s\n",
                      "ggml-xdna",
                      key.c_str(),
                      supported ? "true" : "false",
                      supported ? "XDNA runs this" : "CPU fallback");
        seen_result[key] = supported ? 1 : 0;
    } else if (it->second == 0 && supported) {
        GGML_LOG_INFO("%s: supports_op(%s) = true -> XDNA runs some shapes "
                      "(earlier rejects were size/type/broadcast filters)\n",
                      "ggml-xdna", key.c_str());
        it->second = 1;
    }

    if (level >= 2 && (total_queries == 1 || total_queries % 1024 == 0)) {
        GGML_LOG_INFO("%s: supports_op stats: queries=%llu rejects=%llu (unique ops=%zu)\n",
                      "ggml-xdna",
                      (unsigned long long) total_queries,
                      (unsigned long long) total_rejects,
                      reject_counts.size());
    }

    if (level >= 2) {
        GGML_LOG_DEBUG("%s: supports_op(%s name=%s) = %s\n",
                       "ggml-xdna",
                       key.c_str(),
                       op->name ? op->name : "?",
                       supported ? "true" : "false");
    }
}

#if defined(GGML_XDNA_EXPERIMENTAL)
// The kernels take two equally sized contiguous F32 vectors and write a third.
static bool ggml_xdna_binary_shape_ok(const struct ggml_tensor * op, int64_t op_min) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    return src0 && src1 &&
           op->type   == GGML_TYPE_F32 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           // Require identical shapes, not just matching element counts:
           // same nelements can still differ in layout (e.g. broadcast).
           ggml_are_same_shape(src0, src1) &&
           ggml_are_same_shape(op, src0) &&
           ggml_nelements(op) >= op_min;
}
#endif

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_xdna_device_context * dctx = (ggml_backend_xdna_device_context *)dev->context;

    bool supported = false;

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supported = true;
            break;

#if defined(GGML_XDNA_EXPERIMENTAL)
        case GGML_OP_RMS_NORM:
            // The design bakes in the row length, and it normalises a whole row
            // per submission, so only exactly-sized contiguous rows qualify.
            supported = op->src[0] &&
                        op->type == GGML_TYPE_F32 &&
                        op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) &&
                        ggml_is_contiguous(op->src[0]) &&
                        ggml_are_same_shape(op, op->src[0]) &&
                        (size_t) op->ne[0] == dctx->rms_row &&
                        ggml_nelements(op) >= dctx->op_min &&
                        !ggml_xdna_uses_repack_buft(dctx, op) &&
                        ggml_xdna_device_npu(dctx, GGML_XDNA_SLOT_RMS_NORM) != nullptr;
            if (!supported && dctx->own_graph && dctx->npu_present && dctx->npu_enabled &&
                !ggml_xdna_uses_repack_buft(dctx, op)) {
                supported = true; // nested CPU
            }
            break;

        case GGML_OP_ADD:
        case GGML_OP_MUL:
            // Only claim the op once its kernel is actually live. This gates on
            // npu_present, npu_enabled, the per-op toggle and a successful
            // XRT/kernel init. Cheap in practice: init is triggered at backend
            // init, so this is just the call_once fast path (a safety net if
            // supports_op is queried for a device whose backend was never
            // created).
            supported = ggml_xdna_device_npu(dctx, ggml_xdna_slot_for_op(op->op)) != nullptr &&
                        !ggml_xdna_uses_repack_buft(dctx, op) &&
                        ggml_xdna_binary_shape_ok(op, dctx->op_min);
            if (!supported && dctx->own_graph && dctx->npu_present && dctx->npu_enabled &&
                !ggml_xdna_uses_repack_buft(dctx, op)) {
                supported = true; // nested CPU
            }
            break;
#endif

        case GGML_OP_MUL_MAT:
            // Claimed only for weights that were repacked into our buffer, and
            // then for every batch size: nothing else can read that layout, so
            // handing any of these back to the CPU would feed it garbage. A
            // batch shorter than gemm_m_min runs as a host GEMM inside this
            // backend rather than an NPU submit.
            supported = ggml_xdna_gemm_supports(dctx, op)
#if defined(GGML_XDNA_EXPERIMENTAL)
                        || ggml_xdna_act_gemm_ok(dctx, op)
#endif
                        ;
            if (!supported && dctx->own_graph && dctx->npu_present && dctx->npu_enabled &&
                !ggml_xdna_uses_repack_buft(dctx, op)) {
                // Refuse packable 2D weights whose shape fits the NPU GEMM so
                // select_weight_buft skips the default host buft and lands on
                // XDNA_REPACK under -ngl > 0. Act x act and odd shapes
                // (lm_head, gate proj N=16) stay as nested CPU.
                const ggml_tensor * w = op->src[0];
                if (ggml_xdna_gemm_should_repack_weight(dctx, w)) {
                    supported = false;
                } else {
                    supported = true;
                }
            }
            break;

#if defined(GGML_XDNA_EXPERIMENTAL)
        case GGML_OP_FLASH_ATTN_EXT:
            // Stage 3: with act-GEMM on, refuse it so the expanded QK / SOFT_MAX
            // / AV graph is built and its large act x act MUL_MATs reach the NPU
            // GEMM. A fused FLASH kernel is a later step.
            //
            // Otherwise claim it for the nested CPU. Refusing makes
            // resolve_fused_ops turn flash attention off for the whole model,
            // and the expanded graph then needs an n_ctx x n_ubatch scratch:
            // 36 GiB at this model's 256k default context and -ub 4096.
            supported = !dctx->act_gemm && dctx->own_graph &&
                        dctx->npu_present && dctx->npu_enabled &&
                        !ggml_xdna_uses_repack_buft(dctx, op);
            break;
#endif

        default:
            // Prefill ops with no NPU kernel (ROPE, SSM_CONV, GATED_DELTA_NET,
            // ...): claim them so the graph stays one split.
            // With layers assigned to XDNA (-ngl > 0) this also keeps
            // resolve_fused_ops happy: device_fused == device_layer.
            supported = dctx->own_graph && dctx->npu_present && dctx->npu_enabled &&
                        !ggml_xdna_uses_repack_buft(dctx, op);
            break;
    }

    ggml_xdna_log_supports_op(op, supported);
    return supported;
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_xdna_device_context * dctx = (ggml_backend_xdna_device_context *) dev->context;

    return ggml_backend_buft_is_host(buft) || buft == &dctx->repack_buft;
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "XDNA";
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return ggml_xdna_device_context()->npu_present ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_UNUSED(index);

    static ggml_backend_device ggml_backend_xdna_device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_xdna_device_context(),
    };

    return &ggml_backend_xdna_device;
}

// Weights have to be repacked into the layout the GEMM design reads, and for an
// ACCEL device the only way into the weight buffer list is as an extra buft: the
// device's default buffer type is also what the scheduler allocates activations
// in, and those must stay plain host memory for the elementwise kernels.
static ggml_backend_buffer_type_t * ggml_backend_xdna_device_get_extra_bufts(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type_t bufts[2] = { nullptr, nullptr };

    ggml_backend_xdna_device_context * dctx = (ggml_backend_xdna_device_context *) dev->context;
    if (!dctx->npu_present || !dctx->npu_enabled || !dctx->gemm_enabled) {
        return nullptr;
    }

    bool any = false;
    for (const ggml_xdna_gemm_slot & slot : dctx->gemm) {
        any = any || (slot.enabled && slot.found);
    }
    if (!any) {
        return nullptr;
    }

    bufts[0] = ggml_backend_xdna_repack_buffer_type(dev);
    return bufts;
}

static void ggml_backend_xdna_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;
    ctx->n_threads = n_threads;
    if (!ctx->host) {
        return;
    }
    ggml_backend_dev_t  host_dev = ggml_backend_get_device(ctx->host);
    ggml_backend_reg_t  host_reg = host_dev ? ggml_backend_dev_backend_reg(host_dev) : nullptr;
    auto set_n = host_reg
        ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
              host_reg, "ggml_backend_set_n_threads")
        : nullptr;
    if (set_n) {
        set_n(ctx->host, n_threads);
    }
}

static uint64_t ggml_backend_xdna_get_npu_call_count(ggml_backend_t backend) {
    if (!ggml_backend_is_xdna(backend)) {
        return 0;
    }

    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;
    uint64_t n_calls = 0;
    for (const ggml_xdna_gemm_slot & slot : ctx->dev->gemm) {
        if (slot.npu) {
            ggml_xdna_npu_stats stats;
            ggml_xdna_npu_get_stats(slot.npu, &stats);
            n_calls += stats.n_calls;
        }
    }
    return n_calls;
}

static void * ggml_backend_xdna_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) ggml_backend_xdna_device_get_extra_bufts;
    }
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_xdna_set_n_threads;
    }
    if (strcmp(name, "ggml_backend_xdna_get_npu_call_count") == 0) {
        return (void *) ggml_backend_xdna_get_npu_call_count;
    }

    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg ggml_backend_xdna_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xdna_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
