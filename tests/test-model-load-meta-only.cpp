// A metadata-only GGUF has the header, the KV pairs and the tensor infos, but no tensor data.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char * MODEL_PATH   = "test-model-load-meta-only.gguf";
static const char * SPLIT_PREFIX = "test-model-load-meta-only-split";
static const int    N_SPLIT      = 2;

static const int64_t N_EMBD      = 32;
static const int64_t N_FF        = 64;
static const int64_t N_VOCAB     = 64;
static const int64_t N_HEAD      = 4;
static const int64_t N_HEAD_KV   = 2;
static const int64_t N_EMBD_HEAD = N_EMBD / N_HEAD;
static const int64_t N_EMBD_GQA  = N_EMBD_HEAD * N_HEAD_KV;

struct tensor_info {
    const char * name;
    int64_t      ne0;
    int64_t      ne1; // 0 for a 1d tensor
};

// one layer, no experts, separate q/k/v: the smallest tensor set the llama arch accepts
static const tensor_info TENSORS[] = {
    { "token_embd.weight",        N_EMBD, N_VOCAB    },
    { "output_norm.weight",       N_EMBD, 0          },
    { "output.weight",            N_EMBD, N_VOCAB    },
    { "blk.0.attn_norm.weight",   N_EMBD, 0          },
    { "blk.0.attn_q.weight",      N_EMBD, N_EMBD     },
    { "blk.0.attn_k.weight",      N_EMBD, N_EMBD_GQA },
    { "blk.0.attn_v.weight",      N_EMBD, N_EMBD_GQA },
    { "blk.0.attn_output.weight", N_EMBD, N_EMBD     },
    { "blk.0.ffn_norm.weight",    N_EMBD, 0          },
    { "blk.0.ffn_gate.weight",    N_EMBD, N_FF       },
    { "blk.0.ffn_down.weight",    N_FF,   N_EMBD     },
    { "blk.0.ffn_up.weight",      N_EMBD, N_FF       },
};

static const int N_TENSORS = sizeof(TENSORS) / sizeof(TENSORS[0]);

// mmap says what the loader turns this mode into, see llama_model_loader::llama_model_loader
struct load_mode_case {
    const char *    name;
    llama_load_mode mode;
    bool            mmap;
};

// the modes that relax the check come first, so a regression reports before any later abort
static const load_mode_case MODES[] = {
    { "NONE",       LLAMA_LOAD_MODE_NONE,       false },
    { "MLOCK",      LLAMA_LOAD_MODE_MLOCK,      false },
    { "DIRECT_IO",  LLAMA_LOAD_MODE_DIRECT_IO,  false },
    { "AUTO",       LLAMA_LOAD_MODE_AUTO,       true  },
    { "MMAP",       LLAMA_LOAD_MODE_MMAP,       true  },
    { "MMAP_MLOCK", LLAMA_LOAD_MODE_MMAP_MLOCK, true  },
};

static const int N_MODES = sizeof(MODES) / sizeof(MODES[0]);

enum entry_point {
    ENTRY_FILE,
    ENTRY_SPLITS,
    ENTRY_FILE_PTR,
};

static const char * entry_name(entry_point ep) {
    switch (ep) {
        case ENTRY_FILE:     return "from_file";
        case ENTRY_SPLITS:   return "from_splits";
        case ENTRY_FILE_PTR: return "from_file_ptr";
    }
    return "?";
}

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    fprintf(stderr, "%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        n_fail++;
    }
}

[[noreturn]] static void die(const char * what) {
    fprintf(stderr, "%s\n", what);
    exit(EXIT_FAILURE);
}

// writes tensor infos [first, last) of TENSORS, with no tensor data
static void write_meta_only(const char * path, int first, int last, bool with_hparams, uint16_t split_no, uint16_t split_count) {
    const size_t mem_size = ggml_tensor_overhead() * (N_TENSORS + 1);
    std::vector<uint8_t> mem(mem_size);

    ggml_init_params ip = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ mem.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx  = ggml_init(ip);
    gguf_context * gguf = gguf_init_empty();

    if (with_hparams) {
        gguf_set_val_str(gguf, "general.architecture", "llama");
        gguf_set_val_u32(gguf, "llama.block_count", 1);
        gguf_set_val_u32(gguf, "llama.context_length", 128);
        gguf_set_val_u32(gguf, "llama.embedding_length", N_EMBD);
        gguf_set_val_u32(gguf, "llama.feed_forward_length", N_FF);
        gguf_set_val_u32(gguf, "llama.attention.head_count", N_HEAD);
        gguf_set_val_u32(gguf, "llama.attention.head_count_kv", N_HEAD_KV);
        gguf_set_val_u32(gguf, "llama.rope.dimension_count", N_EMBD_HEAD);
        gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_u32(gguf, "llama.vocab_size", N_VOCAB);
        gguf_set_val_str(gguf, "tokenizer.ggml.model", "no_vocab");
    }

    if (split_count > 0) {
        gguf_set_val_u16(gguf, "split.no", split_no);
        gguf_set_val_u16(gguf, "split.count", split_count);
        gguf_set_val_i32(gguf, "split.tensors.count", N_TENSORS);
    }

    for (int i = first; i < last; i++) {
        const tensor_info & ti = TENSORS[i];
        ggml_tensor * t = ti.ne1 > 0 ?
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ti.ne0, ti.ne1) :
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ti.ne0);
        ggml_set_name(t, ti.name);
        gguf_add_tensor(gguf, t);
    }

    const bool ok = gguf_write_to_file(gguf, path, /*only_meta =*/ true);

    gguf_free(gguf);
    ggml_free(ctx);

    if (!ok) {
        die("failed to write a metadata-only GGUF");
    }
}

static std::string split_path(int split_no) {
    char buf[512];
    if (llama_split_path(buf, sizeof(buf), SPLIT_PREFIX, split_no, N_SPLIT) <= 0) {
        die("failed to build a split path");
    }
    return std::string(buf);
}

static bool loads(entry_point ep, bool no_alloc, llama_load_mode mode) {
    llama_model_params mparams = llama_model_default_params();
    mparams.no_alloc  = no_alloc;
    mparams.load_mode = mode;

    llama_model * model = nullptr;

    switch (ep) {
        case ENTRY_FILE:
            model = llama_model_load_from_file(MODEL_PATH, mparams);
            break;
        case ENTRY_SPLITS: {
            std::vector<std::string>  paths;
            std::vector<const char *> c_paths;
            for (int i = 0; i < N_SPLIT; i++) {
                paths.push_back(split_path(i));
            }
            for (const auto & p : paths) {
                c_paths.push_back(p.c_str());
            }
            model = llama_model_load_from_splits(c_paths.data(), c_paths.size(), mparams);
        } break;
        case ENTRY_FILE_PTR: {
            // the loader does not take ownership of the FILE *
            FILE * f = fopen(MODEL_PATH, "rb");
            if (f == nullptr) {
                die("failed to open the metadata-only GGUF");
            }
            model = llama_model_load_from_file_ptr(f, mparams);
            fclose(f);
        } break;
    }

    if (model == nullptr) {
        return false;
    }
    llama_model_free(model);
    return true;
}

int main() {
    llama_backend_init();

    write_meta_only(MODEL_PATH, 0, N_TENSORS, /*with_hparams =*/ true, 0, 0);

    const int n_first = N_TENSORS / 2;
    write_meta_only(split_path(0).c_str(), 0,       n_first,   /*with_hparams =*/ true,  0, N_SPLIT);
    write_meta_only(split_path(1).c_str(), n_first, N_TENSORS, /*with_hparams =*/ false, 1, N_SPLIT);

    const entry_point ENTRIES[] = { ENTRY_FILE, ENTRY_SPLITS, ENTRY_FILE_PTR };

    for (int i = 0; i < N_MODES; i++) {
        const load_mode_case & m = MODES[i];

        for (int a = 0; a < 2; a++) {
            const bool no_alloc = a == 1;

            // nothing reads tensor data only when no_alloc is set and the mode does not mmap
            const bool expect = no_alloc && !m.mmap;

            for (const entry_point ep : ENTRIES) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%-13s %-10s no_alloc=%d -> %s",
                         entry_name(ep), m.name, no_alloc ? 1 : 0, expect ? "loads" : "rejected");
                check(loads(ep, no_alloc, m.mode) == expect, buf);
            }
        }
    }

    remove(MODEL_PATH);
    for (int i = 0; i < N_SPLIT; i++) {
        remove(split_path(i).c_str());
    }
    llama_backend_free();

    fprintf(stderr, "%s\n", n_fail == 0 ? "all tests passed" : "some tests failed");
    return n_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
