// A metadata-only GGUF has the header, the KV pairs and the tensor infos, but no tensor data.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * MODEL_PATH        = "test-model-load-meta-only.gguf";
static const char * SPLIT_PREFIX      = "test-model-load-meta-only-split";
static const int    N_SPLIT           = 2;

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

static int n_fail = 0;

static void check(bool ok, const char * what) {
    fprintf(stderr, "%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        n_fail++;
    }
}

[[noreturn]] static void die(const char * what) {
    fprintf(stderr, "%s\n", what);
    exit(EXIT_FAILURE);
}

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

static llama_model_params make_params(bool no_alloc, llama_load_mode load_mode) {
    llama_model_params mparams = llama_model_default_params();
    mparams.no_alloc  = no_alloc;
    mparams.load_mode = load_mode;
    return mparams;
}

static bool loads(bool no_alloc, llama_load_mode load_mode) {
    llama_model * model = llama_model_load_from_file(MODEL_PATH, make_params(no_alloc, load_mode));
    if (model == nullptr) {
        return false;
    }
    llama_model_free(model);
    return true;
}

static bool loads_split(bool no_alloc, llama_load_mode load_mode) {
    std::vector<std::string> paths;
    std::vector<const char *> c_paths;
    for (int i = 0; i < N_SPLIT; i++) {
        paths.push_back(split_path(i));
    }
    for (const auto & p : paths) {
        c_paths.push_back(p.c_str());
    }

    llama_model * model = llama_model_load_from_splits(c_paths.data(), c_paths.size(), make_params(no_alloc, load_mode));
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

    check(loads(/*no_alloc =*/ true, LLAMA_LOAD_MODE_NONE), "no_alloc without mmap loads a metadata-only GGUF");
    check(loads_split(/*no_alloc =*/ true, LLAMA_LOAD_MODE_NONE), "no_alloc without mmap loads metadata-only splits");

    check(!loads(/*no_alloc =*/ false, LLAMA_LOAD_MODE_NONE), "a real load still rejects a metadata-only GGUF");
    check(!loads_split(/*no_alloc =*/ false, LLAMA_LOAD_MODE_NONE), "a real load still rejects metadata-only splits");

    check(!loads(/*no_alloc =*/ true, LLAMA_LOAD_MODE_MMAP), "mmap still rejects a metadata-only GGUF");

    remove(MODEL_PATH);
    for (int i = 0; i < N_SPLIT; i++) {
        remove(split_path(i).c_str());
    }
    llama_backend_free();

    fprintf(stderr, "%s\n", n_fail == 0 ? "all tests passed" : "some tests failed");
    return n_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
