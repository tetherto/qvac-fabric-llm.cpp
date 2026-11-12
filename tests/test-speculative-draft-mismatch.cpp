// checks common_speculative_draft_mismatch, the header-only check that a DFlash/DSpark draft model or an external MTP
// head fits the model, and that common_speculative_types_from_gguf reads a malformed block_count as no type
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "speculative.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

static constexpr int64_t N_EMBD  = 64;
static constexpr int64_t N_VOCAB = 8;

using fill_fn = std::function<void(gguf_context *, ggml_context *)>;

static void add_tensor(gguf_context * gguf, ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, name);
    gguf_add_tensor(gguf, t);
}

static void set_tokens(gguf_context * gguf, int64_t n) {
    std::vector<const char *> tokens(n, "t");
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
}

static std::vector<std::string> written;

// a header-only GGUF built by fill, removed when the test ends
static std::string write_gguf(const char * name, const fill_fn & fill) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx  = ggml_init(ip);
    gguf_context * gguf = gguf_init_empty();
    fill(gguf, ctx);
    const std::string path = name;
    written.push_back(path);
    const bool ok = gguf_write_to_file(gguf, path.c_str(), true);
    gguf_free(gguf);
    ggml_free(ctx);
    if (!ok) {
        fprintf(stderr, "failed to write %s\n", name);
    }
    return path;
}

// a DSpark draft model reading target layers 3, 10 and 28 at width N_EMBD
static fill_fn dspark(int64_t n_vocab = N_VOCAB, bool block_size = true, bool d2t = false) {
    return [=](gguf_context * gguf, ggml_context * ctx) {
        gguf_set_val_str(gguf, "general.architecture", "dflash");
        gguf_set_val_u32(gguf, "dflash.embedding_length", N_EMBD);
        gguf_set_val_u32(gguf, "dflash.block_count", 5);
        if (block_size) {
            gguf_set_val_u32(gguf, "dflash.block_size", 7);
        }
        const int32_t layers[] = { 3, 10, 28 };
        gguf_set_arr_data(gguf, "dflash.target_layers", GGUF_TYPE_INT32, layers, 3);
        set_tokens(gguf, n_vocab);
        add_tensor(gguf, ctx, "markov_w1.weight");
        if (d2t) {
            add_tensor(gguf, ctx, "d2t");
        }
    };
}

// a model with one MTP layer and a wider embedding_length_out, as hyper-connections give
static fill_fn target(uint32_t n_block, int64_t n_embd_out = 4 * N_EMBD) {
    return [=](gguf_context * gguf, ggml_context *) {
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
        gguf_set_val_u32(gguf, "qwen35.embedding_length", N_EMBD);
        gguf_set_val_u32(gguf, "qwen35.embedding_length_out", (uint32_t) n_embd_out);
        gguf_set_val_u32(gguf, "qwen35.block_count", n_block);
        gguf_set_val_u32(gguf, "qwen35.nextn_predict_layers", 1);
        set_tokens(gguf, N_VOCAB);
    };
}

static std::string mismatch(common_speculative_type type, const std::string & draft, const std::string & tgt) {
    common_params params;
    params.speculative.types = { type };
    params.speculative.draft.mparams.path = draft;
    return common_speculative_draft_mismatch(params, tgt);
}

static int failures = 0;

static void expect(const char * what, const std::string & got, const std::string & want) {
    const bool ok = got == want;
    printf("%s: %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) {
        printf("  got:  '%s'\n  want: '%s'\n", got.c_str(), want.c_str());
        ++failures;
    }
}

int main() {
    const auto DSPARK = COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
    const auto DFLASH = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    const auto MTP    = COMMON_SPECULATIVE_TYPE_DRAFT_MTP;

    // 29 blocks less 1 nextn layer: 28 layers, so layer 28 is the final hidden state
    const std::string fits    = write_gguf("test-sdm-target-29.gguf", target(29));
    const std::string shallow = write_gguf("test-sdm-target-28.gguf", target(28));
    const std::string draft   = write_gguf("test-sdm-dspark.gguf", dspark());

    // the model's embedding_length_out is 4 * N_EMBD: the draft model is compared with embedding_length
    expect("dspark fits", mismatch(DSPARK, draft, fits), "");
    expect("a dspark model run as dflash fits", mismatch(DFLASH, draft, fits), "");
    expect("the nextn layers are not target layers", mismatch(DSPARK, draft, shallow),
            "it reads target layer 28, the model has 27 layers");

    const std::string other_vocab = write_gguf("test-sdm-dspark-vocab.gguf", dspark(2 * N_VOCAB));
    expect("another vocabulary", mismatch(DSPARK, other_vocab, fits), "16 tokens, the model's 8");
    const std::string mapped = write_gguf("test-sdm-dspark-d2t.gguf", dspark(2 * N_VOCAB, true, true));
    expect("a d2t map does not change the vocabulary", mismatch(DSPARK, mapped, fits), "16 tokens, the model's 8");

    const std::string narrow = write_gguf("test-sdm-target-narrow.gguf", [](gguf_context * gguf, ggml_context * ctx) {
        target(29)(gguf, ctx);
        gguf_set_val_u32(gguf, "qwen35.embedding_length", N_EMBD / 2);
    });
    expect("another width, all reasons", mismatch(DSPARK, other_vocab, narrow),
            "16 tokens, the model's 8; hidden width 64, the model's 32");

    const std::string no_block = write_gguf("test-sdm-dspark-no-block.gguf", dspark(N_VOCAB, false));
    expect("a markov head needs a block size", mismatch(DFLASH, no_block, fits),
            "it has a DSpark Markov head but no positive dflash.block_size");
    // the loader takes only a u32, and the graph's std::stoi only a value within int32
    const auto block = [](const char * name, const std::function<void(gguf_context *)> & set) {
        return write_gguf(name, [&](gguf_context * gguf, ggml_context * ctx) {
            dspark(N_VOCAB, false)(gguf, ctx);
            set(gguf);
        });
    };
    expect("a u16 block size is no block size",
            mismatch(DSPARK, block("test-sdm-block-u16.gguf", [](gguf_context * gguf) { gguf_set_val_u16(gguf, "dflash.block_size", 7); }), fits),
            "it has a DSpark Markov head but no positive dflash.block_size");
    expect("a block size past int32 is no block size",
            mismatch(DSPARK, block("test-sdm-block-big.gguf", [](gguf_context * gguf) { gguf_set_val_u32(gguf, "dflash.block_size", 3000000000u); }), fits),
            "it has a DSpark Markov head but no positive dflash.block_size");

    // a split draft model: the metadata is in the first file, and the Markov head may be in any file
    const auto split = [](bool block_size) {
        const std::string prefix = block_size ? "test-sdm-split-bs" : "test-sdm-split";
        write_gguf((prefix + "-00002-of-00002.gguf").c_str(), [](gguf_context * gguf, ggml_context * ctx) {
            gguf_set_val_u16(gguf, "split.no", 1);
            gguf_set_val_u16(gguf, "split.count", 2);
            add_tensor(gguf, ctx, "markov_w1.weight");
        });
        return write_gguf((prefix + "-00001-of-00002.gguf").c_str(), [=](gguf_context * gguf, ggml_context *) {
            gguf_set_val_str(gguf, "general.architecture", "dflash");
            gguf_set_val_u32(gguf, "dflash.embedding_length", N_EMBD);
            if (block_size) {
                gguf_set_val_u32(gguf, "dflash.block_size", 7);
            }
            set_tokens(gguf, N_VOCAB);
            gguf_set_val_u16(gguf, "split.no", 0);
            gguf_set_val_u16(gguf, "split.count", 2);
        });
    };
    expect("a markov head in another split needs a block size", mismatch(DSPARK, split(false), fits),
            "it has a DSpark Markov head but no positive dflash.block_size");
    expect("a split dspark model with a block size fits", mismatch(DSPARK, split(true), fits), "");

    // gguf_get_arr_data asserts on a string array
    const std::string str_layers = write_gguf("test-sdm-dspark-str-layers.gguf", [](gguf_context * gguf, ggml_context * ctx) {
        dspark()(gguf, ctx);
        const char * layers[] = { "3", "99" };
        gguf_set_arr_str(gguf, "dflash.target_layers", layers, 2);
    });
    expect("string target_layers read as none", mismatch(DSPARK, str_layers, shallow), "");

    // no tokenizer tokens: the vocabulary is <arch>.vocab_size, as the vocab loader reads it
    const std::string no_vocab = write_gguf("test-sdm-dspark-no-vocab.gguf", [](gguf_context * gguf, ggml_context * ctx) {
        dspark()(gguf, ctx);
        gguf_remove_key(gguf, "tokenizer.ggml.tokens");
        gguf_set_val_u32(gguf, "dflash.vocab_size", 3);
    });
    expect("vocab_size without tokens", mismatch(DSPARK, no_vocab, fits), "3 tokens, the model's 8");

    // an external MTP head reads the model's output hidden state
    const fill_fn head = [](gguf_context * gguf, ggml_context *) {
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
        gguf_set_val_u32(gguf, "qwen35.embedding_length", 4 * N_EMBD);
        gguf_set_val_u32(gguf, "qwen35.block_count", 1);
        set_tokens(gguf, N_VOCAB);
    };
    const std::string mtp_head = write_gguf("test-sdm-mtp-head.gguf", head);
    expect("an MTP head at the output width fits", mismatch(MTP, mtp_head, fits), "");
    expect("an MTP head at another output width", mismatch(MTP, mtp_head, write_gguf("test-sdm-target-out.gguf", target(29, 2 * N_EMBD))),
            "hidden width 256, the model's 128");

    expect("other types are not checked", mismatch(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, other_vocab, fits), "");
    expect("an unreadable draft model is left to the load", mismatch(DSPARK, "test-sdm-missing.gguf", fits), "");

    // common_speculative_types_from_gguf: a missing, non-u32 or zero block_count is no type rather than an assert
    const auto types = [](const char * name, const fill_fn & fill) {
        return common_speculative_types_from_gguf(write_gguf(name, fill));
    };
    const auto n_types = [&](const char * name, const fill_fn & fill) {
        return std::to_string(types(name, fill).size());
    };
    expect("no block_count", n_types("test-sdm-types-none.gguf", [](gguf_context * gguf, ggml_context *) {
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
    }), "0");
    expect("a u16 block_count", n_types("test-sdm-types-u16.gguf", [](gguf_context * gguf, ggml_context *) {
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
        gguf_set_val_u16(gguf, "qwen35.block_count", 3);
    }), "0");
    expect("a zero block_count", n_types("test-sdm-types-zero.gguf", [](gguf_context * gguf, ggml_context *) {
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
        gguf_set_val_u32(gguf, "qwen35.block_count", 0);
    }), "0");
    const auto mtp_types = types("test-sdm-types-mtp.gguf", [](gguf_context * gguf, ggml_context * ctx) {
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
        gguf_set_val_u32(gguf, "qwen35.block_count", 3);
        add_tensor(gguf, ctx, "blk.2.nextn.eh_proj.weight");
    });
    expect("the last block's nextn projection is MTP",
            mtp_types.size() == 1 && mtp_types[0] == MTP ? "mtp" : "other", "mtp");

    for (const auto & path : written) {
        std::remove(path.c_str());
    }

    printf("%s\n", failures == 0 ? "all tests passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
