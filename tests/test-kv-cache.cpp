#include "llama-kv-cache.h"
#include "llama-io.h"
#include "llama-model.h"

#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static constexpr uint32_t TEST_LAYER     = 0;
static constexpr uint32_t TEST_KV_SIZE   = 4;
static constexpr uint32_t TEST_N_SEQ_MAX = 1;
static constexpr uint32_t TEST_N_HEAD    = 2;
static constexpr uint32_t TEST_N_HEAD_KV = 2;
static constexpr uint32_t TEST_HEAD_DIM  = 128;
static constexpr uint32_t TEST_N_ROT     = 64;
static constexpr uint32_t TEST_N_TOKENS  = 3;
static constexpr uint32_t TEST_N_POS     = 4;

static constexpr llama_seq_id TEST_SEQ_ID = 0;
static constexpr llama_pos IMAGE_T_ORIGIN = 100;
static constexpr llama_pos IMAGE_Y_ORIGIN = 30;
static constexpr llama_pos IMAGE_X_ORIGIN = 10;
static constexpr llama_pos IMAGE_SHIFT    = -17;

static constexpr uint32_t T_AXIS     = 0;
static constexpr uint32_t Y_AXIS     = 1;
static constexpr uint32_t X_AXIS     = 2;
static constexpr uint32_t OTHER_AXIS = 3;

class vector_writer : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t start = data.size();
        data.resize(start + size);
        ggml_backend_tensor_get(tensor, data.data() + start, offset, size);
    }

    size_t n_bytes() override {
        return data.size();
    }

    std::vector<uint8_t> data;
};

class vector_reader : public llama_io_read_i {
public:
    explicit vector_reader(const std::vector<uint8_t> & bytes) : data(bytes) {}

    const uint8_t * read(size_t size) override {
        if (offset + size > data.size()) {
            throw std::runtime_error("vector_reader: read past end");
        }
        const uint8_t * result = data.data() + offset;
        offset += size;
        return result;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return offset;
    }

private:
    const std::vector<uint8_t> & data;
    size_t offset = 0;
};

static std::unique_ptr<llama_model> make_test_model(enum llama_rope_type rope_type = LLAMA_ROPE_TYPE_IMROPE, bool no_alloc = true) {
    llama_model_params params = llama_model_default_params();
    params.no_alloc = no_alloc;
    auto model = std::make_unique<llama_model>(params);

    llama_hparams & hparams = model->hparams;
    hparams.n_ctx_train          = 512;
    hparams.n_embd               = TEST_N_HEAD * TEST_HEAD_DIM;
    hparams.n_layer_all          = 1;
    hparams.n_head_arr[0]        = TEST_N_HEAD;
    hparams.n_head_kv_arr[0]     = TEST_N_HEAD_KV;
    hparams.n_embd_head_k_full   = TEST_HEAD_DIM;
    hparams.n_embd_head_v_full   = TEST_HEAD_DIM;
    hparams.n_rot_full           = TEST_N_ROT;
    hparams.rope_type            = rope_type;
    hparams.rope_freq_base_train = 1000000.0f;
    hparams.rope_freq_scale_train = 1.0f;
    hparams.rope_sections        = { 16, 24, 24, 0 };

    // the cache reads the per-layer FP8 cache scales
    model->layers.resize(hparams.n_layer_all);

    return model;
}

static llama_kv_cache make_test_cache(const llama_model & model, ggml_type type_k) {
    return llama_kv_cache(
            model,
            model.no_alloc(),
            model.hparams,
            type_k,
            GGML_TYPE_F16,
            /* v_trans = */ false,
            /* offload = */ false,
            /* unified = */ true,
            TEST_KV_SIZE,
            TEST_N_SEQ_MAX,
            /* n_swa = */ 0,
            LLAMA_SWA_TYPE_NONE,
            /* mem_other = */ nullptr,
            /* filter    = */ {},
            /* reuse     = */ {},
            /* share     = */ {});
}

static llama_ubatch make_image_grid_ubatch() {
    llama_ubatch ubatch = {};
    ubatch.data = std::make_shared<llama_ubatch::data_t>();

    auto & data = *ubatch.data;
    data.token.resize(TEST_N_TOKENS);
    data.pos.resize(TEST_N_TOKENS * TEST_N_POS);
    data.n_seq_id.resize(TEST_N_TOKENS, 1);
    data.seq_id.resize(TEST_N_TOKENS);
    data.seq_id_data.resize(TEST_N_TOKENS, TEST_SEQ_ID);
    data.seq_id_unq.push_back(TEST_SEQ_ID);
    // assign() fill-constructs the buffer; resize(n, 0) on this int8_t vector trips a
    // -Wstringop-overflow false positive on GCC 16 (-O3) via _M_fill_append's move path.
    data.output.assign(TEST_N_TOKENS, 0);

    for (uint32_t i = 0; i < TEST_N_TOKENS; ++i) {
        data.seq_id[i] = &data.seq_id_data[i];
        data.pos[i + TEST_N_TOKENS * T_AXIS]     = IMAGE_T_ORIGIN;
        data.pos[i + TEST_N_TOKENS * Y_AXIS]     = IMAGE_Y_ORIGIN + i / 2;
        data.pos[i + TEST_N_TOKENS * X_AXIS]     = IMAGE_X_ORIGIN + i % 2;
        data.pos[i + TEST_N_TOKENS * OTHER_AXIS] = 0;
    }

    ubatch.b_equal_seqs = false;
    ubatch.n_tokens     = TEST_N_TOKENS;
    ubatch.n_seq_tokens = TEST_N_TOKENS;
    ubatch.n_seqs       = 1;
    ubatch.n_seqs_unq   = 1;
    ubatch.n_pos        = TEST_N_POS;
    ubatch.token        = data.token.data();
    ubatch.pos          = data.pos.data();
    ubatch.n_seq_id     = data.n_seq_id.data();
    ubatch.seq_id       = data.seq_id.data();
    ubatch.seq_id_unq   = data.seq_id_unq.data();
    ubatch.output       = data.output.data();

    return ubatch;
}

static llama_kv_cache::slot_info make_slot_info() {
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.strm = { 0 };
    sinfo.idxs = { { 0, 1, 2 } };
    return sinfo;
}

static std::vector<uint8_t> write_sequence_state(const llama_kv_cache & kv) {
    vector_writer writer;
    kv.state_write(writer, TEST_SEQ_ID);
    return std::move(writer.data);
}

static void read_sequence_state(llama_kv_cache & kv, const std::vector<uint8_t> & bytes) {
    vector_reader reader(bytes);
    kv.state_read(reader, TEST_SEQ_ID);
    GGML_ASSERT(reader.n_bytes() == bytes.size());
}

static void test_mrope_k_shift_input_layout() {
    auto model = make_test_model();
    llama_kv_cache kv = make_test_cache(*model, GGML_TYPE_F16);

    const llama_ubatch ubatch = make_image_grid_ubatch();
    kv.apply_ubatch(make_slot_info(), ubatch);
    kv.seq_add(TEST_SEQ_ID, 0, 200, IMAGE_SHIFT);

    std::vector<int32_t> data(TEST_KV_SIZE * TEST_N_POS);

    kv.set_input_k_shift_data(data.data());

    for (uint32_t i = 0; i < TEST_N_TOKENS; ++i) {
        GGML_ASSERT(data[i + TEST_KV_SIZE * T_AXIS]     == IMAGE_SHIFT);
        GGML_ASSERT(data[i + TEST_KV_SIZE * Y_AXIS]     == IMAGE_SHIFT);
        GGML_ASSERT(data[i + TEST_KV_SIZE * X_AXIS]     == IMAGE_SHIFT);
        GGML_ASSERT(data[i + TEST_KV_SIZE * OTHER_AXIS] == 0);
    }
    for (uint32_t dim = 0; dim < TEST_N_POS; ++dim) {
        GGML_ASSERT(data[TEST_N_TOKENS + TEST_KV_SIZE * dim] == 0);
    }
}

static void test_quantized_k_shift_width_pads_to_block() {
    auto model = make_test_model();

    llama_kv_cache f16_cache = make_test_cache(*model, GGML_TYPE_F16);
    GGML_ASSERT(f16_cache.get_k_shift_width(TEST_LAYER) == TEST_N_ROT);

    llama_kv_cache q8_cache = make_test_cache(*model, GGML_TYPE_Q8_0);
    GGML_ASSERT(q8_cache.get_k_shift_width(TEST_LAYER) == TEST_N_ROT);

    llama_kv_cache tbq_cache = make_test_cache(*model, GGML_TYPE_TBQ4_0);
    GGML_ASSERT(tbq_cache.get_k_shift_width(TEST_LAYER) == TEST_HEAD_DIM);

    llama_kv_cache pq_cache = make_test_cache(*model, GGML_TYPE_PQ4_0);
    GGML_ASSERT(pq_cache.get_k_shift_width(TEST_LAYER) == TEST_HEAD_DIM);
}

static void test_sequence_state_roundtrip_preserves_mrope_ext() {
    auto model = make_test_model(LLAMA_ROPE_TYPE_IMROPE, false);

    llama_kv_cache source = make_test_cache(*model, GGML_TYPE_F16);
    source.apply_ubatch(make_slot_info(), make_image_grid_ubatch());

    const std::vector<uint8_t> saved = write_sequence_state(source);

    llama_kv_cache restored = make_test_cache(*model, GGML_TYPE_F16);
    read_sequence_state(restored, saved);

    const std::vector<uint8_t> resaved = write_sequence_state(restored);
    GGML_ASSERT(resaved == saved);
}

// upstream (ggml-org/llama.cpp#28392): FP8 KV cache support checks and shared K/V cache scales
static void init_model(llama_model & model, uint32_t n_layer) {
    model.hparams = {};
    model.hparams.n_layer_all = n_layer;
    model.hparams.n_embd_head_k_full = 4;
    model.hparams.n_embd_head_v_full = 4;
    model.hparams.n_head_arr.fill(1);
    model.hparams.n_head_kv_arr.fill(1);
    std::iota(model.hparams.kv_source_impl.begin(), model.hparams.kv_source_impl.end(), 0);
    model.layers.resize(n_layer);
}

static void test_fp8_cache_support() {
    for (int bit = 0; bit < 64; ++bit) {
        GGML_ASSERT(ctz64(uint64_t{1} << bit) == bit);
        GGML_ASSERT(ctz64(UINT64_MAX << bit) == bit);
    }


    {
        std::string errors;
        llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                *static_cast<std::string *>(user_data) += text;
            }
        }, &errors);

        for (llm_arch arch : { LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK32, LLM_ARCH_DEEPSEEK4,
                LLM_ARCH_DEEPSEEK41, LLM_ARCH_GLM_DSA, LLM_ARCH_GLM5_NEXT, LLM_ARCH_DOTS3NOTE,
                LLM_ARCH_MISTRAL_LARGE4, LLM_ARCH_HY_V4, LLM_ARCH_MINIMAX_M3, LLM_ARCH_QWEN4EXP,
                LLM_ARCH_DFLASH }) {
            std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
            init_model(*model, 1);
            if (arch == LLM_ARCH_DEEPSEEK2) {
                model->hparams.n_embd_head_k_mla_impl = 4;
                model->hparams.n_embd_head_v_mla_impl = 4;
            }

            for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                    if (type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
                        continue;
                    }
                    auto params = llama_context_default_params();
                    params.type_k = type_k;
                    params.type_v = type_v;
                    errors.clear();
                    llama_context * ctx = llama_init_from_model(model.get(), params);
                    GGML_ASSERT(ctx == nullptr);
                    if (type_k == type_v || (!model->hparams.is_mla() && arch != LLM_ARCH_DEEPSEEK4 && arch != LLM_ARCH_DEEPSEEK41)) {
                        GGML_ASSERT(errors.find("FP8 cache is not supported for architecture") != std::string::npos);
                    } else {
                        GGML_ASSERT(errors.find("model does not support different K") != std::string::npos);
                    }
                }
            }
        }

        llama_log_set(nullptr, nullptr);
    }

    {
        std::unique_ptr<llama_model> target(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        std::unique_ptr<llama_model> assistant(llama_model_create(LLM_ARCH_GEMMA4_ASSISTANT, llama_model_default_params()));
        init_model(*target, 3);
        init_model(*assistant, 2);

        ggml_context_ptr ctx(ggml_init({ 4096, nullptr, false }));
        target->layers[2].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        target->layers[2].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[1].k_cache_s = assistant->layers[0].k_cache_s;
        assistant->layers[1].v_cache_s = assistant->layers[0].v_cache_s;
        *static_cast<float *>(target->layers[2].k_cache_s->data) = 0.5f;
        *static_cast<float *>(target->layers[2].v_cache_s->data) = 0.25f;
        *static_cast<float *>(assistant->layers[0].k_cache_s->data) = 2.0f;
        *static_cast<float *>(assistant->layers[0].v_cache_s->data) = 4.0f;

        for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
            for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                llama_kv_cache source(*target, false, target->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr,
                        [](uint32_t il) { return il == 0 ? 2 : -1; }, nullptr);
                llama_kv_cache shared(*assistant, false, assistant->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, &source, nullptr, nullptr,
                        [](int32_t il) { return il == 0 ? 2 : 1; });

                GGML_ASSERT(source.get_k_scale(2) == (type_k == GGML_TYPE_F8_E4M3 ? target->layers[2].k_cache_s : nullptr));
                GGML_ASSERT(source.get_v_scale(2) == (type_v == GGML_TYPE_F8_E4M3 ? target->layers[2].v_cache_s : nullptr));
                GGML_ASSERT(source.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(source.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_storage(0) == source.get_k_storage(2));
                GGML_ASSERT(shared.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(shared.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_scale(1) == nullptr);
                GGML_ASSERT(shared.get_v_scale(1) == nullptr);
            }
        }
    }

}

int main() {
    llama_backend_init();

    test_mrope_k_shift_input_layout();
    test_quantized_k_shift_width_pads_to_block();
    test_sequence_state_roundtrip_preserves_mrope_ext();
    test_fp8_cache_support();

    llama_backend_free();
    return 0;
}
