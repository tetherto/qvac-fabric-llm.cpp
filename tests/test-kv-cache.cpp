#include "llama-kv-cache.h"
#include "llama-io.h"
#include "llama-model.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

static constexpr uint32_t TEST_LAYER     = 0;
static constexpr uint32_t TEST_KV_SIZE   = 4;
static constexpr uint32_t TEST_N_SEQ_MAX = 1;
static constexpr uint32_t TEST_N_PAD     = 1;
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

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
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
    auto model = std::make_unique<llama_model>(params);

    llama_hparams & hparams = model->hparams;
    hparams.no_alloc             = no_alloc;
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

    return model;
}

static llama_kv_cache make_test_cache(const llama_model & model, ggml_type type_k) {
    return llama_kv_cache(
            model,
            model.hparams,
            type_k,
            GGML_TYPE_F16,
            /* v_trans = */ false,
            /* offload = */ false,
            /* unified = */ true,
            TEST_KV_SIZE,
            TEST_N_SEQ_MAX,
            TEST_N_PAD,
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

int main() {
    test_mrope_k_shift_input_layout();
    test_quantized_k_shift_width_pads_to_block();
    test_sequence_state_roundtrip_preserves_mrope_ext();

    return 0;
}
