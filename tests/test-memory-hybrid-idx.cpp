#include "llama-memory-hybrid-idx.h"
#include "llama-io.h"
#include "llama-model.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

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


static llama_ubatch make_batch(llama_seq_id seq_id, llama_pos pos, uint32_t count, bool image = false) {
    llama_batch_allocr balloc(4);
    auto batch = balloc.ubatch_reserve(count, 1);
    batch.seq_id_unq[0] = seq_id;
    batch.seq_idx[seq_id] = 0;
    for (uint32_t i = 0; i < count; ++i) {
        batch.token[i] = 1;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = batch.seq_id_unq;
        batch.pos[i] = image ? pos : pos + i;
        batch.pos[i + count] = image ? pos + i/2 : pos + i;
        batch.pos[i + 2*count] = image ? pos + i%2 : pos + i;
    }
    if (image) {
        batch.token = nullptr;
        batch.data->embd.resize(count*64);
        batch.embd = batch.data->embd.data();
    }
    return batch;
}

static void apply(llama_memory_hybrid_idx & mem, const llama_ubatch & batch) {
    std::vector<llama_ubatch> batches = { batch };
    GGML_ASSERT(mem.get_mem_recr()->prepare(batches));
    auto slots = mem.get_mem_attn()->prepare(batches);
    GGML_ASSERT(!slots.empty());
    llama_memory_hybrid_idx_context ctx(&mem, slots, slots, batches);
    GGML_ASSERT(ctx.apply());
    GGML_ASSERT(ctx.can_use_qsa(batch) == mem.can_use_qsa(batch));
}

static std::vector<uint8_t> save(const llama_memory_hybrid_idx & mem, llama_seq_id seq = -1,
                               llama_state_seq_flags flags = 0) {
    vector_writer writer;
    mem.state_write(writer, seq, flags);
    return writer.data;
}

static void restore(llama_memory_hybrid_idx & mem, const std::vector<uint8_t> & state, llama_seq_id seq = -1,
                    llama_state_seq_flags flags = 0) {
    vector_reader reader(state);
    mem.state_read(reader, seq, flags);
    GGML_ASSERT(reader.n_bytes() == state.size());
}

static void test_fallback(bool unified) {
    llama_model model(llama_model_default_params());
    auto & hp = model.hparams;
    hp.n_layer_all = 1;
    hp.n_embd = 64;
    hp.n_head_arr[0] = 1;
    hp.n_head_kv_arr[0] = 1;
    hp.n_embd_head_k_full = 64;
    hp.n_embd_head_v_full = 64;
    hp.n_rot_full = 64;
    hp.rope_type = LLAMA_ROPE_TYPE_IMROPE;
    hp.rope_sections = { 16, 24, 24, 0 };
    hp.indexer_head_size = 64;
    hp.dsv4_compress_ratios[0] = 4;

    llama_memory_hybrid_idx mem(model,
            GGML_TYPE_F32, GGML_TYPE_F32, false, 64, 1, 0, LLAMA_SWA_TYPE_NONE,
            GGML_TYPE_F32, GGML_TYPE_F32, 3, 3, 4, false, unified,
            [](int32_t) { return true; }, [](int32_t) { return false; }, [](int32_t) { return true; });

    GGML_ASSERT((mem.get_pooled_k(0) != nullptr) == unified);
    GGML_ASSERT(mem.get_pooled_rows() == (unified ? 18 : 0));

    if (unified) {
        mem.pooled_valid(2) = 7;
        mem.clear(false);
        GGML_ASSERT(mem.pooled_valid(2) == 0);
    }

    auto text = make_batch(0, 0, 4);
    auto image = make_batch(0, 4, 4, true);
    auto next = make_batch(0, 6, 1);
    GGML_ASSERT(mem.can_use_qsa(text));
    GGML_ASSERT(!mem.can_use_qsa(image));
    apply(mem, text);

    if (unified) {
        mem.pooled_valid(0) = 1;
    }

    // Slot planning must not mark a sequence before the image batch is applied.
    GGML_ASSERT(!mem.get_mem_attn()->prepare({ image }).empty());
    GGML_ASSERT(mem.can_use_qsa(next));
    const auto text_state = save(mem, 0);
    const auto text_partial = save(mem, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    apply(mem, image);
    GGML_ASSERT(!mem.can_use_qsa(next));
    apply(mem, next);

    // Reservation still builds the sparse graph even when real decoding is dense.
    auto full = mem.init_full();
    GGML_ASSERT(static_cast<llama_memory_hybrid_idx_context *>(full.get())->can_use_qsa(image));

    const auto image_state = save(mem, 0);
    const auto image_partial = save(mem, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    mem.seq_cp(0, 1, -1, -1);
    if (unified) {
        GGML_ASSERT(mem.pooled_valid(0) == 1);
        GGML_ASSERT(mem.pooled_valid(1) == 0);
    }
    auto copied = make_batch(1, 7, 1);
    GGML_ASSERT(!mem.can_use_qsa(copied));
    auto independent = make_batch(2, 0, 1);
    GGML_ASSERT(mem.can_use_qsa(independent) == !unified);

    const auto full_state = save(mem);
    mem.clear(false);
    GGML_ASSERT(mem.can_use_qsa(next));
    restore(mem, full_state);
    GGML_ASSERT(!mem.can_use_qsa(next));
    GGML_ASSERT(!mem.can_use_qsa(copied));
    GGML_ASSERT(mem.can_use_qsa(independent) == !unified);

    // Removing all image cells must not lose the sticky fallback flag.
    GGML_ASSERT(mem.seq_rm(0, 4, 6));
    GGML_ASSERT(!mem.can_use_qsa(next));
    mem.seq_add(0, 0, -1, 2);
    GGML_ASSERT(!mem.can_use_qsa(next));
    const auto removed_image_state = save(mem, 0);
    mem.clear(false);
    restore(mem, removed_image_state, 2);
    GGML_ASSERT(!mem.can_use_qsa(independent));

    mem.clear(false);
    restore(mem, image_state, 1);
    GGML_ASSERT(!mem.can_use_qsa(copied));
    restore(mem, text_partial, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    GGML_ASSERT(!mem.can_use_qsa(copied));
    restore(mem, text_state, 1);
    GGML_ASSERT(mem.can_use_qsa(copied));
    restore(mem, image_partial, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    GGML_ASSERT(!mem.can_use_qsa(copied));

    mem.seq_keep(1);
    GGML_ASSERT(!mem.can_use_qsa(copied));
    GGML_ASSERT(mem.seq_rm(1, -1, -1));
    GGML_ASSERT(mem.can_use_qsa(copied));
    const auto empty_state = save(mem, 0);
    restore(mem, image_state, 1);
    restore(mem, empty_state, 1);
    GGML_ASSERT(mem.can_use_qsa(copied));
    GGML_ASSERT(mem.get_mem_attn()->seq_token_count(1) == 0);
    GGML_ASSERT(mem.get_mem_idx()->seq_token_count(1) == 0);

    // A failed/old restore must clear every cache and its flag together.
    restore(mem, image_state, 1);
    auto truncated = image_state;
    truncated.pop_back();
    bool failed = false;
    try {
        restore(mem, truncated, 1);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    GGML_ASSERT(failed);
    GGML_ASSERT(mem.get_mem_attn()->seq_token_count(1) == 0);
    GGML_ASSERT(mem.get_mem_idx()->seq_token_count(1) == 0);
    GGML_ASSERT(mem.can_use_qsa(copied));
    auto legacy = image_state;
    legacy.erase(legacy.begin(), legacy.begin() + 3*sizeof(uint32_t) + sizeof(uint8_t));
    failed = false;
    try {
        restore(mem, legacy, 1);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    GGML_ASSERT(failed);

    // Explicit spatial positions must also trigger fallback when token IDs exist.
    image.token = image.data->token.data();
    GGML_ASSERT(!mem.can_use_qsa(image));
    apply(mem, image);
    GGML_ASSERT(!mem.can_use_qsa(next));
    mem.clear(true);
    GGML_ASSERT(mem.can_use_qsa(next));
}

static void test_block_topk_inputs() {
    llama_model model(llama_model_default_params());
    auto & hp = model.hparams;
    hp.n_layer_all = 1;
    hp.n_embd = 64;
    hp.n_head_arr[0] = 1;
    hp.n_head_kv_arr[0] = 1;
    hp.n_embd_head_k_full = 64;
    hp.n_embd_head_v_full = 64;
    hp.n_rot_full = 64;
    hp.rope_type = LLAMA_ROPE_TYPE_IMROPE;
    hp.rope_sections = { 16, 24, 24, 0 };
    hp.indexer_head_size = 64;
    hp.dsv4_compress_ratios[0] = 4;

    llama_memory_hybrid_idx mem(model,
            GGML_TYPE_F32, GGML_TYPE_F32, false, 64, 1, 0, LLAMA_SWA_TYPE_NONE,
            GGML_TYPE_F32, GGML_TYPE_F32, 3, 3, 4, false, true,
            [](int32_t) { return true; }, [](int32_t) { return false; }, [](int32_t) { return true; });

    auto batch = make_batch(0, 0, 6);
    std::vector<llama_ubatch> batches = { batch };
    GGML_ASSERT(mem.get_mem_recr()->prepare(batches));
    auto slots = mem.get_mem_attn()->prepare(batches);
    GGML_ASSERT(!slots.empty());

    llama_memory_hybrid_idx_context mctx(&mem, slots, slots, batches);
    GGML_ASSERT(mctx.apply());

    ggml_init_params ip = {
        /*.mem_size   =*/ 16*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(ip));
    GGML_ASSERT(ctx);

    constexpr int64_t n_kv = 64;
    constexpr int64_t ratio = 4;
    constexpr int64_t n_blocks = n_kv/ratio;
    constexpr int64_t n_tokens = 6;

    ggml_tensor * cell_blk    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_kv, 1);
    ggml_tensor * block_cells = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ratio*n_blocks, 1);
    ggml_tensor * tail_cells  = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, ratio - 1, n_tokens, 1);
    ggml_tensor * bias        = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_blocks, n_tokens, 1);
    ggml_tensor * sentinel_cells = ggml_new_tensor_2d(
            ctx.get(), GGML_TYPE_I32, ratio*(n_blocks + 1), 1);
    ggml_tensor * sentinel_bias = ggml_new_tensor_3d(
            ctx.get(), GGML_TYPE_F32, n_blocks + 1, 1, 1);

    ggml_backend_buffer_ptr buf(
            ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(buf);

    mctx.set_input_qsa(
            cell_blk, nullptr, nullptr, block_cells, tail_cells, bias, &batch, ratio,
            /*blk_bias =*/ true, /*block_topk =*/ true);

    const auto & cells = mem.get_mem_idx()->get_cells(0);
    std::array<int32_t, n_tokens> pos_cell;
    pos_cell.fill(-1);
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (!cells.is_empty(cell) && cells.seq_has(cell, 0)) {
            const llama_pos pos = cells.pos_get(cell);
            if (pos >= 0 && pos < n_tokens) {
                pos_cell[pos] = cell;
            }
        }
    }
    for (const int32_t cell : pos_cell) {
        GGML_ASSERT(cell >= 0);
    }

    const int32_t * blocks = (const int32_t *) block_cells->data;
    for (int64_t j = 0; j < ratio; ++j) {
        GGML_ASSERT(blocks[j] == pos_cell[j]);
        GGML_ASSERT(blocks[ratio + j] == -1);
    }

    const int32_t * tails = (const int32_t *) tail_cells->data;
    const std::array<std::array<int32_t, ratio - 1>, n_tokens> expected = {{
        {{ pos_cell[0], -1,          -1          }},
        {{ pos_cell[0], pos_cell[1], -1          }},
        {{ pos_cell[0], pos_cell[1], pos_cell[2] }},
        {{ -1,          -1,          -1          }},
        {{ pos_cell[4], -1,          -1          }},
        {{ pos_cell[4], pos_cell[5], -1          }},
    }};
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t j = 0; j < ratio - 1; ++j) {
            GGML_ASSERT(tails[i*(ratio - 1) + j] == expected[i][j]);
        }
    }

    const float * block_bias = (const float *) bias->data;
    GGML_ASSERT(std::isinf(block_bias[2*n_blocks + 0]) && block_bias[2*n_blocks + 0] < 0.0f);
    GGML_ASSERT(block_bias[3*n_blocks + 0] == 0.0f);
    GGML_ASSERT(block_bias[4*n_blocks + 0] == 0.0f);
    GGML_ASSERT(std::isinf(block_bias[4*n_blocks + 1]) && block_bias[4*n_blocks + 1] < 0.0f);

    auto decode = make_batch(0, 5, 1);
    mctx.set_input_qsa(
            nullptr, nullptr, nullptr, sentinel_cells, nullptr, sentinel_bias, &decode, ratio,
            /*blk_bias =*/ true, /*block_topk =*/ true);

    const int32_t * sentinel = (const int32_t *) sentinel_cells->data;
    for (int64_t j = 0; j < ratio; ++j) {
        GGML_ASSERT(sentinel[j] == pos_cell[j]);
    }
    GGML_ASSERT(sentinel[n_blocks*ratio + 0] == pos_cell[4]);
    GGML_ASSERT(sentinel[n_blocks*ratio + 1] == pos_cell[5]);
    GGML_ASSERT(sentinel[n_blocks*ratio + 2] == -1);
    GGML_ASSERT(sentinel[n_blocks*ratio + 3] == -1);

    const float * sentinel_scores = (const float *) sentinel_bias->data;
    GGML_ASSERT(sentinel_scores[0] == 0.0f);
    GGML_ASSERT(std::isinf(sentinel_scores[1]) && sentinel_scores[1] < 0.0f);
    GGML_ASSERT(sentinel_scores[n_blocks] == 1e9f);
}

int main() {
    test_fallback(true);
    test_fallback(false);
    test_block_topk_inputs();
    printf("QSA multimodal cache tests passed\n");
}
