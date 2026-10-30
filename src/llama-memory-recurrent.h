#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    uint32_t  seq_token_count(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // deferred state commit for gated delta net layers (see ggml_gated_delta_net_deferred): instead of
    // a state snapshot per rollback position, the s rows hold the committed base state and d_l holds
    // the rows recorded by the last ubatch. each cell has two record slots so that a ubatch never
    // overwrites the slot it replays from. rollback then only changes how many rows get replayed.
    bool    gdn_deferred = false;
    bool    gdn_inplace  = false; // update the base states in place (single cell only)
    int64_t gdn_rec_size = 0; // floats per record slot

    int64_t gdn_conv_channels() const { return hparams.ssm_d_inner + 2*hparams.ssm_n_group*hparams.ssm_d_state; }

    std::vector<uint8_t>  rec_slot; // per cell: the record slot holding the current rows
    std::vector<uint32_t> rec_rows; // per cell: rows recorded by the last ubatch

    // fills the base-state copy indices and the {pending, read_row, write_row} triples of a ubatch,
    // then advances the record metadata. must run before s_copy() consumes the rollback indices.
    void set_input_gdn_deferred(int32_t * s_copy_base, int32_t * rec_idx, uint32_t n_rs, uint32_t n_seqs, uint32_t n_seq_tokens);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;
    // gated delta net records, [gdn_rec_size, 2*size] (gdn_deferred only)
    std::vector<ggml_tensor *> d_l;
    // conv input histories, [(n_rs_seq + d_conv) * conv channels, 2*size] (gdn_deferred only): the conv
    // window of a cell starts at its pending row count, the r rows are only used for state save/load
    std::vector<ggml_tensor *> h_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;
    size_t size_d_bytes() const;

    // the base state of a cell brought up to date with its pending recorded rows (gdn_deferred)
    std::vector<float> gdn_deferred_state(int32_t il, uint32_t cell_id, uint32_t pending) const;

    // conv state <-> conv input history of a cell (gdn_deferred), in the r row layout
    std::vector<float> gdn_deferred_conv_state(int32_t il, uint32_t cell_id, uint32_t pending) const;
    void               gdn_deferred_set_conv_state(int32_t il, uint32_t cell_id, const float * conv_state);

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    // s_cells/s_pending: per written cell, the s row and the rows still to replay into it (gdn_deferred only)
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges,
            const std::vector<uint32_t> & s_cells = {}, const std::vector<uint32_t> & s_pending = {}) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;
    ggml_tensor * get_d_l(int32_t il) const;
    ggml_tensor * get_h_l(int32_t il) const;

    bool    get_gdn_deferred() const;
    int64_t get_gdn_k() const; // record rows per slot (n_rs_seq + 1)

    int32_t s_copy(int i) const;

    void set_input_gdn_deferred(int32_t * s_copy_base, int32_t * rec_idx, const llama_ubatch & ubatch) const;

    // true when the deferred gdn op updates the base states in place instead of gathering them,
    // which needs every cell to keep its own base state (no copies, no extra cells)
    bool get_s_base_inplace(uint32_t n_seqs) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
