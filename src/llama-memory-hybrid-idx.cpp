#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

// Reject older QSA states that do not preserve the multimodal fallback flag.
// This header is specific to the indexer memory; other state formats are unchanged.
static constexpr uint32_t HYBRID_IDX_STATE_MAGIC   = 0x51494458; // QIDX
static constexpr uint32_t HYBRID_IDX_STATE_VERSION = 1;

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    n_seq_max(n_seq_max),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // Complete QSA blocks are immutable. Cache their normalized and roped
    // summary keys once, with each layer's rows beside its raw indexer cache.
    // Multi-stream caches keep the existing full recompute path.
    if (mem_idx && mem_idx->get_n_stream() == 1) {
        uint32_t ratio = 0;
        for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
            if (!model.hparams.is_recr(il) && model.hparams.dsv4_compress_ratios[il] > 0) {
                ratio = model.hparams.dsv4_compress_ratios[il];
                break;
            }
        }

        const uint32_t idx_dim = model.hparams.indexer_head_size;
        if (ratio > 0 && idx_dim > 0) {
            // One extra row covers a partial tail; the final row is a dustbin
            // for fixed-size padded writes.
            pooled_rows  = kv_size/ratio + 2;
            pooled_ratio = ratio;

            std::vector<ggml_backend_buffer_type_t> bufts;

            for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
                if (model.hparams.is_recr(il) || model.hparams.dsv4_compress_ratios[il] == 0) {
                    continue;
                }

                ggml_tensor * k = mem_idx->get_k_storage((int32_t) il);
                if (k == nullptr) {
                    continue;
                }

                const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(k->buffer);

                size_t ci = 0;
                while (ci < bufts.size() && bufts[ci] != buft) {
                    ++ci;
                }

                if (ci == bufts.size()) {
                    bufts.push_back(buft);

                    ggml_init_params ip = {
                        /*.mem_size   =*/ 2*model.hparams.n_layer()*ggml_tensor_overhead(),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    pooled_ctxs.emplace_back(ggml_init(ip));
                }

                ggml_tensor * t = ggml_new_tensor_2d(
                        pooled_ctxs[ci].get(), GGML_TYPE_F32, idx_dim, pooled_rows);
                ggml_format_name(t, "idx_pooled_l%u", il);
                pooled_k[(int32_t) il] = t;
            }

            size_t total_bytes = 0;
            for (size_t ci = 0; ci < bufts.size(); ++ci) {
                pooled_bufs.emplace_back(
                        ggml_backend_alloc_ctx_tensors_from_buft(pooled_ctxs[ci].get(), bufts[ci]));
                GGML_ASSERT(pooled_bufs.back() && "failed to allocate pooled indexer key cache");

                // Invalid rows are masked but still consumed by arithmetic, so
                // initialize them to finite values.
                ggml_backend_buffer_clear(pooled_bufs.back().get(), 0);
                total_bytes += ggml_backend_buffer_get_size(pooled_bufs.back().get());
            }

            if (!pooled_k.empty()) {
                LLAMA_LOG_INFO(
                        "%s: pooled indexer key cache, %zu layers x %u rows on %zu buffers, %.2f MiB\n",
                        __func__, pooled_k.size(), pooled_rows, pooled_bufs.size(),
                        total_bytes/1024.0/1024.0);
            }
        }
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    qsa_disabled.reset();

    if (mem_idx) {
        mem_idx->clear(data);
    }

    pooled_reset(-1);
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    pooled_rm(seq_id, p0, p1);

    const bool res = get_mem_attn()->seq_rm(seq_id, p0, p1);
    if (res && p0 <= 0 && (p1 < 0 || p1 == std::numeric_limits<llama_pos>::max())) {
        if (seq_id < 0) {
            qsa_disabled.reset();
        } else {
            qsa_disabled.reset(seq_id);
        }
    }

    return res;
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);

        if (qsa_disabled[seq_id_src] && get_mem_attn()->seq_pos_max(seq_id_dst) >= 0) {
            qsa_disabled.set(seq_id_dst);
        }
    }

    // The destination shares physical rows with the source. Refill its
    // summaries from its own cells before trusting them.
    pooled_reset(seq_id_dst);
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    const bool disabled = qsa_disabled[seq_id];
    qsa_disabled.reset();
    qsa_disabled.set(seq_id, disabled);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    const int64_t keep = pooled_w.count(seq_id) ? pooled_w[seq_id] : 0;
    pooled_w.clear();
    pooled_w[seq_id] = keep;
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    pooled_reset(seq_id);
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    pooled_reset(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & buf : pooled_bufs) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (mem_idx) {
        const uint32_t count = seq_id < 0 ? n_seq_max : 1;
        io.write(&HYBRID_IDX_STATE_MAGIC, sizeof(HYBRID_IDX_STATE_MAGIC));
        io.write(&HYBRID_IDX_STATE_VERSION, sizeof(HYBRID_IDX_STATE_VERSION));
        io.write(&count, sizeof(count));
        for (uint32_t s = 0; s < count; ++s) {
            const uint8_t disabled = qsa_disabled[seq_id < 0 ? s : seq_id];
            io.write(&disabled, sizeof(disabled));
        }
    }

    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section follows the hybrid caches
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        std::bitset<LLAMA_MAX_SEQ> restored_qsa_disabled;
        if (mem_idx) {
            uint32_t magic, version, count;
            io.read(&magic, sizeof(magic));
            io.read(&version, sizeof(version));
            io.read(&count, sizeof(count));
            if (magic != HYBRID_IDX_STATE_MAGIC || version != HYBRID_IDX_STATE_VERSION ||
                    count != (seq_id < 0 ? n_seq_max : 1)) {
                throw std::runtime_error("hybrid indexer state header mismatch");
            }
            for (uint32_t s = 0; s < count; ++s) {
                uint8_t disabled;
                io.read(&disabled, sizeof(disabled));
                if (disabled > 1) {
                    throw std::runtime_error("invalid hybrid indexer QSA flag");
                }
                restored_qsa_disabled.set(seq_id < 0 ? s : seq_id, disabled != 0);
            }
        }

        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            // An empty saved sequence must also replace the destination cells.
            // Otherwise clearing its flag could expose an old image to QSA.
            if (seq_id >= 0) {
                state_drop(seq_id);
            }
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

        if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
            // Recurrent-only rollback leaves the attention cells in place.
            qsa_disabled |= restored_qsa_disabled;
        } else if (seq_id < 0) {
            qsa_disabled = restored_qsa_disabled;
        } else {
            qsa_disabled.set(seq_id, restored_qsa_disabled[seq_id]);
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    // Full restores replace indexer cells with arbitrary contents. A partial
    // speculative restore leaves them in place and rolls back through seq_rm.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        pooled_reset(seq_id);
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    qsa_disabled.reset(seq_id);
    pooled_reset(seq_id);
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

ggml_tensor * llama_memory_hybrid_idx::get_pooled_k(int32_t il) const {
    const auto it = pooled_k.find(il);
    return it == pooled_k.end() ? nullptr : it->second;
}

int64_t & llama_memory_hybrid_idx::pooled_valid(llama_seq_id seq_id) const {
    return pooled_w[seq_id];
}

void llama_memory_hybrid_idx::pooled_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (pooled_k.empty()) {
        return;
    }

    if (seq_id < 0) {
        pooled_reset(-1);
        return;
    }

    if (p0 <= 0 && (p1 < 0 || p1 == std::numeric_limits<llama_pos>::max())) {
        pooled_w[seq_id] = 0;
        return;
    }

    const int64_t first_changed = std::max<llama_pos>(p0, 0)/pooled_ratio;
    auto & w = pooled_w[seq_id];
    w = std::min(w, first_changed);
}

void llama_memory_hybrid_idx::pooled_reset(llama_seq_id seq_id) {
    if (seq_id < 0) {
        pooled_w.clear();
    } else {
        pooled_w[seq_id] = 0;
    }
}

static bool llama_qsa_is_multimodal(const llama_ubatch & ubatch, uint32_t i) {
    return ubatch.token == nullptr || (ubatch.is_pos_2d() &&
            (ubatch.pos[i + ubatch.n_tokens] != ubatch.pos[i] ||
             ubatch.pos[i + 2*ubatch.n_tokens] != ubatch.pos[i]));
}

void llama_memory_hybrid_idx::update_qsa(const llama_ubatch & ubatch) {
    if (!mem_idx) {
        return;
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (llama_qsa_is_multimodal(ubatch, i)) {
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                qsa_disabled.set(ubatch.seq_id[i][s]);
            }
        }
    }
}

bool llama_memory_hybrid_idx::can_use_qsa(const llama_ubatch & ubatch) const {
    if (!mem_idx) {
        return false;
    }

    // The current block gather also visits foreign cells in a unified stream.
    if (mem_idx->get_n_stream() == 1 && qsa_disabled.any()) {
        return false;
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (llama_qsa_is_multimodal(ubatch, i)) {
            return false;
        }
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            if (qsa_disabled[ubatch.seq_id[i][s]]) {
                return false;
            }
        }
    }

    return true;
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    is_full(true),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    if (res && ctx_idx && !is_full) {
        mem->update_qsa(get_ubatch());
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

bool llama_memory_hybrid_idx_context::can_use_qsa(const llama_ubatch & ubatch) const {
    // Reservation must still account for the largest sparse graph, even if the
    // current sequences use dense attention.
    return get_idx() != nullptr && (is_full || mem->can_use_qsa(ubatch));
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(can_use_qsa(*ubatch));

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = blk_cells != nullptr ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos   != nullptr ? (int32_t *) blk_pos->data   : nullptr;
    float   * dst_bias      = (float   *) bias->data;

    // block b covers [b*ratio, (b+1)*ratio), so its first token is at b*ratio
    // multimodal sequences bypass QSA; their temporal positions do not count tokens
    if (dst_blk_pos != nullptr) {
        for (int64_t sec = 0; sec < 4; ++sec) {
            for (int64_t s = 0; s < n_ns; ++s) {
                for (int64_t b = 0; b < n_blocks; ++b) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = (int32_t) (b*r);
                }
            }
        }
    }

    // one pass per stream: cell j is a different token in each, so no mapping is shared
    std::vector<int32_t> blk_of(n_kv);
    std::vector<int32_t> filled(n_blocks);
    std::vector<int32_t> local_blk_cells(r*n_blocks);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk = dst_cell_blk + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells != nullptr
            ? dst_blk_cells + s*(r*n_blocks)
            : local_blk_cells.data();

        // an incomplete block cannot be pooled; the bias below forces those tail cells in
        // -1 means no usable block, and block 0 only keeps the gather in range
        std::fill(blk_of.begin(),  blk_of.end(),  -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t   b = p/r;

            if (b >= n_blocks) {
                oor = true;
                continue;
            }

            blk_of[j] = (int32_t) b;
            cur_blk_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        // per-block mode keeps an unpooled cell's real block, so the block's own -inf reaches it
        // per-cell mode carries that -inf itself and only needs the gather in range
        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 && filled[blk_of[j]] < r && !blk_bias) {
                blk_of[j] = -1;
            }
            cur_cell_blk[j] = blk_of[j] < 0 ? 0 : blk_of[j];
        }

        if (dirty_cells != nullptr) {
            GGML_ASSERT(n_ns == 1);
            GGML_ASSERT(dirty_pos != nullptr && dirty_rows != nullptr);

            const int64_t n_dirty_max = dirty_rows->ne[0];
            const int64_t dustbin     = (int64_t) get_pooled_rows() - 1;

            int32_t * dst_d_cells = (int32_t *) dirty_cells->data;
            int32_t * dst_d_pos   = (int32_t *) dirty_pos->data;
            int64_t * dst_d_rows  = (int64_t *) dirty_rows->data;

            int64_t n_complete = 0;
            for (int64_t b = n_blocks; b-- > 0;) {
                if (filled[b] == r) {
                    n_complete = b + 1;
                    break;
                }
            }

            auto & w = mem->pooled_valid(seq_of_stream);
            w = std::min(w, n_complete);

            const int64_t n_dirty = n_complete - w;
            GGML_ASSERT(n_dirty <= n_dirty_max);

            for (int64_t i = 0; i < n_dirty_max; ++i) {
                const bool    live     = i < n_dirty;
                const int64_t b        = w + i;
                const bool    complete = live && b < n_blocks && filled[b] == r;

                dst_d_rows[i] = live ? b : dustbin;

                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_d_pos[sec*n_dirty_max + i] = complete ? (int32_t) (b*r) : 0;
                }
                for (int64_t j = 0; j < r; ++j) {
                    dst_d_cells[i*r + j] = complete ? cur_blk_cells[b*r + j] : 0;
                }
            }

            w = n_complete;
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos    q      = ubatch->pos[i];

            // the tail is an incomplete block and is always visible, as in the reference
            const llama_pos tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = b*r >= tail_start ? 1e9f : (filled[b] < r ? -INFINITY : 0.0f);
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    v = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                }

                cur_bias[j] = v;
            }
        }
    }
}

ggml_tensor * llama_memory_hybrid_idx_context::get_pooled_k(int32_t il) const {
    return mem != nullptr && get_idx() != nullptr ? mem->get_pooled_k(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::get_pooled_rows() const {
    return mem != nullptr ? mem->get_pooled_rows() : 0;
}

uint32_t llama_memory_hybrid_idx_context::qsa_pooled_n_dirty_max(
        const llama_ubatch & ubatch, uint32_t ratio) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr);

    // Graph reservation uses a mock ubatch without sequence or position data.
    if (ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr || ubatch.pos == nullptr) {
        return (ubatch.n_tokens + ratio - 1)/ratio + 1;
    }

    const llama_seq_id seq = ubatch.seq_id[0][0];

    llama_pos q_max = -1;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        q_max = std::max(q_max, ubatch.pos[i]);
    }

    const int64_t n_complete = (int64_t) (q_max + 1)/ratio;
    const int64_t w = std::min(mem->pooled_valid(seq), n_complete);

    return (uint32_t) std::max<int64_t>(1, n_complete - w);
}
