#include "models.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <stdexcept>

// Laya decision checkpoints: https://github.com/NandhaKishorM/laya
//
// Each sequence is one question over one state:
//
//   [CLS] <type> question: <instructions> [SEP] [MASK] option0 [MASK] option1 ... [SEP] <state> [SEP]
//
// The ModernBert encoder output gets the embedding of the question type, runs through the decision
// blocks, and the hidden state at every [MASK] marker is scored into one logit per option. With
// LLAMA_POOLING_TYPE_RANK every sequence yields n_cls_out = n_act + n_max_options floats:
//
//   [act logits (n_act), option logits in marker order (n_max_options, zero after the sequence's last option)]

void llama_model_laya::load_arch_hparams(llama_model_loader & ml) {
    llama_model_modern_bert::load_arch_hparams(ml);

    // the checkpoints use the exact (erf) GELU of torch.nn.GELU
    if (hparams.llm_ffn_op == LLM_FFN_GEGLU) {
        hparams.llm_ffn_op = LLM_FFN_GEGLU_ERF;
    }

    ml.get_key(LLM_KV_DECISION_BLOCK_COUNT, n_decision_layer);
    ml.get_key(LLM_KV_DECISION_ACT_COUNT,   n_act);
    ml.get_key(LLM_KV_DECISION_MAX_OPTIONS, n_max_options);
    ml.get_arr(LLM_KV_DECISION_QTYPE_TOKENS, qtype_tokens);

    if (qtype_tokens.size() != 3) {
        throw std::runtime_error(format("laya: expected the first tokens of the 3 question types, got %zu", qtype_tokens.size()));
    }
    // the decision blocks are placed with the encoder layer of the same index
    if (n_decision_layer > (uint32_t) hparams.n_layer()) {
        throw std::runtime_error(format("laya: %u decision blocks, at most %u (the encoder layers) are supported", n_decision_layer, hparams.n_layer()));
    }
    // at least 2 so that the top-2 features are defined, at most 255 as in the option-count feature
    if (n_max_options < 2 || n_max_options > 255) {
        throw std::runtime_error(format("laya: %u option slots, expected 2 to 255", n_max_options));
    }
    // any number of act outputs (decision.act_out must match it), as long as the output row size fits an int32
    if (n_act < 1 || (uint64_t) n_act + n_max_options > (uint64_t) INT32_MAX) {
        throw std::runtime_error(format("laya: act head with %u outputs, expected at least 1 and fewer than %u", n_act, (uint32_t) INT32_MAX - n_max_options));
    }

    hparams.n_cls_out = n_act + n_max_options;
}

void llama_model_laya::load_arch_tensors(llama_model_loader & ml) {
    llama_model_modern_bert::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    // widths the reference hard-codes (4*d, 256), read from the tensors; a missing tensor is reported by create_tensor
    auto width = [&](const LLM_TN_IMPL & name) -> int64_t {
        const ggml_tensor * meta = ml.get_tensor_meta(name.str().c_str());
        return meta ? meta->ne[1] : 0;
    };
    const int64_t n_ff_dec  = n_decision_layer > 0 ? width(tn(LLM_TENSOR_DECISION_FFN_UP, "weight", 0)) : 0;
    const int64_t n_act_hid = width(tn(LLM_TENSOR_DECISION_ACT, "weight"));

    decision_type_embd = create_tensor(tn(LLM_TENSOR_DECISION_TYPE_EMBD, "weight"), {n_embd, 3}, 0);

    decision_layers.resize(n_decision_layer);
    for (uint32_t i = 0; i < n_decision_layer; ++i) {
        auto & layer = decision_layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_DECISION_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_DECISION_ATTN_NORM, "bias",   i), {n_embd}, 0);
        layer.wqkv        = create_tensor(tn(LLM_TENSOR_DECISION_ATTN_QKV,  "weight", i), {n_embd, 3*n_embd}, 0);
        layer.bqkv        = create_tensor(tn(LLM_TENSOR_DECISION_ATTN_QKV,  "bias",   i), {3*n_embd}, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_DECISION_ATTN_OUT,  "weight", i), {n_embd, n_embd}, 0);
        layer.bo          = create_tensor(tn(LLM_TENSOR_DECISION_ATTN_OUT,  "bias",   i), {n_embd}, 0);
        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_DECISION_FFN_NORM,  "weight", i), {n_embd}, 0);
        layer.ffn_norm_b  = create_tensor(tn(LLM_TENSOR_DECISION_FFN_NORM,  "bias",   i), {n_embd}, 0);
        layer.ffn_up      = create_tensor(tn(LLM_TENSOR_DECISION_FFN_UP,    "weight", i), {n_embd, n_ff_dec}, 0);
        layer.ffn_up_b    = create_tensor(tn(LLM_TENSOR_DECISION_FFN_UP,    "bias",   i), {n_ff_dec}, 0);
        layer.ffn_down    = create_tensor(tn(LLM_TENSOR_DECISION_FFN_DOWN,  "weight", i), {n_ff_dec, n_embd}, 0);
        layer.ffn_down_b  = create_tensor(tn(LLM_TENSOR_DECISION_FFN_DOWN,  "bias",   i), {n_embd}, 0);
    }

    decision_scorer_norm   = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_NORM, "weight"), {n_embd}, 0);
    decision_scorer_norm_b = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_NORM, "bias"),   {n_embd}, 0);
    decision_scorer        = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,      "weight"), {n_embd, n_embd}, 0);
    decision_scorer_b      = create_tensor(tn(LLM_TENSOR_DECISION_SCORER,      "bias"),   {n_embd}, 0);
    decision_scorer_out    = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT,  "weight"), {n_embd, 1}, 0);
    decision_scorer_out_b  = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT,  "bias"),   {1}, 0);

    // the act head also reads 4 features of the option distribution
    decision_act       = create_tensor(tn(LLM_TENSOR_DECISION_ACT,     "weight"), {n_embd + 4, n_act_hid}, 0);
    decision_act_b     = create_tensor(tn(LLM_TENSOR_DECISION_ACT,     "bias"),   {n_act_hid}, 0);
    decision_act_out   = create_tensor(tn(LLM_TENSOR_DECISION_ACT_OUT, "weight"), {n_act_hid, n_act}, 0);
    decision_act_out_b = create_tensor(tn(LLM_TENSOR_DECISION_ACT_OUT, "bias"),   {n_act}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_laya::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// nn.LayerNorm in the reference head: PyTorch's default eps, independent of the encoder's
static ggml_tensor * laya_head_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, 1e-5f);
    return ggml_add(ctx, ggml_mul(ctx, x, w), b);
}

// per-sequence inputs of the decision head, derived from the tokens of each sequence
class llm_graph_input_laya : public llm_graph_input_i {
public:
    llm_graph_input_laya(const llama_model_laya & model, int64_t n_opt) : model(model), n_opt(n_opt) {}
    virtual ~llm_graph_input_laya() = default;

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;
        const int64_t n_seqs   = ubatch->n_seqs_unq;

        const llama_token mask = model.vocab.token_mask();

        // the first two positions of each sequence: [CLS] and the question type
        std::vector<llama_pos> pos_first(n_seqs, std::numeric_limits<llama_pos>::max());
        for (int64_t i = 0; i < n_tokens; ++i) {
            for (int32_t j = 0; j < ubatch->n_seq_id[i]; ++j) {
                const int32_t s = ubatch->seq_idx[ubatch->seq_id[i][j]];
                pos_first[s] = std::min(pos_first[s], ubatch->pos[i]);
            }
        }

        std::vector<int32_t> seq_qtype(n_seqs, -1);
        std::vector<std::vector<std::pair<llama_pos, int32_t>>> markers(n_seqs);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const llama_token tok = ubatch->token ? ubatch->token[i] : LLAMA_TOKEN_NULL;

            for (int32_t j = 0; j < ubatch->n_seq_id[i]; ++j) {
                const int32_t s = ubatch->seq_idx[ubatch->seq_id[i][j]];

                if (ubatch->pos[i] == pos_first[s] + 1) {
                    for (size_t q = 0; q < model.qtype_tokens.size(); ++q) {
                        if (tok == model.qtype_tokens[q]) {
                            seq_qtype[s] = q;
                        }
                    }
                }
                if (tok == mask) {
                    markers[s].emplace_back(ubatch->pos[i], i);
                }
            }
        }

        for (int64_t s = 0; s < n_seqs; ++s) {
            if (seq_qtype[s] < 0) {
                LLAMA_LOG_WARN("%s: sequence does not start with a laya question type, using 'choice'\n", __func__);
                seq_qtype[s] = 0;
            }
        }

        // a token shared by several sequences takes the question type of its first one
        std::vector<int32_t> qtypes(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            qtypes[i] = seq_qtype[ubatch->seq_idx[ubatch->seq_id[i][0]]];
        }

        // padding points at row 0 and is masked out by opt_keep
        std::vector<int32_t> rows(n_opt*n_seqs, 0);
        std::vector<float>   keep(n_opt*n_seqs, 0.0f);
        for (int64_t s = 0; s < n_seqs; ++s) {
            // input that did not come through common_laya_predict (e.g. /embeddings with many "[MASK]" in the text)
            // may carry more markers than the output row holds: score the first ones instead of aborting
            std::sort(markers[s].begin(), markers[s].end());
            if ((int64_t) markers[s].size() > n_opt) {
                LLAMA_LOG_WARN("%s: a sequence has %zu [MASK] option markers, only the first %" PRId64 " are scored\n",
                        __func__, markers[s].size(), n_opt);
                markers[s].resize(n_opt);
            }
            for (size_t j = 0; j < markers[s].size(); ++j) {
                rows[s*n_opt + j] = markers[s][j].second;
                keep[s*n_opt + j] = 1.0f;
            }
        }

        ggml_backend_tensor_set(qtype, qtypes.data(), 0, ggml_nbytes(qtype));
        if (opt_rows) {
            ggml_backend_tensor_set(opt_rows, rows.data(), 0, ggml_nbytes(opt_rows));
            ggml_backend_tensor_set(opt_keep, keep.data(), 0, ggml_nbytes(opt_keep));
        }
    }

    ggml_tensor * qtype    = nullptr; // I32 [n_tokens]      question type of the sequence of each token
    ggml_tensor * opt_rows = nullptr; // I32 [n_opt*n_seqs]  row of each option marker, in position order
    ggml_tensor * opt_keep = nullptr; // F32 [n_opt, n_seqs] 1 for a marker, 0 for padding

    const llama_model_laya & model;
    const int64_t n_opt;
};

llama_model_laya::graph::graph(const llama_model & model, const llm_graph_params & params) : llama_model_modern_bert::graph(params) {
    const auto & laya = static_cast<const llama_model_laya &>(model);

    // the decision blocks attend over every token, so reduce to the output rows only at the end
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * cur = build_encoder(model, nullptr);

    const bool decide = cparams.embeddings && pooling_type == LLAMA_POOLING_TYPE_RANK;

    const int64_t n_seqs = ubatch.n_seqs_unq;

    // every sequence gets all the option slots, padded and masked out by opt_keep: the graph shape must not depend on
    // the token values, or the graph reserved with dummy tokens is too small for a real batch
    const int64_t n_opt = laya.n_max_options;

    auto inp = std::make_unique<llm_graph_input_laya>(laya, n_opt);

    inp->qtype = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->qtype);

    if (decide) {
        inp->opt_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_opt*n_seqs);
        ggml_set_input(inp->opt_rows);

        inp->opt_keep = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_opt, n_seqs);
        ggml_set_input(inp->opt_keep);
    }

    ggml_tensor * inp_qtype = inp->qtype;
    ggml_tensor * opt_rows  = inp->opt_rows;
    ggml_tensor * opt_keep  = inp->opt_keep;

    res->add_input(std::move(inp));

    cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, laya.decision_type_embd, inp_qtype));
    cb(cur, "decision_inp", -1);

    // decision blocks: bidirectional attention over the whole sequence, no positional encoding
    const int64_t n_head_dec  = std::max<int64_t>(1, n_embd/64);
    const int64_t n_embd_head = n_embd/n_head_dec;

    for (uint32_t i = 0; i < laya.n_decision_layer; ++i) {
        const auto & layer = laya.decision_layers[i];

        // the block weights live with encoder layer i
        const int il = i;

        ggml_tensor * inpL = cur;

        cur = laya_head_norm(ctx0, cur, layer.attn_norm, layer.attn_norm_b);
        cb(cur, "decision_attn_norm", il);

        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur);
        qkv = ggml_add(ctx0, qkv, layer.bqkv);
        cb(qkv, "decision_wqkv", il);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_dec, n_tokens, n_embd_head*ggml_element_size(qkv), qkv->nb[1], 0*n_embd*ggml_element_size(qkv));
        ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_dec, n_tokens, n_embd_head*ggml_element_size(qkv), qkv->nb[1], 1*n_embd*ggml_element_size(qkv));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_dec, n_tokens, n_embd_head*ggml_element_size(qkv), qkv->nb[1], 2*n_embd*ggml_element_size(qkv));

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, inp_attn->get_kq_mask(), nullptr, nullptr, 0,
                1.0f/sqrtf(float(n_embd_head)), il);
        cb(cur, "decision_kqv_out", il);

        cur = build_lora_mm(layer.wo, cur);
        cur = ggml_add(ctx0, cur, layer.bo);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "decision_ffn_inp", il);

        cur = laya_head_norm(ctx0, ffn_inp, layer.ffn_norm, layer.ffn_norm_b);
        cb(cur, "decision_ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   layer.ffn_up_b,   nullptr,
                nullptr,        nullptr,          nullptr,
                layer.ffn_down, layer.ffn_down_b, nullptr,
                nullptr,
                LLM_FFN_RELU, LLM_FFN_SEQ, il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "decision_out", il);
    }

    if (decide) {
        // one logit per option marker, masked like logits.masked_fill(~marker_mask, -1e4)
        ggml_tensor * opt = ggml_get_rows(ctx0, cur, opt_rows);
        opt = laya_head_norm(ctx0, opt, laya.decision_scorer_norm, laya.decision_scorer_norm_b);
        opt = ggml_add(ctx0, build_lora_mm(laya.decision_scorer, opt), laya.decision_scorer_b);
        opt = ggml_gelu_erf(ctx0, opt);
        opt = ggml_add(ctx0, build_lora_mm(laya.decision_scorer_out, opt), laya.decision_scorer_out_b);
        opt = ggml_reshape_2d(ctx0, opt, n_opt, n_seqs);

        // the output keeps 0 in every slot past a sequence's options, whatever else shares the batch
        ggml_tensor * opt_out = ggml_mul(ctx0, opt, opt_keep);
        cb(opt_out, "decision_opt_logits", -1);

        ggml_tensor * opt_masked = ggml_add(ctx0, opt_out, ggml_scale_bias(ctx0, opt_keep, 1e4f, -1e4f));

        // act features: top-1 probability, top-1 margin, normalized entropy, option count / 255
        ggml_tensor * p = ggml_soft_max(ctx0, opt_masked);

        ggml_tensor * top = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, p, 2));
        top = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, p, 1, n_opt, n_seqs), top);
        top = ggml_reshape_2d(ctx0, top, 2, n_seqs);

        ggml_tensor * top1 = ggml_view_2d(ctx0, top, 1, n_seqs, top->nb[1], 0);
        ggml_tensor * top2 = ggml_view_2d(ctx0, top, 1, n_seqs, top->nb[1], ggml_element_size(top));

        ggml_tensor * k = ggml_clamp(ctx0, ggml_sum_rows(ctx0, opt_keep), 2.0f, INFINITY);

        ggml_tensor * ent = ggml_mul(ctx0, p, ggml_log(ctx0, ggml_clamp(ctx0, p, 1e-9f, INFINITY)));
        ent = ggml_div(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, ent), -1.0f), ggml_log(ctx0, k));

        ggml_tensor * feats = ggml_concat(ctx0, ggml_cont(ctx0, top1), ggml_sub(ctx0, top1, top2), 0);
        feats = ggml_concat(ctx0, feats, ent, 0);
        feats = ggml_concat(ctx0, feats, ggml_scale(ctx0, k, 1.0f/255.0f), 0);
        cb(feats, "decision_act_feats", -1);

        ggml_tensor * act = ggml_get_rows(ctx0, cur, build_inp_cls());
        act = ggml_concat(ctx0, act, feats, 0);
        act = ggml_add(ctx0, build_lora_mm(laya.decision_act, act), laya.decision_act_b);
        act = ggml_gelu_erf(ctx0, act);
        act = ggml_add(ctx0, build_lora_mm(laya.decision_act_out, act), laya.decision_act_out_b);
        cb(act, "decision_act_logits", -1);

        ggml_tensor * out = ggml_concat(ctx0, act, opt_out, 0);
        cb(out, "result_embd_pooled", -1);

        res->t_embd_pooled = out;
        ggml_build_forward_expand(gf, out);
    }

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
