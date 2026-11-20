#include "models.h"

#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <cmath>

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act == "swigluoai") {
            hparams.llm_ffn_op = LLM_FFN_SWIGLU_OAI_MOE;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,       hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);

    const bool has_markov   = ml.get_tensor_meta("markov_w1.weight")       != nullptr;
    const bool has_selector = ml.get_tensor_meta("selector_hidden.weight") != nullptr;
    const bool has_lilicorr = ml.get_key(LLM_KV_LILICORR_HIDDEN_SIZE, hparams.lilicorr_hidden_size, false);
    if (has_markov) {
        variant = VARIANT_DSPARK;
    } else if (has_selector) {
        variant = VARIANT_DFLASH2;
    } else if (has_lilicorr) {
        variant = VARIANT_LILICORR;
        load_hparams_lilicorr(ml);
    }

    // DSpine packs [top_k candidates, top_k x top_k transition scores] into one n_embd row per block position
    if (ml.get_key(LLM_KV_DSPINE_HEAD_TOPK, hparams.dspine_head_topk, false)) {
        ml.get_key(LLM_KV_DSPINE_EPS, hparams.dspine_eps, false);
        bool causal = false;
        ml.get_key(LLM_KV_ATTENTION_CAUSAL, causal, false);
        const uint32_t top_k = hparams.dspine_head_topk;
        if (!causal || !std::isfinite(hparams.dspine_eps) || !(hparams.dspine_eps > 0.0f) || top_k < 2 || top_k > 64 || (top_k & (top_k - 1)) != 0 || hparams.dflash_block_size < 2 || has_lilicorr ||
                hparams.dflash_selector_top_k > 0 || (uint64_t) top_k * (top_k + 1) > hparams.n_embd) {
            throw std::runtime_error("invalid DSpine DFlash metadata");
        }
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        load_hparams_dsv4(ml);
        backbone = hparams.dsv4_hc_single_pass ? BACKBONE_DSV41 : BACKBONE_DSV4;
    } else if (ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false)) {
        load_hparams_mistral_large4(ml);
        backbone = BACKBONE_MISTRAL_LARGE4;
    } else if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        // optional interleaved sliding-window attention with per-layer pattern array.
        // DFlash has a single rope, so the SWA rope == main rope.
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern, hparams.n_layer(), false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    switch (variant) {
        case VARIANT_DSPARK:   load_tensors_dspark(n_vocab_draft); break;
        case VARIANT_DFLASH2:  load_tensors_dflash2();             break;
        case VARIANT_LILICORR: load_tensors_lilicorr();            break;
        case VARIANT_DFLASH:                                       break;
    }

    if (hparams.dspine_head_topk > 0) {
        const struct ggml_tensor * codebook_meta = ml->get_tensor_meta("dspine_codebook.weight");
        const struct ggml_tensor * gate_meta     = ml->get_tensor_meta("dspine_gate.weight");
        // the bridge and lattice are built by the plain DFlash decoder only
        if (!codebook_meta || !gate_meta || d2t_meta || variant != VARIANT_DFLASH ||
                hparams.dsv4_hc_mult > 0 || hparams.n_lora_kv > 0) {
            throw std::runtime_error("DSpine draft needs its code bridge, a plain DFlash decoder and the full target vocabulary");
        }
        const int64_t code_size = codebook_meta->ne[0];
        const int64_t rank      = gate_meta->ne[1];
        // checked before any tensor is created: the buffer probe of a zero-size weight aborts
        if (code_size <= 0 || rank <= 0 || n_vocab < (int64_t) hparams.dspine_head_topk) {
            throw std::runtime_error("invalid DSpine code bridge size");
        }

        dspine_codebook  = create_tensor(tn(LLM_TENSOR_DSPINE_CODEBOOK,  "weight"), { code_size, n_vocab }, 0);
        dspine_gate      = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "weight"), { code_size, rank }, 0);
        dspine_gate_b    = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "bias"),   { rank }, 0);
        dspine_pred_gate = create_tensor(tn(LLM_TENSOR_DSPINE_PRED_GATE, "weight"), { code_size, rank }, 0);
        dspine_prev      = create_tensor(tn(LLM_TENSOR_DSPINE_PREV,      "weight"), { code_size, rank }, 0);
        for (int i = 0; i < n_layer; ++i) {
            layers[i].dspine_read  = create_tensor(tn(LLM_TENSOR_DSPINE_READ,  "weight", i), { n_embd, code_size }, 0);
            layers[i].dspine_write = create_tensor(tn(LLM_TENSOR_DSPINE_WRITE, "weight", i), { rank, n_embd }, 0);
        }

        LLAMA_LOG_INFO("%s: DSpine code bridge: code size = %lld, rank = %lld, head top-k = %u\n", __func__,
                (long long) code_size, (long long) rank, hparams.dspine_head_topk);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    fc_in_s         = create_tensor(tn(LLM_TENSOR_FC,              "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    switch (backbone) {
        case BACKBONE_BASE:           load_tensors_layers_base();           break;
        case BACKBONE_DSV4:
        case BACKBONE_DSV41:          load_tensors_layers_dsv4();           break;
        case BACKBONE_MISTRAL_LARGE4: load_tensors_layers_mistral_large4(); break;
    }
}

void llama_model_dflash::load_tensors_layers_base() {
    LLAMA_LOAD_LOCALS;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, TENSOR_NOT_REQUIRED);
        layer.rope_freqs     = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head_k/2 }, TENSOR_NOT_REQUIRED | (i > 0 ? TENSOR_DUPLICATED : 0));

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// target features of an injection batch: host embd rows, or rows of the target's layer input device buffer
static ggml_tensor * build_inp_target_features(llm_graph_context & g, llm_graph_input_embd & inp, int64_t n_embd_inp) {
    if (g.ubatch.embd_row) {
        GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp);

        inp.embd_rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, g.n_tokens);
        ggml_set_input(inp.embd_rows);

        return ggml_get_rows(g.ctx0, g.layer_inp_src, inp.embd_rows);
    }

    inp.embd = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, n_embd_inp, g.n_tokens);
    ggml_set_input(inp.embd);

    return inp.embd;
}

// injection batch: the target features fused through the encoder
ggml_tensor * llama_model_dflash::build_inject_inp(llm_graph_context & g, const llama_model & model) {
    const int64_t n_embd_inp = g.hparams.n_embd_inp_enc();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    ggml_tensor * inp_target = build_inp_target_features(g, *inp, n_embd_inp);
    g.cb(inp_target, "inp_target_features", -1);

    g.res->add_input(std::move(inp));

    ggml_tensor * inp_g = g.build_lora_mm(model.fc, inp_target, model.fc_s);
    return g.build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
}

// tok_embd from the target model (shared via ctx_other) when the draft has none
ggml_tensor * llama_model_dflash::get_tok_embd(const llm_graph_context & g, const llama_model & model) {
    if (model.tok_embd) {
        return model.tok_embd;
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash draft requires the target model's token embeddings");
    return model_other->tok_embd;
}

// lm_head and its scale from the target model (shared via ctx_other) when the draft has none
std::pair<ggml_tensor *, ggml_tensor *> llama_model_dflash::get_lm_head(const llm_graph_context & g, const llama_model & model) {
    if (model.output) {
        return { model.output, model.output_s };
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->output != nullptr && "DFlash draft requires the target model's output projection");
    return { model_other->output, model_other->output_s };
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
llama_model_dflash::graph_base::graph_base(const llm_graph_params & params) : llm_graph_context(params) {
    n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    inp_pos = build_inp_pos();

    // optional iSWA: pick the matching attention input
    use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
}

ggml_tensor * llama_model_dflash::graph_base::build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) {
    if (!hparams.has_rope(il)) {
        return cur;
    }

    return rope_type == GGML_ROPE_TYPE_MROPE
        ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext(ctx0, cur, pos, model.layers[0].rope_freqs,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
}

// KV cache injection
ggml_tensor * llama_model_dflash::graph_base::build_inject() {
    ggml_tensor * inp_g = build_inject_inp(*this, model);
    cb(inp_g, "inp_g_embeddings", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, inp_g, layer.wv_s);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        if (use_iswa) {
            // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
            const bool    is_swa = hparams.is_swa(il);
            const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
            // rotate K/V into the cache's rotated space
            ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
            ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
            if (k_rot) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
        } else {
            // rotate K/V into the cache's rotated space
            if (inp_attn->self_k_rot[il]) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
            }
            if (inp_attn->self_v_rot[il]) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        }
    }

    res->t_embd = inp_g;

    ggml_build_forward_expand(gf, inp_g);

    return inp_g;
}

ggml_tensor * llama_model_dflash::graph_base::build_inp_tokens() {
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = inp_tokens;
    res->add_input(std::move(inp));

    return inp_tokens;
}

// DSpine: the last layer's state before its code write, reused by the refinement
struct dspine_write_state {
    ggml_tensor * raw   = nullptr; // [n_embd, n_tokens]
    ggml_tensor * rms   = nullptr; // [1, n_tokens]
    ggml_tensor * query = nullptr; // [rank, n_tokens]
};

// rows [beg, beg + n) of every block of a [ne0, n_rows, n_blocks] tensor
static ggml_tensor * dspine_block_rows(ggml_context * ctx0, ggml_tensor * x, int64_t beg, int64_t n) {
    return ggml_cont(ctx0, ggml_view_3d(ctx0, x, x->ne[0], n, x->ne[2], x->nb[1], x->nb[2], beg * x->nb[1]));
}

// sqrt(code_size) * codebook[anchor] for every block: [code_size, n_blocks]
static ggml_tensor * build_dspine_anchor_codes(ggml_context * ctx0, const llama_model & model, ggml_tensor * tokens, int64_t n_rows, int64_t n_blocks) {
    ggml_tensor * anchors = ggml_cont_1d(ctx0, ggml_view_2d(ctx0, tokens, 1, n_blocks, n_rows * tokens->nb[0], 0), n_blocks);
    return ggml_scale(ctx0, ggml_get_rows(ctx0, model.dspine_codebook, anchors), sqrtf((float) model.dspine_codebook->ne[0]));
}

// DSpine code write after a layer: block row t > 0 gets a message gated by its own code and the code of row t - 1
static ggml_tensor * build_dspine_write(llm_graph_context & g, const llama_model & model, ggml_tensor * raw,
        ggml_tensor * anchor_codes, int il, dspine_write_state & state) {
    ggml_context * ctx0 = g.ctx0;

    const auto & layer = model.layers[il];

    const int64_t n_embd    = raw->ne[0];
    const int64_t n_tokens  = raw->ne[1];
    const int64_t code_size = anchor_codes->ne[0];
    const int64_t n_blocks  = anchor_codes->ne[1];
    const int64_t n_rows    = n_tokens / n_blocks;
    const int64_t rank      = model.dspine_gate->ne[1];
    const float   eps       = g.hparams.dspine_eps;

    // the reference adds eps after the square root, so this is neither ggml_rms_norm nor ggml_l2_norm
    ggml_tensor * rms        = ggml_sqrt(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, raw)), 1.0f / n_embd));
    ggml_tensor * normalized = ggml_div(ctx0, raw, ggml_scale_bias(ctx0, rms, 1.0f, eps));

    ggml_tensor * proj  = g.build_lora_mm(layer.dspine_read, normalized);
    ggml_tensor * codes = ggml_div(ctx0, ggml_scale(ctx0, proj, sqrtf((float) code_size)),
            ggml_scale_bias(ctx0, ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, proj))), 1.0f, eps));

    // row 0 of each block carries the exact code of the anchor token
    codes = ggml_concat(ctx0, ggml_reshape_3d(ctx0, anchor_codes, code_size, 1, n_blocks),
            dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, codes, code_size, n_rows, n_blocks), 1, n_rows - 1), 1);

    ggml_tensor * query  = g.build_lora_mm(model.dspine_gate,      codes);
    ggml_tensor * sender = g.build_lora_mm(model.dspine_pred_gate, codes);
    ggml_tensor * value  = g.build_lora_mm(model.dspine_prev,      codes);

    // move the sender terms one row down inside each block: row 0 gets a zero message
    const auto shift = [&](ggml_tensor * x) {
        return ggml_pad_ext(ctx0, dspine_block_rows(ctx0, x, 0, n_rows - 1), 0, 0, 1, 0, 0, 0, 0, 0);
    };
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, query, shift(sender)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_reshape_2d(ctx0, ggml_mul(ctx0, gate, shift(value)), rank, n_tokens);

    ggml_tensor * delta = g.build_lora_mm(layer.dspine_write, msg);

    state.raw   = raw;
    state.rms   = rms;
    state.query = ggml_reshape_2d(ctx0, query, rank, n_tokens);

    return ggml_add(ctx0, raw, ggml_mul(ctx0, delta, rms));
}

// DSpine refinement: redo the last code write with the exact code of each predecessor candidate, then score
// the top_k candidates of each block row. Packed as [candidates, top_k x top_k scores] for the CPU-side walk.
static void build_dspine_lattice(llm_graph_context & g, const llama_model & model, ggml_tensor * output,
        ggml_tensor * output_s, ggml_tensor * anchor_codes, const dspine_write_state & state) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    const auto & last = model.layers[g.hparams.n_layer() - 1];

    const int64_t n_embd    = g.n_embd;
    const int64_t n_tokens  = g.n_tokens;
    const int64_t code_size = anchor_codes->ne[0];
    const int64_t n_blocks  = anchor_codes->ne[1];
    const int64_t n_rows    = n_tokens / n_blocks;
    const int64_t n_pos     = n_rows - 1; // block rows 1..n_rows-1 are drafts
    const int64_t rank      = model.dspine_gate->ne[1];
    const int64_t top_k     = g.hparams.dspine_head_topk;

    ggml_tensor * cand     = ggml_reshape_3d(ctx0, ggml_top_k(ctx0, res->t_logits, top_k), top_k, n_rows, n_blocks);
    ggml_tensor * cand_pos = dspine_block_rows(ctx0, cand, 1, n_pos);

    // predecessor codes: the anchor for row 1, the candidates of row t - 1 for row t
    ggml_tensor * codes = ggml_repeat_4d(ctx0, ggml_reshape_4d(ctx0, anchor_codes, code_size, 1, 1, n_blocks), code_size, top_k, 1, n_blocks);
    if (n_pos > 1) {
        ggml_tensor * ids  = ggml_reshape_1d(ctx0, dspine_block_rows(ctx0, cand, 1, n_pos - 1), top_k * (n_pos - 1) * n_blocks);
        ggml_tensor * prev = ggml_scale(ctx0, ggml_get_rows(ctx0, model.dspine_codebook, ids), sqrtf((float) code_size));
        codes = ggml_concat(ctx0, codes, ggml_reshape_4d(ctx0, prev, code_size, top_k, n_pos - 1, n_blocks), 2);
    }

    ggml_tensor * query = ggml_reshape_4d(ctx0, dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, state.query, rank, n_rows, n_blocks), 1, n_pos), rank, 1, n_pos, n_blocks);
    ggml_tensor * raw   = ggml_reshape_4d(ctx0, dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, state.raw, n_embd, n_rows, n_blocks), 1, n_pos), n_embd, 1, n_pos, n_blocks);
    ggml_tensor * rms   = ggml_reshape_4d(ctx0, dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, state.rms, 1, n_rows, n_blocks), 1, n_pos), 1, 1, n_pos, n_blocks);

    // the CPU repack kernels take no 4D src1; merging only the position and block dims keeps ne11 = top_k for the GPU kernels
    const int64_t n_mats   = n_pos * n_blocks;
    ggml_tensor * codes_3d = ggml_reshape_3d(ctx0, codes, code_size, top_k, n_mats);
    ggml_tensor * sender   = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_pred_gate, codes_3d), rank, top_k, n_pos, n_blocks);
    ggml_tensor * value    = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_prev,      codes_3d), rank, top_k, n_pos, n_blocks);

    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, sender, query), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_reshape_3d(ctx0, ggml_mul(ctx0, gate, value), rank, top_k, n_mats);

    ggml_tensor * delta   = ggml_reshape_4d(ctx0, g.build_lora_mm(last.dspine_write, msg), n_embd, top_k, n_pos, n_blocks);
    ggml_tensor * refined = ggml_add(ctx0, ggml_mul(ctx0, delta, rms), raw);
    refined = g.build_norm(refined, model.output_norm, NULL, LLM_NORM_RMS, -1);

    ggml_tensor * head = ggml_get_rows(ctx0, output, ggml_reshape_1d(ctx0, cand_pos, top_k * n_pos * n_blocks));
    head = ggml_reshape_4d(ctx0, head, n_embd, top_k, n_pos, n_blocks);

    // scores[candidate, predecessor] of each draft row, computed like the logits in build_lora_mm
    ggml_tensor * scores = ggml_mul_mat(ctx0, head, g.build_hadamard_input(output, refined));
    if (output_s) {
        scores = ggml_mul(ctx0, scores, output_s);
    }

    ggml_tensor * packed = ggml_concat(ctx0, ggml_cast(ctx0, cand_pos, GGML_TYPE_F32),
            ggml_reshape_3d(ctx0, scores, top_k * top_k, n_pos, n_blocks), 0);
    packed = ggml_pad_ext(ctx0, packed, 0, n_embd - top_k * (top_k + 1), 1, 0, 0, 0, 0, 0);
    packed = ggml_reshape_2d(ctx0, packed, n_embd, n_tokens);
    g.cb(packed, "dspine_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);
}

ggml_tensor * llama_model_dflash::graph_base::build_layers(ggml_tensor * inp_tokens) {
    const float kq_scale = hparams.f_attention_scale != 0.0f ? hparams.f_attention_scale : 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * tok_embd = get_tok_embd(*this, model);
    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);

    // a Hadamard-latent table (also the target's, borrowed) stores rotated rows; restore the primal basis
    if (hdmd) {
        if (const auto it = hdmd->inv.find(tok_embd); it != hdmd->inv.end()) {
            inpL = llama_mul_mat_hadamard(ctx0, inpL, it->second.rot);
            if (it->second.signs) {
                inpL = ggml_mul(ctx0, inpL, it->second.signs);
            }
        }
    }
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    // DSpine writes run on whole [anchor, mask...] blocks only, so graph reservation keeps the plain decoder
    const int64_t n_blocks    = ubatch.n_seqs_unq;
    const int64_t n_blk_rows  = n_blocks > 0 ? n_tokens / n_blocks : 0;
    const bool    use_dspine  = model.dspine_codebook && n_blocks > 0 && n_tokens % n_blocks == 0 &&
            n_blk_rows >= 2 && n_blk_rows <= (int64_t) hparams.dflash_block_size;

    dspine_anchor_codes = use_dspine ? build_dspine_anchor_codes(ctx0, model, inp_tokens, n_blk_rows, n_blocks) : nullptr;
    dspine_write_state dspine_state;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm, layer.wq_s);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, noise_norm, layer.wv_s);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }

        Qcur = build_rope(Qcur, inp_pos, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        cb(cur, "l_out", il);

        if (use_dspine) {
            cur = build_dspine_write(*this, model, cur, dspine_anchor_codes, il, dspine_state);
            cb(cur, "dspine_out", il);
        }

        inpL = cur;
    }

    dspine_raw   = dspine_state.raw;
    dspine_rms   = dspine_state.rms;
    dspine_query = dspine_state.query;

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    return cur;
}

// the target token ids of the draft vocabulary, in draft-logit column order
class llm_graph_input_draft_vocab : public llm_graph_input_i {
public:
    explicit llm_graph_input_draft_vocab(const std::vector<int32_t> & ranges) : ids_host(llama_draft_vocab_ids(ranges)) {}
    virtual ~llm_graph_input_draft_vocab() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        ggml_backend_tensor_set(ids, ids_host.data(), 0, ggml_nbytes(ids));
    }

    ggml_tensor * ids = nullptr; // I32 [n_draft_vocab]

    const std::vector<int32_t> ids_host;
};

static ggml_tensor * build_inp_draft_vocab(llm_graph_context & g, const std::vector<int32_t> & ranges) {
    auto inp = std::make_unique<llm_graph_input_draft_vocab>(ranges);

    inp->ids = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, (int64_t) inp->ids_host.size());
    ggml_set_input(inp->ids);

    ggml_tensor * ids = inp->ids;
    g.res->add_input(std::move(inp));

    return ids;
}

// build_draft_vocab_logits views row ranges of the stored head: its owner must keep the row layout, and the view
// skips the input rotation and LoRA delta of build_lora_mm
static bool dflash_head_rows_viewable(const llm_graph_context & g, const llama_model & owner, ggml_tensor * output) {
    if (owner.has_backend_layout(output)) {
        return false;
    }
    if (g.hdmd && g.hdmd->rot.count(output) > 0) {
        return false;
    }
    for (const auto & lora : *g.loras) {
        if (lora.first->get_weight(output) != nullptr) {
            return false;
        }
    }
    return true;
}

// logits over the draft vocabulary: one mat-mul per row range of the head
static ggml_tensor * build_draft_vocab_logits(ggml_context * ctx, ggml_tensor * output, ggml_tensor * output_s,
        ggml_tensor * cur, const std::vector<int32_t> & ranges) {
    ggml_tensor * logits = nullptr;
    for (size_t r = 0; r < ranges.size(); r += 2) {
        ggml_tensor * w = ggml_view_2d(ctx, output, output->ne[0], ranges[r + 1] - ranges[r], output->nb[1], ranges[r]*output->nb[1]);
        ggml_tensor * l = ggml_mul_mat(ctx, w, cur);
        logits = logits ? ggml_concat(ctx, logits, l, 0) : l;
    }
    return output_s ? ggml_mul(ctx, logits, output_s) : logits;
}

// logits over a reduced vocabulary placed at their target token ids of an n_vocab-wide -inf row
static ggml_tensor * build_vocab_scatter(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * ids, int64_t n_vocab) {
    const int64_t n_draft_vocab = logits->ne[0];
    const int64_t n_rows        = logits->ne[1];

    GGML_ASSERT(ids->ne[0] == n_draft_vocab);

    ggml_tensor * full = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_vocab, n_rows), -INFINITY);
    full = ggml_set_rows(ctx, full,
            ggml_reshape_3d(ctx, logits, 1,             n_draft_vocab, n_rows),
            ggml_reshape_3d(ctx, ids,    n_draft_vocab, 1,             1));

    return ggml_reshape_2d(ctx, full, n_vocab, n_rows);
}

ggml_tensor * llama_model_dflash::graph_base::build_logits(ggml_tensor * cur) {
    const auto [output, output_s] = get_lm_head(*this, model);
    const llama_model * head_owner = model.output ? &model : llama_get_model(cparams.ctx_other);

    // DSpine looks up the codebook rows of the head's candidates
    if (model.dspine_codebook && output->ne[1] != model.dspine_codebook->ne[1]) {
        throw std::runtime_error("DSpine draft codebook does not match the target vocabulary");
    }
    // the lattice reads head rows with get_rows, which a repacked or row-split buffer does not store in row order
    if (model.dspine_codebook && head_owner->has_backend_layout(output)) {
        throw std::runtime_error("DSpine draft needs the target output head in a plain buffer, not a repacked or row-split one (load the target with --no-repack and without -sm row)");
    }

    // DFlash2 draft vocabulary: logits over row ranges of the head only
    const bool use_draft_vocab = !cparams.draft_vocab.empty() && dflash_head_rows_viewable(*this, *head_owner, output);
    draft_ids = use_draft_vocab ? build_inp_draft_vocab(*this, cparams.draft_vocab) : nullptr;

    cur = use_draft_vocab ? build_draft_vocab_logits(ctx0, output, output_s, cur, cparams.draft_vocab) : build_lora_mm(output, cur, output_s);

    // DFlash2 feeds these logits to the selector, so they need the target's output
    // transforms; DFlash1 and DSpark read them through the sampler instead
    if (model.dflash_selector_hidden) {
        if (hparams.f_logit_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        }
        if (hparams.f_final_logit_softcapping > 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }
    }

    logits_draft = cur;

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        cur = build_vocab_scatter(ctx0, cur, model.d2t, model.vocab.n_tokens());
    }
    // requested logits keep the target vocabulary width, with -inf outside the draft vocabulary
    if (draft_ids && n_outputs > 0) {
        cur = build_vocab_scatter(ctx0, cur, draft_ids, model.vocab.n_tokens());
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // DSpine: the refined candidate lattice of the code bridge, on whole-block token batches only
    if (dspine_anchor_codes) {
        build_dspine_lattice(*this, model, output, output_s, dspine_anchor_codes, { dspine_raw, dspine_rms, dspine_query });
    }

    return cur;
}

llama_model_dflash::graph_dflash::graph_dflash(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            switch (backbone) {
                case BACKBONE_BASE:           break;
                case BACKBONE_DSV4:           return std::make_unique<graph_dsv4>(params);
                case BACKBONE_DSV41:          return std::make_unique<graph_dsv41>(params);
                case BACKBONE_MISTRAL_LARGE4: return std::make_unique<graph_mistral_large4>(params);
            }
            switch (variant) {
                case VARIANT_DFLASH:   return std::make_unique<graph_dflash>(params);
                case VARIANT_DFLASH2:  return std::make_unique<graph_dflash2>(params);
                case VARIANT_DSPARK:   return std::make_unique<graph_dspark>(params);
                case VARIANT_LILICORR: return std::make_unique<graph_lilicorr>(params);
            }
            GGML_ABORT("invalid DFlash variant");
        default:
            GGML_ABORT("invalid graph type");
    };
}
