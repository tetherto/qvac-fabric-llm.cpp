#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_conv_deferred(
        llm_graph_input_rs * inp,
        ggml_tensor *        qkv_mixed,
        ggml_tensor *        conv_kernel,
        int64_t              n_norm,
        int64_t              head_dim,
        float                eps,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t head   = mctx_cur->get_head();
    const int32_t rs_z   = mctx_cur->get_rs_z();

    ggml_tensor * hist = mctx_cur->get_h_l(il);

    // both history slots of a cell are adjacent rows
    ggml_tensor * hist2 = ggml_reshape_2d(ctx0, hist, 2 * hist->ne[0], hist->ne[1] / 2);

    // a fresh cell starts from a zero conv window.
    // like the extra copies below, the nodes are zero-sized when unused so the graph matches the reserved one
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0,
        ggml_view_1d(ctx0, hist2, hist2->ne[0] * (rs_z >= 0), rs_z * hist2->nb[1] * (rs_z >= 0)), 0));

    ggml_tensor * out = ggml_gdn_conv_deferred(ctx0, qkv_mixed, conv_kernel, hist, inp->rec_idx,
            mctx_cur->get_gdn_k(), n_norm, head_dim, eps);
    cb(out, "conv_deferred", il);

    // cells that only receive a copy of another cell's state also need its history.
    // copied after the conv, which may read a source row that is one of these cells
    ggml_build_forward_expand(gf, out);
    ggml_tensor * extra = ggml_get_rows(ctx0, hist2, inp->s_copy_base_extra);
    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, extra,
                ggml_view_2d(ctx0, hist2, hist2->ne[0], n_rs - n_seqs, hist2->nb[1], (head + n_seqs) * hist2->nb[1])));

    return out;
}

ggml_tensor * llm_build_delta_net_base::build_rs_ssm(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        int64_t              n_seqs) {
    if (!inp->rec_idx) {
        return build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    }

    const auto * mctx_cur = inp->mctx;

    if (inp->s_base_inplace) {
        // every cell keeps its own base state: no gather, the op reads and writes the cache rows
        const int32_t rs_z = mctx_cur->get_rs_z();
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0,
            ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s()*(rs_z >= 0), rs_z*ssm_states_all->nb[1]*(rs_z >= 0)), 0));

        return ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                            mctx_cur->get_head()*ssm_states_all->nb[1]);
    }

    // one gather of all n_rs base states: the ubatch states, then the extra cells (see build_rs)
    return build_rs(ssm_states_all, inp->s_copy_base, inp->s_copy_base_main, hparams.n_embd_s(), n_seqs,
                    mctx_cur->get_n_rs(), mctx_cur->get_head(), mctx_cur->get_size(), mctx_cur->get_rs_z());
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il,
        ggml_tensor *        dt,
        ggml_tensor *        a) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();

    GGML_ASSERT(dt == nullptr || inp->rec_idx != nullptr);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (inp->rec_idx) {
        // deferred commit: replay the kept rows of the previous ubatch, run the block, record its
        // last rows and store the new base state (see ggml_gated_delta_net_deferred)
        const int64_t D = S_v * S_v * H_v;
        const int64_t K = mctx_cur->get_gdn_k();

        ggml_tensor * rec = mctx_cur->get_d_l(il);

        ggml_tensor * gdn_out = ggml_gated_delta_net_deferred(ctx0, q, k, v, g, b, s, rec, inp->rec_idx, dt, a, K);
        cb(gdn_out, "gdn_deferred_out", il);

        ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
            S_v, H_v, n_seq_tokens, n_seqs,
            ggml_row_size(gdn_out->type, S_v),
            ggml_row_size(gdn_out->type, S_v * H_v),
            ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
            0);
        cb(output, "attn_output", il);

        const int64_t n_attn = S_v * H_v * n_seq_tokens * n_seqs;
        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

        ggml_tensor * base = ggml_view_2d(ctx0, gdn_out, D, n_seqs,
            ggml_row_size(gdn_out->type, D), ggml_row_size(gdn_out->type, n_attn));

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, base,
                    ggml_view_2d(ctx0, ssm_states_all, D, n_seqs, ssm_states_all->nb[1], (size_t) kv_head * row_size)));

        // cells that only receive a copy of another cell's state also need its records
        const int64_t n_rs = mctx_cur->get_n_rs();
        ggml_tensor * rec2 = ggml_reshape_2d(ctx0, rec, 2 * rec->ne[0], rec->ne[1] / 2);
        ggml_tensor * rec_extra = ggml_get_rows(ctx0, rec2, inp->s_copy_base_extra);
        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, rec_extra,
                    ggml_view_2d(ctx0, rec2, rec2->ne[0], n_rs - n_seqs, rec2->nb[1], (kv_head + n_seqs) * rec2->nb[1])));

        return output;
    }

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}
