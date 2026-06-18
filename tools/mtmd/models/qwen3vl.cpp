#include "models.h"

ggml_cgraph * clip_graph_qwen3vl::build() {
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    // [QWEN_VIDEO] a video batch is a pair of frames merged along the temporal axis
    // into a single output tile; otherwise the batch holds same-size image tiles.
    const bool is_video        = img_batch != nullptr && img_batch->is_video;
    const int batch_size       = is_video ? 1 : n_batch;
    // batch_size > 1 encodes multiple same-size tiles in a single forward pass.
    // QKV/FFN run fully batched; attention is per-tile (each tile attends only to its own tokens).
    // M-RoPE offsets are mathematically inert in the vision encoder: relative attention cancels
    // any per-tile absolute offset. Tile arrangement reaches the LM via decoder positions in mtmd.cpp.
    const int n_pos            = n_patches;   // patches per tile
    // M-RoPE: 4 coords per patch. Positions are local tile coordinates and
    // identical for every tile, so all tiles share one positions block.
    const int num_position_ids = n_pos * 4;

    norm_type norm_t = NORM_TYPE_NORMAL;

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = nullptr;
    if (is_video) {
        inp = build_inp_temporal_merge();
    } else {
        // still image tiles: both temporal kernels see the same frame, per tile
        ggml_tensor * inp_raw = build_inp_raw();
        inp = ggml_add(ctx0,
                ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
                ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));

        GGML_ASSERT(img.nx() % (patch_size * 2) == 0);
        GGML_ASSERT(img.ny() % (patch_size * 2) == 0);
    }

    // spatial merge
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // add patch bias
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }

    // absolute position embedding: same local positions for every tile, broadcast over batch_size
    ggml_tensor * learned_pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    learned_pos_embd = ggml_cont_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, n_patches_y, 1);
    learned_pos_embd = ggml_reshape_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, 2, n_patches_y / 2);
    learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = ggml_cont_3d(
        ctx0, learned_pos_embd,
        n_embd, n_patches_x * n_patches_y, 1);
    // broadcast-add: learned_pos_embd is [n_embd, n_pos, 1], inp is [n_embd, n_pos, batch_size]
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "inp_pos_emb", -1);

    ggml_tensor * inpL = inp;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    // deepstack features (stack along the feature dimension), [n_embd * len(deepstack_layers), n_patches_x * n_patches_y, batch_size]
    ggml_tensor * deepstack_features = nullptr;
    const int merge_factor = hparams.n_merge > 0 ? hparams.n_merge * hparams.n_merge : 4; // default 2x2=4 for qwen3vl

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            cur = build_mm(layer.qkv_w, cur);   // [3*n_embd, n_pos, batch_size]
            cur = ggml_add(ctx0, cur, layer.qkv_b);

            // Extract Q/K/V as 4D views [d_head, n_head, n_pos, batch_size].
            // QKV and FFN projections run fully batched; attention is per-tile (each tile
            // attends only to its own n_pos patches - no cross-tile attention).
            ggml_tensor * Qcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, batch_size,
                    /* nb1 */ ggml_row_size(cur->type, d_head),
                    /* nb2 */ cur->nb[1],
                    /* nb3 */ cur->nb[2],
                    /* off */ 0);
            ggml_tensor * Kcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, batch_size,
                    /* nb1 */ ggml_row_size(cur->type, d_head),
                    /* nb2 */ cur->nb[1],
                    /* nb3 */ cur->nb[2],
                    /* off */ ggml_row_size(cur->type, n_embd));
            ggml_tensor * Vcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, batch_size,
                    /* nb1 */ ggml_row_size(cur->type, d_head),
                    /* nb2 */ cur->nb[1],
                    /* nb3 */ cur->nb[2],
                    /* off */ ggml_row_size(cur->type, 2 * n_embd));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Per-tile attention: each tile attends only to its own n_pos tokens.
            // QKV/FFN linear layers above are fully batched and run in parallel.
            ggml_tensor * attn_out = nullptr;
            for (int b = 0; b < batch_size; b++) {
                ggml_tensor * Q_b = ggml_view_3d(ctx0, Qcur, d_head, n_head, n_pos,
                    Qcur->nb[1], Qcur->nb[2], (size_t)b * Qcur->nb[3]);
                ggml_tensor * K_b = ggml_view_3d(ctx0, Kcur, d_head, n_head, n_pos,
                    Kcur->nb[1], Kcur->nb[2], (size_t)b * Kcur->nb[3]);
                ggml_tensor * V_b = ggml_view_3d(ctx0, Vcur, d_head, n_head, n_pos,
                    Vcur->nb[1], Vcur->nb[2], (size_t)b * Vcur->nb[3]);

                Q_b = ggml_rope_multi(ctx0, Q_b, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
                K_b = ggml_rope_multi(ctx0, K_b, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

                cb(Q_b, "Qcur_rope", il);
                cb(K_b, "Kcur_rope", il);

                ggml_tensor * out_b = build_attn(layer.o_w, layer.o_b,
                    Q_b, K_b, V_b, nullptr, kq_scale, il);
                out_b = ggml_reshape_2d(ctx0, out_b, n_embd, n_pos);

                if (batch_size == 1) {
                    attn_out = out_b;
                } else {
                    if (attn_out == nullptr) {
                        attn_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_pos * batch_size);
                    }
                    // in-place set: copies only this tile's slice instead of
                    // duplicating the whole accumulator on every iteration
                    attn_out = ggml_set_2d_inplace(ctx0, attn_out, out_b,
                        n_embd * sizeof(float),
                        (size_t)b * n_embd * n_pos * sizeof(float));
                }
            }
            cb(attn_out, "attn_out", il);

            // [n_embd, n_pos * batch_size] -> [n_embd, n_pos, batch_size]
            cur = ggml_reshape_3d(ctx0, attn_out, n_embd, n_pos, batch_size);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        if (layer.has_deepstack()) {
            ggml_tensor * feat = ggml_reshape_3d(ctx0, cur, n_embd * merge_factor, n_pos / merge_factor, batch_size);
            feat = build_norm(feat, layer.deepstack_norm_w, layer.deepstack_norm_b, norm_t, eps, il);
            feat = build_ffn(feat,
                layer.deepstack_fc1_w, layer.deepstack_fc1_b,
                nullptr, nullptr,
                layer.deepstack_fc2_w, layer.deepstack_fc2_b,
                ffn_op_type::FFN_GELU, il);

            if(!deepstack_features) {
                deepstack_features = feat;
            } else {
                // concat along the feature dimension
                deepstack_features = ggml_concat(ctx0, deepstack_features, feat, 0);
            }
        }

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // multimodal projection
    ggml_tensor * embeddings = inpL;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);

    embeddings = build_ffn(embeddings,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        ffn_op_type::FFN_GELU, -1);

    if (deepstack_features) {
        embeddings = ggml_concat(ctx0, embeddings, deepstack_features, 0);
    } // concat along the feature dimension

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
