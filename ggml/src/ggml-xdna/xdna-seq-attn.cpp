#include "xdna-seq.h"

// Per-token TXN stream builder for the merged conv+norm+gdn decode kernel
// (fused_layer.xclbin, kernels/fused_layer.py). With the fused azg_n /
// out_words geometry it drives attn_gdn_gated.xclbin (kernels/attn_gdn_gated.py)
// and appends the on-chip gated phase. The emission mirrors the python
// builders word for word, so the two builders are interchangeable at runtime.

#include <algorithm>
#include <vector>

// xdna_dma_dir aliases matching the python/IRON usage in this backend:
// MM2S = host -> device fill, S2MM = device -> host drain.
static constexpr xdna_dma_dir DIR_MM2S = xdna_dma_dir::MM2S;
static constexpr xdna_dma_dir DIR_S2MM = xdna_dma_dir::S2MM;

// BD word fields (verified against the design's compiled stream): d1 = burst
// flag only, d2 = ax_cache 2.
static void emit_bd(xdna_seq * seq, int col, int bd, uint32_t len_words,
                    uint32_t arg, uint32_t byte_off) {
    xdna_bd bd_;
    bd_.buf_len   = len_words;                 // 4-byte words
    bd_.buf_off   = byte_off;
    bd_.d0_stride = 1;                         // 1D transfer, burst + ax_cache 2
    bd_.d1_stride = 1;
    bd_.d2_stride = 1;
    bd_.ax_cache  = 2;
    xdna_seq_blockwrite(seq, (uint32_t) col, 0, (uint32_t) bd, &bd_);
    xdna_seq_ddr_patch(seq, (uint32_t) col, 0, (uint32_t) bd, arg, byte_off);
}

// A strided descriptor: `d1` runs of `d0` words, `d1_stride` words apart. The
// encoding is the one IRON emits for a two-dimensional TensorAccessPattern -
// sizes in 32-bit words, strides in words - read back out of a built artifact.
static void emit_bd_2d(xdna_seq * seq, int col, int bd, uint32_t arg,
                       uint32_t byte_off, uint32_t d0, uint32_t d1,
                       uint32_t d1_stride) {
    xdna_bd bd_;
    bd_.buf_len   = d0 * d1;
    bd_.buf_off   = byte_off;
    bd_.d0_size   = d0;
    bd_.d0_stride = 1;
    bd_.d1_size   = d1;
    bd_.d1_stride = d1_stride;
    bd_.d2_stride = 1;
    bd_.ax_cache  = 2;
    xdna_seq_blockwrite(seq, (uint32_t) col, 0, (uint32_t) bd, &bd_);
    xdna_seq_ddr_patch(seq, (uint32_t) col, 0, (uint32_t) bd, arg, byte_off);
}

static bool valid_geom(const xdna_attn_gdn_geom * g) {
    if (!g || g->conv_cols <= 0 || g->norm_cols <= 0 || g->gdn_cols <= 0 ||
        g->conv_cols + g->norm_cols + g->gdn_cols != g->n_cols ||
        g->n_block % g->conv_cols != 0 || g->n_vh % g->norm_cols != 0 ||
        g->n_block % g->n_vh != 0 || g->feed_n < 3 * g->sv + g->sv ||
        g->head_norm != 3 * g->sv + 3 || g->pkv_n != 3 * g->sv + 3 ||
        g->pkvb_n != g->n_obj * g->pkv_n) {
        return false;
    }
    return true;
}

bool xdna_attn_gdn_build(xdna_seq * seq, const xdna_attn_gdn_geom * g,
                             int schedule, int phase) {
    // phase < 0 builds the whole token; 0..3 build conv, norm, gdn or the
    // gated epilogue alone. The cores loop forever waiting on their fifos, so
    // a phase submitted on its own is the same work in the same order - which
    // is what makes a per-stage timing possible at all.
    const bool want_conv  = phase < 0 || phase == 0;
    const bool want_norm  = phase < 0 || phase == 1;
    const bool want_gdn   = phase < 0 || phase == 2;
    const bool want_gated = phase < 0 || phase == 3;
    if (!seq || !valid_geom(g) || schedule < 0 || schedule > 7 || phase > 3) {
        return false;
    }
    seq->n_cols = (uint32_t) g->n_cols;
    const bool fused   = g->azg_n > 0 && g->out_words > 0;
    const int conv_gpo = g->conv_gpo > 0 ? g->conv_gpo : 1;
    const int conv_gp  = g->n_block / g->conv_cols;  // 24 conv groups / col
    const int hp       = g->n_vh / g->norm_cols;     // 8 heads / norm col
    const int norm_col0 = g->conv_cols;
    const int gdn_col0  = norm_col0 + g->norm_cols;
    const int gdn_slots = g->n_vh * g->n_obj / g->gdn_cols;  // 32 / gdn col

    const int group = schedule == 1 ? 1 : 2;

    // ---- conv: cols 0..conv_cols-1 (phase A) ----
    // A round is conv_gpo consecutive feed groups: contiguous in the feed
    // buffer, and gpo consecutive heads of one region in x, so each of the
    // three transfers is still a single descriptor.
    // Each round of a batch gets its own three descriptors. Sharing them
    // between rounds means reprogramming one whose transfer may not have
    // started, which the hardware does not notice and silently drops.
    auto conv_fill = [&](int col, int ga, int b = 0) {
        const uint32_t bd = (uint32_t)(3 * b);
        const uint32_t off = (uint32_t)(ga * g->feed_n * 4);
        if (g->feed_slot >= g->feed_n) {
            emit_bd(seq, col, (int) bd, (uint32_t)(conv_gpo * g->feed_n), 0, off);
        } else {
            emit_bd_2d(seq, col, (int) bd, 0, off, (uint32_t) g->feed_slot,
                       (uint32_t) conv_gpo, (uint32_t) g->feed_n);
        }
        xdna_seq_push_queue(seq, (uint32_t) col, 0, bd, DIR_MM2S, 0, false, 0);
    };
    auto conv_drain = [&](int col, int ga, int b = 0) {
        const int h = ga % g->n_vh;
        const int r = ga / g->n_vh;   // 0 = q, 1 = k, 2 = v region
        const uint32_t xoff = (uint32_t)((h * g->head_norm + r * g->sv) * 4);
        const int xc = g->conv_x_col[col], xh = g->conv_x_ch[col];
        const int hc = g->conv_h_col[col], hh = g->conv_h_ch[col];
        const uint32_t bx = (uint32_t)(3 * b + 1);
        const uint32_t bh = (uint32_t)(3 * b + 2);
        if (conv_gpo == 1) {
            emit_bd(seq, xc, (int) bx, (uint32_t) g->sv, 1, xoff);
        } else {
            emit_bd_2d(seq, xc, (int) bx, 1, xoff, (uint32_t) g->sv,
                       (uint32_t) conv_gpo, (uint32_t) g->head_norm);
        }
        xdna_seq_issue_token(seq, (uint32_t) xc, 0, DIR_S2MM, (uint32_t) xh, 0xF);
        xdna_seq_push_queue(seq, (uint32_t) xc, 0, bx, DIR_S2MM,
                            (uint32_t) xh, true, 0);
        if (conv_gpo == 1) {
            emit_bd(seq, hc, (int) bh, (uint32_t)(3 * g->sv), 0,
                    (uint32_t)(ga * g->feed_n * 4));
        } else {
            emit_bd_2d(seq, hc, (int) bh, 0, (uint32_t)(ga * g->feed_n * 4),
                       (uint32_t)(3 * g->sv), (uint32_t) conv_gpo,
                       (uint32_t) g->feed_n);
        }
        xdna_seq_issue_token(seq, (uint32_t) hc, 0, DIR_S2MM, (uint32_t) hh, 0xF);
        xdna_seq_push_queue(seq, (uint32_t) hc, 0, bh, DIR_S2MM,
                            (uint32_t) hh, true, 0);
    };
    auto conv_wait = [&](int col) {
        xdna_seq_wait_token(seq, (uint32_t) g->conv_x_col[col], 0, DIR_S2MM,
                            (uint32_t) g->conv_x_ch[col]);
        xdna_seq_wait_token(seq, (uint32_t) g->conv_h_col[col], 0, DIR_S2MM,
                            (uint32_t) g->conv_h_ch[col]);
    };
    auto conv_slot = [&](int col, int s) {
        conv_fill(col, col * conv_gp + s);
        conv_drain(col, col * conv_gp + s);
        conv_wait(col);
    };
    if (!want_conv) {
        // nothing
    } else if (conv_gpo != 1 && schedule != 4) {
        return false;   // the diagnostic schedules assume one group an object
    } else if (schedule == 7) {
        // One descriptor per direction per column instead of seventy-two. A
        // shim descriptor is a byte stream the MemTile cuts into objects, so a
        // strided one can cover every group of a column at once: the history
        // writes back into the feed slots, which are conv_gp runs of 3*sv a
        // feed apart, and the x writes go to (head, region) slots, which are
        // runs of consecutive heads one head_norm apart - two runs a column
        // for this geometry.
        //
        // This is the last of the four things conv's time could be made of.
        // It is not arithmetic (emptying the kernel changes nothing), not
        // bytes (295 KB is 23 us at the array's rate), not objects in flight
        // (a fifo depth of 4 with five slots banked changes nothing) - which
        // leaves the per-transfer cost of 72 small descriptors a column.
        for (int col = 0; col < g->conv_cols; col++) {
            int bd = 0;
            emit_bd(seq, col, bd, (uint32_t)(conv_gp * g->feed_n), 0,
                    (uint32_t)(col * conv_gp * g->feed_n * 4));
            xdna_seq_push_queue(seq, (uint32_t) col, 0, (uint32_t) bd, DIR_MM2S,
                                0, false, 0);
            bd++;

            int n_x = 0;
            for (int s = 0; s < conv_gp; ) {
                const int ga0 = col * conv_gp + s;
                const int r0  = ga0 / g->n_vh;
                const int h0  = ga0 % g->n_vh;
                int n = 1;
                while (s + n < conv_gp) {
                    const int ga = col * conv_gp + s + n;
                    if (ga / g->n_vh != r0 || ga % g->n_vh != h0 + n) {
                        break;
                    }
                    n++;
                }
                emit_bd_2d(seq, col, bd, 1,
                           (uint32_t)((h0 * g->head_norm + r0 * g->sv) * 4),
                           (uint32_t) g->sv, (uint32_t) n,
                           (uint32_t) g->head_norm);
                xdna_seq_issue_token(seq, (uint32_t) col, 0, DIR_S2MM, 1, 0xF);
                xdna_seq_push_queue(seq, (uint32_t) col, 0, (uint32_t) bd,
                                    DIR_S2MM, 1, true, 0);
                bd++;
                n_x++;
                s += n;
            }

            emit_bd_2d(seq, col, bd, 0,
                       (uint32_t)(col * conv_gp * g->feed_n * 4),
                       (uint32_t)(3 * g->sv), (uint32_t) conv_gp,
                       (uint32_t) g->feed_n);
            xdna_seq_issue_token(seq, (uint32_t) col, 0, DIR_S2MM, 0, 0xF);
            xdna_seq_push_queue(seq, (uint32_t) col, 0, (uint32_t) bd,
                                DIR_S2MM, 0, true, 0);

            for (int k = 0; k < n_x; k++) {
                xdna_seq_wait_token(seq, (uint32_t) col, 0, DIR_S2MM, 1);
            }
            xdna_seq_wait_token(seq, (uint32_t) col, 0, DIR_S2MM, 0);
        }
    } else if (schedule == 6) {
        // One feed descriptor per conv column instead of one per group. The
        // groups a column consumes are consecutive in the feed buffer, and a
        // shim descriptor is a byte stream that the MemTile fifo cuts into
        // objects, so 24 transfers become one. Nothing else changes: the
        // drains still go a group at a time. This is here to measure whether
        // the stream's own instruction count is what the core's time is made
        // of - the descriptor bank of schedule 5 removed waits without
        // removing transfers and changed nothing at all.
        for (int col = 0; col < g->conv_cols; col++) {
            emit_bd(seq, col, 0, (uint32_t)(conv_gp * g->feed_n), 0,
                    (uint32_t)(col * conv_gp * g->feed_n * 4));
            xdna_seq_push_queue(seq, (uint32_t) col, 0, 0, DIR_MM2S, 0, false, 0);
        }
        for (int s = 0; s < conv_gp; s++) {
            for (int col = 0; col < g->conv_cols; col++) {
                conv_drain(col, col * conv_gp + s);
            }
            for (int col = 0; col < g->conv_cols; col++) {
                conv_wait(col);
            }
        }
    } else if (schedule == 5) {
        // A descriptor bank for the conv slots: each slot keeps its own three
        // descriptors, so several are in flight before anything waits. conv is
        // half of the stream's transfers and more than half of its token
        // waits, and a column's descriptor file has room for five slots.
        const int B = 5;
        for (int s0 = 0; s0 < conv_gp; s0 += B) {
            const int nb = std::min(B, conv_gp - s0);
            for (int b = 0; b < nb; b++) {
                for (int col = 0; col < g->conv_cols; col++) {
                    const int ga = col * conv_gp + s0 + b;
                    emit_bd(seq, col, 3 * b, (uint32_t) g->feed_n, 0,
                            (uint32_t)(ga * g->feed_n * 4));
                    xdna_seq_push_queue(seq, (uint32_t) col, 0, (uint32_t)(3 * b),
                                        DIR_MM2S, 0, false, 0);
                }
            }
            for (int b = 0; b < nb; b++) {
                for (int col = 0; col < g->conv_cols; col++) {
                    const int ga = col * conv_gp + s0 + b;
                    const int h = ga % g->n_vh;
                    const int r = ga / g->n_vh;   // 0 = q, 1 = k, 2 = v region
                    const uint32_t xoff = (uint32_t)((h * g->head_norm + r * g->sv) * 4);
                    emit_bd(seq, col, 3 * b + 1, (uint32_t) g->sv, 1, xoff);
                    xdna_seq_issue_token(seq, (uint32_t) col, 0, DIR_S2MM, 1, 0xF);
                    xdna_seq_push_queue(seq, (uint32_t) col, 0, (uint32_t)(3 * b + 1),
                                        DIR_S2MM, 1, true, 0);
                    emit_bd(seq, col, 3 * b + 2, (uint32_t)(3 * g->sv), 0,
                            (uint32_t)(ga * g->feed_n * 4));
                    xdna_seq_issue_token(seq, (uint32_t) col, 0, DIR_S2MM, 0, 0xF);
                    xdna_seq_push_queue(seq, (uint32_t) col, 0, (uint32_t)(3 * b + 2),
                                        DIR_S2MM, 0, true, 0);
                }
            }
            for (int b = 0; b < nb; b++) {
                for (int col = 0; col < g->conv_cols; col++) {
                    conv_wait(col);
                }
            }
        }
    } else if (schedule == 0) {
        for (int col = 0; col < g->conv_cols; col++) {
            for (int s = 0; s < conv_gp; s++) {
                conv_slot(col, s);
            }
        }
    } else {
        for (int g0 = 0; g0 < conv_gp; g0 += group * conv_gpo) {
            for (int s = g0, b = 0; s < g0 + group * conv_gpo && s < conv_gp;
                 s += conv_gpo, b++) {
                for (int col = 0; col < g->conv_cols; col++) {
                    conv_fill(col, col * conv_gp + s, b);
                }
            }
            for (int s = g0, b = 0; s < g0 + group * conv_gpo && s < conv_gp;
                 s += conv_gpo, b++) {
                for (int col = 0; col < g->conv_cols; col++) {
                    conv_drain(col, col * conv_gp + s, b);
                }
            }
            for (int s = g0; s < g0 + group * conv_gpo && s < conv_gp; s += conv_gpo) {
                for (int col = 0; col < g->conv_cols; col++) {
                    conv_wait(col);
                }
            }
        }
    }

    // ---- norm (phase B) ----
    if (want_norm)
    // One stream in and one out for the stage: a round is norm_cols adjacent
    // heads, contiguous in both buffers, which a MemTile hands one head to
    // each column. The input sits on norm_col0's shim and the output on the
    // next column's, both channel 0.
    {
    const int nfill_col  = norm_col0;
    const int ndrain_col = norm_col0 + 1;
    // One head per object when the chunks stay on chip: the four norm cores
    // each need the whole head, so the fill is a broadcast rather than a split.
    const int nh_fill = g->pkv_onchip ? g->n_vh : hp;
    const int nw_fill = g->pkv_onchip ? g->head_norm : g->norm_cols * g->head_norm;
    // On chip the fills move into the gdn loop below and are paced by its
    // waits: pushed here in a block of sixteen they fill the shim's queue
    // before the gdn block's state fills are issued, and nothing can drain.
    for (int h = 0; !g->pkv_onchip && h < nh_fill; h++) {
        emit_bd(seq, nfill_col, 0, (uint32_t) nw_fill, 1,
                (uint32_t)(h * nw_fill * 4));
        xdna_seq_push_queue(seq, (uint32_t) nfill_col, 0, 0, DIR_MM2S, 0, false, 0);
        if (!g->pkv_onchip) {
            emit_bd(seq, ndrain_col, 0, (uint32_t)(g->norm_cols * g->pkvb_n), 2,
                    (uint32_t)(h * g->norm_cols * g->pkvb_n * 4));
            xdna_seq_issue_token(seq, (uint32_t) ndrain_col, 0, DIR_S2MM, 0, 0xF);
            xdna_seq_push_queue(seq, (uint32_t) ndrain_col, 0, 0, DIR_S2MM, 0, true, 0);
            xdna_seq_wait_token(seq, (uint32_t) ndrain_col, 0, DIR_S2MM, 0);
        }
    }
    }

    // ---- gdn (phase C) ----
    if (want_gdn)
    // The design carries one stream for the whole gdn block instead of one per
    // column: an object is a round of gdn_cols consecutive chunks, which a
    // MemTile hands out one chunk per column, so chunk gi + r*gdn_cols still
    // lands on column gi. Four consecutive chunks are contiguous in every
    // buffer they touch, so a round is a single descriptor. Each stream sits
    // on its own column's MemTile - and therefore its own column's shim
    // channel 0 - which is what took the block from eight shim channels each
    // way to four.
    {
    const int pkv_col   = gdn_col0;
    const int azg_col   = gdn_col0 + 3;
    const int gc        = g->gdn_cols;
    // The state is split across sg streams in each direction, each serving
    // gg = gdn_cols/sg of the columns. A round is still gdn_cols consecutive
    // chunks, so a stream's half of it is contiguous in the state buffer.
    // Where they sit is what the design pins (kernels/attn_gdn_gated.py) and
    // what the artifact pins: fills on the state and sout columns, drains
    // on the same two the other way round.
    const int sg = g->gdn_state_streams > 0 ? g->gdn_state_streams : 1;
    const int gg = gc / sg;
    // A shim tile has one port for all of its channels, so what has to be
    // spread is the tile, not the channel: splitting the state onto two
    // channels of the same two tiles measured worse than not splitting it at
    // all (209 us against 177). Four tiles, none of them carrying both
    // directions - the norm stage moving on chip is what freed the two the
    // drains use. Keep in step with S_COLS/O_COLS in attn_gdn_gated.py.
    const int s_col[2] = { gdn_col0 + 1, gdn_col0 + 2 };
    const int s_ch [2] = { 0, 0 };
    const int o_col[2] = { sg == 1 ? gdn_col0 + 2 : gdn_col0 + 3, gdn_col0 };
    const int o_ch [2] = { sg == 1 ? 0 : 1, 0 };
    if (sg > 2 || gc % sg != 0) {
        return false;
    }
    if (fused && want_gated && g->attn_onchip) {
        // The gated tile takes a head's attn from the array as the gdn block
        // produces it, so its z/gamma fill has to be in flight before the
        // rounds start. Left where it was otherwise - with attn in DDR the
        // tile cannot start until the whole block has drained anyway.
        // On the attn column's shim, not the gated tile's: the GEMV's eight
        // weight streams take a channel on every column and this is the one
        // the recurrent design can spare (attn_gdn_gated.py).
        emit_bd(seq, azg_col, 0, (uint32_t)(g->n_vh * g->azg_n), 4, 0);
        xdna_seq_push_queue(seq, (uint32_t) azg_col, 0, 0, DIR_MM2S, 0,
                            false, 0);
    }
    // Rounds in flight before waiting; each costs one descriptor per stream.
    const int B = (schedule >= 4) ? 4 : 1;

    for (int r0 = 0; r0 < gdn_slots; r0 += B) {
        const int nb = std::min(B, gdn_slots - r0);
        for (int b = 0; b < nb; b++) {
            const int c0 = (r0 + b) * gc;
            if (!g->pkv_onchip) {
                emit_bd(seq, pkv_col, b, (uint32_t)(gc * g->pkv_n), 2,
                        (uint32_t)(c0 * g->pkv_n * 4));
                xdna_seq_push_queue(seq, (uint32_t) pkv_col, 0, (uint32_t) b,
                                    DIR_MM2S, 0, false, 0);
            } else if (((r0 + b) % (g->n_obj / gc)) == 0) {
                // A head is n_obj/gc rounds, and its norm cores need it before
                // this round can run: one fill per head, here, where the gdn
                // waits pace it. Four descriptors, because at most two are in
                // flight between waits.
                const int h = (r0 + b) / (g->n_obj / gc);
                const uint32_t bd = (uint32_t)(4 + (h & 3));
                emit_bd(seq, norm_col0, (int) bd, (uint32_t) g->head_norm, 1,
                        (uint32_t)(h * g->head_norm * 4));
                xdna_seq_push_queue(seq, (uint32_t) norm_col0, 0, bd,
                                    DIR_MM2S, 0, false, 0);
            }
            for (int sgi = 0; sgi < sg; sgi++) {
                emit_bd(seq, s_col[sgi], b, (uint32_t)(gg * g->rows / 2), 3,
                        (uint32_t)((c0 + sgi * gg) * g->rows * 2));
                xdna_seq_push_queue(seq, (uint32_t) s_col[sgi], 0, (uint32_t) b,
                                    DIR_MM2S, (uint32_t) s_ch[sgi], false, 0);
            }
        }
        for (int b = 0; b < nb; b++) {
            const int c0   = (r0 + b) * gc;
            const int head = c0 / g->n_obj;
            const int j    = c0 % g->n_obj;
            for (int sgi = 0; sgi < sg; sgi++) {
                // The two directions share a column now, and a shim tile's
                // descriptor file is shared between its channels, so the
                // drains sit above the fills rather than alongside them.
                const uint32_t bd = (uint32_t)(B + b);
                emit_bd(seq, o_col[sgi], (int) bd, (uint32_t)(gg * g->rows / 2), 3,
                        (uint32_t)((c0 + sgi * gg) * g->rows * 2));
                xdna_seq_issue_token(seq, (uint32_t) o_col[sgi], 0, DIR_S2MM,
                                     (uint32_t) o_ch[sgi], 0xF);
                xdna_seq_push_queue(seq, (uint32_t) o_col[sgi], 0, bd,
                                    DIR_S2MM, (uint32_t) o_ch[sgi], true, 0);
            }
            if (!g->attn_onchip) {
                const uint32_t aoff = (uint32_t)((fused ? head * g->azg_n + j * g->chunk
                                                        : head * g->sv    + j * g->chunk) * 4);
                emit_bd(seq, azg_col, b, (uint32_t)(gc * g->chunk), 4, aoff);
                xdna_seq_issue_token(seq, (uint32_t) azg_col, 0, DIR_S2MM, 0, 0xF);
                xdna_seq_push_queue(seq, (uint32_t) azg_col, 0, (uint32_t) b,
                                    DIR_S2MM, 0, true, 0);
            }
        }
        for (int b = 0; b < nb; b++) {
            for (int sgi = 0; sgi < sg; sgi++) {
                xdna_seq_wait_token(seq, (uint32_t) o_col[sgi], 0, DIR_S2MM,
                                    (uint32_t) o_ch[sgi]);
            }
            if (!g->attn_onchip) {
                xdna_seq_wait_token(seq, (uint32_t) azg_col, 0, DIR_S2MM, 0);
            }
        }
    }
    }

    if (fused && want_gated) {
        // ---- gated phase: azg fill (whole arg4) on the first norm column's
        // spare MM2S ch1, then the aq/d_a OUT drain on its S2MM ch1.  The
        // gdn drains of every schedule above already waited, so the azg attn
        // lanes are in DDR before this fill reads them.
        const int gcol = norm_col0;   // the gated tile parks on col 2
        if (!g->attn_onchip) {
            const uint32_t azg_words = (uint32_t)(g->n_vh * g->azg_n);
            emit_bd(seq, gdn_col0 + 3, 0, azg_words, 4, 0);
            xdna_seq_push_queue(seq, (uint32_t)(gdn_col0 + 3), 0, 0, DIR_MM2S,
                                0, false, 0);
        }
        // OUT drain (arg5): the aperture base the argument needs is folded in
        // by xdna_seq_ddr_patch now, so this is a plain offset like the rest.
        // The activation the appended projection reads gets its own
        // descriptor: one object drained by two descriptors back to back on
        // the same channel, the second covering only the activation window,
        // so the handover has a completion of its own to wait on.
        const uint32_t act_w = (uint32_t)(g->gated_act_bytes / 4);
        {
            xdna_bd bd_;
            bd_.buf_len   = (uint32_t) g->out_words - act_w;
            bd_.buf_off   = (uint32_t) g->out_base;
            bd_.d0_stride = 1;
            bd_.d1_stride = 1;
            bd_.d2_stride = 1;
            bd_.ax_cache  = 2;
            xdna_seq_blockwrite(seq, (uint32_t) gcol, 0, 0, &bd_);
            xdna_seq_ddr_patch(seq, (uint32_t) gcol, 0, 0, 5,
                               (uint32_t) g->out_base);
        }
        // Channel 0, not 1: the norm drain used to occupy it on this column
        // and the gated drain took what was left. With norm consolidated onto
        // the next column's shim this one is free again, and IRON places the
        // drain there - the two builders have to agree on the channel or the
        // stream waits on a token no one issues.
        if (act_w) {
            // Its own fifo, its own column, and a plain patch: this is the
            // descriptor an appended projection reads back.
            // Descriptor 1, not 0: this column's shim carries the azg fill
            // on 0, and that transfer feeds the gated tile head by head - it
            // is still in flight when this stage runs, and reprogramming its
            // descriptor loses it.
            const int acol = g->gated_act_col;
            emit_bd(seq, acol, 1, act_w, (uint32_t) g->gated_act_arg,
                    (uint32_t) g->gated_act_off);
            xdna_seq_issue_token(seq, (uint32_t) acol, 0, DIR_S2MM, 0, 0xF);
            xdna_seq_push_queue(seq, (uint32_t) acol, 0, 1, DIR_S2MM, 0, true, 0);
        }
        // The core releases the activation object before its output, so the
        // activation's transfer is pushed first.
        xdna_seq_issue_token(seq, (uint32_t) gcol, 0, DIR_S2MM, 0, 0xF);
        xdna_seq_push_queue(seq, (uint32_t) gcol, 0, 0, DIR_S2MM, 0, true, 0);
        if (act_w) {
            xdna_seq_wait_token(seq, (uint32_t) g->gated_act_col, 0,
                                DIR_S2MM, 0);
        }
        xdna_seq_wait_token(seq, (uint32_t) gcol, 0, DIR_S2MM, 0);
    }
    return true;
}
