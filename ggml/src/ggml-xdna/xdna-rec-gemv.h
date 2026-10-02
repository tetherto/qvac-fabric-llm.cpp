#pragma once

// The projection stages of a fused recurrent layer - ssm_out and the whole FFN
// - run on the decode GEMV artifact (xdna-gemv.h) instead of their own
// xclbins.
//
// Two measured reasons, both of which the GEMV design already answers. A
// hardware context is per xclbin and the array is reconfigured whenever
// consecutive dispatches come from different ones; a layer that alternates
// between three pays that three times per token, measured at 638 us for
// ssm_out and 1354 us for the FFN by timing an immediate repeat of the same
// dispatch. And with the context already resident the fused FFN still spent
// 2065 us on ~8 MB of weights, about 4 GB/s, where the GEMV stream measures
// 56 GB/s on the same weights.
//
// One artifact serves every shape here, so ssm_out, the gate/up activation and
// the down projection are three dispatches on one context. The FFN's
// nonlinearity closes on the cores (the epilogue), so the gate and up halves
// never come back to the host; only the residual adds stay here, 1024 floats
// each.

#include "xdna-gemv.h"

#include <cstdint>
#include <vector>

struct ggml_tensor;
struct xdna_kernel_pool;

struct xdna_rec_gemv {
    xdna_gemv *      so    = nullptr;  // ssm_out    [K_gate x d_out]
    // The FFN is one dispatch when the pair can be built: the epilogue writes
    // its quantized activation straight into what the down projection reads,
    // so the two share an instruction stream. `act` and `down` are the
    // fallback, two dispatches with the activation going through the host.
    xdna_gemv_pair * ffn   = nullptr;
    xdna_gemv *      act   = nullptr;  // gate | up  [d_out x 2*d_ff] -> silu(g)*u
    xdna_gemv *      down  = nullptr;  // down       [d_ff x d_out]
    int              n_out = 0;        // the down projection's real width

    std::vector<float>   a;            // the gdn output, dequantized from its int8 codes
    std::vector<float>   mid;          // the FFN activation the epilogue returns
    std::vector<float>   acc;          // a projection's output, before the residual add
    std::vector<uint8_t> settled;      // staging for a settled read of a buffer
};

// Pack the four weights into the GEMV layouts and load the runners. The
// weights must be types the GEMV route covers (Q4_K for gate/up; Q4_K, Q5_K or
// Q6_K for ssm_out and down) and their shapes must tile the array exactly.
// Returns nullptr when any of that does not hold.
xdna_rec_gemv * xdna_rec_gemv_create(struct xdna_kernel_pool *  pool,
                                     const struct ggml_tensor * w_so,
                                     const struct ggml_tensor * w_gate,
                                     const struct ggml_tensor * w_up,
                                     const struct ggml_tensor * w_down);

void xdna_rec_gemv_free(xdna_rec_gemv * m);

// An attention layer's tail as one dispatch: attn_output drains straight into
// the FFN's raw activation tiles and the FFN pair follows in the same stream -
// the stream a GDN layer ends with, without the core in front of it. The
// residual add, the post-attention norm and SwiGLU are the array's (the
// prologue tile and the epilogue); the host writes the projection's input and
// the tiles' host half, and adds the two residuals from what comes back.
struct xdna_rec_tail {
    xdna_rec_gemv *      gv   = nullptr;  // so = attn_output, and the FFN pair
    xdna_kernel *        kern = nullptr;  // pooled, shared by every layer with this stream
    xdna_buffer *        w    = nullptr;  // [attn_output weights | FFN weights]
    xdna_buffer *        a    = nullptr;  // attn_output's activation, then gamma
    xdna_buffer *        res  = nullptr;  // the residual rows (not owned)
    std::vector<uint8_t> host_a;
    xrt::run             run;
};

xdna_rec_tail * xdna_rec_tail_create(struct xdna_kernel_pool *  pool,
                                     const struct ggml_tensor * w_o,
                                     const struct ggml_tensor * w_gate,
                                     const struct ggml_tensor * w_up,
                                     const struct ggml_tensor * w_down,
                                     struct xdna_buffer *       res,
                                     const float *              gamma,
                                     float                      eps);
void            xdna_rec_tail_free(xdna_rec_tail * t);

// h_attn = hres + W_o * act; h_out = h_attn + FFN(rms_norm(h_attn) * gamma),
// the boundary on the prologue through the residual rows `res`.
bool xdna_rec_tail_run(xdna_rec_tail * t, const float * act, const float * hres, float * h_attn, float * h_out);

// The layer's in-projection - attn_qkv and attn_gate against the normed input
// - packed for the head of the core's own stream. Its values come off the
// array split by stream: the last output stream carries z and every other one
// qkv, so each stream drains into one of the core's inputs (the per-block qkv
// window of the feed, the per-head z lane of azg) and the projection has no
// output buffer at all. False when the shapes do not split that way.
struct xdna_rec_inproj {
    xdna_gemv_geom       geom;
    std::vector<uint8_t> packed;
    int                  n_qkv = 0;  // qkv values
    int                  n_z   = 0;  // z values
};

bool xdna_rec_inproj_pack(const struct ggml_tensor * w_qkv, const struct ggml_tensor * w_z, xdna_rec_inproj & out);

// The ssm_out GEMV, for appending its stream to the core's.
struct xdna_gemv * xdna_rec_gemv_so(xdna_rec_gemv * m);

// Collect a projection the fused core dispatch has already run: reads the
// device output and adds the residual, which is all xdna_rec_gemv_so_run does
// once its dispatch is somebody else's.
bool xdna_rec_gemv_so_collect(xdna_rec_gemv * m,
                              const void *    out,
                              size_t          off,
                              const void *    act,
                              const float *   hres,
                              float *         h_attn);

// h_attn = hres + W_so * a. `act_tiles`, when given and the weight is the
// 4-bit form, is the activation the core's gated stage already wrote in the
// GEMV's tile layout, so nothing is dequantized or repacked here; otherwise
// the int8 codes and their row scale are used.
bool xdna_rec_gemv_so_run(xdna_rec_gemv * m,
                          const void *    act_tiles,
                          const int8_t *  aq,
                          float           d_a,
                          const float *   hres,
                          float *         h_attn);

// h_out = h_attn + W_down * (silu(W_gate * hff) * (W_up * hff)).
bool xdna_rec_gemv_ffn_run(xdna_rec_gemv * m, const float * hff, const float * h_attn, float * h_out);

// The projection's result read out of the FFN's activation tiles, where the
// fused dispatch drained it. `acc` takes n_out floats.
bool xdna_rec_gemv_acc_from_tiles(xdna_rec_gemv * m, float * acc);

// Both results of a layer whose FFN rode the fused dispatch - the projection
// from the activation tiles and the FFN's output from the tail after them -
// out of one settled read of the buffer rather than one each.
bool xdna_rec_gemv_fused_results(xdna_rec_gemv * m, float * acc, float * ffn_out);

// h_out = h_attn + W_down * (silu(W_gate * x) * (W_up * x)) with
// x = rms_norm(acc + hres) * gamma computed on the array's prologue tile, so
// the host does nothing between the layer's two dispatches.
bool xdna_rec_gemv_ffn_run_raw(xdna_rec_gemv * m,
                               const float *   acc,
                               const float *   hres,
                               const float *   gamma,
                               const float *   h_attn,
                               float *         h_out);
