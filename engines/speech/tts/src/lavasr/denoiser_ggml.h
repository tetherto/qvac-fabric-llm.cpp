#pragma once

// LavaSR denoiser neural net (UL-UNAS U-Net) as a ggml compute graph, the GPU twin of the
// scalar denoiser_net_forward (denoiser_core.cpp).  STFT/resample + overlap-add stay CPU DSP.

#include "denoiser_core.h"

#include "ggml.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

namespace detail {

// `fused` selects the existing fused-op family. An optional GRU-only support
// query can independently select a real GRU candidate when `fused` is false;
// rejection keeps the standard-op recurrence and does not expand the candidate.
using GruSupport = std::function<bool(const ggml_tensor *)>;

// Queries the actual direct-convolution candidate. Rejection preserves the
// portable F32 IM2COL/matmul graph for older runtimes and unsupported layouts.
using ConvSupport = std::function<bool(const ggml_tensor *)>;

// The actual fused candidate is optional; older runtimes keep seven F32 ops.
using AffineSupport = std::function<bool(const ggml_tensor *)>;

// Query the actual F32 mean candidate; rejection retains SUM_ROWS + SCALE.
using MeanSupport = std::function<bool(const ggml_tensor *)>;

ggml_tensor * mean_rows(ggml_context * ctx, ggml_tensor * x, bool portable = false,
                        const MeanSupport & mean_support = {});

// Mean squared energy over channels: x:[F,T,C,Bc] -> [F,T*Bc].
// The portable layout folds spatial axes before transposing channels inward.
ggml_tensor * channel_energy_mean(ggml_context * ctx, ggml_tensor * x, bool portable = false,
                                  const MeanSupport & mean_support = {});

// Batched PyTorch GRU (gate order r,z,n; zero init state) over B weight-sharing sequences.
// x:[I,B,L], Wih:[I,3H], Whh:[H,3H], Bih/Bhh:[3H] -> y:[H,B,L]; reverse writes each step at its t.
ggml_tensor * gru_batched(ggml_context * ctx, ggml_tensor * x, ggml_tensor * Wih,
                          ggml_tensor * Whh, ggml_tensor * Bih, ggml_tensor * Bhh,
                          bool reverse, bool fused = true, const GruSupport & gru_support = {});

// Grouped/depthwise 2-D conv, causal in time, symmetric freq pad, freq stride only.
// x:[F,T,Cin], W:[kf,kt,Cin/g,Cout] (PyTorch [Cout,Cin/g,kt,kf]) -> [Fout,T,Cout].
ggml_tensor * conv2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * W,
                     ggml_tensor * bias, int stride_f, int pad_f, int groups, bool fused = true,
                     bool portable = false, const ConvSupport & conv_support = {});

// Transposed 2-D conv (decoder freq-upsampler) = zero-insert freq-upsample + causal conv2d.
// Wc is the pre-reindexed regular-conv kernel; x:[F,T,Cin] -> [(F-1)*stride_f+kf-2*pad_f,T,Cout].
ggml_tensor * conv_transpose2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * Wc,
                               ggml_tensor * bias, int stride_f, int pad_f, int groups, bool fused = true,
                               bool portable = false, const ConvSupport & conv_support = {});

// Per-(c,f) affine + per-channel PReLU.  aw,ab: ggml [F,C] (PyTorch [C,F]); slope:[C].
ggml_tensor * affine_prelu(ggml_context * ctx, ggml_tensor * x, ggml_tensor * aw,
                           ggml_tensor * ab, ggml_tensor * slope, bool fused = true,
                           const AffineSupport & affine_support = {});

// Channel shuffle (2 groups): o[2c]=x[c], o[2c+1]=x[half+c].  x:[F,T,C,Bc].
// The portable fallback copies whole F*T planes, preserving chunk batches.
ggml_tensor * shuffle2(ggml_context * ctx, ggml_tensor * x, bool fused = true,
                        bool portable = false);

// Resolves a weight tensor by name; returns nullptr if absent (mirrors Runner::Wopt).
using WResolver = std::function<ggml_tensor *(const std::string &)>;

// Materialize a contiguous [C,F,T,Bc] DPGRNN result as [F,T,C,Bc]. The portable
// layout folds F*T so the existing backend can use a batched 2-D transpose.
ggml_tensor * dpgrnn_output_layout(ggml_context * ctx, ggml_tensor * x, bool portable = false);

// Dual-path grouped RNN bottleneck (denoiser_core.cpp dpgrnn): grouped BiGRU over freq +
// grouped uni-GRU over time, each FC + LayerNorm-(F,C) + residual.  x:[F,T,C] -> [F,T,C].
ggml_tensor * dpgrnn(ggml_context * ctx, ggml_tensor * x, const std::string & prefix,
                     const WResolver & W, float ln_eps, bool fused = true,
                     const GruSupport & gru_support = {}, bool portable = false);

// Causal time-frequency attention (denoiser_core.cpp ctfa): out = at(temporal) * x * af(freq),
// each gate a GRU/BiGRU + FC + sigmoid over an energy-mean.  x:[F,T,C] -> [F,T,C].
// With gf, the gate subtrees are pinned into the graph first so the two mask
// multiplies come out adjacent (lets the backend fuse the pair).
ggml_tensor * ctfa(ggml_context * ctx, ggml_tensor * x, const std::string & prefix,
                   const WResolver & W, int freq_comp_ratio, bool fused = true,
                   ggml_cgraph * gf = nullptr, bool portable = false,
                   const GruSupport & gru_support = {}, const MeanSupport & mean_support = {});

} // namespace detail

class DenoiserGgml {
public:
    // Build the backend + upload weights.  n_gpu_layers: >0 = GPU (Adreno OpenCL,
    // CPU fallback), <0 = ggml-CPU (the CPU counterpart of the GPU graph). The
    // portable graph uses HTP-supported F32 decompositions and is selectable on
    // CPU for parity tests; HTP always enables it. Throws.
    static std::unique_ptr<DenoiserGgml> create(const DenoiserWeights & w,
                                                int n_gpu_layers, bool verbose = false,
                                                bool portable_graph = false);
    ~DenoiserGgml();
    DenoiserGgml(const DenoiserGgml &)             = delete;
    DenoiserGgml & operator=(const DenoiserGgml &) = delete;

    // Same contract as denoiser_net_forward: real/imag [L*spec_bins] (t*F+f) in
    // and out, one zero-state chunk of L frames.
    void chunk_forward(const std::vector<float> & real_in, const std::vector<float> & imag_in,
                       int L, std::vector<float> & real_out, std::vector<float> & imag_out);

    // Batched form: n_chunks independent zero-state chunks stacked as [n_chunks][L*F],
    // run in ONE graph (chunk-batch in ne3) so the dispatch cost is paid once.
    void batch_forward(const std::vector<float> & real_in, const std::vector<float> & imag_in,
                       int L, int n_chunks, std::vector<float> & real_out, std::vector<float> & imag_out);

    bool         is_gpu() const;
    const char * backend_name() const;

private:
    DenoiserGgml();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::lavasr
