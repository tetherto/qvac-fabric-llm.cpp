#pragma once
// ggml building blocks shared by the Audio8 transformers.
//
// The rotation is spelled out here instead of going through ggml_rope because
// Audio8's reference rounds its cosine/sine tables to bfloat16, and the graph
// has to consume those exact values to follow the reference trajectory. The
// tables arrive already permuted to the split-half layout by the converter.

#include "ggml.h"

#include <string>
#include <vector>

#include "audio8/internal.h"

namespace tts_cpp {
namespace audio8 {
namespace detail {

// Host bytes one scratch arena holds: the tensor headers plus the graph. The
// projector prices replayed graphs with this, so it cannot drift from what the
// constructor below actually allocates.
size_t scratch_arena_bytes(int nodes);

// A graph and the arena its nodes live in, sized for `nodes` tensors.
struct scratch {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;

    explicit scratch(int nodes);
    ~scratch();
    scratch(const scratch &) = delete;
    scratch & operator=(const scratch &) = delete;

    bool ok() const { return ctx && graph; }

    // Hands the arena to a caller that will outlive this scope; the caller then
    // owns the context and frees it with ggml_free.
    void release() { ctx = nullptr; graph = nullptr; }
};

ggml_tensor * input_i32(ggml_context * ctx, const char * name, int count);
ggml_tensor * input_f32(ggml_context * ctx, const char * name, int ne0, int ne1);

// Keeps a tensor alive past the allocator's reuse of intermediate buffers so
// the host can read it once the graph has run.
ggml_tensor * mark_output(ggml_cgraph * graph, ggml_tensor * t);

void write_input(ggml_cgraph * graph, const char * name, const void * data, size_t bytes);
void read_output(ggml_tensor * t, std::vector<float> & out);
void read_ids(ggml_tensor * t, std::vector<int32_t> & out);

// Inputs are written between these two calls: the allocator hands out the
// buffers, then the graph runs against them.
bool prepare_graph(ggml_backend_t backend, ::tts_cpp::detail::sched_fallback & sched,
                   ggml_backend_buffer_t weight_buffer, ggml_gallocr_t allocr,
                   ggml_cgraph * graph, const char * stage, bool & use_sched,
                   std::string * error);
bool compute_graph(ggml_backend_t backend, ::tts_cpp::detail::sched_fallback & sched,
                   ggml_cgraph * graph, bool use_sched, int n_threads,
                   const char * stage, std::string * error);

struct rope_planes {
    ggml_tensor * cos = nullptr;
    ggml_tensor * signed_sin = nullptr;
};

// One row per position of the baked tables, each widened to [head_dim, 1,
// count] so it broadcasts across heads and both halves of a head read the same
// plane: the cosine row twice, the sine row negated then plain. Widening costs
// three nodes per graph and saves five in every attention block.
rope_planes rope_window(ggml_context * ctx, ggml_tensor * cos_table,
                        ggml_tensor * sin_table, int first, int count);

// x is [head_dim, n_head, count], rotated as x * cos + swap(x) * signed_sin,
// where swap exchanges the halves of each head. Negating the lower sines is
// exact, so this is the arithmetic the per-half spelling did.
ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, const rope_planes & rope);

ggml_tensor * precise_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b);
ggml_tensor * multiply_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                           bool precise);

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps);

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x,
                     ggml_tensor * bias);

// SwiGLU: w2(silu(w1(x)) * w3(x)), as the single gated node so the two
// projections and the gate reach the backend as one fusable run.
ggml_tensor * swiglu(ggml_context * ctx, ggml_tensor * w1, ggml_tensor * w2,
                     ggml_tensor * w3, ggml_tensor * x);

// The same SwiGLU from w1 and w3 stacked into one [w1; w3] weight: one matmul,
// whose lower half gates the upper half.
ggml_tensor * swiglu_stacked(ggml_context * ctx, ggml_tensor * w13, ggml_tensor * w2,
                             ggml_tensor * x);

struct attention_shape {
    int n_head = 0;
    int n_kv = 0;
    int head_dim = 0;
    int width = 0;
    int n_past = 0;
    int layer = 0;
    bool precise_values = false;
};

// Projects x, rotates, appends to the cache and reads the whole prefix back.
// Returns [dim, width]; mask is [n_past + width, width], and may be null for a
// single position, which sees the whole prefix.
ggml_tensor * attention(ggml_context * ctx, ggml_cgraph * graph,
                        const attention_weights & weights, ggml_tensor * x,
                        const rope_planes & rope, const kv_cache & cache,
                        const attention_shape & shape, ggml_tensor * mask);

// Attention over a fresh, graph-local K/V pair; used by the codec's window
// transformers, which see their whole sequence at once.
ggml_tensor * windowed_attention(ggml_context * ctx, const attention_weights & weights,
                                 ggml_tensor * x, const rope_planes & rope,
                                 const attention_shape & shape, ggml_tensor * mask);

// windowed_attention for positions that follow earlier ones: held_keys (rotated)
// and held_values are their [head_dim, n_kv, held] keys and values, or null,
// and mask is [held + width, width]. Also returns the keys and values it
// attended over, the held ones followed by x's.
struct windowed_step {
    ggml_tensor * out = nullptr;
    ggml_tensor * keys = nullptr;
    ggml_tensor * values = nullptr;
};

windowed_step windowed_attention_after(ggml_context * ctx, const attention_weights & weights,
                                       ggml_tensor * x, const rope_planes & rope,
                                       const attention_shape & shape, ggml_tensor * held_keys,
                                       ggml_tensor * held_values, ggml_tensor * mask);

// Fills a [keys, queries] f32 mask with 0 where a query may attend and
// -INFINITY elsewhere. `window` of 0 means unlimited history.
void fill_causal_mask(float * mask, int keys, int queries, int first_query, int window);

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
