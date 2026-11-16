#pragma once

// Where the merged prefill design's attention (xdna-pfx-attn.cpp) reads its objects, which its host pack or the row
// command (xdna-pfx-row.cpp) writes: argument 0's W objects, one a core a pass, and argument 1's two A streams (one
// a D half) of headers and 32-key K and V^T tiles.

#include <cstddef>
#include <cstdint>

struct xdna_buffer;

namespace xdna_pfx_layout {

constexpr int    ATTN_COLS = 6;              // attention columns, kernels/pfx.py: 0, 1, 2, 4, 5, 6
constexpr int    PAIRS     = 2 * ATTN_COLS;  // row pairs a pass
constexpr int    KVH       = 2;
constexpr int    R         = 32;             // rows a pair
constexpr int    NK        = 32;             // keys a tile
constexpr size_t OBJ_BYTES = 64 * 64 * 2;    // an A object: one K or V^T tile's D half
constexpr size_t TB_BYTES  = 10752;          // a core's W object, the length of every W send
constexpr int    HDR_OBJS  = 2;              // pgemm's header and the attention's, ahead of each A stream's tiles
constexpr size_t BO_ALIGN  = 4096;

inline size_t w_obj_off(size_t npass, int ci, size_t p, int i) {
    return (((size_t) ci * npass + p) * 4 + (size_t) i) * TB_BYTES;
}

// An A stream's object index of KV head g's tile t: its K, then V^T
inline size_t kv_obj(int64_t n_tiles, int g, int64_t t) {
    return HDR_OBJS + ((size_t) g * (size_t) n_tiles + (size_t) t) * 2;
}

inline size_t a_stream_objs(int64_t n_tiles) {
    return kv_obj(n_tiles, KVH, 0);
}

}  // namespace xdna_pfx_layout

// One attention node's objects in one buffer: the W objects from byte 0, the two A streams from a_off
struct xdna_pfx_attn_dst {
    xdna_buffer * bo           = nullptr;
    size_t        w_bytes      = 0;
    size_t        a_off        = 0;
    size_t        a_bytes      = 0;
    size_t        stream_bytes = 0;  // one A stream
    int64_t       n_tiles      = 0;
    int           pp           = 0;  // positions a pair (R / heads a KV head)
    int           npg          = 0;  // passes a KV head
};

// The layout for n_tokens rows, n_kv keys and hg query heads a KV head (bo null)
inline xdna_pfx_attn_dst xdna_pfx_attn_geometry(int64_t n_tokens, int64_t n_kv, int hg) {
    using namespace xdna_pfx_layout;
    xdna_pfx_attn_dst d;
    d.pp           = R / hg;
    d.npg          = (int) ((n_tokens + PAIRS * d.pp - 1) / (PAIRS * d.pp));
    d.n_tiles      = (n_kv + NK - 1) / NK;
    d.w_bytes      = w_obj_off((size_t) (KVH * d.npg), ATTN_COLS, 0, 0);
    d.a_off        = (d.w_bytes + BO_ALIGN - 1) / BO_ALIGN * BO_ALIGN;
    d.stream_bytes = a_stream_objs(d.n_tiles) * OBJ_BYTES;
    d.a_bytes      = 2 * d.stream_bytes;
    return d;
}

// The W object of KV head g's positions [pos, pos + pp) (pos a multiple of pp) and D half h: pair pidx of its pass
// is core 2 (pidx % 2) + h of attention column pidx / 2
inline size_t xdna_pfx_attn_q_obj(const xdna_pfx_attn_dst & d, int g, int64_t pos, int h) {
    using namespace xdna_pfx_layout;
    const int64_t per_pass = (int64_t) PAIRS * d.pp, k = pos / per_pass;
    const int     pidx     = (int) ((pos % per_pass) / d.pp);
    return w_obj_off((size_t) (KVH * d.npg), pidx / 2, (size_t) (g * d.npg + k), 2 * (pidx % 2) + h);
}

// KV head g's tile t, D half h: its K object, or with vt its V^T object
inline size_t xdna_pfx_attn_kv_obj(const xdna_pfx_attn_dst & d, int g, int64_t t, int h, bool vt) {
    using namespace xdna_pfx_layout;
    return d.a_off + (size_t) h * d.stream_bytes + (kv_obj(d.n_tiles, g, t) + (vt ? 1 : 0)) * OBJ_BYTES;
}
