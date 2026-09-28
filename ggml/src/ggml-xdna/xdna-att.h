#pragma once

// Decode attention on the fused layer's GEMV pool (kernels/attn-dec.cc): one
// query token of Qwen3.5's full-attention layers - eight query heads over two
// kv heads of 256 - against llama's f16 KV cache, read where it lies in the
// backend's host BOs. The sixteen pool cores each take every sixteenth chunk
// of five positions; their partial softmax states are combined here.

#include <cstddef>

struct xdna_att;
struct xdna_buffer;
struct xdna_device;
struct xdna_kernel_pool;

xdna_att * xdna_att_create(struct xdna_kernel_pool * pool, struct xdna_device * dev);
void       xdna_att_free(xdna_att * a);

// The cache positions one dispatch can take.
int xdna_att_max_positions(void);

// out[h * 256 + d] = softmax(scale * q_h . K) V for the first n_valid cache
// positions. `k` / `v` are the BO and byte offset of position 0 of the layer's
// K and V (position stride 1024 B, kv-head stride 512 B, f16). The caller has
// synced the rows the host wrote. False when the dispatch cannot be built or
// run - the caller falls back.
bool xdna_att_run(xdna_att * a, struct xdna_buffer * kbo, size_t koff,
                  struct xdna_buffer * vbo, size_t voff, const float * q,
                  float scale, int n_valid, float * out);
