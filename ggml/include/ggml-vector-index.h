#pragma once
//
// ggml-vector-index: TurboQuant-style ANN vector-index C API.
//
// POC NOTE
// --------
// This is an experimental public C API for fabric's vector index. The
// implementation under `ggml/src/ggml-vector-index.cpp` is intentionally naive
// (full f32 storage, scalar dot-product, min-heap top-k). The stacked follow-up
// PRs expand the storage modes, error handling, and persistence format before
// downstream bindings should treat the contract as stable.
//
// Threading: instances are NOT thread-safe. Callers must serialize access
// to a given handle. Multiple handles can be used concurrently.
//
// Endianness: persistence format is fixed little-endian. Values are converted
// explicitly when snapshots are read and written.

#include <stdint.h>

#ifndef GGML_API
#    ifdef GGML_SHARED
#        if defined(_WIN32) && !defined(__MINGW32__)
#            ifdef GGML_BUILD
#                define GGML_API __declspec(dllexport) extern
#            else
#                define GGML_API __declspec(dllimport) extern
#            endif
#        else
#            define GGML_API __attribute__((visibility("default"))) extern
#        endif
#    else
#        define GGML_API extern
#    endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a vector index instance.
struct ggml_vec_index;
typedef struct ggml_vec_index ggml_vec_index_t;

// Error codes returned from int-valued APIs. 0 = OK. Negative = failure.
enum ggml_vec_index_error {
    GGML_VEC_INDEX_OK            = 0,
    GGML_VEC_INDEX_E_INVALID_ARG = -1,
    GGML_VEC_INDEX_E_DUPLICATE   = -2,
    GGML_VEC_INDEX_E_IO          = -3,
    GGML_VEC_INDEX_E_OOM         = -4,
    GGML_VEC_INDEX_E_INTERNAL    = -5,
    GGML_VEC_INDEX_E_NOT_FOUND   = -6,
    GGML_VEC_INDEX_E_NOT_DURABLE = -10,
};

// Lifecycle.
//
// `dim` must be > 0. `bit_width` must be 32 in this foundation
// implementation. Returns NULL on bad args.
GGML_API ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width);

GGML_API void ggml_vec_index_free(ggml_vec_index_t * idx);

// Mutation.
//
// Adds `n` vectors of length `dim` each (row-major, contiguous in `vectors`),
// associating each with the corresponding `ids[i]` (caller-owned external id).
// Returns 0 on success. Returns GGML_VEC_INDEX_E_DUPLICATE if any id already
// exists in the index; in that case the index is unchanged (atomic add).
// `n == 0` is a no-op and does not require non-NULL `vectors` or `ids`.
// All vector components must be finite. UINT64_MAX is reserved for search
// result padding and is not a valid id.
GGML_API int ggml_vec_index_add(ggml_vec_index_t * idx, const float * vectors, int n, const uint64_t * ids);

// Removes the entry for `id` via swap-with-last (slot indices are NOT stable
// across removes; external ids ARE). Returns GGML_VEC_INDEX_OK if removed,
// GGML_VEC_INDEX_E_NOT_FOUND if not present, or
// GGML_VEC_INDEX_E_INVALID_ARG if `id` is the reserved UINT64_MAX value.
GGML_API int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id);

// Returns 1 if the id is in the index, 0 otherwise. Read-only.
GGML_API int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id);

// No-op for the POC. Placeholder for future cache warming / codebook
// resolution after a bulk add. Reserved as a mutating op (warm-up may
// materialize derived state inside the index).
GGML_API void ggml_vec_index_prepare(ggml_vec_index_t * idx);

// Top-k search. `queries` is `n_q * dim` row-major. `out_scores` and
// `out_ids` are caller-allocated buffers of size `n_q * k`. Each row is
// sorted descending by score (higher = closer / more similar). If the index
// holds fewer than k entries, the remaining slots in each row are filled
// with -FLT_MAX scores and UINT64_MAX ids. Only the UINT64_MAX id identifies
// padding because a real clamped score may also equal -FLT_MAX. Read-only
// against the index (does not mutate state).
//
// Score semantics: scalar dot product accumulated in double and clamped to the
// finite f32 range. Callers that want cosine similarity must L2-normalize their
// vectors before insert AND before query; the index does NOT normalize
// internally. All query components must be finite. Ties are not ordered
// deterministically. `n_q == 0` is a no-op and does not require non-NULL
// buffers. If the function returns an error, the output buffers are
// unspecified and must not be consumed.
GGML_API int ggml_vec_index_search(const ggml_vec_index_t * idx,
                                   const float *            queries,
                                   int                      n_q,
                                   int                      k,
                                   float *                  out_scores,
                                   uint64_t *               out_ids);

// Persistence. Format is .tvim version 1; see bottom of this header. V1
// snapshots are limited to 4 GiB serialized size; larger indexes are rejected
// by write and load. Write replaces `path` through a same-directory temporary
// file so a pre-rename failure does not truncate an existing snapshot. Returns
// GGML_VEC_INDEX_E_NOT_DURABLE if the replacement is visible but syncing its
// parent directory fails.
GGML_API int ggml_vec_index_write(const ggml_vec_index_t * idx, const char * path);

// Returns NULL on I/O, format, or validation failure. Detailed load errors are
// not surfaced by this foundation API.
GGML_API ggml_vec_index_t * ggml_vec_index_load(const char * path);

// Stats.
GGML_API int ggml_vec_index_len(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_dim(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_bit_width(const ggml_vec_index_t * idx);

// File format (.tvim version 1, all little-endian):
//
//   offset  size   field
//   ------  -----  -------------------------------------------------------
//   0       4      magic = "TVPI" (bytes 0x54, 0x56, 0x50, 0x49)
//   4       1      version = 1
//   5       1      bit_width
//   6       2      reserved (zero)
//   8       4      dim (uint32)
//   12      4      n_vectors (uint32)
//   16      N*D*4  vectors (float32, row-major)
//   ...     N*8    ids (uint64)
//
// Where N = n_vectors and D = dim. There is no checksum in v1; future
// versions may append one without breaking back-compat (version-gated).

#ifdef __cplusplus
}
#endif
