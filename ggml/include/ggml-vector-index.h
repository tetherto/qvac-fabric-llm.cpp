#pragma once
//
// ggml-vector-index: vector-index C API.
//
// This public C API supports full f32 storage (`bit_width=32`), q8 storage
// (`bit_width=8`), and packed q4
// storage (`bit_width=4`) with CPU search directly against quantized codes.
// q8 and q4 use NEON when available. Supported x86 CMake builds compile AVX2
// kernels separately and runtime-dispatch them from a non-AVX2 baseline.
//
// Threading: read-only APIs on the same handle can run concurrently. Mutations,
// persistence writes, compaction, and IVF builds are serialized with reads and
// with each other. The caller must still keep the handle alive for the duration
// of every API call. Prepared filter handles must also remain alive for the
// full duration of any `ggml_vec_index_search_prepared_filtered` call using
// them; do not free a filter concurrently with a search that uses it.
//
// Endianness: persistence format is fixed little-endian. Loaders decode
// little-endian fields into host values.

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
#            define GGML_API __attribute__ ((visibility ("default"))) extern
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

// Prepared filtered-search handle. Valid only while the source index remains
// alive and at the generation it was created from; any successful add/remove
// invalidates existing filters.
struct ggml_vec_index_filter;
typedef struct ggml_vec_index_filter ggml_vec_index_filter_t;

// Error codes returned from int-valued APIs. 0 = OK. Negative = failure.
enum ggml_vec_index_error {
    GGML_VEC_INDEX_OK                = 0,
    GGML_VEC_INDEX_E_INVALID_ARG     = -1,
    GGML_VEC_INDEX_E_DUPLICATE       = -2,
    GGML_VEC_INDEX_E_IO              = -3,
    GGML_VEC_INDEX_E_OOM             = -4,
    GGML_VEC_INDEX_E_INTERNAL        = -5,
    GGML_VEC_INDEX_E_NOT_FOUND       = -6,
    GGML_VEC_INDEX_E_BAD_MAGIC       = -7,
    GGML_VEC_INDEX_E_BAD_VERSION     = -8,
    GGML_VEC_INDEX_E_PARTIAL_COMPACT = -9,
    GGML_VEC_INDEX_E_NOT_DURABLE     = -10,
};

// Returns a stable string for ggml_vec_index_error values.
GGML_API const char * ggml_vec_index_error_to_string(int error);

// Lifecycle.
//
// `dim` must be > 0. `bit_width` must be 4, 8, or 32. `bit_width=4` and
// `bit_width=8` store per-vector symmetric quantized codes with one f32 scale
// per vector. `bit_width=32` stores full f32 vectors. Returns NULL on bad args.
GGML_API ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width);

GGML_API void ggml_vec_index_free(ggml_vec_index_t * idx);

// Mutation.
//
// Adds `n` vectors of length `dim` each (row-major, contiguous in `vectors`),
// associating each with the corresponding `ids[i]` (caller-owned external id).
// Returns 0 on success. Returns GGML_VEC_INDEX_E_DUPLICATE if any id already
// exists in the index; in that case the index is unchanged (atomic add).
// All vector components must be finite. UINT64_MAX is reserved for search
// result padding and is not a valid id. `n == 0` is a no-op on handles that
// accept plain mutations. Live index length and total allocated slots are capped
// at INT_MAX; compact after bulk removes to reclaim tombstoned slots.
GGML_API int ggml_vec_index_add(ggml_vec_index_t * idx, const float * vectors, int n, const uint64_t * ids);

// Removes the entry for `id` by marking its internal slot deleted. Physical
// storage is compacted only when writing a snapshot. Returns GGML_VEC_INDEX_OK
// if removed, GGML_VEC_INDEX_E_NOT_FOUND if not present, negative on error.
GGML_API int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id);

// Physically removes deleted slots from in-memory storage. This does not write
// to disk. If any slots are removed, prepared filters and IVF state are
// invalidated. Returns 0 on success, negative on error.
GGML_API int ggml_vec_index_compact(ggml_vec_index_t * idx);

// Reserved for future incremental persistence. Currently returns
// GGML_VEC_INDEX_E_INVALID_ARG without mutating `idx` or writing `delta_path`.
GGML_API int ggml_vec_index_add_logged(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    const char       * delta_path);

// Reserved for future incremental persistence. Currently returns
// GGML_VEC_INDEX_E_INVALID_ARG without mutating `idx` or writing `delta_path`.
GGML_API int ggml_vec_index_remove_logged(
    ggml_vec_index_t * idx,
    uint64_t           id,
    const char       * delta_path);

// Returns 1 if the id is in the index, 0 otherwise. NULL handles return 0.
// Read-only.
GGML_API int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id);

// Compatibility no-op. Existing callers do not need to call this; use
// `ggml_vec_index_build_ivf` when ANN search preparation is needed.
GGML_API void ggml_vec_index_prepare(ggml_vec_index_t * idx);

// Builds an in-memory IVF-flat approximate nearest-neighbor structure for the
// same dot-product score used by exact search. IVF assigns vectors and queries
// to arithmetic centroids with dot-product scoring; low nprobe values are a
// recall/latency heuristic, not a metric-correct guarantee. This is not
// persisted in .tvim files; call again after loading if ANN search is needed.
// This is allowed on mmap-loaded handles because it only builds heap-owned
// search state. Successful add/remove calls invalidate the IVF structure.
// `n_lists` is capped to the current index length. `n_iter` controls centroid
// refinement; 0 uses deterministic initial centroids only.
GGML_API int ggml_vec_index_build_ivf(
    ggml_vec_index_t * idx,
    int                n_lists,
    int                n_iter);

// Top-k search. `queries` is `n_q * dim` row-major. `out_scores` and
// `out_ids` are caller-allocated buffers of size `n_q * k`. Each row is
// sorted descending by score (higher = closer / more similar), with equal
// scores ordered by ascending id. If the index holds fewer than k entries, the
// remaining slots in each row are filled with UINT64_MAX ids; callers must use
// out_ids[i] == UINT64_MAX to identify padding. Padded score slots are filled
// with -FLT_MAX for compatibility, but that value can also be a legitimate
// finite dot product. Read-only against the index (does not mutate state).
// Exact search scans all live entries; use filtered or IVF search to reduce
// the candidate set.
//
// Score semantics: dot product. For f32 storage this is a full-precision dot
// product. For q4/q8 storage, the query remains f32 and the dot product is
// computed against dequantized indexed components:
// `query[i] * (q_code * per_vector_scale)`, without expanding the stored
// matrix back to f32. Callers that want cosine similarity must L2-normalize
// their vectors before insert AND before query; the index does NOT normalize
// internally. All query components must be finite. SIMD and scalar reduction
// order can produce small score differences across CPU architectures.
GGML_API int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids);

// Filtered top-k search. Only entries whose ids appear in `allowed_ids` are
// considered. Missing ids are ignored; duplicate filter ids are treated once.
// `allowed_ids` may be NULL only when `n_allowed == 0`, which produces only
// sentinel results. The same filter is applied to every query row.
GGML_API int ggml_vec_index_search_filtered(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    const uint64_t         * allowed_ids,
    int                      n_allowed,
    float                  * out_scores,
    uint64_t               * out_ids);

// Prepared filtered search. Creating a filter maps, sorts, and deduplicates
// `allowed_ids` once, so callers can reuse it for repeated searches over the
// same allowlist. The source index must outlive every filter created from it.
// Stale filters return GGML_VEC_INDEX_E_INVALID_ARG.
GGML_API ggml_vec_index_filter_t * ggml_vec_index_filter_create(
    const ggml_vec_index_t * idx,
    const uint64_t         * allowed_ids,
    int                      n_allowed);

GGML_API void ggml_vec_index_filter_free(ggml_vec_index_filter_t * filter);

GGML_API int ggml_vec_index_search_prepared_filtered(
    const ggml_vec_index_t        * idx,
    const ggml_vec_index_filter_t * filter,
    const float                   * queries,
    int                             n_q,
    int                             k,
    float                         * out_scores,
    uint64_t                      * out_ids);

// IVF-flat ANN top-k search. `ggml_vec_index_build_ivf` must have been called
// after the most recent mutation. `nprobe` controls how many centroid lists are
// searched; higher values improve recall and lower the latency win. `nprobe`
// must be >= 1. If nprobe is greater than the number of built lists, all lists
// are searched, so candidate coverage matches exact search.
GGML_API int ggml_vec_index_search_ivf(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    int                      nprobe,
    float                  * out_scores,
    uint64_t               * out_ids);

// Persistence. The current implementation supports f32 (`bit_width=32`)
// legacy .tvim version 1 snapshots only; q4/q8 snapshots return
// GGML_VEC_INDEX_E_INVALID_ARG. V1 snapshots are limited to 4 GiB serialized
// size; larger states are rejected by write and load. See bottom of this
// header.
GGML_API int ggml_vec_index_write(
    ggml_vec_index_t * idx,
    const char       * path);

// Loads f32 legacy .tvim version 1 snapshots.
// Returns 0 on success and stores the loaded handle in `out`.
GGML_API int ggml_vec_index_load_ex(
    const char         * path,
    ggml_vec_index_t  ** out);

// Returns NULL on failure.
GGML_API ggml_vec_index_t * ggml_vec_index_load(const char * path);

// Reserved for future mmap snapshot loading. Currently returns
// GGML_VEC_INDEX_E_BAD_VERSION and stores NULL in `out`.
GGML_API int ggml_vec_index_load_mmap_ex(
    const char         * path,
    ggml_vec_index_t  ** out);

// Returns NULL; mmap loading is not implemented yet.
GGML_API ggml_vec_index_t * ggml_vec_index_load_mmap(const char * path);

// Reserved for future .tvid delta replay. Currently returns
// GGML_VEC_INDEX_E_INVALID_ARG and stores NULL in `out`.
GGML_API int ggml_vec_index_load_with_delta_ex(
    const char         * snapshot_path,
    const char         * delta_path,
    ggml_vec_index_t  ** out);

GGML_API ggml_vec_index_t * ggml_vec_index_load_with_delta(
    const char * snapshot_path,
    const char * delta_path);

// Reserved for future delta compaction. Currently returns
// GGML_VEC_INDEX_E_INVALID_ARG.
GGML_API int ggml_vec_index_compact_delta(
    ggml_vec_index_t * idx,
    const char       * snapshot_path,
    const char       * delta_path);

// Stats. NULL handles return 0.
GGML_API int ggml_vec_index_len(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_dim(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_bit_width(const ggml_vec_index_t * idx);

// File format (.tvim legacy version 1, all little-endian):
//
//   offset  size   field
//   ------  -----  -------------------------------------------------------
//   0       4      magic = "TVPI" (bytes 0x54, 0x56, 0x50, 0x49)
//   4       1      version = 1
//   5       1      bit_width = 32
//   6       1      reserved (zero)
//   7       1      reserved (zero)
//   8       4      dim (uint32)
//   12      4      n_vectors (uint32)
//   16      ...    vectors: N*D float32 values, row-major
//   ...     N*8    ids (uint64)
//
// Where N = n_vectors and D = dim. Readers reject unknown versions, non-f32
// bit widths, nonzero reserved bytes, duplicate or reserved ids, non-finite
// vector components, truncated payloads, and trailing bytes.

#ifdef __cplusplus
}
#endif
