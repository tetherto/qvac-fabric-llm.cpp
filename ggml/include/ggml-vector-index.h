#pragma once
//
// ggml-vector-index: vector-index C API.
//
// This public C API supports full f32 storage (`bit_width=32`), q8 storage
// (`bit_width=8`), packed q4 storage (`bit_width=4`), and distinct TurboVec
// q2/q4 modes with CPU search directly against quantized codes.
// q8 and q4 use NEON when available. Supported x86 CMake builds compile AVX2
// kernels separately and runtime-dispatch them from a non-AVX2 baseline.
//
// Threading: read-only APIs on the same handle can run concurrently. Mutations,
// persistence writes, compaction, and IVF builds are serialized with reads and
// with each other. Writer fairness and bounded mutation latency are not
// guaranteed while concurrent readers continue to arrive. The caller must
// still keep the handle alive for the duration of every API call. Prepared
// filter handles must also remain alive for the full duration of any
// `ggml_vec_index_search_prepared_filtered` call using them; do not free a
// filter concurrently with a search that uses it.
//
// Endianness: persistence format is fixed little-endian. Stream loaders decode
// fields into host values; mmap loading requires a little-endian host.

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

// Creates a TurboQuant q2 vector index. This is distinct from the generic
// q4/q8 modes created by `ggml_vec_index_create`: vectors are normalized,
// rotated, quantized with Lloyd-Max q2 codebooks, and searched against rotated
// queries. This implementation requires a 64-bit target,
// `0 < dim <= 1024 && dim % 8 == 0`, and supports add/search/filter/IVF plus
// regular snapshot write/load. TurboVec materializes a dense `dim x dim`
// rotation matrix on first use. Snapshots from earlier releases with larger
// dimensions can be loaded and rewritten, but add/search/IVF are unsupported.
// `ggml_vec_index_prepare` is best-effort and does not report allocation status.
// mmap loading and logged mutations are reserved for later format work.
GGML_API ggml_vec_index_t * ggml_vec_index_create_turbovec_q2(int dim);

// Creates a TurboQuant q4 vector index. This is distinct from the generic
// `bit_width=4` mode created by `ggml_vec_index_create`: vectors are normalized,
// rotated, quantized with Lloyd-Max q4 codebooks, and searched against rotated
// queries. This implementation requires a 64-bit target,
// `0 < dim <= 1024 && dim % 8 == 0`, and supports add/search/filter/IVF plus
// regular snapshot write/load. TurboVec materializes a dense `dim x dim`
// rotation matrix on first use. Snapshots from earlier releases with larger
// dimensions can be loaded and rewritten, but add/search/IVF are unsupported.
// `ggml_vec_index_prepare` is best-effort and does not report allocation status.
// mmap loading and logged mutations are reserved for later format work.
GGML_API ggml_vec_index_t * ggml_vec_index_create_turbovec_q4(int dim);

GGML_API void ggml_vec_index_free(ggml_vec_index_t * idx);

// Mutation.
//
// Adds `n` vectors of length `dim` each (row-major, contiguous in `vectors`),
// associating each with the corresponding `ids[i]` (caller-owned external id).
// Returns 0 on success. Returns GGML_VEC_INDEX_E_DUPLICATE if any id already
// exists in the index; in that case the index is unchanged (atomic add).
// All vector components must be finite. TurboVec q2/q4 also require
// `abs(component) < 1e16`. UINT64_MAX is reserved for search result padding and
// is not a valid id. `n == 0` is a no-op on handles that accept plain
// mutations. Live index length and total allocated slots are capped at INT_MAX;
// compact after bulk removes to reclaim tombstoned slots.
GGML_API int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids);

// Removes the entry for `id` by marking its internal slot deleted. Physical
// storage is compacted only when writing a snapshot. Returns GGML_VEC_INDEX_OK
// if removed, GGML_VEC_INDEX_E_NOT_FOUND if not present, negative on error.
GGML_API int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id);

// Physically removes deleted slots from in-memory storage. This does not write
// to disk. If any slots are removed, prepared filters and IVF state are
// invalidated. Returns 0 on success, negative on error.
GGML_API int ggml_vec_index_compact(ggml_vec_index_t * idx);

// Logged mutations for incremental persistence. These update `idx` and append
// a durable v4 delta record to `delta_path`. Replay the log on top of a full
// .tvim snapshot with `ggml_vec_index_load_with_delta`.
//
// A new log can only start from a handle whose current state was loaded from or
// successfully written to a snapshot. Write a new snapshot after plain mutations.
// Delta logs are state-bound and single-writer per snapshot lineage. Use one
// evolving writer handle for a given {snapshot, delta_path} pair. Once bound,
// logged mutations and compaction with a different delta path return
// GGML_VEC_INDEX_E_INVALID_ARG. Equivalent hardlink aliases identify the same
// log. If another handle or process appends to the same log, stale writers are
// caught up when possible; otherwise they are rejected and must reload with
// `ggml_vec_index_load_with_delta` before appending again.
// A successful catch-up remains applied even if the requested mutation then
// returns an error such as GGML_VEC_INDEX_E_DUPLICATE or E_NOT_FOUND.
// Cross-process protection relies on cooperative OS file locks. Store delta
// logs on local filesystems and do not modify `.tvid` files outside this API.
// If an append error occurs after a complete replayable record is observed, the
// mutation may remain applied and the handle may require reload before further
// writes.
// Once a handle participates in delta logging, use logged mutations for
// content changes and compact_delta for snapshots; plain add/remove/compact/write
// return GGML_VEC_INDEX_E_INVALID_ARG.
// New logged mutations require v4 logs; compact the snapshot+delta pair first
// when carrying a legacy log.
GGML_API int ggml_vec_index_add_logged(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    const char       * delta_path);

// Removes `id` and appends the mutation to the handle's v4 delta log. Uses the
// same snapshot-lineage, path-binding, and durability contract as
// `ggml_vec_index_add_logged`.
GGML_API int ggml_vec_index_remove_logged(
    ggml_vec_index_t * idx,
    uint64_t           id,
    const char       * delta_path);

// Returns 1 if the id is in the index, 0 otherwise. NULL handles return 0.
// Read-only.
GGML_API int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id);

// Optional cache warmup. TurboVec q2/q4 precompute rotation and codebook state;
// other storage modes ignore this call. Existing callers do not need to call
// this; use `ggml_vec_index_build_ivf` when ANN search preparation is needed.
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
// internally. All query components must be finite; TurboVec q2/q4 also require
// `abs(component) < 1e16`. SIMD and scalar reduction order can produce small
// score differences across CPU architectures.
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

// Persistence. Writers use .tvim v2 for f32/q4/q8 and v3 for TurboVec.
// Delta-bound handles must use `ggml_vec_index_compact_delta`; ordinary
// snapshot writes return GGML_VEC_INDEX_E_INVALID_ARG.
// Returns GGML_VEC_INDEX_E_NOT_DURABLE when the file was atomically replaced
// but the parent-directory sync failed. Legacy v1 snapshots are limited to 4
// GiB serialized size; larger v1 states are rejected by load.
GGML_API int ggml_vec_index_write(
    ggml_vec_index_t * idx,
    const char       * path);

// Loads v2/v3 files and migrates v1 f32 snapshots. Legacy bit_width=8 snapshots
// are quantized to q8; all other legacy bit widths migrate to f32/32-bit.
// Returns 0 on success and stores the loaded handle in `out`.
GGML_API int ggml_vec_index_load_ex(
    const char         * path,
    ggml_vec_index_t  ** out);

// Returns NULL on failure.
GGML_API ggml_vec_index_t * ggml_vec_index_load(const char * path);

// Loads a v2 .tvim snapshot with its vector section memory-mapped read-only.
// Legacy v1 snapshots return GGML_VEC_INDEX_E_BAD_VERSION.
// IDs and quantization scales are copied into memory for lookup and scoring.
// Data-mutating and logged mutation APIs return
// GGML_VEC_INDEX_E_INVALID_ARG on mmap-backed handles.
// Heap-only search preparation such as `ggml_vec_index_build_ivf` is allowed.
// `ggml_vec_index_write` can snapshot mmap-backed handles, but callers must
// write to a different path than the mapped source file and the handle must
// not be delta-bound.
// This loader is snapshot-only: it does not replay .tvid delta logs. Use
// `ggml_vec_index_load_with_delta` when loading a snapshot plus delta log;
// that path materializes the resulting index in memory.
// Requires a little-endian host and, on POSIX, a filesystem that supports
// flock(). Use `ggml_vec_index_load` on other hosts.
// Returns 0 on success and stores the loaded handle in `out`.
GGML_API int ggml_vec_index_load_mmap_ex(
    const char         * path,
    ggml_vec_index_t  ** out);

// Returns NULL on failure or unsupported file format.
GGML_API ggml_vec_index_t * ggml_vec_index_load_mmap(const char * path);

// Loads a full .tvim snapshot and replays an append-only delta log. Missing
// delta logs are treated as empty. The returned handle is bound to that delta
// log: plain add, remove, compact, and snapshot write operations return
// GGML_VEC_INDEX_E_INVALID_ARG.
// Returns 0 on success and stores the loaded handle in `out`.
GGML_API int ggml_vec_index_load_with_delta_ex(
    const char         * snapshot_path,
    const char         * delta_path,
    ggml_vec_index_t  ** out);

GGML_API ggml_vec_index_t * ggml_vec_index_load_with_delta(
    const char * snapshot_path,
    const char * delta_path);

// Writes a fresh snapshot and resets the .tvid delta log. Returns
// GGML_VEC_INDEX_E_PARTIAL_COMPACT if the snapshot was replaced but its
// durability could not be confirmed, or if resetting the delta log failed.
GGML_API int ggml_vec_index_compact_delta(
    ggml_vec_index_t * idx,
    const char       * snapshot_path,
    const char       * delta_path);

// Stats. NULL handles return 0.
GGML_API int ggml_vec_index_len(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_dim(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_bit_width(const ggml_vec_index_t * idx);

// File format (.tvim versions 2 and 3, all little-endian):
// "TQ+" here is the TurboVec per-coordinate calibration scheme, unrelated to
// ggml tensor types GGML_TYPE_TQ1_0 and GGML_TYPE_TQ2_0.
//
//   offset  size   field
//   ------  -----  -------------------------------------------------------
//   0       4      magic = "TVPI" (bytes 0x54, 0x56, 0x50, 0x49)
//   4       1      version (2 for f32/q4/q8, 3 for TurboVec)
//   5       1      bit_width (2 for TurboVec q2, 4, 8, or 32)
//   6       1      storage kind (1 = f32, 2 = q8, 3 = q4, 4 = TurboVec q4, 5 = TurboVec q2)
//   7       1      flags (bit 0 = checksum trailer present)
//   8       4      dim (uint32)
//   12      4      n_vectors (uint32)
//   16      4      qparam_type (0 = none, 1 = per-vector f32 scale)
//   20      4      qparam_bytes_per_vector (0 or 4)
//   24      4      bytes_per_component (0 for packed q4/TurboVec, 1 for q8, 4 for f32)
//   28      4      TQ+ calibration bytes (v3; zero in v2)
//   32      ...    qparams:
//                    - f32: empty
//                    - q4/q8: N float32 scales
//                    - TurboVec q2/q4: N float32 score-correction scales
//   ...     ...    TQ+ calibration (v3): D float32 shifts, then D float32 scales
//   ...     ...    vectors:
//                    - f32: N*D float32 values, row-major
//                    - q8:  N*D int8 codes, row-major
//                    - q4:  N*ceil(D/2) packed unsigned nibbles, row-major
//                    - TurboVec q4: N*(D/2) Lloyd-Max codes in bit-plane layout
//                    - TurboVec q2: N*(D/4) Lloyd-Max codes in bit-plane layout
//   ...     N*8    ids (uint64)
//   ...     4      header CRC32C, when flag bit 0 is set
//   ...     4      qparams CRC32C, when flag bit 0 is set
//   ...     4      vectors CRC32C, when flag bit 0 is set
//   ...     4      ids CRC32C, when flag bit 0 is set
//
// Where N = n_vectors and D = dim. q8 uses symmetric per-vector quantization:
// scale = max(abs(v)) / 127, code = round(v / scale) clamped to [-127, 127].
// q4 uses scale = max(abs(v)) / 7, code = round(v / scale) clamped to [-7, 7],
// stored as unsigned nibble `code + 8` (0 is invalid). Zero vectors use
// scale = 1 and all-zero dequantized codes. Each CRC32C covers exactly its
// corresponding serialized section; the qparams CRC also covers v3 calibration,
// the header CRC covers bytes [0, 32), and
// the CRC32C of an empty section is zero.
// Legacy v2 files with flags=0 and no checksum trailer remain readable.
// Writers emit checksummed v2/v3 files. Readers reject unknown versions and
// flag bits; they also accept legacy v1 f32 snapshots. Legacy bit_width=8
// snapshots migrate to q8, while all other legacy widths migrate to f32.
//
// Delta log (.tvid version 4, all little-endian):
//
//   file header:
//     0   4   magic = "TVDL"
//     4   1   version = 4
//     5   1   bit_width (4, 8, or 32)
//     6   2   reserved (zero)
//     8   4   dim (uint32)
//     12  4   reserved (zero)
//     16  32  base snapshot state identity
//
//   record header:
//     0   1   op (1 = add, 2 = remove)
//     1   3   reserved (zero)
//     4   4   n (add count; remove uses 1)
//     8   8   payload bytes
//     16  4   CRC32C over record header bytes [0, 16), state identity, and payload
//     20  4   reserved (zero)
//     24  32  state identity after applying this record
//
//   add payload:
//     - f32: N uint64 ids, then N*D float32 vectors
//     - q8:  N uint64 ids, then N float32 scales, then N*D int8 codes
//     - q4:  N uint64 ids, then N float32 scales, then
//            N*ceil(D/2) packed unsigned nibbles
//   remove payload: one uint64 id
//
// The base snapshot state identity binds the log to the snapshot state it
// extends. It stores the active count plus the three maintained 64-bit state
// hash aggregates, with sums maintained modulo 2^64. Record state identities
// let loading validate each replayed record's post-state
// and recognize a compacted snapshot when a process crashed before replacing
// the old delta log. Readers also accept legacy .tvid v1 logs, whose state
// field is a full-index CRC32C, v2 logs, whose add payloads always store f32
// vectors, and v3 logs, whose state field is a 32-bit token.

#ifdef __cplusplus
}
#endif
