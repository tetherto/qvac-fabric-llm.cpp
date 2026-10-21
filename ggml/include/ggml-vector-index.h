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

#include <stddef.h>
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

// This library is built with CXX_VISIBILITY_PRESET hidden, unlike ggml / ggml-base.
// In a static build GGML_API carries no visibility attribute, so the preset would
// leave the public API hidden as well: on Mach-O those symbols become private
// extern, and a consumer relinking the archive into a shared module cannot export
// them even when its -exported_symbols_list names them. Annotate the API directly
// so it stays exportable while the implementation internals remain hidden.
#ifndef GGML_VEC_INDEX_API
#    if defined(_WIN32) && !defined(__MINGW32__)
#        define GGML_VEC_INDEX_API GGML_API
#    else
#        define GGML_VEC_INDEX_API __attribute__ ((visibility ("default"))) extern
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
GGML_VEC_INDEX_API const char * ggml_vec_index_error_to_string(int error);

// Lifecycle.
//
// `dim` must be > 0. `bit_width` must be 4, 8, or 32. `bit_width=4` and
// `bit_width=8` store per-vector symmetric quantized codes with one f32 scale
// per vector. `bit_width=32` stores full f32 vectors. Returns NULL on bad args.
GGML_VEC_INDEX_API ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width);

// Creates a TurboQuant index with `bit_width` 2, 3 or 4: a port of upstream
// turbovec 1.1.2 that encodes, snapshots and syncs byte for byte as turbovec
// does. Vectors and queries are rotated with turbovec's deterministic
// block-Hadamard transform, quantized with Lloyd-Max codebooks (optionally
// under a TQ+ calibration, see `ggml_vec_index_turbovec_calibrate`), and
// searched against the rotated query without decoding. Requires a 64-bit
// target and `0 < dim <= 16384 && dim % 8 == 0`. Supports add, search,
// filtered search, IVF, snapshots, in-memory snapshots and incremental sync;
// mmap loading and delta logs are not supported. `ggml_vec_index_prepare` is
// best-effort and does not report allocation status. Returns NULL on bad args.
GGML_VEC_INDEX_API ggml_vec_index_t * ggml_vec_index_create_turbovec(int dim, int bit_width);

// Shorthands for `ggml_vec_index_create_turbovec(dim, 2)` and `(dim, 4)`. The
// TurboVec q4 mode is distinct from the generic `bit_width=4` mode created by
// `ggml_vec_index_create`.
GGML_VEC_INDEX_API ggml_vec_index_t * ggml_vec_index_create_turbovec_q2(int dim);
GGML_VEC_INDEX_API ggml_vec_index_t * ggml_vec_index_create_turbovec_q4(int dim);

// Fits the TurboVec TQ+ per-coordinate calibration from `n` sample vectors
// (row-major, `dim` floats each) and commits it. Every later add is encoded
// under it, and rows already stored are re-encoded from their codes. Pass a
// random, representative sample (around 1000 rows); a sorted or clustered
// sample fits quantiles that hurt recall. Calibrate before adding when you
// can: re-encoding stored rows is a second quantization. May be called again
// to refit. Requires `n >= 2`. Returns GGML_VEC_INDEX_E_INVALID_ARG for a
// non-TurboVec or legacy (v3-loaded) index, invalid values, or a sample whose
// fit is the identity. Invalidates prepared filters and IVF state.
GGML_VEC_INDEX_API int ggml_vec_index_turbovec_calibrate(
    ggml_vec_index_t * idx,
    const float * sample,
    int n);

// Staged search for a TurboVec 1.x index, on by default. From 32,768
// vectors at 2 or 4 bits and dimensions that are multiples of 32, on hosts
// where upstream turbovec runs it (aarch64, where 4 bits needs the
// dot-product extension, and x86 with AVX-512 F, BW, VBMI and VNNI; not
// under TURBOVEC_2BIT_VM8=1), a search scans the sign planes for a
// shortlist, ranks it with the lower planes and rescores the best few
// exactly: returned scores are the full scan's, but the ids can differ from
// a full scan's (on random data 77-94% of a top 10 overlaps it, as with
// upstream). Pass 0 to always scan the whole index; that also drops the
// planes caches. Legacy and 3-bit indexes accept the call but never search
// in stages.
GGML_VEC_INDEX_API int ggml_vec_index_turbovec_set_staged(ggml_vec_index_t * idx, int enabled);

// Returns 1 when a TurboVec index holds a committed TQ+ calibration.
GGML_VEC_INDEX_API int ggml_vec_index_turbovec_is_calibrated(const ggml_vec_index_t * idx);

GGML_VEC_INDEX_API void ggml_vec_index_free(ggml_vec_index_t * idx);

// Worker threads search, add and TurboVec calibration may use, including
// the calling thread; 1 (the default) runs every call on the calling thread.
// Queries are split across threads; when a search has fewer queries or
// batches than threads, each splits the blocks of a TurboVec index of
// 32,768 or more vectors instead, staged searches included. Results do not
// depend on the thread count. `n_threads` must be in 1..256.
GGML_VEC_INDEX_API int ggml_vec_index_set_threads(ggml_vec_index_t * idx, int n_threads);
GGML_VEC_INDEX_API int ggml_vec_index_threads(const ggml_vec_index_t * idx);

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
GGML_VEC_INDEX_API int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids);

// Removes the entry for `id` by marking its internal slot deleted. Physical
// storage is compacted only when writing a snapshot. A TurboVec 1.x index
// instead removes at once, as upstream turbovec does: the last row moves into
// the freed slot, which changes the order `ggml_vec_index_ids` reports.
// Returns GGML_VEC_INDEX_OK if removed, GGML_VEC_INDEX_E_NOT_FOUND if not
// present, negative on error.
GGML_VEC_INDEX_API int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id);

// Physically removes deleted slots from in-memory storage. This does not write
// to disk. If any slots are removed, prepared filters and IVF state are
// invalidated. Returns 0 on success, negative on error.
GGML_VEC_INDEX_API int ggml_vec_index_compact(ggml_vec_index_t * idx);

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
GGML_VEC_INDEX_API int ggml_vec_index_add_logged(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    const char       * delta_path);

// Removes `id` and appends the mutation to the handle's v4 delta log. Uses the
// same snapshot-lineage, path-binding, and durability contract as
// `ggml_vec_index_add_logged`. Same return convention as
// `ggml_vec_index_remove`: GGML_VEC_INDEX_OK if removed,
// GGML_VEC_INDEX_E_NOT_FOUND if not present, negative on error.
GGML_VEC_INDEX_API int ggml_vec_index_remove_logged(
    ggml_vec_index_t * idx,
    uint64_t           id,
    const char       * delta_path);

// Returns 1 if the id is in the index, 0 otherwise. NULL handles return 0.
// Read-only.
GGML_VEC_INDEX_API int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id);

// Optional cache warmup. TurboVec q2/q4 precompute rotation and codebook state;
// other storage modes ignore this call. Existing callers do not need to call
// this; use `ggml_vec_index_build_ivf` when ANN search preparation is needed.
GGML_VEC_INDEX_API void ggml_vec_index_prepare(ggml_vec_index_t * idx);

// Builds an in-memory IVF-flat approximate nearest-neighbor structure for the
// same dot-product score used by exact search. IVF assigns vectors and queries
// to arithmetic centroids with dot-product scoring; low nprobe values are a
// recall/latency heuristic, not a metric-correct guarantee. This is not
// persisted in .tvim files; call again after loading if ANN search is needed.
// This is allowed on mmap-loaded handles because it only builds heap-owned
// search state. Successful add/remove calls invalidate the IVF structure.
// `n_lists` is capped to the current index length. `n_iter` controls centroid
// refinement; 0 uses deterministic initial centroids only.
GGML_VEC_INDEX_API int ggml_vec_index_build_ivf(
    ggml_vec_index_t * idx,
    int                n_lists,
    int                n_iter);

// Top-k search. `queries` is `n_q * dim` row-major. `out_scores` and
// `out_ids` are caller-allocated buffers of size `n_q * k`. Each row is
// sorted descending by score (higher = closer / more similar), with equal
// scores ordered by ascending id; a TurboVec 1.x index orders them by row, as
// upstream turbovec does (insertion order, as reshuffled by removals). `k`
// must be at least 1. If the index holds fewer than k entries, the
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
GGML_VEC_INDEX_API int ggml_vec_index_search(
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
GGML_VEC_INDEX_API int ggml_vec_index_search_filtered(
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
GGML_VEC_INDEX_API ggml_vec_index_filter_t * ggml_vec_index_filter_create(
    const ggml_vec_index_t * idx,
    const uint64_t         * allowed_ids,
    int                      n_allowed);

GGML_VEC_INDEX_API void ggml_vec_index_filter_free(ggml_vec_index_filter_t * filter);

GGML_VEC_INDEX_API int ggml_vec_index_search_prepared_filtered(
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
GGML_VEC_INDEX_API int ggml_vec_index_search_ivf(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    int                      nprobe,
    float                  * out_scores,
    uint64_t               * out_ids);

// Persistence. Writers use .tvim v2 for f32/q4/q8. TurboVec indexes write
// upstream turbovec's v8 container, readable by turbovec 1.1 and later;
// indexes loaded from a v3 snapshot keep writing v3. A v8 snapshot reserves
// two fixed-size commit headers, mostly zero, which the writer leaves as a
// file hole where the filesystem supports one.
// Delta-bound handles must use `ggml_vec_index_compact_delta`; ordinary
// snapshot writes return GGML_VEC_INDEX_E_INVALID_ARG.
// Returns GGML_VEC_INDEX_E_NOT_DURABLE when the file was atomically replaced
// but the parent-directory sync failed. Legacy v1 snapshots are limited to 4
// GiB serialized size; larger v1 states are rejected by load.
GGML_VEC_INDEX_API int ggml_vec_index_write(
    ggml_vec_index_t * idx,
    const char       * path);

// How durable `ggml_vec_index_write_with_durability` makes a snapshot before
// it returns (upstream turbovec's `Durability`).
enum ggml_vec_index_durability {
    // Temp file, fsync, atomic rename, parent-directory fsync: what
    // `ggml_vec_index_write` does.
    GGML_VEC_INDEX_DURABILITY_DURABLE = 0,
    // The same temp file and atomic rename without either fsync. The
    // destination never holds a torn snapshot and the previous file survives
    // a process crash, but a power loss shortly after the call returns may
    // lose the new file. A fast write does not allow starting a delta log.
    GGML_VEC_INDEX_DURABILITY_FAST = 1,
};

// `ggml_vec_index_write` with an explicit durability level, for every index
// kind. Syncs are always durable.
GGML_VEC_INDEX_API int ggml_vec_index_write_with_durability(
    ggml_vec_index_t              * idx,
    const char                    * path,
    enum ggml_vec_index_durability  durability);

// In-memory snapshots of a TurboVec index (not a v3-loaded one): exactly the
// bytes `ggml_vec_index_write` puts in the file, padding included. The
// buffer is allocated by the library; free it with `ggml_vec_index_bytes_free`.
// Other storage modes return GGML_VEC_INDEX_E_INVALID_ARG.
GGML_VEC_INDEX_API int ggml_vec_index_to_bytes(
    const ggml_vec_index_t * idx,
    uint8_t               ** out_bytes,
    size_t                 * out_size);

GGML_VEC_INDEX_API void ggml_vec_index_bytes_free(uint8_t * bytes);

// Exact byte size `ggml_vec_index_to_bytes` returns, without serializing.
// Returns 0 for indexes it does not support.
GGML_VEC_INDEX_API size_t ggml_vec_index_serialized_size(const ggml_vec_index_t * idx);

// Loads a TurboVec v8 snapshot from memory with the same validation as
// `ggml_vec_index_load_ex`.
GGML_VEC_INDEX_API int ggml_vec_index_from_bytes(
    const uint8_t      * bytes,
    size_t               size,
    ggml_vec_index_t  ** out);

// Incremental save of a TurboVec index (not a v3-loaded one) to `path`
// (upstream turbovec `sync`). The first sync to a path, and any sync after a
// calibrate, writes the whole file and claims it; later syncs write only the
// blocks and commit header that changed, durable after one fsync. A crash
// at any byte leaves the previous commit loadable. Loading a synced file
// binds the loaded index to it, so syncs continue incrementally. Returns
// GGML_VEC_INDEX_E_IO when another writer has advanced or replaced the file
// (load it to adopt its state), and GGML_VEC_INDEX_E_NOT_DURABLE as
// `ggml_vec_index_write` does. One writer per path.
GGML_VEC_INDEX_API int ggml_vec_index_sync(
    ggml_vec_index_t * idx,
    const char       * path);

// Loads v2/v3 and v7/v8 files and migrates v1 f32 snapshots. Legacy bit_width=8 snapshots
// are quantized to q8; all other legacy bit widths migrate to f32/32-bit.
// Returns 0 on success and stores the loaded handle in `out`.
GGML_VEC_INDEX_API int ggml_vec_index_load_ex(
    const char         * path,
    ggml_vec_index_t  ** out);

// Returns NULL on failure.
GGML_VEC_INDEX_API ggml_vec_index_t * ggml_vec_index_load(const char * path);

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
GGML_VEC_INDEX_API int ggml_vec_index_load_mmap_ex(
    const char         * path,
    ggml_vec_index_t  ** out);

// Returns NULL on failure or unsupported file format.
GGML_VEC_INDEX_API ggml_vec_index_t * ggml_vec_index_load_mmap(const char * path);

// Loads a full .tvim snapshot and replays an append-only delta log. Missing
// delta logs are treated as empty. The returned handle is bound to that delta
// log: plain add, remove, compact, and snapshot write operations return
// GGML_VEC_INDEX_E_INVALID_ARG.
// Returns 0 on success and stores the loaded handle in `out`.
GGML_VEC_INDEX_API int ggml_vec_index_load_with_delta_ex(
    const char         * snapshot_path,
    const char         * delta_path,
    ggml_vec_index_t  ** out);

GGML_VEC_INDEX_API ggml_vec_index_t * ggml_vec_index_load_with_delta(
    const char * snapshot_path,
    const char * delta_path);

// Writes a fresh snapshot and resets the .tvid delta log. Returns
// GGML_VEC_INDEX_E_PARTIAL_COMPACT if the snapshot was replaced but its
// durability could not be confirmed, or if resetting the delta log failed.
GGML_VEC_INDEX_API int ggml_vec_index_compact_delta(
    ggml_vec_index_t * idx,
    const char       * snapshot_path,
    const char       * delta_path);

// Stats. NULL handles return 0.
GGML_VEC_INDEX_API int ggml_vec_index_len(const ggml_vec_index_t * idx);
GGML_VEC_INDEX_API int ggml_vec_index_dim(const ggml_vec_index_t * idx);
GGML_VEC_INDEX_API int ggml_vec_index_bit_width(const ggml_vec_index_t * idx);

// Storage mode of an index. `bit_width` alone cannot tell generic q4 from
// TurboVec q4.
enum ggml_vec_index_storage {
    GGML_VEC_INDEX_STORAGE_NONE        = 0,
    GGML_VEC_INDEX_STORAGE_F32         = 1,
    GGML_VEC_INDEX_STORAGE_Q8          = 2,
    GGML_VEC_INDEX_STORAGE_Q4          = 3,
    GGML_VEC_INDEX_STORAGE_TURBOVEC_Q2 = 4,
    GGML_VEC_INDEX_STORAGE_TURBOVEC_Q3 = 5,
    GGML_VEC_INDEX_STORAGE_TURBOVEC_Q4 = 6,
};

// Returns a `ggml_vec_index_storage` value; GGML_VEC_INDEX_STORAGE_NONE for
// a NULL handle.
GGML_VEC_INDEX_API int ggml_vec_index_storage(const ggml_vec_index_t * idx);

// Returns 1 for a TurboVec index loaded from a v3 snapshot, which keeps the
// turbovec 0.9 encoding (see `ggml_vec_index_create_turbovec`).
GGML_VEC_INDEX_API int ggml_vec_index_turbovec_is_legacy(const ggml_vec_index_t * idx);

// Copies the live ids into `out_ids`, up to `capacity` of them, and stores
// the live count in `out_count`. Pass `capacity` 0 to query the count. Ids
// come in a stable order that changes only with mutations.
GGML_VEC_INDEX_API int ggml_vec_index_ids(
    const ggml_vec_index_t * idx,
    uint64_t               * out_ids,
    size_t                   capacity,
    size_t                 * out_count);

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
