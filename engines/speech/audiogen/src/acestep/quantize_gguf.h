#pragma once

// Requantization core behind the acestep-quantize CLI: reads a BF16 stage
// GGUF, applies the per-tensor policy from quantize_policy.h, and streams the
// quantized copy one tensor at a time so peak memory stays at the largest
// single tensor. Ported from tools/quantize.cpp in acestep.cpp (MIT).
//
// Kept apart from the CLI so the offset/padding planning and the streaming
// writer are exercised by test_acestep_units.cpp against a synthetic GGUF
// round-trip, not only observed on converted files. All resources are held by
// RAII guards so every error path releases the mapping and GGUF contexts.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

#include "gguf_mmap.h"
#include "quantize_policy.h"

namespace tts_cpp::acestep {

struct QuantizeStats {
    int     n_tensors   = 0;
    int     n_quantized = 0;
    int     n_promoted  = 0;
    int64_t bytes_in    = 0;
    int64_t bytes_out   = 0;
};

namespace quantize_detail {

struct GgufFree {
    void operator()(gguf_context * p) const { gguf_free(p); }
};

struct GgmlFree {
    void operator()(ggml_context * p) const { ggml_free(p); }
};

struct FileClose {
    void operator()(FILE * p) const { fclose(p); }
};

struct MappingGuard {
    MappedFile file;

    ~MappingGuard() {
        if (file.data) {
            mapped_file_close(file);
        }
    }
};

inline bool source_type_convertible(enum ggml_type type) {
    return type == GGML_TYPE_BF16 || type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

inline bool tensor_to_f32(const void * src, float * dst, int64_t n, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F32:
            memcpy(dst, src, (size_t) n * sizeof(float));
            return true;
        default:
            return false;
    }
}

struct TensorPlan {
    enum ggml_type target;
    bool           quantize;
    bool           promote;
};

inline std::string read_arch(const gguf_context * inp) {
    const int64_t idx = gguf_find_key(inp, "general.architecture");
    if (idx < 0) {
        return "unknown";
    }
    return gguf_get_val_str(inp, idx);
}

inline int read_block_count(const gguf_context * inp, const std::string & arch) {
    const std::string key = arch + ".block_count";
    const int64_t     idx = gguf_find_key(inp, key.c_str());
    if (idx < 0) {
        return 0;
    }
    return (int) gguf_get_val_u32(inp, idx);
}

inline std::vector<TensorPlan> plan_tensors(gguf_context *       out,
                                            const gguf_context * inp,
                                            ggml_context *       meta,
                                            const std::string &  arch,
                                            const QuantVariant & variant,
                                            int                  n_layers) {
    const int               n_tensors = (int) gguf_get_n_tensors(inp);
    std::vector<TensorPlan> plans((size_t) n_tensors, { GGML_TYPE_COUNT, false, false });

    for (int i = 0; i < n_tensors; i++) {
        const char *  name   = gguf_get_tensor_name(inp, i);
        ggml_tensor * t      = ggml_get_tensor(meta, name);
        const int     n_dims = ggml_n_dims(t);

        gguf_add_tensor(out, t);

        const enum ggml_type target = quant_pick_type(name, n_dims, arch.c_str(), variant, n_layers);

        if (target == GGML_TYPE_COUNT) {
            if (quant_should_promote_f32(n_dims) &&
                (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16)) {
                gguf_set_tensor_type(out, name, GGML_TYPE_F32);
                plans[(size_t) i] = { GGML_TYPE_F32, false, true };
            }
            continue;
        }

        const bool aligned = t->ne[0] % ggml_blck_size(target) == 0;
        if (source_type_convertible(t->type) && aligned) {
            gguf_set_tensor_type(out, name, target);
            plans[(size_t) i] = { target, true, false };
        }
    }

    return plans;
}

// Pads to the next alignment boundary in fixed-size zero chunks, so any
// general.alignment value is handled without an alignment-sized buffer.
inline size_t write_padding(FILE * fout, size_t data_pos, size_t alignment) {
    constexpr size_t ZERO_CHUNK    = 64;
    const uint8_t    zeros[ZERO_CHUNK] = {};

    const size_t pad = (alignment - (data_pos % alignment)) % alignment;
    for (size_t left = pad; left > 0;) {
        const size_t n = left < ZERO_CHUNK ? left : ZERO_CHUNK;
        fwrite(zeros, 1, n, fout);
        left -= n;
    }
    return pad;
}

inline bool stream_tensor_data(FILE *                          fout,
                               const gguf_context *            inp,
                               ggml_context *                  meta,
                               const uint8_t *                 base,
                               size_t                          data_off,
                               size_t                          alignment,
                               const std::vector<TensorPlan> & plans,
                               QuantizeStats &                 stats,
                               std::string &                   error) {
    const int n_tensors = (int) gguf_get_n_tensors(inp);
    size_t    data_pos  = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char *  name     = gguf_get_tensor_name(inp, i);
        ggml_tensor * t        = ggml_get_tensor(meta, name);
        const int64_t nel      = ggml_nelements(t);
        const size_t  src_size = ggml_nbytes(t);
        const void *  src      = base + data_off + gguf_get_tensor_offset(inp, i);

        stats.bytes_in += (int64_t) src_size;
        data_pos += write_padding(fout, data_pos, alignment);

        const TensorPlan & plan = plans[(size_t) i];

        if (plan.promote) {
            std::vector<float> f32((size_t) nel);
            if (!tensor_to_f32(src, f32.data(), nel, t->type)) {
                error = std::string("cannot promote ") + name;
                return false;
            }
            const size_t out_size = (size_t) nel * sizeof(float);
            fwrite(f32.data(), 1, out_size, fout);
            data_pos += out_size;
            stats.bytes_out += (int64_t) out_size;
            stats.n_promoted++;
        } else if (plan.quantize) {
            std::vector<float> f32((size_t) nel);
            if (!tensor_to_f32(src, f32.data(), nel, t->type)) {
                error = std::string("cannot quantize ") + name;
                return false;
            }
            const int64_t        n_per_row = t->ne[0];
            const int64_t        nrows     = nel / n_per_row;
            const size_t         qsize     = ggml_row_size(plan.target, n_per_row) * (size_t) nrows;
            std::vector<uint8_t> qbuf(qsize);
            ggml_quantize_chunk(plan.target, f32.data(), qbuf.data(), 0, nrows, n_per_row, nullptr);
            fwrite(qbuf.data(), 1, qsize, fout);
            data_pos += qsize;
            stats.bytes_out += (int64_t) qsize;
            stats.n_quantized++;
        } else {
            fwrite(src, 1, src_size, fout);
            data_pos += src_size;
            stats.bytes_out += (int64_t) src_size;
        }
    }

    return true;
}

}  // namespace quantize_detail

// Quantizes inp_path into out_path with the given variant. On failure returns
// false, fills `error`, and removes any partial output file.
inline bool quantize_gguf_file(const std::string &  inp_path,
                               const std::string &  out_path,
                               const QuantVariant & variant,
                               QuantizeStats &      stats,
                               std::string &        error) {
    using namespace quantize_detail;

    MappingGuard mapping;
    if (!mapped_file_open(mapping.file, inp_path, "acestep-quantize")) {
        error = "cannot map " + inp_path;
        return false;
    }

    ggml_context *   meta_raw = nullptr;
    gguf_init_params params   = { /*no_alloc=*/true, /*ctx=*/&meta_raw };

    std::unique_ptr<gguf_context, GgufFree> inp(gguf_init_from_file(inp_path.c_str(), params));
    std::unique_ptr<ggml_context, GgmlFree> meta(meta_raw);
    if (!inp) {
        error = "failed to read " + inp_path;
        return false;
    }

    const std::string arch     = read_arch(inp.get());
    const int         n_layers = read_block_count(inp.get(), arch);

    std::unique_ptr<gguf_context, GgufFree> out(gguf_init_empty());
    gguf_set_kv(out.get(), inp.get());
    gguf_set_val_u32(out.get(), "general.quantization_version", 2);
    gguf_set_val_u32(out.get(), "general.file_type", (uint32_t) variant.ftype);

    const std::vector<TensorPlan> plans =
        plan_tensors(out.get(), inp.get(), meta.get(), arch, variant, n_layers);
    stats.n_tensors = (int) plans.size();

    if (!gguf_write_to_file(out.get(), out_path.c_str(), /*only_meta=*/true)) {
        error = "failed to write metadata to " + out_path;
        return false;
    }

    std::unique_ptr<FILE, FileClose> fout(fopen(out_path.c_str(), "ab"));
    if (!fout) {
        error = "failed to open " + out_path + " for append";
        remove(out_path.c_str());
        return false;
    }

    const bool ok = stream_tensor_data(fout.get(), inp.get(), meta.get(), mapping.file.data,
                                       gguf_get_data_offset(inp.get()),
                                       gguf_get_alignment(out.get()), plans, stats, error);
    fout.reset();

    if (!ok) {
        remove(out_path.c_str());
        return false;
    }
    return true;
}

}  // namespace tts_cpp::acestep
