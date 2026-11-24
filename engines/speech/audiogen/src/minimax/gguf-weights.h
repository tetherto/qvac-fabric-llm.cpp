#pragma once

#include "gguf.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

struct GGUFModel {
    struct gguf_context * gguf;
    struct ggml_context * meta;
    uint8_t *             mapping;
    size_t                file_size;
    size_t                data_offset;
#ifdef _WIN32
    HANDLE fh;
    HANDLE mh;
#else
    int fd;
#endif
};

static bool gf_tensor_range_fits(size_t file_size, size_t data_offset, size_t tensor_offset, size_t tensor_size) {
    if (data_offset > file_size) {
        return false;
    }
    const size_t data_size = file_size - data_offset;
    if (tensor_offset > data_size) {
        return false;
    }
    return tensor_size <= data_size - tensor_offset;
}

static void gf_close(GGUFModel * gf) {
    if (gf->gguf) {
        gguf_free(gf->gguf);
    }
    if (gf->meta) {
        ggml_free(gf->meta);
    }
#ifdef _WIN32
    if (gf->mapping) {
        UnmapViewOfFile(gf->mapping);
    }
    if (gf->mh) {
        CloseHandle(gf->mh);
    }
    if (gf->fh && gf->fh != INVALID_HANDLE_VALUE) {
        CloseHandle(gf->fh);
    }
#else
    if (gf->mapping) {
        munmap(gf->mapping, gf->file_size);
    }
    if (gf->fd >= 0) {
        close(gf->fd);
    }
#endif
    *gf = {};
}

static bool gf_load(GGUFModel * gf, const char * path) {
    *gf = {};

#ifdef _WIN32
    gf->fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (gf->fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[GGUF] Cannot open %s\n", path);
        return false;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(gf->fh, &li);
    gf->file_size = (size_t) li.QuadPart;
    gf->mh        = CreateFileMappingA(gf->fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!gf->mh) {
        CloseHandle(gf->fh);
        fprintf(stderr, "[GGUF] CreateFileMapping failed %s\n", path);
        return false;
    }
    gf->mapping = (uint8_t *) MapViewOfFile(gf->mh, FILE_MAP_READ, 0, 0, 0);
    if (!gf->mapping) {
        CloseHandle(gf->mh);
        CloseHandle(gf->fh);
        fprintf(stderr, "[GGUF] MapViewOfFile failed %s\n", path);
        return false;
    }
#else
    gf->fd = open(path, O_RDONLY);
    if (gf->fd < 0) {
        fprintf(stderr, "[GGUF] Cannot open %s\n", path);
        return false;
    }
    struct stat sb;
    fstat(gf->fd, &sb);
    gf->file_size = (size_t) sb.st_size;
    gf->mapping   = (uint8_t *) mmap(NULL, gf->file_size, PROT_READ, MAP_PRIVATE, gf->fd, 0);
    if (gf->mapping == MAP_FAILED) {
        close(gf->fd);
        gf->mapping = NULL;
        fprintf(stderr, "[GGUF] Mmap failed %s\n", path);
        return false;
    }
#endif

    struct ggml_context *   meta   = NULL;
    struct gguf_init_params params = { true, &meta };
    gf->gguf                       = gguf_init_from_file(path, params);
    if (!gf->gguf) {
        fprintf(stderr, "[GGUF] Failed to parse %s\n", path);
        gf_close(gf);
        return false;
    }
    gf->meta        = meta;
    gf->data_offset = gguf_get_data_offset(gf->gguf);

    int64_t n = gguf_get_n_tensors(gf->gguf);

    for (int64_t i = 0; i < n; i++) {
        const char *         tname = gguf_get_tensor_name(gf->gguf, i);
        struct ggml_tensor * t     = ggml_get_tensor(gf->meta, tname);
        size_t               toff  = gguf_get_tensor_offset(gf->gguf, i);
        size_t               tsize = ggml_nbytes(t);
        if (!gf_tensor_range_fits(gf->file_size, gf->data_offset, toff, tsize)) {
            fprintf(stderr,
                    "[GGUF] FATAL: '%s' is truncated or corrupt.\n"
                    "       tensor '%s' does not fit at data offset %zu, tensor offset %zu, size %zu in %zu bytes.\n"
                    "       Re-download the file and verify its size or checksum.\n",
                    path, tname, gf->data_offset, toff, tsize, gf->file_size);
            gf_close(gf);
            return false;
        }
    }

    fprintf(stderr, "[GGUF] %s: %lld tensors, data at offset %zu\n", path, (long long) n, gf->data_offset);
    return true;
}

static bool gf_load_metadata(GGUFModel * gf, const char * path) {
    *gf = {};
#ifndef _WIN32
    gf->fd = -1;
#endif
    struct ggml_context *   meta   = NULL;
    struct gguf_init_params params = { true, &meta };
    gf->gguf                       = gguf_init_from_file(path, params);
    if (!gf->gguf) {
        fprintf(stderr, "[GGUF] Failed to parse %s\n", path);
        return false;
    }
    gf->meta        = meta;
    gf->data_offset = gguf_get_data_offset(gf->gguf);
    return true;
}

static const uint8_t * gf_tensor_source(const GGUFModel & gf, int64_t idx) {
    if (!gf.mapping || idx < 0) {
        return nullptr;
    }
    return gf.mapping + gf.data_offset + gguf_get_tensor_offset(gf.gguf, idx);
}

static struct ggml_tensor * gf_load_tensor(WeightCtx *         wctx,
                                           const GGUFModel &   gf,
                                           const std::string & name,
                                           const int64_t *     shape_override  = nullptr,
                                           int                 n_dims_override = 0) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        fprintf(stderr, "[GGUF] FATAL: tensor '%s' not found\n", name.c_str());
        return nullptr;
    }

    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        fprintf(stderr, "[GGUF] FATAL: tensor '%s' not in meta context\n", name.c_str());
        return nullptr;
    }

    int     n_dims;
    int64_t ne[4] = { 1, 1, 1, 1 };

    if (shape_override && n_dims_override > 0) {
        n_dims = n_dims_override;
        for (int i = 0; i < n_dims; i++) {
            ne[i] = shape_override[i];
        }
    } else {
        n_dims = ggml_n_dims(src);
        for (int i = 0; i < n_dims; i++) {
            ne[i] = src->ne[i];
        }
    }

    struct ggml_tensor * tensor = ggml_new_tensor(wctx->ctx, src->type, n_dims, ne);
    ggml_set_name(tensor, name.c_str());

    wctx_queue_copy(wctx, tensor, gf_tensor_source(gf, idx), ggml_nbytes(src), 0);
    return tensor;
}

static struct ggml_tensor * gf_try_load_tensor(WeightCtx * wctx, const GGUFModel & gf, const std::string & name) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        return nullptr;
    }
    return gf_load_tensor(wctx, gf, name);
}

static struct ggml_tensor * gf_load_tensor_f32(WeightCtx * wctx, const GGUFModel & gf, const std::string & name) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        fprintf(stderr, "[GGUF] FATAL: tensor '%s' not found\n", name.c_str());
        return nullptr;
    }
    struct ggml_tensor * src    = ggml_get_tensor(gf.meta, name.c_str());
    int                  n_dims = ggml_n_dims(src);
    int64_t              ne[4]  = { 1, 1, 1, 1 };
    for (int i = 0; i < n_dims; i++) {
        ne[i] = src->ne[i];
    }

    if (src->type == GGML_TYPE_F32) {
        return gf_load_tensor(wctx, gf, name);
    }

    if (src->type != GGML_TYPE_BF16 && src->type != GGML_TYPE_F16) {
        fprintf(stderr, "[GGUF] WARNING: gf_load_tensor_f32 unsupported type %d for '%s', loading as-is\n", src->type,
                name.c_str());
        return gf_load_tensor(wctx, gf, name);
    }

    struct ggml_tensor * tensor = ggml_new_tensor(wctx->ctx, GGML_TYPE_F32, n_dims, ne);
    ggml_set_name(tensor, name.c_str());

    size_t  n    = ggml_nelements(src);
    auto    buf  = std::make_unique<float[]>(n);
    float * data = buf.get();

    size_t       offset = gguf_get_tensor_offset(gf.gguf, idx);
    const void * raw    = gf.mapping + gf.data_offset + offset;

    if (src->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) raw;
        for (size_t i = 0; i < n; i++) {
            data[i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &p[i]);
        }
    } else {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, data, (int) n);
    }

    wctx->pending.push_back({ tensor, data, n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return tensor;
}

static const void * gf_get_data(const GGUFModel & gf, const char * name) {
    return gf_tensor_source(gf, gguf_find_tensor(gf.gguf, name));
}

static struct ggml_tensor * gf_load_qkv_fused(WeightCtx *         wctx,
                                              const GGUFModel &   gf,
                                              const std::string & q_name,
                                              const std::string & k_name,
                                              const std::string & v_name) {
    struct ggml_tensor * q_src = ggml_get_tensor(gf.meta, q_name.c_str());
    struct ggml_tensor * k_src = ggml_get_tensor(gf.meta, k_name.c_str());
    struct ggml_tensor * v_src = ggml_get_tensor(gf.meta, v_name.c_str());
    if (!q_src || !k_src || !v_src) {
        fprintf(stderr, "[GGUF] FATAL: QKV tensor not found: %s / %s / %s\n", q_name.c_str(), k_name.c_str(),
                v_name.c_str());
        return nullptr;
    }

    if (q_src->ne[0] != k_src->ne[0] || k_src->ne[0] != v_src->ne[0]) {
        return nullptr;
    }
    if (q_src->type != k_src->type || k_src->type != v_src->type) {
        return NULL;
    }

    int64_t              ne0       = q_src->ne[0];
    int64_t              fused_ne1 = q_src->ne[1] + k_src->ne[1] + v_src->ne[1];
    int64_t              ne[2]     = { ne0, fused_ne1 };
    struct ggml_tensor * fused     = ggml_new_tensor(wctx->ctx, q_src->type, 2, ne);

    size_t row_size = ggml_row_size(q_src->type, ne0);
    size_t q_bytes  = q_src->ne[1] * row_size;
    size_t k_bytes  = k_src->ne[1] * row_size;
    size_t v_bytes  = v_src->ne[1] * row_size;

    auto get_data = [&](const std::string & name) -> const void * {
        int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
        size_t  off = gguf_get_tensor_offset(gf.gguf, idx);
        return gf.mapping + gf.data_offset + off;
    };

    wctx->pending.push_back({ fused, get_data(q_name), q_bytes, 0 });
    wctx->pending.push_back({ fused, get_data(k_name), k_bytes, q_bytes });
    wctx->pending.push_back({ fused, get_data(v_name), v_bytes, q_bytes + k_bytes });
    return fused;
}

static struct ggml_tensor * gf_load_pair_fused(WeightCtx *         wctx,
                                               const GGUFModel &   gf,
                                               const std::string & a_name,
                                               const std::string & b_name) {
    struct ggml_tensor * a_src = ggml_get_tensor(gf.meta, a_name.c_str());
    struct ggml_tensor * b_src = ggml_get_tensor(gf.meta, b_name.c_str());
    if (!a_src || !b_src) {
        return NULL;
    }
    if (a_src->ne[0] != b_src->ne[0] || a_src->type != b_src->type) {
        return NULL;
    }

    int64_t              ne0   = a_src->ne[0];
    int64_t              ne[2] = { ne0, a_src->ne[1] + b_src->ne[1] };
    struct ggml_tensor * fused = ggml_new_tensor(wctx->ctx, a_src->type, 2, ne);

    size_t row_size = ggml_row_size(a_src->type, ne0);
    size_t a_bytes  = a_src->ne[1] * row_size;
    size_t b_bytes  = b_src->ne[1] * row_size;

    auto get_data = [&](const std::string & name) -> const void * {
        int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
        size_t  off = gguf_get_tensor_offset(gf.gguf, idx);
        return gf.mapping + gf.data_offset + off;
    };

    wctx->pending.push_back({ fused, get_data(a_name), a_bytes, 0 });
    wctx->pending.push_back({ fused, get_data(b_name), b_bytes, a_bytes });
    return fused;
}

static uint32_t gf_get_u32(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_UINT32) {
        return 0;
    }
    return gguf_get_val_u32(gf.gguf, idx);
}

static float gf_get_f32(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_FLOAT32) {
        return 0.0f;
    }
    return gguf_get_val_f32(gf.gguf, idx);
}

static const char * gf_get_str(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_STRING) {
        return "";
    }
    return gguf_get_val_str(gf.gguf, idx);
}

static bool gf_get_bool(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_BOOL) {
        return false;
    }
    return gguf_get_val_bool(gf.gguf, idx);
}
