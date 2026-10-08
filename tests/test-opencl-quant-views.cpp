// Reads views of an OpenCL Q1_0 / PQ2_0 weight back and compares them byte for byte with the uploaded weight. The
// backend stores these weights converted, so a view must be restored with its weight's layout. Skips without OpenCL.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// fewer than 128 rows, so a Q1_0 weight keeps its packed layout rather than the Adreno transposed one
static constexpr int64_t WEIGHT_K    = 256;
static constexpr int64_t WEIGHT_ROWS = 64;
static constexpr int64_t VIEW_ROWS   = WEIGHT_ROWS/2;
static constexpr int     CTX_TENSORS = 4;

struct view_case {
    const char * name;
    int64_t      ne0;
    int64_t      ne1;
    int64_t      first_row; // weight row the view starts at
    int64_t      read_row;  // first view row read back
    int64_t      read_rows; // view rows read back, 0 reads to the end
};

static const view_case VIEW_CASES[] = {
    { "reshape to wider rows",        2*WEIGHT_K, WEIGHT_ROWS/2, 0,         0, 0 },
    { "first rows",                   WEIGHT_K,   VIEW_ROWS,     0,         0, 0 },
    { "last rows",                    WEIGHT_K,   VIEW_ROWS,     VIEW_ROWS, 0, 0 },
    { "one row inside the last rows", WEIGHT_K,   VIEW_ROWS,     VIEW_ROWS, 1, 1 },
};

static const ggml_type WEIGHT_TYPES[] = { GGML_TYPE_Q1_0, GGML_TYPE_PQ2_0 };

static ggml_backend_dev_t find_opencl_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "OpenCL") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static ggml_context * make_context() {
    ggml_init_params params = {
        /*.mem_size   =*/ CTX_TENSORS*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(params);
}

// both types are an fp16 scale followed by quant bytes; every block gets its own finite scale and byte pattern
static void fill_block(uint8_t * blk, size_t blk_size, int64_t ib) {
    const ggml_fp16_t d = ggml_fp32_to_fp16(0.5f + (float) (ib % 7)/8.0f);
    std::memcpy(blk, &d, sizeof(d));
    for (size_t j = sizeof(d); j < blk_size; ++j) {
        blk[j] = (uint8_t) (ib*31 + j*7 + 3);
    }
}

static std::vector<uint8_t> make_weight_bytes(const ggml_tensor * weight) {
    const size_t  blk_size = ggml_type_size(weight->type);
    const int64_t n_blocks = ggml_nelements(weight)/ggml_blck_size(weight->type);
    std::vector<uint8_t> bytes(ggml_nbytes(weight));
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        fill_block(bytes.data() + ib*blk_size, blk_size, ib);
    }
    return bytes;
}

static bool read_matches(const ggml_tensor * t, size_t offset, size_t size, const uint8_t * expected) {
    std::vector<uint8_t> got(size);
    ggml_backend_tensor_get(t, got.data(), offset, size);
    return std::memcmp(got.data(), expected, size) == 0;
}

// the view is made after the upload, as a graph makes its views of a model weight
static bool check_view(ggml_tensor * weight, const std::vector<uint8_t> & bytes, const view_case & vc) {
    ggml_context * ctx = make_context();
    ggml_tensor * view = ggml_view_2d(ctx, weight, vc.ne0, vc.ne1, ggml_row_size(weight->type, vc.ne0),
                                      vc.first_row*weight->nb[1]);
    GGML_ASSERT(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);

    const int64_t read_rows = vc.read_rows != 0 ? vc.read_rows : vc.ne1 - vc.read_row;
    const size_t  offset    = vc.read_row*view->nb[1];
    const size_t  size      = read_rows*view->nb[1];
    const bool    ok        = read_matches(view, offset, size, bytes.data() + view->view_offs + offset);

    std::printf("%s %-6s %s\n", ok ? "ok  " : "FAIL", ggml_type_name(weight->type), vc.name);
    ggml_free(ctx);
    return ok;
}

static int check_views(ggml_tensor * weight, const std::vector<uint8_t> & bytes) {
    int failures = 0;
    for (const view_case & vc : VIEW_CASES) {
        failures += check_view(weight, bytes, vc) ? 0 : 1;
    }
    return failures;
}

static int check_type(ggml_backend_t backend, ggml_type type) {
    ggml_context * ctx = make_context();
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, WEIGHT_K, WEIGHT_ROWS);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    const std::vector<uint8_t> bytes = make_weight_bytes(weight);
    ggml_backend_tensor_set(weight, bytes.data(), 0, bytes.size());

    const bool whole_ok = read_matches(weight, 0, bytes.size(), bytes.data());
    std::printf("%s %-6s whole weight\n", whole_ok ? "ok  " : "FAIL", ggml_type_name(type));
    const int failures = (whole_ok ? 0 : 1) + check_views(weight, bytes);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return failures;
}

static int check_types(ggml_backend_t backend) {
    int failures = 0;
    for (ggml_type type : WEIGHT_TYPES) {
        failures += check_type(backend, type);
    }
    return failures;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = find_opencl_device();
    if (dev == nullptr) {
        std::printf("no OpenCL device found - test skipped\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend != nullptr);

    const int failures = check_types(backend);
    ggml_backend_free(backend);

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
