#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace {

constexpr int32_t kGroupSize = 256;
constexpr int     kH256      = 256;
constexpr double  kNativeTolerance = 3e-5;
constexpr double  kCompatTolerance = 2e-3;
constexpr double  kRotationTolerance = 1e-5;
constexpr double  kQ8LinearTolerance = 1e-2;
constexpr double  kH256Tolerance = 2e-5;
constexpr double  kPlainMatmulTolerance = 1e-2;
constexpr size_t  kContextSize = 16 * 1024 * 1024;

struct graph_ctx {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    graph_ctx() {
        ggml_init_params params = { kContextSize, nullptr, true };
        ctx = ggml_init(params);
    }

    ~graph_ctx() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ctx);
    }
};

void hadamard_4(float * v, size_t stride) {
    const float a = v[0], b = v[stride], c = v[2*stride], d = v[3*stride];
    v[0]        = ( a + b + c - d)*0.5f;
    v[stride]   = ( a + b - c + d)*0.5f;
    v[2*stride] = ( a - b + c + d)*0.5f;
    v[3*stride] = (-a + b + c + d)*0.5f;
}

void rotate_stage(float * v, int n, size_t stride) {
    for (size_t base = 0; base < (size_t) n; base += 4*stride) {
        for (size_t i = 0; i < stride; ++i) {
            hadamard_4(v + base + i, stride);
        }
    }
}

void rotate_block(float * v, int n) {
    for (size_t stride = 1; stride < (size_t) n; stride *= 4) {
        rotate_stage(v, n, stride);
    }
}

void rotate_row(float * v, int64_t n, int group_size) {
    for (int64_t k0 = 0; k0 < n; k0 += group_size) {
        rotate_block(v + k0, group_size);
    }
}

float dense_h256_coefficient(int row, int col) {
    static const int signs[4][4] = {
        { 1,  1,  1, -1},
        { 1,  1, -1,  1},
        { 1, -1,  1,  1},
        {-1,  1,  1,  1},
    };
    int sign = 1;
    for (int digit = 0; digit < 4; ++digit) {
        sign *= signs[row % 4][col % 4];
        row /= 4;
        col /= 4;
    }
    return sign / 16.0f;
}

float pattern_value(size_t i, size_t mul, size_t add, size_t mod) {
    return float(int64_t((i*mul + add) % mod) - int64_t(mod/2)) * 0.03125f;
}

std::vector<float> pattern(size_t n, size_t mul, size_t add, size_t mod) {
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = pattern_value(i, mul, add, mod);
    }
    return data;
}

bool graph_supported(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            return false;
        }
    }
    return true;
}

bool compute(ggml_backend_t backend, graph_ctx & g, ggml_cgraph * graph, const std::function<void()> & upload) {
    g.buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (!g.buffer) {
        return false;
    }
    upload();
    return ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
}

std::vector<float> download(const ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

int report(const char * name, bool ok, double max_err) {
    std::printf("    %-48s %s (max err %.2e)\n", name, ok ? "ok" : "FAILED", max_err);
    return ok ? 0 : 1;
}

int report_skip(const char * name) {
    std::printf("    %-48s not supported, skipped\n", name);
    return 0;
}

int test_repack_scale_range() {
    constexpr int64_t k = 32;
    const float scales[] = { 0.0007f, 0.000001f, 0.000000001f };
    std::vector<int8_t> weights(k*3, 127);
    std::vector<uint8_t> packed(ggml_convrot_repack_q8_0(nullptr, nullptr, k, 3, nullptr));
    ggml_convrot_repack_q8_0(weights.data(), scales, k, 3, packed.data());

    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, k);
    float rounded[3];
    for (int row = 0; row < 3; ++row) {
        ggml_fp16_t half;
        std::memcpy(&half, packed.data() + row*row_size, sizeof(half));
        rounded[row] = ggml_fp16_to_fp32(half);
    }
    const double normal_error    = std::fabs(rounded[0] - scales[0]) / scales[0];
    const double subnormal_error = std::fabs(rounded[1] - scales[1]) / scales[1];
    const bool ok = normal_error < 1e-3 && subnormal_error > normal_error && rounded[1] > 0.0f && rounded[2] == 0.0f;
    return report("repack F32 -> F16 scale range", ok, normal_error);
}

double rotation_error(const std::vector<float> & data, const std::vector<float> & out, int64_t k, int64_t rows,
                      int64_t planes, int64_t storage_rows, int64_t row_step, int32_t group_size) {
    double max_err = 0.0;
    for (int64_t r = 0; r < rows*planes; ++r) {
        const int64_t p = r / rows, row = r % rows;
        const float * src = data.data() + (p*storage_rows + row*row_step)*k;
        std::vector<float> ref(src, src + k);
        rotate_row(ref.data(), k, group_size);
        for (int64_t i = 0; i < k; ++i) {
            max_err = std::fmax(max_err, std::fabs(out[r*k + i] - ref[i]) / (1.0 + std::fabs(ref[i])));
        }
    }
    return max_err;
}

int test_rotation(ggml_backend_t backend, int32_t group_size, bool inplace, int64_t groups, int64_t rows) {
    char name[64];
    std::snprintf(name, sizeof(name), "rotation group=%d groups=%lld%s", group_size, (long long) groups, inplace ? " in place" : "");

    const int64_t k = groups*group_size, planes = 2;
    const int64_t row_step = inplace ? 1 : 2;
    graph_ctx g;
    ggml_tensor * storage = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, k, rows*row_step, planes);
    ggml_tensor * x = inplace ? storage : ggml_view_3d(g.ctx, storage, k, rows, planes, 2*storage->nb[1], storage->nb[2], 0);
    ggml_tensor * y = inplace ? ggml_convrot_inplace(g.ctx, x, group_size) : ggml_convrot(g.ctx, x, group_size);
    ggml_cgraph * graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(graph, y);
    if (!graph_supported(backend, graph)) {
        return report_skip(name);
    }

    const std::vector<float> data = pattern(ggml_nelements(storage), 37, 11, 101);
    const bool computed = compute(backend, g, graph, [&] { ggml_backend_tensor_set(storage, data.data(), 0, ggml_nbytes(storage)); });
    if (!computed) {
        return report(name, false, 0.0);
    }
    const double max_err = rotation_error(data, download(y), k, rows, planes, storage->ne[1], row_step, group_size);
    return report(name, max_err <= kRotationTolerance, max_err);
}

int test_rotations(ggml_backend_t backend) {
    int failures = 0;
    for (int32_t group_size : { 4, 16, 64, 256, 1024 }) {
        failures += test_rotation(backend, group_size, false, 2, 3);
        failures += test_rotation(backend, group_size, true, 2, 3);
    }
    failures += test_rotation(backend, 4, false, 256, 1);
    return failures;
}

struct convrot_weights {
    int64_t k, n;
    std::vector<int8_t> w;
    std::vector<float>  s;
};

convrot_weights make_weights(int64_t k, int64_t n, float scale_base, float scale_step) {
    convrot_weights cw = { k, n, std::vector<int8_t>(k*n), std::vector<float>(n) };
    for (int64_t row = 0; row < n; ++row) {
        cw.s[row] = row == 1 ? 0.0f : scale_base + scale_step*float(row*7 % 13);
        for (int64_t i = 0; i < k; ++i) {
            cw.w[row*k + i] = row == 1 ? 0 : int8_t((i*17 + row*31) % 255 - 127);
        }
    }
    return cw;
}

std::vector<float> rotated_weight_row(const convrot_weights & cw, int64_t row, bool round_scale_to_f16) {
    const float scale = round_scale_to_f16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(cw.s[row])) : cw.s[row];
    std::vector<float> w(cw.k);
    for (int64_t i = 0; i < cw.k; ++i) {
        w[i] = float(cw.w[row*cw.k + i]) * scale;
    }
    rotate_row(w.data(), cw.k, kGroupSize);
    return w;
}

double dot(const std::vector<float> & w, const float * x) {
    double acc = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        acc += double(w[i]) * double(x[i]);
    }
    return acc;
}

double sum_abs_products(const std::vector<float> & w, const float * x) {
    double acc = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        acc += std::fabs(double(w[i]) * double(x[i]));
    }
    return acc;
}

struct oracle {
    std::vector<double> value;
    std::vector<double> sum_abs;
    double max_abs = 0.0;
};

oracle weight_rotation_oracle(const convrot_weights & cw, const std::vector<float> & x, int64_t columns, bool round_scale_to_f16) {
    oracle ref = { std::vector<double>(cw.n*columns), std::vector<double>(cw.n*columns) };
    for (int64_t row = 0; row < cw.n; ++row) {
        const std::vector<float> w = rotated_weight_row(cw, row, round_scale_to_f16);
        for (int64_t c = 0; c < columns; ++c) {
            ref.value[c*cw.n + row]   = dot(w, x.data() + c*cw.k);
            ref.sum_abs[c*cw.n + row] = sum_abs_products(w, x.data() + c*cw.k);
            ref.max_abs = std::fmax(ref.max_abs, std::fabs(ref.value[c*cw.n + row]));
        }
    }
    return ref;
}

bool zero_row_is_zero(const std::vector<float> & out, int64_t n, int64_t columns) {
    for (int64_t c = 0; c < columns; ++c) {
        if (out[c*n + 1] != 0.0f) {
            return false;
        }
    }
    return true;
}

double q8_linear_error(const std::vector<float> & out, const oracle & ref, bool & ok) {
    double max_err = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double err = std::fabs(out[i] - ref.value[i]);
        ok = ok && std::isfinite(out[i]) && err <= kQ8LinearTolerance*ref.max_abs;
        max_err = std::fmax(max_err, err / ref.max_abs);
    }
    return max_err;
}

int test_q8_linear(ggml_backend_t backend, int64_t m) {
    char name[64];
    std::snprintf(name, sizeof(name), "repacked Q8_0 linear M=%lld", (long long) m);

    const convrot_weights cw = make_weights(2*kGroupSize, 37, 0.0007f, 0.00001f);
    graph_ctx g;
    ggml_tensor * x  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, cw.k, m);
    ggml_tensor * wq = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, cw.k, cw.n);
    ggml_tensor * y  = ggml_convrot_mul_mat(g.ctx, wq, x, kGroupSize);
    ggml_cgraph * graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(graph, y);
    if (!graph_supported(backend, graph)) {
        return report_skip(name);
    }

    const size_t packed_size = ggml_convrot_repack_q8_0(nullptr, nullptr, cw.k, cw.n, nullptr);
    std::vector<uint8_t> packed(packed_size);
    ggml_convrot_repack_q8_0(cw.w.data(), cw.s.data(), cw.k, cw.n, packed.data());
    const std::vector<float> xv = pattern(cw.k*m, 13, 5, 97);

    bool ok = packed_size == ggml_nbytes(wq) && compute(backend, g, graph, [&] {
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(wq, packed.data(), 0, ggml_nbytes(wq));
    });
    if (!ok) {
        return report(name, false, 0.0);
    }
    const std::vector<float> out = download(y);
    const double max_err = q8_linear_error(out, weight_rotation_oracle(cw, xv, m, true), ok);
    return report(name, ok && zero_row_is_zero(out, cw.n, m), max_err);
}

void write_activation(ggml_type type, float value, uint8_t * dst) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, &value, sizeof(value));
        return;
    }
    const ggml_fp16_t half = ggml_fp32_to_fp16(value);
    std::memcpy(dst, &half, sizeof(half));
}

float activation_value(ggml_type type, float value) {
    return type == GGML_TYPE_F32 ? value : ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

struct activations {
    std::vector<uint8_t> storage;
    std::vector<float>   values;
};

activations make_activations(ggml_type type, int64_t k, int64_t columns, int64_t column_step) {
    const size_t elem = ggml_type_size(type);
    activations a = { std::vector<uint8_t>(k*columns*column_step*elem, 0), std::vector<float>(k*columns) };
    for (int64_t c = 0; c < columns; ++c) {
        for (int64_t i = 0; i < k; ++i) {
            const float value = pattern_value(c*k + i, 13, 29, 97);
            write_activation(type, value, a.storage.data() + (c*column_step*k + i)*elem);
            a.values[c*k + i] = activation_value(type, value);
        }
    }
    return a;
}

double convrot_error(const std::vector<float> & out, const oracle & ref, bool f16_compat, bool & ok) {
    double max_err = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double scale = f16_compat ? 1.0 + ref.sum_abs[i] : 1.0 + std::fabs(ref.value[i]);
        const double err = std::fabs(out[i] - ref.value[i]) / scale;
        ok = ok && std::isfinite(out[i]) && err <= (f16_compat ? kCompatTolerance : kNativeTolerance);
        max_err = std::fmax(max_err, err);
    }
    return max_err;
}

int test_mul_mat_convrot(ggml_backend_t backend, ggml_type type, int64_t columns, bool strided, bool f16_compat) {
    char name[64];
    std::snprintf(name, sizeof(name), "mul_mat_convrot %s M=%lld%s%s", ggml_type_name(type), (long long) columns,
                  strided ? " strided" : "", f16_compat ? " f16-compat" : "");

    const convrot_weights cw = make_weights(2*kGroupSize, 37, 0.0039f, 0.0001f);
    const int64_t column_step = strided ? 2 : 1;
    graph_ctx g;
    ggml_tensor * storage = ggml_new_tensor_2d(g.ctx, type, cw.k, columns*column_step);
    ggml_tensor * x = strided ? ggml_view_2d(g.ctx, storage, cw.k, columns, 2*storage->nb[1], 0) : storage;
    ggml_tensor * w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I8, cw.k, cw.n);
    ggml_tensor * s = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, cw.n);
    ggml_tensor * y = ggml_mul_mat_convrot(g.ctx, x, w, s, kGroupSize);
    ggml_mul_mat_convrot_set_f16_compat(y, f16_compat);
    ggml_cgraph * graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(graph, y);
    if (!graph_supported(backend, graph)) {
        return report_skip(name);
    }

    const activations a = make_activations(type, cw.k, columns, column_step);
    bool ok = compute(backend, g, graph, [&] {
        ggml_backend_tensor_set(storage, a.storage.data(), 0, ggml_nbytes(storage));
        ggml_backend_tensor_set(w, cw.w.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(s, cw.s.data(), 0, ggml_nbytes(s));
    });
    if (!ok) {
        return report(name, false, 0.0);
    }
    const std::vector<float> out = download(y);
    const double max_err = convrot_error(out, weight_rotation_oracle(cw, a.values, columns, false), f16_compat, ok);
    return report(name, ok && zero_row_is_zero(out, cw.n, columns), max_err);
}

int test_mul_mat_convrot_empty_batch(ggml_backend_t backend) {
    const char * name = "mul_mat_convrot empty batch";
    graph_ctx g;
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kGroupSize, 0);
    ggml_tensor * w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I8, kGroupSize, 3);
    ggml_tensor * s = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 3);
    ggml_tensor * y = ggml_mul_mat_convrot(g.ctx, x, w, s, kGroupSize);
    ggml_cgraph * graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(graph, y);
    if (!graph_supported(backend, graph)) {
        return report_skip(name);
    }
    const bool ok = y->ne[0] == 3 && y->ne[1] == 0 && compute(backend, g, graph, [] {});
    return report(name, ok, 0.0);
}

int test_mul_mat_convrots(ggml_backend_t backend) {
    int failures = 0;
    for (bool f16_compat : { false, true }) {
        failures += test_mul_mat_convrot(backend, GGML_TYPE_F32, 1, false, f16_compat);
        failures += test_mul_mat_convrot(backend, GGML_TYPE_F32, 5, true, f16_compat);
        failures += test_mul_mat_convrot(backend, GGML_TYPE_F16, 4, false, f16_compat);
        failures += test_mul_mat_convrot(backend, GGML_TYPE_F32, 48, false, f16_compat);
    }
    return failures + test_mul_mat_convrot_empty_batch(backend);
}

std::vector<float> dense_h256() {
    std::vector<float> h(kH256*kH256);
    for (int row = 0; row < kH256; ++row) {
        for (int col = 0; col < kH256; ++col) {
            h[row*kH256 + col] = dense_h256_coefficient(row, col);
        }
    }
    return h;
}

struct h256_errors {
    double max_abs_err = 0.0;
    double max_abs_ref = 0.0;
};

h256_errors h256_error(const std::vector<float> & h, const std::vector<float> & x, float scale, const std::vector<float> & out) {
    h256_errors e;
    for (size_t r = 0; r < out.size(); ++r) {
        const size_t row = r / kH256, col = r % kH256;
        double expected = 0.0;
        for (int i = 0; i < kH256; ++i) {
            expected += double(h[col*kH256 + i]) * double(x[row*kH256 + i]*scale);
        }
        e.max_abs_err = std::fmax(e.max_abs_err, std::fabs(out[r] - expected));
        e.max_abs_ref = std::fmax(e.max_abs_ref, std::fabs(expected));
    }
    return e;
}

double h256_tolerance(ggml_backend_t backend, double max_abs_ref) {
    const bool exact = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    return exact ? kH256Tolerance : kPlainMatmulTolerance * max_abs_ref;
}

int test_h256_hint(ggml_backend_t backend) {
    const char * name = "mul_mat H256 hint after a producer";
    constexpr int64_t rows = 257, batches = 3;
    constexpr float scale = 1.25f;

    graph_ctx g;
    ggml_tensor * h = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kH256, kH256);
    ggml_tensor * x = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, kH256, rows, batches);
    ggml_tensor * y = ggml_mul_mat(g.ctx, h, ggml_scale(g.ctx, x, scale));
    ggml_mul_mat_set_hint(y, GGML_HINT_SRC0_IS_CONVROT_H256);
    ggml_cgraph * graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(graph, y);
    if (!graph_supported(backend, graph)) {
        return report_skip(name);
    }

    const std::vector<float> hv = dense_h256();
    const std::vector<float> xv = pattern(ggml_nelements(x), 17, 31, 97);
    const bool computed = compute(backend, g, graph, [&] {
        ggml_backend_tensor_set(h, hv.data(), 0, ggml_nbytes(h));
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    });
    if (!computed) {
        return report(name, false, 0.0);
    }
    const h256_errors e = h256_error(hv, xv, scale, download(y));
    return report(name, e.max_abs_err <= h256_tolerance(backend, e.max_abs_ref), e.max_abs_err);
}

int test_backend(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        return 0;
    }
    std::printf("backend %s (%s)\n", ggml_backend_name(backend), ggml_backend_dev_description(dev));
    int failures = test_rotations(backend);
    for (int64_t m : { 1, 2, 48 }) {
        failures += test_q8_linear(backend, m);
    }
    failures += test_mul_mat_convrots(backend);
    failures += test_h256_hint(backend);
    ggml_backend_free(backend);
    return failures;
}

int test_backends() {
    int failures = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        failures += test_backend(ggml_backend_dev_get(i));
    }
    return failures;
}

} // namespace

int main() {
    ggml_backend_load_all();

    const int failures = test_repack_scale_range() + test_backends();
    std::puts(failures == 0 ? "ConvRot tests passed" : "ConvRot tests FAILED");
    return failures == 0 ? 0 : 1;
}
