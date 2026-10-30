#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <vector>

static constexpr int WIDTH = 4096;
static constexpr int EXPERTS = 2;
static constexpr int CHOSEN_EXPERT = 1;
static constexpr int RUNS = 4;
static constexpr int GRAPH_NODES = 16;
static constexpr float TOLERANCE = 1e-3f;
static constexpr float MIN_MAGNITUDE = 1e-2f;
static constexpr float WEIGHT_FREQUENCY = 0.013f;
static constexpr float WEIGHT_PHASE = 0.7f;
static constexpr float WEIGHT_AMPLITUDE = 0.05f;
static constexpr float INPUT_FREQUENCY = 0.37f;
static constexpr float INPUT_AMPLITUDE = 2.0f;
static constexpr float IDENTITY_SCALE = 1.0f;

struct AliasCase {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * activations = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<float> expected;
};

static void fill_weight_row(std::vector<float> & values, int row, int salt) {
    for (int col = 0; col < WIDTH; ++col) {
        const float phase = WEIGHT_FREQUENCY * (float) (row * WIDTH + col) + WEIGHT_PHASE * (float) salt;
        values[(size_t) row * WIDTH + col] = std::sin(phase) * WEIGHT_AMPLITUDE;
    }
}

static std::vector<float> weight_values(int salt) {
    std::vector<float> values((size_t) WIDTH * WIDTH);
    for (int row = 0; row < WIDTH; ++row) {
        fill_weight_row(values, row, salt);
    }
    return values;
}

static int gate_salt(int expert) {
    return 2 * expert + 1;
}

static int up_salt(int expert) {
    return 2 * expert + 2;
}

static std::vector<float> stacked_values() {
    std::vector<float> values;
    for (int expert = 0; expert < EXPERTS; ++expert) {
        const std::vector<float> gate = weight_values(gate_salt(expert));
        const std::vector<float> up = weight_values(up_salt(expert));
        values.insert(values.end(), gate.begin(), gate.end());
        values.insert(values.end(), up.begin(), up.end());
    }
    return values;
}

static std::vector<float> input_values() {
    std::vector<float> values(WIDTH);
    for (int i = 0; i < WIDTH; ++i) {
        values[i] = std::cos(INPUT_FREQUENCY * (float) i) * INPUT_AMPLITUDE;
    }
    return values;
}

static std::vector<unsigned char> encoded(ggml_type type, const std::vector<float> & values) {
    const int64_t rows = (int64_t) values.size() / WIDTH;
    std::vector<unsigned char> bytes(ggml_row_size(type, WIDTH) * rows);
    ggml_quantize_chunk(type, values.data(), bytes.data(), 0, rows, WIDTH, nullptr);
    return bytes;
}

static std::vector<float> round_trip(ggml_type type, const std::vector<float> & values) {
    const std::vector<unsigned char> bytes = encoded(type, values);
    std::vector<float> decoded(values.size());
    ggml_get_type_traits(type)->to_float(bytes.data(), decoded.data(), (int64_t) decoded.size());
    return decoded;
}

static float row_dot(const std::vector<float> & weights, int row, const std::vector<float> & x) {
    double sum = 0.0;
    for (int col = 0; col < WIDTH; ++col) {
        sum += (double) weights[(size_t) row * WIDTH + col] * x[col];
    }
    return (float) sum;
}

static std::vector<float> reference_swiglu(const std::vector<float> & gate, const std::vector<float> & up,
                                           const std::vector<float> & x) {
    std::vector<float> out(WIDTH);
    for (int row = 0; row < WIDTH; ++row) {
        const float g = row_dot(gate, row, x);
        out[row] = g / (1.0f + std::exp(-g)) * row_dot(up, row, x);
    }
    return out;
}

static float worst_relative_error(const std::vector<float> & actual, const std::vector<float> & expected) {
    float worst = 0.0f;
    for (int i = 0; i < WIDTH; ++i) {
        const float error = std::fabs(actual[i] - expected[i]) / std::fmax(std::fabs(expected[i]), MIN_MAGNITUDE);
        worst = std::isfinite(actual[i]) ? std::fmax(worst, error) : INFINITY;
    }
    return worst;
}

static float worst_error_over_runs(ggml_backend_t backend, ggml_cgraph * graph, const AliasCase & c) {
    float worst = 0.0f;
    std::vector<float> actual(WIDTH);
    for (int run = 0; run < RUNS; ++run) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return INFINITY;
        ggml_backend_tensor_get(c.out, actual.data(), 0, actual.size() * sizeof(float));
        worst = std::fmax(worst, worst_relative_error(actual, c.expected));
    }
    return worst;
}

static bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a->buffer == b->buffer && a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static ggml_context * new_context() {
    return ggml_init({ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false),
                      nullptr, true});
}

static void upload(ggml_tensor * tensor, const std::vector<unsigned char> & bytes) {
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

static void upload(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static std::vector<float> expert_rows(const std::vector<float> & stacked, int expert, int part) {
    const size_t matrix = (size_t) WIDTH * WIDTH;
    const auto first = stacked.begin() + (std::ptrdiff_t) ((2 * (size_t) expert + (size_t) part) * matrix);
    return std::vector<float>(first, first + (std::ptrdiff_t) matrix);
}

static bool build_split(AliasCase & c, ggml_backend_t backend, ggml_type type) {
    c.ctx = new_context();
    auto * w_gate = ggml_new_tensor_2d(c.ctx, type, WIDTH, WIDTH);
    auto * w_up = ggml_new_tensor_2d(c.ctx, type, WIDTH, WIDTH);
    auto * input = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, WIDTH, 1);
    c.buffer = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
    if (!c.buffer) return false;
    const std::vector<float> gate = weight_values(gate_salt(0)), up = weight_values(up_salt(0)), x = input_values();
    upload(w_gate, encoded(type, gate));
    upload(w_up, encoded(type, up));
    upload(input, x);
    c.activations = ggml_scale(c.ctx, input, IDENTITY_SCALE);
    c.out = ggml_swiglu_split(c.ctx, ggml_mul_mat(c.ctx, w_gate, c.activations),
                              ggml_mul_mat(c.ctx, w_up, c.activations));
    c.expected = reference_swiglu(round_trip(type, gate), round_trip(type, up), x);
    return true;
}

static ggml_tensor * stacked_half(ggml_context * ctx, ggml_tensor * gate_up, int half) {
    return ggml_view_3d(ctx, gate_up, WIDTH, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2],
                        (size_t) half * WIDTH * gate_up->nb[0]);
}

static bool build_stacked(AliasCase & c, ggml_backend_t backend, ggml_type type) {
    c.ctx = new_context();
    auto * experts = ggml_new_tensor_3d(c.ctx, type, WIDTH, 2 * WIDTH, EXPERTS);
    auto * ids = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, 1, 1);
    auto * input = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, WIDTH, 1, 1);
    c.buffer = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
    if (!c.buffer) return false;
    const std::vector<float> stacked = stacked_values(), x = input_values();
    const int32_t chosen = CHOSEN_EXPERT;
    upload(experts, encoded(type, stacked));
    ggml_backend_tensor_set(ids, &chosen, 0, sizeof(chosen));
    upload(input, x);
    c.activations = ggml_scale(c.ctx, input, IDENTITY_SCALE);
    auto * gate_up = ggml_mul_mat_id(c.ctx, experts, c.activations, ids);
    c.out = ggml_swiglu_split(c.ctx, stacked_half(c.ctx, gate_up, 0), stacked_half(c.ctx, gate_up, 1));
    const std::vector<float> decoded = round_trip(type, stacked);
    c.expected = reference_swiglu(expert_rows(decoded, CHOSEN_EXPERT, 0), expert_rows(decoded, CHOSEN_EXPERT, 1), x);
    return true;
}

static void release(AliasCase & c) {
    if (c.buffer) ggml_backend_buffer_free(c.buffer);
    if (c.ctx) ggml_free(c.ctx);
}

static bool product_survives_activation_slot(ggml_backend_t backend, AliasCase & c, const char * label,
                                             ggml_type type) {
    ggml_set_output(c.out);
    auto * graph = ggml_new_graph_custom(c.ctx, GRAPH_NODES, false);
    ggml_build_forward_expand(graph, c.out);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    const bool aliased = allocated && overlaps(c.out, c.activations);
    const float error = allocated ? worst_error_over_runs(backend, graph, c) : INFINITY;
    const bool ok = aliased && error <= TOLERANCE;
    std::printf("%s %s %s: the product %s the activations' slot, worst relative error %g\n", ok ? "PASS" : "FAIL",
                label, ggml_type_name(type), aliased ? "reuses" : "does not reuse", error);
    ggml_gallocr_free(alloc);
    return ok;
}

static bool check_case(ggml_backend_t backend, ggml_type type, bool stacked) {
    AliasCase c;
    const bool built = stacked ? build_stacked(c, backend, type) : build_split(c, backend, type);
    const bool ok = built && product_survives_activation_slot(backend, c, stacked ? "stacked" : "split", type);
    release(c);
    return ok;
}

static bool check_type(ggml_backend_t backend, ggml_type type) {
    const bool split = check_case(backend, type, false);
    const bool stacked = check_case(backend, type, true);
    return split && stacked;
}

static bool check_types(ggml_backend_t backend) {
    bool ok = true;
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        ok = check_type(backend, type) && ok;
    }
    return ok;
}

int main() {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("MTL0");
    if (!dev) return 77;
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 1;
    const bool ok = check_types(backend);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
