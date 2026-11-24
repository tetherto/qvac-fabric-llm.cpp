#include "minimax/mm3-cond-graph.h"
#include "minimax/mm3-vocoder-graph.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace {

constexpr int64_t INPUT_LENGTH = 17;
constexpr int64_t INPUT_CHANNELS = 8;
constexpr int64_t OUTPUT_CHANNELS = 4;
constexpr int64_t CONDITION_CHANNELS = 4096;
constexpr int64_t CONDITION_OUTPUT_CHANNELS = 4;
constexpr int64_t CONDITION_FRAMES = 17;
constexpr float ABSOLUTE_TOLERANCE = 1e-4f;
// Looser than the vocoder's: the condition projection reduces over 12288
// elements per output, so float accumulation order alone moves the result more.
constexpr float CONDITION_ABSOLUTE_TOLERANCE = 1e-3f;
// Graph node budgets, sized to each case's node count with headroom.
constexpr size_t CONDITION_MAX_NODES = 16;
constexpr size_t VOCODER_MAX_NODES = 64;

std::vector<float> make_values(size_t count, float frequency, float amplitude = 1.0f) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = amplitude * std::sin(static_cast<float>(i + 1) * frequency);
    }
    return values;
}

ggml_backend_t find_metal_backend() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        const char * name = registry ? ggml_backend_reg_name(registry) : nullptr;
        if (name && (std::strcmp(name, "MTL") == 0 || std::strcmp(name, "Metal") == 0)) {
            return ggml_backend_dev_init(device, nullptr);
        }
    }
    return nullptr;
}

// Both parity graphs are matmul-based: mm3_cond_conv1d is im2col + MUL_MAT, and
// mm3_voc_convt opens with a MUL_MAT. ggml's Metal backend gates GGML_OP_MUL_MAT
// on has_simdgroup_reduction, which it derives from MTLGPUFamilyApple7; hosted
// macOS runners expose a paravirtual GPU reporting only Apple5, so every MUL_MAT
// is unsupported there regardless of shape or type.
//
// Ask before computing. ggml_backend_graph_compute() does not consult
// supports_op — it encodes node by node and ggml_abort()s on the first one the
// device cannot take, which surfaces as a bare backtrace with no diagnosis. The
// engine never hits this because it computes through ggml_backend_sched, which
// does consult supports_op and reroutes such nodes to the CPU.
bool metal_supports_mul_mat(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) return false;
    ggml_tensor * lhs = ggml_new_tensor_2d(context, GGML_TYPE_F32, 32, 4);
    ggml_tensor * rhs = ggml_new_tensor_2d(context, GGML_TYPE_F32, 32, 4);
    const bool supported = ggml_backend_supports_op(backend, ggml_mul_mat(context, lhs, rhs));
    ggml_free(context);
    return supported;
}

// The tensors a case exposes to the shared runner: two inputs to fill and the
// output to read back. The runner owns everything else.
struct ParityGraph {
    ggml_tensor * weights;
    ggml_tensor * input;
    ggml_tensor * output;
};

// Builds a case's graph inside a context the runner supplies.
using GraphBuilder = std::function<ParityGraph(ggml_context *)>;

// Shared scaffolding for every parity case: size and open a no_alloc context,
// let the case build its graph, allocate, fill both inputs with deterministic
// values, compute on `backend`, and read the output back. Returns an empty
// vector if the graph did not compute, which the comparator reports as a size
// mismatch.
//
// The amplitudes are per-case: the condition projection reduces over 12288
// elements, so unit-amplitude inputs there would push the accumulation into a
// range where f32 rounding dwarfs the tolerance.
std::vector<float> run_parity_graph(ggml_backend_t backend,
                                    size_t max_nodes,
                                    const GraphBuilder & build,
                                    float weight_amplitude,
                                    float input_amplitude) {
    const size_t context_size =
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params params = {context_size, nullptr, true};
    ggml_context * context = ggml_init(params);
    ggml_cgraph * graph = ggml_new_graph_custom(context, max_nodes, false);

    const ParityGraph built = build(context);
    ggml_set_name(built.weights, "weights");
    ggml_set_name(built.input, "input");
    ggml_set_name(built.output, "output");
    ggml_set_input(built.weights);
    ggml_set_input(built.input);
    ggml_set_output(built.output);
    ggml_build_forward_expand(graph, built.output);

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocator, graph);
    ggml_gallocr_alloc_graph(allocator, graph);

    const std::vector<float> weight_values = make_values(
        static_cast<size_t>(ggml_nelements(built.weights)), 0.013f, weight_amplitude);
    const std::vector<float> input_values = make_values(
        static_cast<size_t>(ggml_nelements(built.input)), 0.031f, input_amplitude);
    ggml_backend_tensor_set(built.weights, weight_values.data(), 0, ggml_nbytes(built.weights));
    ggml_backend_tensor_set(built.input, input_values.data(), 0, ggml_nbytes(built.input));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> values(static_cast<size_t>(ggml_nelements(built.output)));
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(built.output, values.data(), 0, ggml_nbytes(built.output));
    } else {
        values.clear();
    }

    ggml_gallocr_free(allocator);
    ggml_free(context);
    return values;
}

// Element-wise CPU-vs-Metal comparison. `label` names the case in failure
// output, e.g. "condition projection" or "vocoder convt stride=8".
bool outputs_match(const std::vector<float> & expected,
                   const std::vector<float> & actual,
                   float tolerance,
                   const char * label) {
    if (expected.size() != actual.size() || expected.empty()) {
        std::fprintf(stderr,
                     "FAIL %s output size cpu=%zu metal=%zu\n",
                     label,
                     expected.size(),
                     actual.size());
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        const float difference = std::fabs(expected[i] - actual[i]);
        // Guard the CPU reference too, not just the Metal result: if expected[i]
        // were NaN then difference is NaN, `difference > tolerance` is false, and
        // a broken oracle would silently score as a match.
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
            difference > tolerance) {
            std::fprintf(stderr,
                         "FAIL %s index=%zu cpu=%.8f metal=%.8f difference=%.8f\n",
                         label,
                         i,
                         expected[i],
                         actual[i],
                         difference);
            return false;
        }
    }
    return true;
}

std::vector<float> run_condition_projection(ggml_backend_t backend) {
    return run_parity_graph(
        backend,
        CONDITION_MAX_NODES,
        [](ggml_context * context) {
            ggml_tensor * weights = ggml_new_tensor_3d(
                context, GGML_TYPE_F32, 3, CONDITION_CHANNELS, CONDITION_OUTPUT_CHANNELS);
            ggml_tensor * input =
                ggml_new_tensor_2d(context, GGML_TYPE_F32, CONDITION_FRAMES, CONDITION_CHANNELS);
            return ParityGraph{weights, input, mm3_cond_conv1d(context, weights, nullptr, input, 1)};
        },
        0.01f,
        0.1f);
}

bool verify_condition_projection(ggml_backend_t cpu, ggml_backend_t metal) {
    const std::vector<float> expected = run_condition_projection(cpu);
    const std::vector<float> actual = run_condition_projection(metal);
    if (!outputs_match(expected, actual, CONDITION_ABSOLUTE_TOLERANCE, "condition projection")) {
        return false;
    }
    std::fprintf(stderr, "PASS MiniMax condition projection\n");
    return true;
}

std::vector<float> run_vocoder_convt(ggml_backend_t backend, int stride) {
    return run_parity_graph(
        backend,
        VOCODER_MAX_NODES,
        [stride](ggml_context * context) {
            const int64_t kernel_size = 2 * stride;
            ggml_tensor * weights = ggml_new_tensor_2d(
                context, GGML_TYPE_F32, INPUT_CHANNELS, kernel_size * OUTPUT_CHANNELS);
            ggml_tensor * input =
                ggml_new_tensor_2d(context, GGML_TYPE_F32, INPUT_LENGTH, INPUT_CHANNELS);
            return ParityGraph{
                weights, input,
                mm3_voc_convt(context, weights, nullptr, input, stride, OUTPUT_CHANNELS)};
        },
        1.0f,
        1.0f);
}

bool verify_vocoder_strides(ggml_backend_t cpu, ggml_backend_t metal) {
    constexpr int STRIDES[] = {8, 4, 2};
    for (int stride : STRIDES) {
        char label[64];
        std::snprintf(label, sizeof(label), "vocoder convt stride=%d", stride);
        const std::vector<float> expected = run_vocoder_convt(cpu, stride);
        const std::vector<float> actual = run_vocoder_convt(metal, stride);
        if (!outputs_match(expected, actual, ABSOLUTE_TOLERANCE, label)) return false;
        std::fprintf(stderr, "PASS MiniMax vocoder convt stride=%d\n", stride);
    }
    return true;
}

}

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::fprintf(stderr, "FAIL CPU backend initialization failed\n");
        return 1;
    }
    ggml_backend_t metal = find_metal_backend();
    if (!metal) {
        ggml_backend_free(cpu);
        std::fprintf(stderr, "SKIP Metal backend is unavailable\n");
        return 77;
    }

    if (!metal_supports_mul_mat(metal)) {
        ggml_backend_dev_t device = ggml_backend_get_device(metal);
        const char * description = device ? ggml_backend_dev_description(device) : "unknown device";
        std::fprintf(stderr,
                     "SKIP Metal device cannot run MUL_MAT (%s): the parity graphs are "
                     "matmul-based and ggml gates GGML_OP_MUL_MAT on simdgroup reduction "
                     "(MTLGPUFamilyApple7+). Matmul parity needs a non-virtualized Apple GPU.\n",
                     description);
        ggml_backend_free(metal);
        ggml_backend_free(cpu);
        return 77;
    }

    const bool passed =
        verify_condition_projection(cpu, metal) && verify_vocoder_strides(cpu, metal);
    ggml_backend_free(metal);
    ggml_backend_free(cpu);
    return passed ? 0 : 1;
}
