#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr int64_t D       = 128;
constexpr int64_t H       = 48;
constexpr int64_t H_K     = 16;
constexpr int     REPLAYS = 4;

struct inputs {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> state;
};

struct outputs {
    std::vector<float> full_attn;
    std::vector<float> full_state;
    std::vector<float> split_attn;
    std::vector<float> split_state;
    bool               cute_allocated = false;
};

static ggml_backend_t cuda_backend() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg != nullptr && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

static inputs make_inputs(int64_t tokens) {
    inputs data;
    data.q.resize(D * H_K * tokens);
    data.k.resize(D * H_K * tokens);
    data.v.resize(D * H * tokens);
    data.g.resize(H * tokens);
    data.beta.resize(H * tokens);
    data.state.resize(D * D * H);

    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < H_K; ++h) {
            double q_norm = 0.0;
            double k_norm = 0.0;
            for (int64_t d = 0; d < D; ++d) {
                const size_t i = (t * H_K + h) * D + d;
                data.q[i]      = sinf((float) (i + 1) * 0.013f);
                data.k[i]      = cosf((float) (i + 3) * 0.017f);
                q_norm += data.q[i] * data.q[i];
                k_norm += data.k[i] * data.k[i];
            }
            q_norm = sqrt(q_norm);
            k_norm = sqrt(k_norm);
            for (int64_t d = 0; d < D; ++d) {
                const size_t i = (t * H_K + h) * D + d;
                data.q[i] /= (float) q_norm;
                data.k[i] /= (float) k_norm;
            }
        }
    }
    for (size_t i = 0; i < data.v.size(); ++i) {
        data.v[i] = 0.25f * sinf((float) (i + 5) * 0.007f);
    }
    for (size_t i = 0; i < data.g.size(); ++i) {
        data.g[i]    = -0.01f - 0.02f * (float) (i % 7) / 6.0f;
        data.beta[i] = 0.2f + 0.6f * (float) (i % 11) / 10.0f;
    }
    for (size_t i = 0; i < data.state.size(); ++i) {
        data.state[i] = 0.01f * cosf((float) (i + 7) * 0.003f);
    }
    return data;
}

static ggml_tensor * token_view(
        ggml_context * ctx, ggml_tensor * tensor, int64_t heads, int64_t begin, int64_t count) {
    return ggml_view_4d(ctx, tensor, D, heads, count, 1, tensor->nb[1], tensor->nb[2], tensor->nb[3],
                        begin * tensor->nb[2]);
}

static ggml_tensor * scalar_token_view(
        ggml_context * ctx, ggml_tensor * tensor, int64_t begin, int64_t count) {
    return ggml_view_4d(ctx, tensor, 1, H, count, 1, tensor->nb[1], tensor->nb[2], tensor->nb[3],
                        begin * tensor->nb[2]);
}

static bool run_graph(
        ggml_backend_t backend, const inputs & data, int64_t tokens, int64_t split, int repeats, outputs & result) {
    ggml_init_params params = {
        64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false),
        nullptr,
        true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * q     = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, D, H_K, tokens, 1);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, D, H_K, tokens, 1);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, D, H, tokens, 1);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, tokens, 1);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, tokens, 1);
    ggml_tensor * state = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, D, D, H, 1);

    ggml_tensor * full = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state, 1);

    ggml_tensor * first = ggml_gated_delta_net(
        ctx.get(), token_view(ctx.get(), q, H_K, 0, split), token_view(ctx.get(), k, H_K, 0, split),
        token_view(ctx.get(), v, H, 0, split), scalar_token_view(ctx.get(), g, 0, split),
        scalar_token_view(ctx.get(), beta, 0, split), state, 1);

    const size_t  split_attn_bytes = D * H * split * sizeof(float);
    ggml_tensor * middle_state = ggml_view_4d(ctx.get(), first, D, D, H, 1, D * sizeof(float), D * D * sizeof(float),
                                              D * D * H * sizeof(float), split_attn_bytes);

    const int64_t remaining = tokens - split;
    ggml_tensor * second = ggml_gated_delta_net(
        ctx.get(), token_view(ctx.get(), q, H_K, split, remaining), token_view(ctx.get(), k, H_K, split, remaining),
        token_view(ctx.get(), v, H, split, remaining), scalar_token_view(ctx.get(), g, split, remaining),
        scalar_token_view(ctx.get(), beta, split, remaining), middle_state, 1);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, full);
    ggml_build_forward_expand(graph, second);

    if (!ggml_backend_supports_op(backend, full) || !ggml_backend_supports_op(backend, second)) {
        return false;
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return false;
    }

    ggml_backend_tensor_set(q, data.q.data(), 0, data.q.size() * sizeof(float));
    ggml_backend_tensor_set(k, data.k.data(), 0, data.k.size() * sizeof(float));
    ggml_backend_tensor_set(v, data.v.data(), 0, data.v.size() * sizeof(float));
    ggml_backend_tensor_set(g, data.g.data(), 0, data.g.size() * sizeof(float));
    ggml_backend_tensor_set(beta, data.beta.data(), 0, data.beta.size() * sizeof(float));
    ggml_backend_tensor_set(state, data.state.data(), 0, data.state.size() * sizeof(float));

    result.cute_allocated = ggml_backend_buffer_get_alloc_size(full->buffer, full) > ggml_nbytes(full);
    std::vector<float> previous_full(ggml_nelements(full));
    std::vector<float> previous_first(ggml_nelements(first));
    std::vector<float> previous_second(ggml_nelements(second));
    for (int i = 0; i < repeats; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_synchronize(backend);
        for (const auto & item : { std::make_pair(full, &previous_full),
                                  std::make_pair(first, &previous_first),
                                  std::make_pair(second, &previous_second) }) {
            std::vector<float> current(ggml_nelements(item.first));
            ggml_backend_tensor_get(item.first, current.data(), 0, ggml_nbytes(item.first));
            if (i > 0 && current != *item.second) {
                fprintf(stderr, "GDN output changed on replay %d\n", i);
                return false;
            }
            *item.second = std::move(current);
        }
    }

    const size_t full_attn_elems = D * H * tokens;
    const size_t state_elems     = D * D * H;
    result.full_attn.resize(full_attn_elems);
    result.full_state.resize(state_elems);
    result.split_attn.resize(full_attn_elems);
    result.split_state.resize(state_elems);

    ggml_backend_tensor_get(full, result.full_attn.data(), 0, full_attn_elems * sizeof(float));
    ggml_backend_tensor_get(full, result.full_state.data(), full_attn_elems * sizeof(float),
                            state_elems * sizeof(float));
    const size_t second_attn_bytes = D * H * remaining * sizeof(float);
    ggml_backend_tensor_get(first, result.split_attn.data(), 0, split_attn_bytes);
    ggml_backend_tensor_get(second, result.split_attn.data() + D * H * split, 0, second_attn_bytes);
    ggml_backend_tensor_get(second, result.split_state.data(), second_attn_bytes, state_elems * sizeof(float));
    return true;
}

static double nmse(const std::vector<float> & reference, const std::vector<float> & actual) {
    double error = 0.0;
    double norm  = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double delta = (double) reference[i] - actual[i];
        error += delta * delta;
        norm += (double) reference[i] * reference[i];
    }
    return norm == 0.0 ? error : error / norm;
}

static int child_main(int64_t tokens, int64_t split) {
    ggml_backend_load_all();
    ggml_backend_ptr cuda(cuda_backend());
    if (!cuda) {
        printf("RESULT skip=no_cuda\n");
        return 0;
    }
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!cpu) {
        return 1;
    }

    const inputs data = make_inputs(tokens);
    outputs      reference;
    outputs      actual;
    if (!run_graph(cpu.get(), data, tokens, split, 1, reference) ||
        !run_graph(cuda.get(), data, tokens, split, REPLAYS, actual)) {
        return 1;
    }

    const double attn_error               = nmse(reference.full_attn, actual.full_attn);
    const double state_error              = nmse(reference.full_state, actual.full_state);
    const double continuation_attn_error  = nmse(actual.full_attn, actual.split_attn);
    const double continuation_state_error = nmse(actual.full_state, actual.split_state);
    const double tolerance = actual.cute_allocated ? 3e-4 : 1e-7;
    const bool   ok = attn_error <= tolerance && state_error <= tolerance &&
                      continuation_attn_error <= tolerance && continuation_state_error <= tolerance;

    printf("RESULT ok=%d cute=%d tokens=%lld split=%lld attn=%.9e state=%.9e split_attn=%.9e split_state=%.9e\n",
           ok ? 1 : 0, actual.cute_allocated ? 1 : 0, (long long) tokens, (long long) split, attn_error, state_error,
           continuation_attn_error, continuation_state_error);
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "--child") == 0) {
        const int64_t tokens = atoll(argv[2]);
        const int64_t split  = atoll(argv[3]);
        if (tokens <= 1 || split <= 0 || split >= tokens) {
            return 1;
        }
        return child_main(tokens, split);
    }
    for (int64_t tokens : { 63, 64, 65, 127, 128, 129, 512, 1024, 2048 }) {
        if (child_main(tokens, tokens / 2) != 0) {
            return 1;
        }
    }
    return 0;
}
