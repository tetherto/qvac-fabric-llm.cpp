// Exercise the production allocation/probe helper with public GGML interfaces
// replaced by a metadata-only backend. No model, device, library or payload is
// needed, and an attempted nonzero allocation in measure mode fails the test.
#ifndef PARAKEET_LSTM_CONCAT_BUFFER_HEADER
#define PARAKEET_LSTM_CONCAT_BUFFER_HEADER "lstm_concat_buffer.h"
#endif
#include PARAKEET_LSTM_CONCAT_BUFFER_HEADER

#include <array>
#include <cstdio>
#include <memory>
#include <vector>

struct ggml_backend_buffer_type { bool repack; };
struct ggml_backend_buffer { ggml_backend_buffer_type_t type; };
struct ggml_backend { int reject_layer; };
struct ggml_context { std::vector<std::unique_ptr<ggml_tensor>> tensors; };

namespace {

constexpr size_t kPhysicalBytes = 8192;
constexpr size_t kInitialBytes = 11;
constexpr int64_t kWidth = 64;
constexpr int64_t kRows = 128;
int failures = 0;
int cases = 0;
int zero_allocations = 0;
int payload_allocations = 0;
int live_buffers = 0;
int probes = 0;
bool fail_allocation = false;
bool fail_probe_context = false;

void expect(bool ok, const char * text) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", text);
    ++failures;
}

ggml_tensor * tensor(ggml_context * ctx) {
    ctx->tensors.emplace_back(new ggml_tensor{});
    return ctx->tensors.back().get();
}

ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type) {
    if (fail_allocation) return nullptr;
    ++live_buffers;
    return new ggml_backend_buffer{type};
}

struct Layer { ggml_tensor * w; };

void run_case(bool measure, int reject_layer, bool repack,
              bool allocation_failure = false, bool context_failure = false) {
    expect(live_buffers == 0, "the previous case released every buffer");
    ++cases;
    probes = zero_allocations = payload_allocations = 0;
    fail_allocation = allocation_failure;
    fail_probe_context = context_failure;
    ggml_context ctx;
    std::array<Layer, 2> layers;
    for (auto & layer : layers) {
        layer.w = tensor(&ctx);
        layer.w->type = GGML_TYPE_Q8_0;
        layer.w->ne[0] = kWidth;
        layer.w->ne[1] = kRows;
    }
    ggml_backend_buffer_type buft{repack};
    ggml_backend backend{reject_layer};
    ggml_backend_buffer_t buffer = nullptr;
    size_t bytes = kInitialBytes;
    const bool success = parakeet::detail::prepare_lstm_concat_buffer(
        &ctx, layers, &backend, &buft, buffer, measure ? &bytes : nullptr);
    const bool expected = reject_layer < 0 && repack && !allocation_failure && !context_failure;
    expect(success == expected, "measuring and allocating honor the same capability result");
    expect(zero_allocations == (measure ? 1 : 0), "only measure mode requests a zero-sized buffer");
    expect(payload_allocations == (measure ? 0 : 1), "measure mode never requests payload allocation");
    expect(bytes == kInitialBytes + (measure && expected ? kPhysicalBytes : 0),
           "only an accepted measured stack contributes its physical size");
    if (success) {
        expect(buffer && live_buffers == 1, "successful preparation retains one owned buffer");
        expect(probes == 2, "every stacked layer is probed");
        for (const auto & layer : layers) {
            expect(layer.w->buffer == buffer, "the probe and graph retain the selected buffer kind");
            expect(layer.w->data != nullptr, "accepted tensors are externally allocated for graph sizing");
        }
        ggml_backend_buffer_free(buffer);
    } else {
        expect(buffer == nullptr && live_buffers == 0, "refusal or probe failure releases the buffer");
        for (const auto & layer : layers) {
            expect(layer.w->buffer == nullptr && layer.w->data == nullptr,
                   "a rejected stack leaves no dangling allocation marker");
        }
    }
    fail_allocation = fail_probe_context = false;
}

}  // namespace

size_t ggml_tensor_overhead(void) { return sizeof(ggml_tensor); }
ggml_context * ggml_init(ggml_init_params params) {
    expect(params.no_alloc, "the support probe creates metadata only");
    return fail_probe_context ? nullptr : new ggml_context;
}
void ggml_free(ggml_context * ctx) { delete ctx; }
ggml_tensor * ggml_get_first_tensor(const ggml_context * ctx) {
    return ctx->tensors.empty() ? nullptr : ctx->tensors.front().get();
}
ggml_tensor * ggml_get_next_tensor(const ggml_context * ctx, ggml_tensor * current) {
    for (size_t i = 0; i + 1 < ctx->tensors.size(); ++i) {
        if (ctx->tensors[i].get() == current) return ctx->tensors[i + 1].get();
    }
    return nullptr;
}
ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t width) {
    ggml_tensor * t = tensor(ctx);
    t->type = type;
    t->ne[0] = width;
    return t;
}
ggml_tensor * ggml_mul_mat(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * t = tensor(ctx);
    t->op = GGML_OP_MUL_MAT;
    t->src[0] = w;
    t->src[1] = x;
    return t;
}
bool ggml_backend_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    const int layer = probes++;
    expect(op->op == GGML_OP_MUL_MAT && op->src[1]->ne[0] == kWidth,
           "the probe uses the actual stacked-GEMV dimensions");
    return layer != backend->reject_layer && op->src[0]->buffer &&
           op->src[0]->buffer->type->repack;
}
size_t ggml_backend_alloc_ctx_tensors_from_buft_size(ggml_context * ctx,
                                                   ggml_backend_buffer_type_t) {
    for (const auto & t : ctx->tensors) {
        expect(t->data == nullptr, "stack payload is not allocated or read before sizing");
    }
    return kPhysicalBytes;
}
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t type, size_t bytes) {
    expect(bytes == 0, "metadata buffer creation must request zero bytes");
    ++zero_allocations;
    return allocate(type);
}
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ggml_context * ctx,
                                                             ggml_backend_buffer_type_t type) {
    ++payload_allocations;
    ggml_backend_buffer_t buffer = allocate(type);
    if (buffer) {
        for (const auto & t : ctx->tensors) {
            t->buffer = buffer;
            t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(128));
        }
    }
    return buffer;
}
void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (!buffer) return;
    expect(live_buffers > 0, "a buffer is released at most once");
    --live_buffers;
    delete buffer;
}

int main() {
    for (bool measure : {false, true}) {
        run_case(measure, -1, true);
        run_case(measure, 0, true);
        run_case(measure, 1, true);
        run_case(measure, -1, false);
        run_case(measure, -1, true, true);
        run_case(measure, -1, true, false, true);
    }
    expect(live_buffers == 0, "all cases finish with no live buffers");
    if (failures) {
        std::fprintf(stderr, "test-lstm-concat-buffer: %d failures in %d cases\n", failures, cases);
        return 1;
    }
    std::printf("test-lstm-concat-buffer: %d cases PASS\n", cases);
    return 0;
}
