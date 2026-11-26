// Metadata-only allocation contracts using a fake public backend API.
// Includes the same chunking code used by the real Audio8 loader.
#include "audio8/repack_weights.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

struct ggml_context { std::vector<ggml_tensor *> tensors; };
struct ggml_backend_buffer_type {
    size_t alignment = 128;
    size_t maximum = 4096;
    size_t guard = 4096;
    bool overflow = false;
    std::vector<size_t> requests;
};
static unsigned payload_allocations = 0;

ggml_tensor * ggml_get_first_tensor(const ggml_context * ctx) {
    return ctx->tensors.empty() ? nullptr : ctx->tensors.front();
}
ggml_tensor * ggml_get_next_tensor(const ggml_context * ctx, ggml_tensor * tensor) {
    auto found = std::find(ctx->tensors.begin(), ctx->tensors.end(), tensor);
    return found != ctx->tensors.end() && ++found != ctx->tensors.end() ? *found : nullptr;
}
int ggml_n_dims(const ggml_tensor * tensor) {
    for (int i = 3; i >= 1; --i) if (tensor->ne[i] != 1) return i + 1;
    return 1;
}
const char * ggml_get_name(const ggml_tensor * tensor) { return tensor->name; }
size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t type) { return type->alignment; }
size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t type) { return type->maximum; }
size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return size_t(tensor->ne[0] * tensor->ne[1]) / 32 * 34;
}
size_t ggml_backend_buft_get_buffer_alloc_size(ggml_backend_buffer_type_t type, size_t size) {
    type->requests.push_back(size);
    return type->overflow ? SIZE_MAX : size + type->guard;
}
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t, size_t) {
    ++payload_allocations;
    return nullptr;
}
void ggml_backend_buffer_set_usage(ggml_backend_buffer_t, enum ggml_backend_buffer_usage) {}
ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t) { return {}; }
enum ggml_status ggml_tallocr_alloc(ggml_tallocr *, ggml_tensor *) { return GGML_STATUS_FAILED; }

using tts_cpp::audio8::detail::repack_weights::allocate_projection_weights;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("line " + std::to_string(__LINE__) + ": " #x); } while (0)

struct fixture {
    ggml_tensor a = {}, b = {}, canonical = {};
    ggml_context ctx;
    ggml_backend_buffer_type type;
    std::vector<ggml_backend_buffer_t> buffers;
    std::string error;
    fixture() {
        for (auto * t : {&a, &b, &canonical}) {
            t->type = GGML_TYPE_Q8_0;
            t->ne[0] = t->ne[1] = 32;
            t->ne[2] = t->ne[3] = 1;
        }
        std::strcpy(a.name, "lm/blk/0/wqkv");
        std::strcpy(b.name, "fast/blk/0/wo");
        std::strcpy(canonical.name, "lm/sem_head");
        ctx.tensors = {&a, &canonical, &b};
    }
    bool measure(size_t & bytes) {
        return allocate_projection_weights(&ctx, &type, buffers, &bytes, &error);
    }
};

int main() {
    try {
        {
            fixture f;
            size_t bytes = 17;
            CHECK(f.measure(bytes));
            CHECK(bytes == 17 + 2304 + 4096);
            CHECK(f.type.requests == std::vector<size_t>{2304});
            CHECK(f.a.data && f.b.data && !f.canonical.data);
            CHECK(f.buffers.empty() && payload_allocations == 0);
        }
        {
            fixture f;
            f.type.maximum = 1280; // two 1152-byte payloads require two chunks
            size_t bytes = 0;
            CHECK(f.measure(bytes));
            CHECK(bytes == 2 * (1152 + 4096));
            CHECK(f.type.requests == (std::vector<size_t>{1152, 1152}));
        }
        {
            fixture f;
            f.type.guard = 0; // providers without an extra physical guard
            size_t bytes = 0;
            CHECK(f.measure(bytes) && bytes == 2304);
        }
        {
            fixture f;
            f.type.overflow = true;
            size_t bytes = 0;
            CHECK(!f.measure(bytes) && bytes == 0);
            CHECK(!f.a.data && !f.b.data && !f.error.empty());
        }
        {
            fixture f;
            size_t bytes = SIZE_MAX - 100;
            CHECK(!f.measure(bytes) && bytes == SIZE_MAX - 100);
        }
        {
            fixture f;
            f.ctx.tensors = {&f.canonical};
            size_t bytes = 23;
            CHECK(f.measure(bytes) && bytes == 23 && f.type.requests.empty());
            CHECK(!f.canonical.data);
        }
        CHECK(payload_allocations == 0);
        std::puts("Audio8 REPACK metadata: 6 guarded chunk/overflow contracts passed; no payload allocation");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
