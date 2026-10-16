// Regression test: backends that keep per-tensor state in tensor->extra (e.g. OpenCL, whose
// init_tensor creates the extra and whose set_tensor replaces it when it re-lays out weights)
// must work behind an RPC server. The server deserializes a fresh ggml_tensor for every command,
// so it has to keep those extras itself.
//
// A fake device wraps the CPU backend but behaves like OpenCL with respect to extras; it is served
// by an in-process RPC server and driven through the regular RPC client.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#    include <unistd.h>
#endif

static std::mutex               g_errors_mutex;
static std::vector<std::string> g_errors;

static void record_error(const char * where, const ggml_tensor * t, const char * what) {
    std::lock_guard<std::mutex> lock(g_errors_mutex);
    g_errors.push_back(std::string(where) + " '" + t->name + "': " + what);
}

// ---- fake buffer: extras like the OpenCL backend ----

struct fake_extra {
    size_t offset; // byte offset of the (parent) tensor in the buffer
};

struct fake_buffer_ctx {
    ggml_backend_buffer_t          inner;    // CPU buffer holding the data
    std::deque<fake_extra>         extras;   // owned extras (stable addresses)
    std::map<size_t, fake_extra *> current;  // latest extra per offset (set_tensor replaces it)
};

static size_t tensor_offset(ggml_backend_buffer_t buffer, const ggml_tensor * t) {
    return (const char *) t->data - (const char *) ggml_backend_buffer_get_base(buffer);
}

// Mirrors how a backend like OpenCL finds the device memory for a tensor: only through its extra.
static bool check_extra(ggml_backend_buffer_t buffer, const ggml_tensor * t, const char * where) {
    auto * ctx   = (fake_buffer_ctx *) buffer->context;
    auto * extra = (fake_extra *) t->extra;
    if (extra == nullptr) {
        record_error(where, t, "missing extra");
        return false;
    }
    const size_t view_offs = t->view_src ? t->view_offs : 0;
    if (extra->offset + view_offs != tensor_offset(buffer, t)) {
        record_error(where, t, "extra does not match the tensor offset");
        return false;
    }
    auto it = ctx->current.find(extra->offset);
    if (it == ctx->current.end() || it->second != extra) {
        record_error(where, t, "stale extra (not the one set_tensor left behind)");
        return false;
    }
    return true;
}

static void fake_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (fake_buffer_ctx *) buffer->context;
    ggml_backend_buffer_free(ctx->inner);
    delete ctx;
}

static void * fake_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_get_base(((fake_buffer_ctx *) buffer->context)->inner);
}

static ggml_status fake_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t) {
    auto * ctx = (fake_buffer_ctx *) buffer->context;
    if (t->view_src != nullptr) {
        t->extra = t->view_src->extra; // views share the parent's extra, as in OpenCL
        return GGML_STATUS_SUCCESS;
    }
    const size_t offset = tensor_offset(buffer, t);
    ctx->extras.push_back({ offset });
    ctx->current[offset] = &ctx->extras.back();
    t->extra             = &ctx->extras.back();
    return GGML_STATUS_SUCCESS;
}

static void fake_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    if (!check_extra(buffer, t, "set_tensor")) {
        return;
    }
    memcpy((char *) t->data + offset, data, size);
    // Whole-tensor uploads get a new extra, like OpenCL's re-laid-out quantized weights.
    if (t->view_src == nullptr && offset == 0 && size == ggml_nbytes(t)) {
        auto * ctx = (fake_buffer_ctx *) buffer->context;
        ctx->extras.push_back({ tensor_offset(buffer, t) });
        ctx->current[tensor_offset(buffer, t)] = &ctx->extras.back();
        t->extra = &ctx->extras.back();
    }
}

static void fake_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    if (!check_extra(buffer, t, "get_tensor")) {
        memset(data, 0, size);
        return;
    }
    memcpy(data, (const char *) t->data + offset, size);
}

static void fake_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    if (check_extra(buffer, t, "memset_tensor")) {
        memset((char *) t->data + offset, value, size);
    }
}

static void fake_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_buffer_clear(((fake_buffer_ctx *) buffer->context)->inner, value);
}

static const ggml_backend_buffer_i fake_buffer_iface = {
    /* .free_buffer   = */ fake_buffer_free,
    /* .get_base      = */ fake_buffer_get_base,
    /* .init_tensor   = */ fake_buffer_init_tensor,
    /* .memset_tensor = */ fake_buffer_memset_tensor,
    /* .set_tensor    = */ fake_buffer_set_tensor,
    /* .get_tensor    = */ fake_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ fake_buffer_clear,
    /* .reset         = */ nullptr,
};

static const char * fake_buft_get_name(ggml_backend_buffer_type_t) { return "FakeExtra"; }

static ggml_backend_buffer_t fake_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx  = new fake_buffer_ctx;
    ctx->inner  = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    return ggml_backend_buffer_init(buft, fake_buffer_iface, ctx, size);
}

static size_t fake_buft_get_alignment(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

static bool fake_buft_is_host(ggml_backend_buffer_type_t) { return false; }

static ggml_backend_buffer_type fake_buft = {
    /* .iface = */ {
        /* .get_name       = */ fake_buft_get_name,
        /* .alloc_buffer   = */ fake_buft_alloc_buffer,
        /* .get_alignment  = */ fake_buft_get_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ fake_buft_is_host,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

// ---- fake backend: checks every tensor's extra, then computes on the CPU ----

static const char * fake_backend_get_name(ggml_backend_t) { return "FakeExtra"; }

static void fake_backend_free(ggml_backend_t backend) {
    ggml_backend_free((ggml_backend_t) backend->context);
    delete backend;
}

static ggml_status fake_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->buffer && node->buffer->buft == &fake_buft) {
            check_extra(node->buffer, node, "graph_compute node");
        }
        for (int j = 0; j < GGML_MAX_SRC && node->src[j]; j++) {
            ggml_tensor * src = node->src[j];
            if (src->buffer && src->buffer->buft == &fake_buft) {
                check_extra(src->buffer, src, "graph_compute src");
            }
        }
    }
    return ggml_backend_graph_compute((ggml_backend_t) backend->context, graph);
}

static ggml_guid_t fake_guid() {
    static ggml_guid guid = { 0x7e, 0x57, 0xe7, 0x7a, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb };
    return &guid;
}

static ggml_backend_device fake_device;

static ggml_backend_t fake_dev_init_backend(ggml_backend_dev_t dev, const char *) {
    ggml_backend_i iface {};
    iface.get_name      = fake_backend_get_name;
    iface.free          = fake_backend_free;
    iface.graph_compute = fake_backend_graph_compute;
    return new ggml_backend { fake_guid(), iface, dev, ggml_backend_cpu_init() };
}

static const char * fake_dev_get_name(ggml_backend_dev_t) { return "FakeExtra0"; }
static const char * fake_dev_get_description(ggml_backend_dev_t) { return "CPU-backed device with OpenCL-style tensor extras"; }
static void fake_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 1ull << 30; }
static enum ggml_backend_dev_type fake_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }

static void fake_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    props->name        = fake_dev_get_name(dev);
    props->description = fake_dev_get_description(dev);
    props->type        = fake_dev_get_type(dev);
    fake_dev_get_memory(dev, &props->memory_free, &props->memory_total);
}

static ggml_backend_buffer_type_t fake_dev_get_buffer_type(ggml_backend_dev_t) { return &fake_buft; }
static bool fake_dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool fake_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &fake_buft; }

// ---- test ----

static bool run(const char * endpoint) {
    constexpr int n = 64;

    ggml_backend_t rpc = ggml_backend_rpc_init(endpoint, 0);
    if (rpc == nullptr) {
        fprintf(stderr, "cannot connect to %s\n", endpoint);
        return false;
    }

    ggml_init_params params = { /* .mem_size = */ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), /* .mem_buffer = */ nullptr, /* .no_alloc = */ true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_set_name(c, "c");
    ggml_tensor * v = ggml_view_1d(ctx, c, n / 2, (n / 2) * sizeof(float)); // second half of c
    ggml_set_name(v, "v");
    ggml_tensor * e = ggml_add(ctx, v, v);
    ggml_set_name(e, "e");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(rpc));
    bool ok = buf != nullptr;

    // Two rounds: the second upload replaces the extras created by the first one.
    for (int round = 0; ok && round < 2; round++) {
        std::vector<float> ha(n), hb(n), he(n / 2, 0.0f);
        for (int i = 0; i < n; i++) {
            ha[i] = (float) (i + round);
            hb[i] = (float) (10 * i);
        }
        ggml_backend_tensor_set(a, ha.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, hb.data(), 0, ggml_nbytes(b));
        ok = ok && ggml_backend_graph_compute(rpc, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(e, he.data(), 0, ggml_nbytes(e));
        for (int i = 0; ok && i < n / 2; i++) {
            const int   k        = n / 2 + i;
            const float expected = 2.0f * (ha[k] + hb[k]);
            if (he[i] != expected) {
                fprintf(stderr, "round %d: e[%d] = %f, expected %f\n", round, i, he[i], expected);
                ok = false;
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(rpc);
    return ok;
}

int main() {
    fake_device.iface                 = {};
    fake_device.iface.get_name        = fake_dev_get_name;
    fake_device.iface.get_description = fake_dev_get_description;
    fake_device.iface.get_memory      = fake_dev_get_memory;
    fake_device.iface.get_type        = fake_dev_get_type;
    fake_device.iface.get_props       = fake_dev_get_props;
    fake_device.iface.init_backend    = fake_dev_init_backend;
    fake_device.iface.get_buffer_type = fake_dev_get_buffer_type;
    fake_device.iface.supports_op     = fake_dev_supports_op;
    fake_device.iface.supports_buft   = fake_dev_supports_buft;
    fake_buft.device                  = &fake_device;

#ifndef _WIN32
    const int port = 30000 + (int) (getpid() % 20000);
#else
    const int port = 47321;
#endif
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);

    ggml_backend_dev_t        dev    = &fake_device;
    ggml_backend_rpc_server_t server = ggml_backend_rpc_server_create(endpoint.c_str(), nullptr, 1, 1, &dev);
    if (server == nullptr) {
        fprintf(stderr, "failed to start RPC server on %s\n", endpoint.c_str());
        return 1;
    }
    std::thread server_thread([server] { ggml_backend_rpc_server_run(server); });

    const bool ok = run(endpoint.c_str());

    ggml_backend_rpc_server_stop(server);
    server_thread.join();
    ggml_backend_rpc_server_free(server);

    for (const auto & err : g_errors) {
        fprintf(stderr, "error: %s\n", err.c_str());
    }
    if (!ok || !g_errors.empty()) {
        fprintf(stderr, "FAILED\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
