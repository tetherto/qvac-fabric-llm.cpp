#include "ggml-impl.h"
#include "ggml-xdna.h"
#include "ggml-backend-impl.h"

#include "xdna-types.h"
#include "xdna-runtime.h"
#include "xdna-ops.h"

#include <mutex>
#include <string>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <unistd.h>
#endif

// Process-wide context shared by all backend instances. The xdna_device holds
// the name/description, so nothing is duplicated here.
struct ggml_backend_xdna_context {
    xdna_device * device = nullptr;
    xdna_kernel_pool * pool = nullptr;   // lazily scanned kernel pool

    xdna_ops ops;   // operator-specific dispatch (GEMM variants, helpers)
};

static ggml_backend_xdna_context * ggml_xdna_device_context(void) {
    static ggml_backend_xdna_context ctx;
    static std::once_flag once;

    std::call_once(once, [&]() {
        ctx.device = xdna_device_open();
        if (ctx.device) {
            ctx.pool = new xdna_kernel_pool;
            ctx.pool->device = ctx.device;
            xdna_kernel_pool_scan(ctx.pool);
            xdna_ops_init(&ctx.ops, ctx.pool);
            if (ctx.ops.gemm_xclbin_decode.empty() && ctx.ops.gemm_xclbin_prefill.empty()) {
                GGML_LOG_WARN("%s: no GEMM kernels found (build with GGML_XDNA=ON)\n", "ggml-xdna");
            }
        }
    });

    return &ctx;
}

// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XDNA";
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    // The context is a process-wide singleton; it is not freed here.
    GGML_UNUSED(backend);
    delete backend;
}

static bool ggml_xdna_is_view_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;

    if (!ctx->device) {
        return GGML_STATUS_SUCCESS;
    }

    // The scheduler routes ops accepted by supports_op plus the view ops that
    // alias their data; views are no-ops here. Kernels are submitted (started)
    // per op and finalized once at the end, so all runs of the graph overlap.
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (ggml_xdna_is_view_op(node->op)) {
            continue;
        }
        if (!xdna_ops_compute(&ctx->ops, node)) {
            xdna_ops_finalize(&ctx->ops);
            return GGML_STATUS_FAILED;
        }
    }
    if (!xdna_ops_finalize(&ctx->ops)) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i xdna_backend_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x7a, 0xff, 0x0d, 0x7e, 0x5a, 0x1b, 0x4c, 0xf6, 0xbd, 0x94, 0x6e, 0xc1, 0xf1, 0xcb, 0x3f, 0xbd };
    return &guid;
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_backend_xdna_context * ctx = ggml_xdna_device_context();

    // When the NPU is absent the backend is still registered (llama.cpp
    // expects every ACCEL device to yield a backend), but supports_op()
    // rejects everything so all work stays on the CPU.
    if (!ctx->device) {
        GGML_LOG_INFO("%s: XDNA backend init: disabled (no NPU)\n", __func__);
    } else {
        GGML_LOG_INFO("%s: XDNA backend init: device=%s\n", __func__, ctx->device->name.c_str());
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

// device interface

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) dev->context;
    return ctx->device ? ctx->device->name.c_str() : "XDNA";
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) dev->context;
    return ctx->device ? ctx->device->description.c_str() : "AMD XDNA (no NPU device)";
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    // The NPU reads/writes host-visible BOs, so report the host memory the
    // device can consume (fit params divides by the free size, so it must be
    // non-zero for the XDNA device to be used with --fit).
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free  = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = (size_t) pages * page_size;
    // "free" system memory is ill-defined, for practical purposes assume all of it is free:
    *free = *total;
#endif // _WIN32
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_xdna_init();
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) dev->context;

    if (!ctx->device || !ctx->pool) {
        return false;
    }

    return xdna_ops_supported(&ctx->ops, op);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "XDNA";
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return ggml_xdna_device_context()->device ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_UNUSED(index);

    static ggml_backend_device ggml_backend_xdna_device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_xdna_device_context(),
    };

    return &ggml_backend_xdna_device;
}

static void * ggml_backend_xdna_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg ggml_backend_xdna_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xdna_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
