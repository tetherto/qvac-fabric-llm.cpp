#include "ibv-drv.h"

#ifdef GGML_RPC_RDMA_DLOPEN

#include "ggml-impl.h"

// System headers first: verbs.h includes them again under the hidden pragma below,
// and libc declarations must keep default visibility.
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#include <mutex>

// The shims below take the libibverbs names so verbs.h and transport.cpp need no
// changes. Hidden visibility keeps them inside this library.
#pragma GCC visibility push(hidden)

#include <infiniband/verbs.h>

// Every non-inline libibverbs function the RPC transport calls, including those
// that verbs.h inline wrappers call (ibv_query_port, _ibv_query_gid_ex, ibv_reg_mr).
// ibv_reg_mr_iova2 is the __ibv_reg_mr branch that only -O0 builds keep.
#define IBV_SYMBOLS(X)            \
    X(_ibv_query_gid_ex)          \
    X(ibv_ack_cq_events)          \
    X(ibv_alloc_pd)               \
    X(ibv_close_device)           \
    X(ibv_create_comp_channel)    \
    X(ibv_create_cq)              \
    X(ibv_create_qp)              \
    X(ibv_dealloc_pd)             \
    X(ibv_dereg_mr)               \
    X(ibv_destroy_comp_channel)   \
    X(ibv_destroy_cq)             \
    X(ibv_destroy_qp)             \
    X(ibv_free_device_list)       \
    X(ibv_get_cq_event)           \
    X(ibv_get_device_list)        \
    X(ibv_get_device_name)        \
    X(ibv_modify_qp)              \
    X(ibv_open_device)            \
    X(ibv_query_gid)              \
    X(ibv_query_port)             \
    X(ibv_reg_mr)                 \
    X(ibv_reg_mr_iova2)           \
    X(ibv_wc_status_str)

#define IBV_PFN(name) static decltype(&name) name##_pfn = nullptr;
IBV_SYMBOLS(IBV_PFN)
#undef IBV_PFN

// extern "C" turns a signature that does not match verbs.h into a compile error.
// Names that verbs.h also defines as macros are in parentheses.
extern "C" {

int _ibv_query_gid_ex(struct ibv_context * context, uint32_t port_num, uint32_t gid_index,
                      struct ibv_gid_entry * entry, uint32_t flags, size_t entry_size) {
    return _ibv_query_gid_ex_pfn(context, port_num, gid_index, entry, flags, entry_size);
}

void ibv_ack_cq_events(struct ibv_cq * cq, unsigned int nevents) {
    ibv_ack_cq_events_pfn(cq, nevents);
}

struct ibv_pd * ibv_alloc_pd(struct ibv_context * context) {
    return ibv_alloc_pd_pfn(context);
}

int ibv_close_device(struct ibv_context * context) {
    return ibv_close_device_pfn(context);
}

struct ibv_comp_channel * ibv_create_comp_channel(struct ibv_context * context) {
    return ibv_create_comp_channel_pfn(context);
}

struct ibv_cq * ibv_create_cq(struct ibv_context * context, int cqe, void * cq_context,
                              struct ibv_comp_channel * channel, int comp_vector) {
    return ibv_create_cq_pfn(context, cqe, cq_context, channel, comp_vector);
}

struct ibv_qp * ibv_create_qp(struct ibv_pd * pd, struct ibv_qp_init_attr * qp_init_attr) {
    return ibv_create_qp_pfn(pd, qp_init_attr);
}

int ibv_dealloc_pd(struct ibv_pd * pd) {
    return ibv_dealloc_pd_pfn(pd);
}

int ibv_dereg_mr(struct ibv_mr * mr) {
    return ibv_dereg_mr_pfn(mr);
}

int ibv_destroy_comp_channel(struct ibv_comp_channel * channel) {
    return ibv_destroy_comp_channel_pfn(channel);
}

int ibv_destroy_cq(struct ibv_cq * cq) {
    return ibv_destroy_cq_pfn(cq);
}

int ibv_destroy_qp(struct ibv_qp * qp) {
    return ibv_destroy_qp_pfn(qp);
}

void ibv_free_device_list(struct ibv_device ** list) {
    ibv_free_device_list_pfn(list);
}

int ibv_get_cq_event(struct ibv_comp_channel * channel, struct ibv_cq ** cq, void ** cq_context) {
    return ibv_get_cq_event_pfn(channel, cq, cq_context);
}

struct ibv_device ** ibv_get_device_list(int * num_devices) {
    return ibv_get_device_list_pfn(num_devices);
}

const char * ibv_get_device_name(struct ibv_device * device) {
    return ibv_get_device_name_pfn(device);
}

int ibv_modify_qp(struct ibv_qp * qp, struct ibv_qp_attr * attr, int attr_mask) {
    return ibv_modify_qp_pfn(qp, attr, attr_mask);
}

struct ibv_context * ibv_open_device(struct ibv_device * device) {
    return ibv_open_device_pfn(device);
}

int ibv_query_gid(struct ibv_context * context, uint8_t port_num, int index, union ibv_gid * gid) {
    return ibv_query_gid_pfn(context, port_num, index, gid);
}

int (ibv_query_port)(struct ibv_context * context, uint8_t port_num, struct _compat_ibv_port_attr * port_attr) {
    return ibv_query_port_pfn(context, port_num, port_attr);
}

struct ibv_mr * (ibv_reg_mr)(struct ibv_pd * pd, void * addr, size_t length, int access) {
    return ibv_reg_mr_pfn(pd, addr, length, access);
}

struct ibv_mr * ibv_reg_mr_iova2(struct ibv_pd * pd, void * addr, size_t length, uint64_t iova, unsigned int access) {
    return ibv_reg_mr_iova2_pfn(pd, addr, length, iova, access);
}

const char * ibv_wc_status_str(enum ibv_wc_status status) {
    return ibv_wc_status_str_pfn(status);
}

} // extern "C"

#pragma GCC visibility pop

bool ibvdrv_init() {
    static bool          loaded = false;
    static std::once_flag once;
    std::call_once(once, [] {
        // never closed: connections keep using it until the process exits
        void * lib = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            GGML_LOG_INFO("ggml-rpc: cannot load libibverbs.so.1, RDMA disabled: %s\n", dlerror());
            return;
        }
#define IBV_RESOLVE(name)                                                                     \
        name##_pfn = (decltype(name##_pfn)) dlsym(lib, #name);                                \
        if (name##_pfn == nullptr) {                                                          \
            GGML_LOG_INFO("ggml-rpc: libibverbs.so.1 has no %s, RDMA disabled\n", #name); \
            return;                                                                           \
        }
        IBV_SYMBOLS(IBV_RESOLVE)
#undef IBV_RESOLVE
        loaded = true;
    });
    return loaded;
}

#else

// libibverbs is linked directly, so its functions are always available.
bool ibvdrv_init() {
    return true;
}

#endif // GGML_RPC_RDMA_DLOPEN
