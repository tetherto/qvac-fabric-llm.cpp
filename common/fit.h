#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
                               bool   prefetch_weights_auto, // enable prefetch when fitting a dense model
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// Automatic acceleration policies, exposed for hardware-independent tests.
bool common_fit_auto_moe_cache(
        const llama_model_params & mparams, const llama_context_params & cparams,
        uint32_t n_expert, size_t n_devices, bool shares_host, bool cache_supported);

bool common_fit_auto_prefetch_weights(
        const llama_context_params & cparams, bool automatic,
        uint32_t n_expert, size_t n_devices, bool shares_host, bool copy_stream);

// Pure decision arithmetic, exposed for tests (tests/test-fit-params.cpp).
// The projected figures are resident demand per row; "shares_host" marks
// devices whose memory is the same physical pool as the host's.

// Deficit (in bytes, >= 0) of the combined host + shared-memory-device budget
// against available host memory. 0 means the combined budget is met.
int64_t common_fit_shared_pool_deficit(
        const std::vector<int64_t> & dev_projected,
        const std::vector<bool>    & shares_host,
                           int64_t   host_free,
                           int64_t   host_projected_resident,
                           int64_t   host_margin);

// Per-device cap for a device that draws from the host pool. The pool budget
// is what stays free after the host's own demand and margin, split evenly
// between the devices that share it. A negative budget is returned unsplit.
int64_t common_fit_shared_pool_target(
                           int64_t   host_free,
                           int64_t   host_projected_resident,
                           int64_t   host_margin,
                            size_t   n_shares_host);

// Context size after the step-2 linear interpolation, guarded against a
// context-independent memory delta (returns n_ctx_min) and clamped to the
// training context. Returns 0 when no reduction can meet the target.
uint32_t common_fit_reduced_n_ctx(
        int64_t  sum_used_target,
        int64_t  sum_projected_used,
        int64_t  sum_projected_used_min_ctx,
        uint32_t hp_nct,
        uint32_t n_ctx_min);

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
