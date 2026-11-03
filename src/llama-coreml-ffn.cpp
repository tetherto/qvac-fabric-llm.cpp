#include "llama-coreml-ffn-impl.h"
#include "llama-context.h"

#include <cstdio>
#include <stdexcept>

llama_coreml_ffn_params llama_coreml_ffn_default_params() {
    // Ultra chips have two 16-core Neural Engines and run two programs at once
    if (llama_coreml_ffn_neural_engine_cores() > 16) {
        return {nullptr, 8192, 2, 256};
    }
    return {nullptr, 11264, 1, 256};
}

bool llama_supports_coreml_ffn() {
#ifdef LLAMA_USE_COREML_FFN
    return true;
#else
    return false;
#endif
}

bool llama_set_coreml_ffn(llama_context * ctx, const llama_coreml_ffn_params * params, char * error, size_t error_size) {
    if (error && error_size) { error[0] = '\0'; }
    if (!ctx) {
        if (error && error_size) { snprintf(error, error_size, "null llama context"); }
        return false;
    }
    return ctx->set_coreml_ffn(params, error, error_size);
}

bool llama_context::set_coreml_ffn(const llama_coreml_ffn_params * params, char * error, size_t error_size) {
    try {
        synchronize();
        decltype(coreml_ffn) candidate(nullptr, llama_coreml_ffn_free);
        if (params) {
            if (!llama_supports_coreml_ffn()) {
                throw std::runtime_error("Core ML FFN support was not built (LLAMA_COREML_FFN=OFF)");
            }
            if (!cparams.op_offload || !cparams.offload_kqv || cparams.training || cparams.training_ctx || cparams.cb_eval ||
                    cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || !ctx_compute.expired() || compute_share_source) {
                throw std::runtime_error("Core ML FFN requires a default inference context with Metal offload, its own scheduler and no eval callback");
            }
            candidate.reset(llama_coreml_ffn_load(model, *params));
        }
        // Graphs contain pointers into the old runtime, so invalidate them before replacing it.
        for (auto & graph : gf_res_prev) { if (graph) { graph->reset(); } }
        gf_res_reserve->reset();
        gf_res_prev_active = nullptr;
        ggml_backend_sched_reset(sched.get());
        coreml_ffn = std::move(candidate);
        sched_need_reserve = true;
        return true;
    } catch (const std::exception & failure) {
        if (error && error_size) { snprintf(error, error_size, "%s", failure.what()); }
        return false;
    }
}

#ifndef LLAMA_USE_COREML_FFN
llama_coreml_ffn * llama_coreml_ffn_load(const llama_model &, const llama_coreml_ffn_params &) { return nullptr; }
int llama_coreml_ffn_neural_engine_cores() { return 0; }
void llama_coreml_ffn_free(llama_coreml_ffn *) {}
bool llama_coreml_ffn_check(llama_coreml_ffn *) { return true; }
ggml_tensor * llama_coreml_ffn_build(llama_coreml_ffn *, ggml_context *, ggml_cgraph *,
        ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int) { return nullptr; }
#endif
