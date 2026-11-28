#include "llama-coreml-ffn.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void fill_tensor(ggml_tensor * tensor, void *) {
    std::vector<float> data(ggml_nelements(tensor), strstr(tensor->name, "norm") ? 1.0f : 0.01f);
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}

static void check_context_lifecycle() {
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "qwen35");
    gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
    for (auto field : {std::make_pair("context_length", 128u), {"embedding_length", 32u}, {"block_count", 2u},
            {"feed_forward_length", 64u}, {"vocab_size", 32u}, {"attention.head_count", 4u}, {"attention.head_count_kv", 2u},
            {"attention.key_length", 8u}, {"attention.value_length", 8u}, {"ssm.conv_kernel", 4u}, {"ssm.inner_size", 64u},
            {"ssm.state_size", 16u}, {"ssm.time_step_rank", 4u}, {"ssm.group_count", 2u}, {"full_attention_interval", 2u}}) {
        gguf_set_val_u32(meta, ("qwen35." + std::string(field.first)).c_str(), field.second);
    }
    gguf_set_val_f32(meta, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    const int32_t sections[] = {2, 1, 1, 0};
    gguf_set_arr_data(meta, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
    ggml_context * tensors = ggml_init({ggml_tensor_overhead() * 64, nullptr, true});
    auto add = [&](const std::string & name, int64_t n0, int64_t n1 = 1) {
        auto * tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, n0, n1);
        ggml_set_name(tensor, name.c_str());
        gguf_add_tensor(meta, tensor);
    };
    add("token_embd.weight", 32, 32);
    add("output_norm.weight", 32);
    add("output.weight", 32, 32);
    for (int i = 0; i < 2; ++i) {
        const auto p = "blk." + std::to_string(i) + ".";
        add(p + "attn_norm.weight", 32);
        add(p + "post_attention_norm.weight", 32);
        add(p + "ffn_gate.weight", 32, 64);
        add(p + "ffn_up.weight", 32, 64);
        add(p + "ffn_down.weight", 64, 32);
        if (i == 0) {
            add(p + "attn_qkv.weight", 32, 128);
            add(p + "attn_gate.weight", 32, 64);
            add(p + "ssm_conv1d.weight", 4, 128);
            add(p + "ssm_dt.bias", 4);
            add(p + "ssm_a", 4);
            add(p + "ssm_beta.weight", 32, 4);
            add(p + "ssm_alpha.weight", 32, 4);
            add(p + "ssm_norm.weight", 16);
            add(p + "ssm_out.weight", 64, 32);
        } else {
            add(p + "attn_q.weight", 32, 64);
            add(p + "attn_k.weight", 32, 16);
            add(p + "attn_v.weight", 32, 16);
            add(p + "attn_output.weight", 32, 32);
            add(p + "attn_q_norm.weight", 8);
            add(p + "attn_k_norm.weight", 8);
        }
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_init_from_user(meta, fill_tensor, nullptr, mp);
    gguf_free(meta);
    ggml_free(tensors);
    check(model, "synthetic model load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 32;
    cp.op_offload = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto * ctx = llama_init_from_model(model, cp);
    check(ctx, "synthetic context creation");
    auto params = llama_coreml_ffn_default_params();
    params.cache_dir = ".";
    char error[256];
    for (int iteration = 0; iteration < 3; ++iteration) {
        check(llama_set_coreml_ffn(ctx, nullptr, error, sizeof(error)), "disable must invalidate old graphs safely");
        check(!llama_set_coreml_ffn(ctx, &params, error, sizeof(error)) && error[0], "ineligible configuration rejected");
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_token tokens[] = {1, 2, 3, 4};
        check(llama_decode(ctx, llama_batch_get_one(tokens, 4)) == 0, "failed configuration must preserve ordinary inference");
        check(llama_get_logits_ith(ctx, -1), "ordinary logits after reconfiguration");
    }
    llama_free(ctx);
    llama_model_free(model);
}

int main() {
    try {
        const auto params = llama_coreml_ffn_default_params();
        check(!params.cache_dir && params.min_tokens == 256 && ((params.metal_channels == 8192 && params.n_packages == 2) ||
            (params.metal_channels == 11264 && params.n_packages == 1)), "default parameters");
        char error[64] = {};
        check(!llama_set_coreml_ffn(nullptr, &params, error, sizeof(error)) && error[0], "null context error");
        check(!llama_set_coreml_ffn(nullptr, &params, nullptr, 0), "null error buffer");
        char tiny[1] = {'x'};
        check(!llama_set_coreml_ffn(nullptr, nullptr, tiny, sizeof(tiny)) && tiny[0] == '\0', "bounded error buffer");
        llama_backend_init();
        check_context_lifecycle();
        llama_backend_free();
        printf("Core ML FFN public API contracts passed (available=%d)\n", llama_supports_coreml_ffn());
        return 0;
    } catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
