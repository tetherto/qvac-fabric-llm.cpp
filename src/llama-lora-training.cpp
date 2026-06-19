#include "llama-lora-training.h"

#include <cstring>
#include <random>
#include <filesystem>


ggml_context * llama_lora_create_context(size_t mem_size) {
    struct ggml_init_params init_params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(init_params);
}

bool llama_lora_validate_training_params(const struct llama_lora_training_params * params) {
    if (!params) {
        LLAMA_LOG_ERROR("LoRA training validation: params is null\n");
        return false;
    }

    if (params->rank <= 0 || params->rank > 1024) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid rank %d (must be 1-1024)\n", params->rank);
        return false;
    }

    if (params->alpha <= 0.0f) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid alpha %f (must be > 0)\n", params->alpha);
        return false;
    }

    if (params->dropout < 0.0f || params->dropout > 1.0f) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid dropout %f (must be [0, 1])\n", params->dropout);
        return false;
    }

    if (params->init_std <= 0.0f || params->init_std > 1.0f) {
        LLAMA_LOG_ERROR("LoRA training validation: invalid init_std %f (must be (0, 1])\n", params->init_std);
        return false;
    }

    if (params->target_modules == 0) {
        LLAMA_LOG_ERROR("LoRA training validation: no target modules specified\n");
        return false;
    }

    return true;
}

void llama_lora_create_tensor_pair(
        struct ggml_context * lora_ctx,
        const char * base_name,
        const struct ggml_tensor * base_tensor,
        int32_t rank,
        struct ggml_tensor ** lora_a,
        struct ggml_tensor ** lora_b) {

    if (!lora_ctx || !base_name || !base_tensor || !lora_a || !lora_b) {
        throw std::invalid_argument("Invalid null arguments provided to llama_lora_create_tensor_pair");
    }

    // Get base tensor dim
    const int64_t d0 = base_tensor->ne[0]; // input dim
    const int64_t d1 = base_tensor->ne[1]; // output dim
    const int64_t d2 = base_tensor->ne[2]; // expert dim (1 for dense, n_expert for MoE)

    char lora_a_name[256], lora_b_name[256];
    int ret_a = snprintf(lora_a_name, sizeof(lora_a_name), "%s.lora_a", base_name);
    int ret_b = snprintf(lora_b_name, sizeof(lora_b_name), "%s.lora_b", base_name);
    if (ret_a < 0 || ret_a >= (int) sizeof(lora_a_name) ||
        ret_b < 0 || ret_b >= (int) sizeof(lora_b_name)) {
        throw std::runtime_error(std::string("LoRA tensor name too long or formatting failed: ") + base_name);
    }

    if (d2 > 1) {
        // MoE expert tensor: per-expert LoRA pair. build_lora_mm_id() routes
        // through ggml_mul_mat_id with the same routing tensor used by the
        // base expert matmul, so shapes must mirror the base 3D layout.
        *lora_a = ggml_new_tensor_3d(lora_ctx, GGML_TYPE_F32, d0, rank, d2);
        *lora_b = ggml_new_tensor_3d(lora_ctx, GGML_TYPE_F32, rank, d1, d2);
    } else {
        // LoRA A: [d0, rank] - projects input to low rank
        *lora_a = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, d0, rank);
        // LoRA B: [rank, d1] - projects from low rank to output
        *lora_b = ggml_new_tensor_2d(lora_ctx, GGML_TYPE_F32, rank, d1);
    }
    ggml_set_name(*lora_a, lora_a_name);
    ggml_set_name(*lora_b, lora_b_name);
}

static bool is_tensor_on_device(const struct ggml_tensor * tensor) {
    return tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer);
}

static bool tensor_module_is(const std::string & name, const char * module) {
    std::string needle = ".";
    needle += module;
    needle += ".";
    return name.find(needle) != std::string::npos;
}

static bool tensor_is_output(const std::string & name) {
    static const std::string prefix = "output.";
    return name.compare(0, prefix.size(), prefix) == 0;
}

static bool lora_should_target(const std::string & tensor_name, uint32_t target_modules) {
    const bool is_blk = tensor_name.compare(0, 4, "blk.") == 0;

    if (is_blk) {
        if ((target_modules & LLAMA_LORA_TARGET_ATTN_Q)            && tensor_module_is(tensor_name, "attn_q"))           return true;
        if ((target_modules & LLAMA_LORA_TARGET_ATTN_K)            && tensor_module_is(tensor_name, "attn_k"))           return true;
        if ((target_modules & LLAMA_LORA_TARGET_ATTN_V)            && tensor_module_is(tensor_name, "attn_v"))           return true;
        if ((target_modules & LLAMA_LORA_TARGET_ATTN_O)            && tensor_module_is(tensor_name, "attn_output"))      return true;
        if ((target_modules & LLAMA_LORA_TARGET_FFN_GATE)          && tensor_module_is(tensor_name, "ffn_gate"))         return true;
        if ((target_modules & LLAMA_LORA_TARGET_FFN_UP)            && tensor_module_is(tensor_name, "ffn_up"))           return true;
        if ((target_modules & LLAMA_LORA_TARGET_FFN_DOWN)          && tensor_module_is(tensor_name, "ffn_down"))         return true;
        if ((target_modules & LLAMA_LORA_TARGET_FFN_GATE_EXPS)     && tensor_module_is(tensor_name, "ffn_gate_exps"))    return true;
        if ((target_modules & LLAMA_LORA_TARGET_FFN_UP_EXPS)       && tensor_module_is(tensor_name, "ffn_up_exps"))      return true;
        if ((target_modules & LLAMA_LORA_TARGET_FFN_DOWN_EXPS)     && tensor_module_is(tensor_name, "ffn_down_exps"))    return true;
        if ((target_modules & LLAMA_LORA_TARGET_FFN_GATE_UP_EXPS)  && tensor_module_is(tensor_name, "ffn_gate_up_exps")) return true;
    }

    if ((target_modules & LLAMA_LORA_TARGET_OUTPUT) && tensor_is_output(tensor_name)) return true;

    return false;
}

static void init_tensor_guassian(struct ggml_tensor * tensor, float std_dev, uint32_t seed) {
    const size_t n_elements = ggml_nelements(tensor);
    std::vector<float> data(n_elements);

    std::mt19937 gen;
    if (seed != 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::normal_distribution<float> dist(0.0f, std_dev);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dist(gen);
    }

    if (is_tensor_on_device(tensor)) {
        ggml_backend_tensor_set(tensor, data.data(), 0, n_elements * sizeof(float));
    } else {
        std::copy(data.begin(), data.end(), (float *)tensor->data);
    }
}

static void init_tensor_zeros(struct ggml_tensor * tensor) {
    const size_t n_elements = ggml_nelements(tensor);

    if (is_tensor_on_device(tensor)) {
        std::vector<float> zeros(n_elements, 0.0f);
        ggml_backend_tensor_set(tensor, zeros.data(), 0, n_elements * sizeof(float));
    } else {
        std::fill_n((float *)tensor->data, n_elements, 0.0f);
    }
}

static void llama_lora_init_tensor_weights(struct ggml_tensor * lora_a, struct ggml_tensor * lora_b, float init_std, uint32_t seed) {
    if (!lora_a || !lora_b || !lora_a->data || !lora_b->data) {
        throw std::invalid_argument("Invalid null tensors or data pointers passed to llama_lora_init_tensor_weights");
    }

    // LoRA initialization: A ~ N(0, init_std), B = 0
    init_tensor_guassian(lora_a, init_std, seed);
    init_tensor_zeros(lora_b);
}

bool llama_lora_allocate_buffers(
        struct llama_adapter_lora * adapter,
        struct llama_model * model) {

    if (!adapter || !model) {
        return false;
    }

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;

    // LoRA tensors are F32/F16 and need a buffer type that:
    //  1. Supports training ops (OPT_STEP_ADAMW, gradient accumulation)
    //  2. Can handle set_tensor for floating-point types
    // CPU_REPACK buffers crash on F32/F16 (NULL repack traits), and GPU
    // buffers may not support OPT_STEP_ADAMW on all devices. Use a plain
    // host-accessible CPU buffer — the scheduler handles cross-backend
    // copies for forward/backward passes automatically.
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    if (adapter->ctxs.empty()) {
        LLAMA_LOG_ERROR("No contexts found in adapter\n");
        return false;
    }
    ggml_context * lora_ctx = adapter->ctxs[0].get();

    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(lora_ctx, buft) };
    if (!buf) {
        LLAMA_LOG_ERROR("Failed to allocate buffer for LoRA adapter\n");
        return false;
    }
    LLAMA_LOG_INFO("LoRA buffer size = %.2f MiB\n", ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
    adapter->bufs.emplace_back(std::move(buf));

    return true;
}

struct llama_adapter_lora * llama_lora_create_adapter(
        struct llama_model * model,
        const struct llama_lora_training_params * params) {

    llama_adapter_lora * adapter = new llama_adapter_lora(model);
    try {
        adapter->alpha = params->alpha;

        // Pre-pass: count tensors that will get a LoRA pair so we can size
        // the no_alloc ggml context to actual need (each LoRA pair adds two
        // tensor descriptors).
        size_t lora_pair_count = 0;
        for (const auto & tensor_pair : model->tensors_by_name) {
            const std::string & tensor_name = tensor_pair.first;
            struct ggml_tensor * base_tensor = tensor_pair.second;
            if (!base_tensor || base_tensor->ne[1] <= 0) continue;
            if (lora_should_target(tensor_name, params->target_modules)) {
                lora_pair_count++;
            }
        }

        if (lora_pair_count == 0) {
            throw std::runtime_error("No suitable tensors found for LoRA");
        }

        const size_t tensor_overhead = ggml_tensor_overhead();
        const size_t margin          = 64 * 1024;
        const size_t estimated_lora_mem = 2 * lora_pair_count * tensor_overhead + margin;

        ggml_context * lora_ctx = llama_lora_create_context(estimated_lora_mem);
        if (!lora_ctx) {
            throw std::runtime_error("Failed to create LoRA context");
        }

        adapter->ctxs.emplace_back(lora_ctx);
        int created_count = 0;

        for (const auto & tensor_pair : model->tensors_by_name) {
            const std::string & tensor_name = tensor_pair.first;
            struct ggml_tensor * base_tensor = tensor_pair.second;

            if (!base_tensor || base_tensor->ne[1] <= 0) {
                continue;
            }

            if (!lora_should_target(tensor_name, params->target_modules)) {
                continue;
            }

            struct ggml_tensor * lora_a = nullptr;
            struct ggml_tensor * lora_b = nullptr;

            llama_lora_create_tensor_pair(lora_ctx, tensor_name.c_str(), base_tensor, params->rank, &lora_a, &lora_b);
            created_count++;
            adapter->ab_map[tensor_name] = llama_adapter_lora_weight(lora_a, lora_b);
        }

        if (created_count == 0) {
            throw std::runtime_error("No suitable tensors found for LoRA adaptation");
        }

        if (!llama_lora_allocate_buffers(adapter, model)) {
            throw std::runtime_error("Failed to allocate LoRA buffers");
        }

        for (const auto & ab_pair : adapter->ab_map) {
            const llama_adapter_lora_weight & weight = ab_pair.second;

            llama_lora_init_tensor_weights(weight.a, weight.b, params->init_std, params->seed);
        }

        model->loras.insert(adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("Failed to create LoRA adapter: %s\n", err.what());
        delete adapter;
        return nullptr;
    }
}

struct llama_adapter_lora * llama_lora_training_init(
        struct llama_context * ctx,
        struct llama_model * model,
        const struct llama_lora_training_params * params) {

    if (!ctx || !model || !params) {
        LLAMA_LOG_ERROR("LoRA training init: invalid parameters\n");
        return nullptr;
    }

    if (!llama_lora_validate_training_params(params)) {
        return nullptr;
    }

    struct llama_adapter_lora * adapter = llama_lora_create_adapter(model, params);
    if (!adapter) {
        return nullptr;
    }

    float scale = 1.0f;
    if (llama_set_adapters_lora(ctx, &adapter, 1, &scale) < 0) {
        LLAMA_LOG_ERROR("Failed to apply LoRA adapter to context\n");
        delete adapter;
        return nullptr;
    }

    LLAMA_LOG_INFO("LoRA adapter contains %zu tensor pairs and is now registered with context\n", adapter->ab_map.size());

    return adapter;
}

bool llama_opt_param_filter_lora(const struct ggml_tensor * tensor, void * userdata) {
    (void) userdata; // Unused param

    if (!tensor) {
        return false;
    }

    const char * name = tensor->name;

    // Check if tensor is LoRA A or B
    // LoRA tensor naming convention: blk.{layer}.{module}.lora_a or .lora_b
    if (strstr(name, ".lora_a") || strstr(name, ".lora_b")) {
        LLAMA_LOG_DEBUG("LoRA filter: including trainable params '%s'\n", name);
        return true;
    }

    return false;
}

bool llama_lora_save_adapter(
    const struct llama_adapter_lora * adapter,
    const char * filename,
    const struct llama_model * model) {

    if (!adapter || !filename || !model) {
        LLAMA_LOG_ERROR("llama_lora_save_adapter: invalid parameters\n");
        return false;
    }

    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        LLAMA_LOG_ERROR("llama_lora_save_adapter: failed to create GGUF context\n");
        return false;
    }

    std::string arch_name = model->arch_name();
    if (arch_name.empty()) {
        LLAMA_LOG_ERROR("llama_lora_save_adapter: failed to get model architecture\n");
        gguf_free(gguf_ctx);
        return false;
    }

    gguf_set_val_str(gguf_ctx, "general.architecture", arch_name.c_str());
    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "general.name", "LoRA Adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");
    gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", adapter->alpha);

    int tensor_count = 0;
    for (const auto & kv : adapter->ab_map) {
        const auto & lora_weight = kv.second;

        if (lora_weight.a && lora_weight.b) {
            gguf_add_tensor(gguf_ctx, lora_weight.a);
            gguf_add_tensor(gguf_ctx, lora_weight.b);
            tensor_count += 2;
        }
    }

    bool success = gguf_write_to_file(gguf_ctx, filename, false);
    if (success) {
        LLAMA_LOG_INFO("Successfully saved LoRA adapter with %d tensors to: %s\n",
                       tensor_count, filename);
    } else {
        LLAMA_LOG_ERROR("Failed to write LoRA adapter to: %s\n", filename);
    }

    gguf_free(gguf_ctx);
    return success;
}

bool llama_lora_save_checkpoint(
    const struct llama_adapter_lora * adapter,
    const char * checkpoint_path,
    const struct llama_model * model,
    struct llama_context * ctx
) {
    if (!adapter || !checkpoint_path || !model || !ctx) {
        LLAMA_LOG_ERROR("llama_lora_save_checkpoint: invalid parameters\n");
        return false;
    }

    std::filesystem::path checkpoint_dir = std::filesystem::path(checkpoint_path);
    if (!checkpoint_dir.empty()) {
        if (!std::filesystem::exists(checkpoint_dir)) {
            if (!std::filesystem::create_directories(checkpoint_dir)) {
                LLAMA_LOG_ERROR("llama_lora_save_checkpoint: failed to create checkpoint directory: %s\n",
                               checkpoint_dir.string().c_str());
                return false;
            }
        }
    }

    std::filesystem::path model_path = checkpoint_dir / "model.gguf";
    bool lora_saved = llama_lora_save_adapter(adapter, model_path.string().c_str(), model);
    if (!lora_saved) {
        LLAMA_LOG_ERROR("llama_lora_save_checkpoint: failed to save LoRA adapter weights to %s\n",
                        model_path.string().c_str());
        return false;
    }

    std::filesystem::path optimizer_path = checkpoint_dir / "optimizer.gguf";
    bool optimizer_saved = ctx->opt_save_state(optimizer_path.string().c_str());
    if (!optimizer_saved) {
        LLAMA_LOG_ERROR("llama_lora_save_checkpoint: failed to save optimizer state to %s\n",
                        optimizer_path.string().c_str());
        return false;
    }

    return true;
}
