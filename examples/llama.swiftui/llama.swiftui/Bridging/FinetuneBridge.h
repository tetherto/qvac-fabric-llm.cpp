#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes returned by llama_swift_run_lora_finetune
enum llama_swift_finetune_error {
    LLAMA_SWIFT_FINETUNE_OK = 0,
    LLAMA_SWIFT_FINETUNE_ERROR_INVALID_ARGUMENT = 1,
    LLAMA_SWIFT_FINETUNE_ERROR_MODEL_LOAD = 2,
    LLAMA_SWIFT_FINETUNE_ERROR_CONTEXT_CREATE = 3,
    LLAMA_SWIFT_FINETUNE_ERROR_DATASET = 4,
    LLAMA_SWIFT_FINETUNE_ERROR_TRAINING_INIT = 5,
    LLAMA_SWIFT_FINETUNE_ERROR_SAVE = 6,
};

struct llama_swift_finetune_options {
    int32_t  n_ctx;
    int32_t  n_threads;
    int32_t  n_batch;
    int32_t  n_ubatch;
    int32_t  epochs;
    int32_t  lora_rank;
    float    lora_alpha;
    float    learning_rate;
    float    val_split;
    uint32_t target_modules;
    int32_t  seed;
    bool     flash_attn;
    int32_t  n_gpu_layers;

    int32_t     checkpoint_save_steps;
    const char* checkpoint_save_dir;
    const char* resume_from_checkpoint;
    bool        auto_resume;
};

typedef void (*llama_swift_finetune_log_callback)(const char * message, void * user_data);

enum llama_swift_finetune_error llama_swift_run_lora_finetune(
    const char * model_path,
    const char * dataset_path,
    const char * output_adapter_path,
    const struct llama_swift_finetune_options * options,
    llama_swift_finetune_log_callback logger,
    void * user_data);

#ifdef __cplusplus
}
#endif
