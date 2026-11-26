#pragma once

#include <cstdlib>
#include <cstring>

// True when the named environment variable is set to a non-empty value other
// than "0" (the on/off convention shared by the MM3_* environment variables).
static bool mm3_env_flag_set(const char * name) {
    if (!name) {
        return false;
    }
    const char * value = std::getenv(name);
    return value && value[0] && value[0] != '0';
}

// The LM defaults to flash attention on the GPU backends where its
// teacher-forced logits were measured against the explicit-softmax path and
// the CPU reference (CUDA and Vulkan); every other backend keeps the
// explicit-softmax path unless MM3_LM_FLASH opts it in.
static bool mm3_lm_flash_attn_default(const char * backend_reg_name) {
    return backend_reg_name &&
           (std::strcmp(backend_reg_name, "CUDA") == 0 || std::strcmp(backend_reg_name, "Vulkan") == 0);
}

// Decides whether a graph should build its attention with flash attention.
// Flash attention is GPU-only; `no_flash_env` always forces it off, and
// `on_flash_env` (optional) opts a default-off graph in.
static bool mm3_use_flash_attn(bool has_gpu, bool default_on, const char * no_flash_env,
                                const char * on_flash_env) {
    if (!has_gpu) {
        return false;
    }
    if (mm3_env_flag_set(no_flash_env)) {
        return false;
    }
    return default_on || mm3_env_flag_set(on_flash_env);
}
