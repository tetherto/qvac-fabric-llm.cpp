#pragma once

// Per-tensor quantization policy for the ACE-Step stage GGUFs, ported from
// tools/quantize.cpp in acestep.cpp (MIT) so the engine's own tool reproduces
// the layouts it was validated against. The policy mirrors llama-quantize:
// important tensors (v_proj, down_proj, and o_proj for L variants) get bumped
// in S/M variants, embed_tokens is held at Q6_K, 1-D norms and biases are
// promoted to F32, and the VAE plus a deny-list of quality-critical tensors
// are never quantized.
//
// Pure functions over tensor names and dimensions, with no ggml context or
// GGUF handle, so the policy is regression tested in
// test/test_acestep_units.cpp rather than only observed on converted files.

#include <cctype>
#include <cstdlib>
#include <cstring>

#include "ggml.h"

namespace tts_cpp::acestep {

struct QuantVariant {
    const char *    name;
    enum ggml_type  base;
    enum ggml_type  bump;   // type for important tensors (COUNT = no bump)
    enum ggml_type  embed;  // type for embed_tokens (COUNT = same as base)
    // 0 = none, 1 = first bump_layer_count layers, 2 = first + last + every
    // 3rd layer (M variants), 3 = every important tensor (L variants)
    int             bump_mode;
    int             bump_layer_count;
    enum ggml_ftype ftype;  // written to general.file_type (u32, GGUF convention)
};

inline const QuantVariant * quant_variants(size_t & count) {
    static const QuantVariant variants[] = {
        { "Q2_K",   GGML_TYPE_Q2_K, GGML_TYPE_Q4_K,  GGML_TYPE_Q6_K, 1, 4, GGML_FTYPE_MOSTLY_Q2_K },
        { "Q3_K_S", GGML_TYPE_Q3_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, GGML_FTYPE_MOSTLY_Q3_K },
        { "Q3_K_M", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 2, 0, GGML_FTYPE_MOSTLY_Q3_K },
        { "Q3_K_L", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 3, 0, GGML_FTYPE_MOSTLY_Q3_K },
        { "Q4_K_S", GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 1, 4, GGML_FTYPE_MOSTLY_Q4_K },
        { "Q4_K_M", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0, GGML_FTYPE_MOSTLY_Q4_K },
        { "Q5_K_S", GGML_TYPE_Q5_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, GGML_FTYPE_MOSTLY_Q5_K },
        { "Q5_K_M", GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0, GGML_FTYPE_MOSTLY_Q5_K },
        { "Q6_K",   GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0, GGML_FTYPE_MOSTLY_Q6_K },
        { "Q8_0",   GGML_TYPE_Q8_0, GGML_TYPE_COUNT, GGML_TYPE_Q8_0, 0, 0, GGML_FTYPE_MOSTLY_Q8_0 },
    };
    count = sizeof(variants) / sizeof(variants[0]);
    return variants;
}

inline const QuantVariant * find_quant_variant(const char * name) {
    size_t               count    = 0;
    const QuantVariant * variants = quant_variants(count);
    for (size_t i = 0; i < count; ++i) {
        bool match = true;
        for (const char *a = name, *b = variants[i].name;; ++a, ++b) {
            const char ca = (char) std::toupper((unsigned char) *a);
            if (ca != *b) {
                match = false;
                break;
            }
            if (*a == '\0') {
                break;
            }
        }
        if (match) {
            return &variants[i];
        }
    }
    return nullptr;
}

// HF tensor name "model.layers.N.xxx" -> N, else -1.
inline int quant_layer_index(const char * name) {
    const char * p = std::strstr(name, "layers.");
    if (!p) {
        return -1;
    }
    return std::atoi(p + 7);
}

inline bool quant_important_sm(const char * name) {
    return std::strstr(name, "v_proj.weight") != nullptr || std::strstr(name, "down_proj.weight") != nullptr;
}

inline bool quant_important_l(const char * name) {
    return quant_important_sm(name) || std::strstr(name, "o_proj.weight") != nullptr;
}

inline bool quant_is_embed(const char * name) {
    return std::strstr(name, "embed_tokens.weight") != nullptr ||
           std::strstr(name, "token_embd.weight") != nullptr;
}

inline bool quant_is_untied_output(const char * name, const char * arch) {
    return std::strcmp(arch, "qwen3") == 0 && std::strcmp(name, "output.weight") == 0;
}

// The acoustic code table is read with get_rows, and CUDA's get_rows accepts
// Q8_0 but no k-quant: a k-quant table would silently fall back to a host copy
// of the whole table on every depth step.
inline bool quant_is_mm3_row_table(const char * name, const char * arch) {
    return std::strcmp(arch, "mm3") == 0 && std::strcmp(name, "depth.audio_embd.weight") == 0;
}

inline bool quant_is_mm3_protected_component(const char * name) {
    return std::strncmp(name, "cond.", 5) == 0 || std::strncmp(name, "voc.", 4) == 0 ||
           std::strcmp(name, "dit.time_fourier.weight") == 0 ||
           std::strcmp(name, "depth.pos_embd.weight") == 0;
}

inline bool quant_should_quantize(const char * name, int n_dims, const char * arch) {
    if (std::strstr(arch, "vae")) {
        return false;
    }
    if (n_dims < 2) {
        return false;
    }
    if (std::strstr(arch, "text-enc") && std::strstr(name, "embed_tokens")) {
        return false;
    }
    if (std::strstr(name, "silence_latent")) {
        return false;
    }
    if (std::strstr(name, "scale_shift_table")) {
        return false;
    }
    if (std::strstr(name, "null_condition_emb")) {
        return false;
    }
    if (std::strcmp(arch, "mm3") == 0 && quant_is_mm3_protected_component(name)) {
        return false;
    }
    return true;
}

inline bool quant_bump_applies(const QuantVariant & v, int layer, int n_layers) {
    switch (v.bump_mode) {
        case 1:
            return layer >= 0 && layer < v.bump_layer_count;
        case 2:
            return layer >= 0 && (layer < n_layers / 9 || layer >= n_layers - n_layers / 7 || layer % 3 == 0);
        case 3:
            return true;
        default:
            return false;
    }
}

// Target type for one tensor, or GGML_TYPE_COUNT to keep it as stored.
inline enum ggml_type quant_pick_type(const char *         name,
                                      int                  n_dims,
                                      const char *         arch,
                                      const QuantVariant & v,
                                      int                  n_layers) {
    if (!quant_should_quantize(name, n_dims, arch)) {
        return GGML_TYPE_COUNT;
    }

    if (quant_is_mm3_row_table(name, arch)) {
        return GGML_TYPE_Q8_0;
    }

    if ((quant_is_embed(name) || quant_is_untied_output(name, arch)) && !std::strstr(arch, "text-enc")) {
        return v.embed != GGML_TYPE_COUNT ? v.embed : v.base;
    }

    const bool important = v.bump_mode == 3 ? quant_important_l(name) : quant_important_sm(name);
    if (important && v.bump != GGML_TYPE_COUNT && quant_bump_applies(v, quant_layer_index(name), n_layers)) {
        return v.bump;
    }

    return v.base;
}

// 1-D norms and biases stored in half precision are promoted to F32.
inline bool quant_should_promote_f32(int n_dims) {
    return n_dims < 2;
}

}  // namespace tts_cpp::acestep
