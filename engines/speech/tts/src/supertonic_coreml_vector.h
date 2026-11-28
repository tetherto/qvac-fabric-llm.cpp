#pragma once
#include <string>
#include <utility>
#include <vector>

struct supertonic_coreml_vector_context;
supertonic_coreml_vector_context * supertonic_coreml_vector_init(
    const std::vector<std::pair<int, std::string>> & paths,
    int heads, float cond_scale, float uncond_scale);
void supertonic_coreml_vector_free(supertonic_coreml_vector_context * ctx);
const char * supertonic_coreml_vector_backend_label(const supertonic_coreml_vector_context * ctx);
int supertonic_coreml_vector_weight_bits(const supertonic_coreml_vector_context * ctx);
// time_embeddings contains count consecutive 64-float host time embeddings.
// A failed call leaves output untouched, allowing whole-loop ggml fallback.
bool supertonic_coreml_vector_run(supertonic_coreml_vector_context * ctx,
    const float * latent, int latent_length, const float * text, int text_length,
    const float * style, const float * latent_mask, const float * time_embeddings,
    const float * theta, int total_steps, int count, std::vector<float> & output);
