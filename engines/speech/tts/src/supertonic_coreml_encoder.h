#pragma once

#include <cstdint>
#include <vector>

// Apple-only implementation; declarations remain usable by portable callers.
struct supertonic_coreml_encoder_context;
supertonic_coreml_encoder_context * supertonic_coreml_encoder_init(const char * path, int64_t vocabulary);
void supertonic_coreml_encoder_free(supertonic_coreml_encoder_context * ctx);
int supertonic_coreml_encoder_capacity(const supertonic_coreml_encoder_context * ctx);
bool supertonic_coreml_encoder_encode(supertonic_coreml_encoder_context * ctx,
                                      const int64_t * ids, int length, const float * style,
                                      std::vector<float> & embedding);
const char * supertonic_coreml_encoder_backend_label(const supertonic_coreml_encoder_context * ctx);
