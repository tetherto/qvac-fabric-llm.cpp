#include "moss/transcribe_tensors.h"

#include "ggml.h"

#include <stdexcept>
#include <string>

namespace parakeet::moss::detail {
namespace {

[[noreturn]] void fail(const ggml_tensor * tensor, const char * problem) {
    throw std::runtime_error("moss transcribe: " + std::string(ggml_get_name(tensor)) + " " + problem);
}

bool is_weight_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || ggml_is_quantized(type);
}

void require_dims(const ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 || tensor->ne[2] != ne2 || tensor->ne[3] != 1) {
        fail(tensor, "has unexpected dimensions");
    }
}

} // namespace

void require_transcribe_weight(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    require_dims(tensor, ne0, ne1, 1);
    if (!is_weight_type(tensor->type)) {
        fail(tensor, "has an unsupported type");
    }
}

void require_transcribe_f32(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    require_dims(tensor, ne0, ne1, 1);
    if (tensor->type != GGML_TYPE_F32) {
        fail(tensor, "must be f32");
    }
}

void require_transcribe_kernel(const ggml_tensor * tensor, int64_t width, int64_t in_channels, int64_t out_channels) {
    require_dims(tensor, width, in_channels, out_channels);
    if (tensor->type != GGML_TYPE_F16) {
        fail(tensor, "must be f16");
    }
}

} // namespace parakeet::moss::detail
