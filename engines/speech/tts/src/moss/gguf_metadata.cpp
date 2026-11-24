#include "moss/gguf_metadata.h"

#include "gguf.h"

#include <stdexcept>
#include <utility>

namespace tts_cpp::moss::detail {
namespace {

std::vector<std::string> read_strings(const gguf_context * file, int64_t id) {
    std::vector<std::string> values(gguf_get_arr_n(file, id));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = gguf_get_arr_str(file, id, i);
    }
    return values;
}

} // namespace

GgufMetadata::GgufMetadata(const gguf_context * file, std::string owner)
    : file_(file), owner_(std::move(owner)) {}

void GgufMetadata::fail(const std::string & message) const {
    throw std::runtime_error(owner_ + ": " + message);
}

int64_t GgufMetadata::require(const std::string & key) const {
    const int64_t id = gguf_find_key(file_, key.c_str());
    if (id < 0) {
        fail("missing metadata: " + key);
    }
    return id;
}

uint32_t GgufMetadata::u32(const std::string & key) const {
    const int64_t id = require(key);
    switch (gguf_get_kv_type(file_, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(file_, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(file_, id);
        default:               fail("unsupported metadata type: " + key);
    }
}

float GgufMetadata::f32(const std::string & key) const {
    const int64_t id = require(key);
    if (gguf_get_kv_type(file_, id) != GGUF_TYPE_FLOAT32) {
        fail("metadata must be a float: " + key);
    }
    return gguf_get_val_f32(file_, id);
}

bool GgufMetadata::boolean(const std::string & key) const {
    const int64_t id = require(key);
    if (gguf_get_kv_type(file_, id) != GGUF_TYPE_BOOL) {
        fail("metadata must be a bool: " + key);
    }
    return gguf_get_val_bool(file_, id);
}

std::string GgufMetadata::str(const std::string & key) const {
    const int64_t id = require(key);
    if (gguf_get_kv_type(file_, id) != GGUF_TYPE_STRING) {
        fail("metadata must be a string: " + key);
    }
    return gguf_get_val_str(file_, id);
}

std::vector<std::string> GgufMetadata::str_array(const std::string & key) const {
    const int64_t id = require(key);
    if (gguf_get_kv_type(file_, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(file_, id) != GGUF_TYPE_STRING) {
        fail("metadata must be a string array: " + key);
    }
    return read_strings(file_, id);
}

std::vector<int32_t> GgufMetadata::int_array(const std::string & key, size_t max_count) const {
    const int64_t id = require(key);
    if (gguf_get_kv_type(file_, id) != GGUF_TYPE_ARRAY) {
        fail("metadata must be an array: " + key);
    }
    const gguf_type type = gguf_get_arr_type(file_, id);
    if (type != GGUF_TYPE_INT32 && type != GGUF_TYPE_UINT32) {
        fail("metadata array must hold 32-bit integers: " + key);
    }
    const size_t count = gguf_get_arr_n(file_, id);
    if (count > max_count) {
        fail("metadata array is too long: " + key);
    }
    const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(file_, id));
    return std::vector<int32_t>(data, data + count);
}

} // namespace tts_cpp::moss::detail
