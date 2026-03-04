#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>
#include <streambuf>
#include <string>
#include <vector>

#include "llama.h"

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora * adapter) { llama_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<llama_model, llama_model_deleter> llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
typedef std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter> llama_adapter_lora_ptr;

LLAMA_API struct llama_model * llama_model_load_from_buffer(std::vector<uint8_t> &&   data,
                                                            struct llama_model_params params);
LLAMA_API bool                 llama_model_load_fulfill_split_future(const char * path, const char * context,
                                                                     std::unique_ptr<std::basic_streambuf<char>> && streambuf);

// Read uint32 metadata directly from GGUF metadata without loading tensors.
enum class MetaResultStatus {
    SUCCESS = 0,
    PATH_NULL = 1,
    NULL_OUT_META_HANDLE = 2,
    GGUF_INIT_FAILED = 3,
    META_HANDLE_NULL = 4,
    KEY_NULL = 5,
    VALUE_NULL = 6,
    KEY_NOT_FOUND = 7,
    KV_TYPE_NOT_UINT32 = 8,
    STREAMBUF_SEEK_FAILED = 9,
    KV_TYPE_NOT_STRING = 10,
};

struct llama_metadata_handle;
struct metadata_handle_deleter {
    void operator()(struct llama_metadata_handle * ctx) const;
};
typedef std::unique_ptr<struct llama_metadata_handle, metadata_handle_deleter> metadata_handle_ptr;

LLAMA_API MetaResultStatus llama_model_meta_get_u32(metadata_handle_ptr const & meta_handle, const char * key, uint32_t * value);
LLAMA_API MetaResultStatus llama_model_meta_get_str(metadata_handle_ptr const & meta_handle, const char * key, std::string * value);
LLAMA_API MetaResultStatus llama_model_meta_from_file(const char * path_model, metadata_handle_ptr * out_meta_handle);
LLAMA_API MetaResultStatus llama_model_meta_from_streambuf(std::basic_streambuf<char> & streambuf, metadata_handle_ptr * out_meta_handle);
