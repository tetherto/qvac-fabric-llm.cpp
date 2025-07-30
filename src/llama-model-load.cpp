#include "llama-model-load.h"

#include "llama-model-loader.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <variant>

llama_gguf_file_load::llama_gguf_file_load(struct ggml_context ** ctx, load_input_t load_input) :
    params({
        /*.no_alloc = */ true,
        /*.ctx      = */ ctx,
    }) {
    using namespace load_input_variant;
    if (std::holds_alternative<llama_fname_load_input>(load_input)) {
        const auto & file_input = std::get<llama_fname_load_input>(load_input);
        meta.reset(gguf_init_from_file(file_input.fname.c_str(), params));
        if (!meta) {
            throw std::runtime_error(format("%s: failed to load model from %s", __func__, file_input.fname.c_str()));
        }
        file = std::make_unique<llama_file_disk>(file_input.fname.c_str(), "rb");
    } else if (std::holds_alternative<llama_buffer_future_load_input>(load_input)) {
        const auto & future_input = std::get<llama_buffer_future_load_input>(load_input);
        auto         future_file =
            std::make_unique<llama_future_file_buffer_ro>(future_input.promise_key, future_input.context);
        std::unique_ptr<llama_file_buffer_ro> file_buffer = future_file->extract();
        meta.reset(gguf_init_from_buffer(*file_buffer->streambuf, params));
        if (!meta) {
            throw std::runtime_error(format("%s: failed to load model from buffer", __func__));
        }
        file = std::move(file_buffer);
    } else {
        const auto & buffer_input = std::get<llama_buffer_load_input>(load_input);
        meta.reset(gguf_init_from_buffer(*buffer_input.streambuf, params));
        if (!meta) {
            throw std::runtime_error(format("%s: failed to load model from buffer", __func__));
        }
        file = std::make_unique<llama_file_buffer_ro>(std::move(buffer_input.streambuf));
    }
}

llama_gguf_file_load llama_gguf_split_load::load_split_gguf(struct ggml_context **     ctx,
                                                            const char *               fname_split,
                                                            load_input_t &             load_input,
                                                            std::vector<std::string> & splits) {
    using namespace load_input_variant;
    if (std::holds_alternative<llama_fname_load_input>(load_input)) {
        return llama_gguf_file_load(ctx, llama_fname_load_input{ fname_split, splits });
    }
    if (std::holds_alternative<llama_buffer_future_load_input>(load_input)) {
        auto future_input = std::get<llama_buffer_future_load_input>(load_input);
        return llama_gguf_file_load(ctx, llama_buffer_future_load_input{ fname_split, future_input.context, splits,
                                                                         future_input.tensor_list_file });
    }
    return llama_gguf_file_load(ctx, load_input);
}

llama_gguf_split_load::llama_gguf_split_load(load_input_t &                             load_input,
                                             load_input_variant::llama_fname_load_input base_split,
                                             uint16_t                                   idx,
                                             std::string                                kv_split_no) :
    load_input(load_input),
    base_split(base_split),
    idx(idx),
    kv_split_no(std::move(kv_split_no)) {}

incremental_splits_tensor_load::incremental_splits_tensor_load(struct ggml_context *       ctx,
                                                               struct llama_model_loader & ml,
                                                               llama_gguf_file_load &      base_split,
                                                               std::set<std::string>       tensor_list) :
    expected_tensors(std::move(tensor_list)) {
    ml.process_loaded_gguf(ctx, base_split, 0);
    _process_split(ctx, ml, 0);
}

struct ggml_context * llama_gguf_split_load::load(llama_model_loader & ml) {
    if (loaded) {
        return ml.contexts[idx].get();
    }

    struct ggml_context * ctx = ml.contexts.back().get();

    const char * fname_split = base_split.splits[idx].c_str();
    LLAMA_LOG_INFO("loading split-file %s\n", fname_split);

    llama_gguf_file_load split_gguf =
        llama_gguf_file_load(load_split_gguf(&ctx, fname_split, load_input, base_split.splits));
    gguf_context_ptr & split_meta = split_gguf.meta;

    if (idx > 0) {
        const int kid = gguf_find_key(split_meta.get(), kv_split_no.c_str());
        if (kid < 0) {
            throw std::runtime_error(format("missing key %s in GGUF split %s", kv_split_no.c_str(), fname_split));
        }
        int idx_gguf = gguf_get_val_u16(split_meta.get(), kid);
        if (idx_gguf != idx) {
            throw std::runtime_error(
                format("invalid split file idx: %d (file: %s), expected %d", idx_gguf, fname_split, idx));
        }
    }

    // Check that this split's idx matches the expected position in ml.files
    if (!ml.files.empty() && idx != ml.files.size()) {
        throw std::runtime_error(
            format("invalid split file loading order: got idx %d but expected %zu based on ml.files size", idx,
                   ml.files.size()));
    }

    ml.process_loaded_gguf(ctx, split_gguf, idx);

    loaded = true;
    return ctx;
}

void incremental_splits_tensor_load::add_split(llama_gguf_split_load splitLoad) {
    // +1 because first split is expected to have been already loaded (not delayed)
    split_info[delayed_files.size() + 1] = SplitInfo();
    delayed_files.emplace_back(std::move(splitLoad));
}

void incremental_splits_tensor_load::_load_split(struct llama_model_loader & ml, uint16_t idx) {
    // -1 because first split is expected to have been already loaded (not delayed and not present in delayed_files)
    const struct ggml_context * ctx = delayed_files[idx - 1].load(ml);
    _process_split(ctx, ml, idx);
}

void incremental_splits_tensor_load::_process_split(const struct ggml_context * ctx,
                                                    struct llama_model_loader & ml,
                                                    uint16_t                    idx) {
    SplitInfo & split = split_info[idx];

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string tensor_name = std::string(cur->name);
        split.total_tensor_count++;

        // Add tensor info with initial loaded state as false
        tensor_info[tensor_name] = TensorInfo{ idx, false };

        auto it = ml.weights_map.find(tensor_name);
        if (it == ml.weights_map.end()) {
            throw std::runtime_error(format("tensor '%s' not found in weights_map", tensor_name.c_str()));
        }
        split.data_size += ggml_nbytes(it->second.tensor);
    }
}

uint16_t incremental_splits_tensor_load::load_tensor_metadata(struct llama_model_loader & ml,
                                                              const char *                tensor_name,
                                                              ggml_tensor **              out_tensor_metadata) {
    if (expected_tensors.find(tensor_name) == expected_tensors.end()) {
        throw std::runtime_error(format("unknown tensor not expected in split files: %s", tensor_name));
    }
    while (!(*out_tensor_metadata) && delayed_loaded < delayed_files.size()) {
        // +1 because first split is expected to have been already loaded (not delayed)
        _load_split(ml, delayed_loaded + 1);
        *out_tensor_metadata = ml.get_tensor_meta(tensor_name);
        delayed_loaded++;
        if (delayed_loaded == delayed_files.size() && ml.weights_map.size() != expected_n_tensors()) {
            throw std::runtime_error(
                format("finished incrementally loading all splits but expected %zu tensors, got %zu",
                       expected_n_tensors(), ml.weights_map.size()));
        }
    }
    uint16_t split_idx = get_split_idx_for_tensor(tensor_name);

    // Mark tensor as loaded and increment split's loaded count
    auto tensor_it = tensor_info.find(tensor_name);
    if (!tensor_it->second.is_loaded) {
        tensor_it->second.is_loaded = true;
        split_info[split_idx].loaded_tensor_count++;
    }

    return split_idx;
}

uint16_t incremental_splits_tensor_load::get_split_idx_for_tensor(const char * tensor_name) const {
    return _get_tensor_info_iterator(tensor_name)->second.split_idx;
}

std::size_t incremental_splits_tensor_load::get_split_data_size(uint16_t split_idx) const {
    return _get_split_info_iterator(split_idx)->second.data_size;
}

void incremental_splits_tensor_load::print_currently_known_tensors() const {
    LLAMA_LOG_INFO("Current incremental loaded tensors:\n");
    for (const auto & it : tensor_info) {
        LLAMA_LOG_INFO("Tensor '%s' in split %d (loaded: %s)\n", it.first.c_str(), it.second.split_idx,
                       it.second.is_loaded ? "yes" : "no");
    }
}

bool incremental_splits_tensor_load::all_tensors_are_loaded(uint16_t split_idx) const {
    auto              it    = _get_split_info_iterator(split_idx);
    const SplitInfo & split = it->second;
    return split.all_tensors_loaded();
}

std::size_t incremental_splits_tensor_load::expected_n_tensors() {
    return expected_tensors.size();
}

void incremental_splits_tensor_load::release_split(struct llama_model_loader & ml, uint16_t split_idx) {
    // Let destructor of the smart pointer do the release of memory
    ml.files[split_idx] = nullptr;
}

std::map<std::string, incremental_splits_tensor_load::TensorInfo>::const_iterator
incremental_splits_tensor_load::_get_tensor_info_iterator(const char * tensor_name) const {
    auto it = tensor_info.find(tensor_name);
    if (it == tensor_info.end()) {
        throw std::runtime_error(format("tensor '%s' not found in tensor_info map", tensor_name));
    }
    return it;
}

std::map<uint16_t, incremental_splits_tensor_load::SplitInfo>::const_iterator
incremental_splits_tensor_load::_get_split_info_iterator(uint16_t split_idx) const {
    auto it = split_info.find(split_idx);
    if (it == split_info.end()) {
        throw std::runtime_error(format("split index %d not found in split_info map", split_idx));
    }
    return it;
}

bool incremental_splits_tensor_load::SplitInfo::all_tensors_loaded() const {
    return loaded_tensor_count >= total_tensor_count;
}

bool incremental_splits_tensor_load::tensor_ignored(
    const std::optional<incremental_splits_tensor_load> & splits_tensor_load,
    const char *                                          tensor_name) {
    return !splits_tensor_load.has_value() ||
           (splits_tensor_load.has_value() &&
            splits_tensor_load->expected_tensors.find(tensor_name) == splits_tensor_load->expected_tensors.end());
}

ggml_context * incremental_splits_tensor_load::get_model_ctx_for_split_buft(ggml_backend_buffer_type_t buft,
                                                                            uint16_t                   split) {
    auto key = std::make_pair(buft, split);
    auto it  = ctx_split_map.find(key);
    if (it == ctx_split_map.end()) {
        const size_t max_n_tensors = _get_split_info_iterator(split)->second.total_tensor_count;
        const size_t ctx_size      = ggml_tensor_overhead() * max_n_tensors;

        ggml_init_params params = {
            /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true,
        };

        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for split-file");
        }

        ctx_split_map[key] = ggml_context_ptr{ ctx };

        return ctx;
    }
    return it->second.get();
}
