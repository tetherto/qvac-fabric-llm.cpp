#pragma once

#include "ggml-cpp.h"
#include "llama-mmap.h"
#include "llama-model-load-input.h"

#include <cstdint>
#include <cstring>
#include <set>

struct llama_model_loader;

/// @brief Immediately loads and stores relevant data in the struct fields.
struct llama_gguf_file_load {
    struct gguf_init_params     params;
    gguf_context_ptr            meta;
    std::unique_ptr<llama_file> file = nullptr;

    llama_gguf_file_load(struct ggml_context ** ctx, load_input_t load_input);
};

/// @brief Stores relevant information to be able to loads a `.gguf` split file when load method is called.
struct llama_gguf_split_load {
    load_input_t                               load_input;
    load_input_variant::llama_fname_load_input base_split;
    uint16_t                                   idx;
    std::string                                kv_split_no;
    bool                                       loaded = false;

    llama_gguf_split_load(load_input_t &                             load_input,
                          load_input_variant::llama_fname_load_input base_split,
                          uint16_t                                   idx,
                          std::string                                kv_split_no);

    static llama_gguf_file_load load_split_gguf(struct ggml_context **     ctx,
                                                const char *               fname_split,
                                                load_input_t &             load_input,
                                                std::vector<std::string> & splits);

    struct ggml_context * load(struct llama_model_loader & ml);
};

/// @brief Handles incremental load of tensor and split-files.
/// @note First split-file is expected to be already available at construction, the remainder of split-files are
/// incrementally load on-demand by calling `load_tensor_metadata`
struct incremental_splits_tensor_load {
    incremental_splits_tensor_load(struct ggml_context *       ctx,
                                   struct llama_model_loader & ml,
                                   llama_gguf_file_load &      base_split,
                                   std::set<std::string>       tensor_list);

    void add_split(llama_gguf_split_load splitLoad);

    /// @brief Incrementally loads file splits until the tensor metadata is found.
    /// Also increments loaded tensor count so that `all_tensors_are_loaded` returns true
    /// when all tensors in a file-split have been requested.
    /// @returns Split idx where the tensor was found
    /// @throw runtime_error if tensor was not found
    uint16_t load_tensor_metadata(struct llama_model_loader & ml,
                                  const char *                tensor_name,
                                  ggml_tensor **              out_tensor_metadata);

    /// @returns True if all tensors of a split have been loaded.
    bool all_tensors_are_loaded(uint16_t split_idx) const;

    /// @returns Max number of tensors as described on the summary tensor-list file.
    std::size_t expected_n_tensors();

    /// @bried Release file memory for a split.
    static void release_split(struct llama_model_loader & ml, uint16_t split_idx);

    void print_currently_known_tensors() const;

    uint16_t get_split_idx_for_tensor(const char * tensor_name) const;

    std::size_t get_split_data_size(uint16_t split_idx) const;

    static bool tensor_ignored(const std::optional<incremental_splits_tensor_load> & splits_tensor_load,
                               const char *                                          tensor_name);

    /// @brief Lalizy get/allocate a context with enough capacity for all tensors of
    /// same type of an individual split. The context can be used to instantiate the
    /// final model tensors and and attach to them backend buffers.
    /// @tparam impl The model implementation type where the context will be stored.
    ggml_context * get_model_ctx_for_split_buft(ggml_backend_buffer_type_t buft, uint16_t split);

    // define a comparator for the buft -> ctx maps to ensure that the order is well-defined:
    struct ggml_backend_buft_split_comparator {
        bool operator()(const std::pair<ggml_backend_buffer_type_t, uint16_t> & lhs,
                        const std::pair<ggml_backend_buffer_type_t, uint16_t> & rhs) const {
            // First compare by buffer type name, then by split index
            const int name_cmp = strcmp(ggml_backend_buft_name(lhs.first), ggml_backend_buft_name(rhs.first));
            if (name_cmp != 0) {
                return name_cmp < 0;
            }
            return lhs.second < rhs.second;
        }
    };

    // public so that it can be processed by the backend storage allocator
    std::map<std::pair<ggml_backend_buffer_type_t, uint16_t>, ggml_context_ptr, ggml_backend_buft_split_comparator>
        ctx_split_map;

  private:
    struct TensorInfo {
        uint16_t split_idx = 0;
        bool     is_loaded = false;
    };

    struct SplitInfo {
        uint32_t total_tensor_count = 0, loaded_tensor_count = 0;

        /// @brief Total ggml tensor data size of this split
        std::size_t data_size = 0;

        bool all_tensors_loaded() const;
    };

    void _load_split(struct llama_model_loader & ml, uint16_t idx);
    void _process_split(const struct ggml_context * ctx, struct llama_model_loader & ml, uint16_t idx);

    /// @brief Get tensor info iterator or throw if not found
    /// @throw runtime_error if tensor not found
    std::map<std::string, TensorInfo>::const_iterator _get_tensor_info_iterator(const char * tensor_name) const;

    /// @brief Get split info iterator or throw if not found
    /// @throw runtime_error if split not found
    std::map<uint16_t, SplitInfo>::const_iterator _get_split_info_iterator(uint16_t split_idx) const;

    std::map<std::string, TensorInfo> tensor_info;
    std::map<uint16_t, SplitInfo>     split_info;

    /// @brief Number of delayed files that have been loaded
    std::size_t delayed_loaded = 0;

    /// @brief Vector of split files to be loaded on demand
    std::vector<llama_gguf_split_load> delayed_files;

    /// @brief Set of expected tensor names loaded from tensor list file
    std::set<std::string> expected_tensors;
};
