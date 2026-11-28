#include "cosyvoice_fit_internal.h"
#include "gguf_stream.h"
#include "test_env_portable.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kColumns = 256;
constexpr int64_t kRows = 6001;
constexpr const char * kFlowWeight = "flow/test/weight";
constexpr const char * kOtherWeight = "lm/test/weight";

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void fill_bf16(ggml_tensor * tensor) {
    auto * data = static_cast<ggml_bf16_t *>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) data[i].bits = static_cast<uint16_t>(i);
}

void add_weight(gguf_context * file, ggml_context * context, const char * name) {
    ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_BF16, kColumns, kRows);
    ggml_set_name(tensor, name);
    fill_bf16(tensor);
    gguf_add_tensor(file, tensor);
}

void write_fixture(const std::string & path) {
    constexpr size_t tensor_count = 2;
    const size_t bytes = tensor_count * kColumns * kRows * sizeof(ggml_bf16_t);
    ggml_init_params params = {bytes + tensor_count * ggml_tensor_overhead(), nullptr, false};
    ggml_context * context = ggml_init(params);
    gguf_context * file = gguf_init_empty();
    add_weight(file, context, kOtherWeight);
    add_weight(file, context, kFlowWeight);
    const bool written = gguf_write_to_file(file, path.c_str(), false);
    gguf_free(file);
    ggml_free(context);
    require(written, "cannot write bf16 fixture");
}

void check_f32_bits(const std::vector<uint32_t> & values) {
    constexpr unsigned bf16_shift = 16;
    for (size_t i = 0; i < values.size(); ++i) {
        require(values[i] == static_cast<uint32_t>(static_cast<uint16_t>(i)) << bf16_shift,
                "bf16 expansion changed a value");
    }
}

void check_f32_tensor(ggml_tensor * tensor) {
    std::vector<uint32_t> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    check_f32_bits(values);
}

void check_loader(const std::string & path) {
    model_ctx loaded = cosyvoice_load_gguf(path);
    cosyvoice_fit_load_measure measured;
    model_ctx metadata = cosyvoice_load_gguf_metadata_only(path, loaded.backend, measured);
    ggml_tensor * flow = cosyvoice_get(loaded, kFlowWeight);
    ggml_tensor * other = cosyvoice_get(loaded, kOtherWeight);
#if defined(__aarch64__) || defined(_M_ARM64)
    require(flow->type == GGML_TYPE_F32, "ARM CPU flow must expand bf16 to f32");
    require(flow->buffer != loaded.map_buf, "expanded flow must not alias bf16 mapping");
    check_f32_tensor(flow);
#else
    require(flow->type == GGML_TYPE_BF16, "non-ARM flow must retain bf16");
#endif
    require(other->type == GGML_TYPE_BF16, "non-flow weights must retain their type");
    require(cosyvoice_get(metadata, kFlowWeight)->type == flow->type,
            "memory projection must use runtime weight type");
    require(measured.device_bytes >= ggml_nbytes(flow) + ggml_nbytes(other),
            "memory projection must cover expanded weights");
    cosyvoice_free(metadata);
    cosyvoice_free(loaded);
}

void check_truncated_stream(const gguf_context * file, const std::string & path, ggml_tensor * tensor) {
    const std::string truncated = path + ".truncated";
    std::filesystem::copy_file(path, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - sizeof(ggml_bf16_t));
    {
        tts_cpp::detail::gguf_stream_reader reader(file, truncated);
        require(!reader.bf16_to_f32(kFlowWeight, tensor), "truncated bf16 payload must fail");
    }
    std::filesystem::remove(truncated);
}

void check_stream(const std::string & path) {
    ggml_context * metadata = nullptr;
    gguf_init_params params = {true, &metadata};
    gguf_context * file = gguf_init_from_file(path.c_str(), params);
    require(file != nullptr, "cannot read bf16 fixture");
    ggml_init_params context_params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context * context = ggml_init(context_params);
    ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, kColumns, kRows);
    model_ctx backend_owner = cosyvoice_load_gguf(path);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend_owner.backend);
    {
        tts_cpp::detail::gguf_stream_reader reader(file, path);
        require(reader.bf16_to_f32(kFlowWeight, tensor), "chunked bf16 expansion failed");
        check_f32_tensor(tensor);
        require(!reader.bf16_to_f32("missing", tensor), "missing tensor must fail");
        require(!reader.bf16_to_f32(kFlowWeight, cosyvoice_get(backend_owner, kOtherWeight)),
                "non-f32 destination must fail");
    }
    check_truncated_stream(file, path, tensor);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    cosyvoice_free(backend_owner);
    gguf_free(file);
    ggml_free(metadata);
}

}

int main() {
    const std::string path = test_tmpdir() + "/test-cosyvoice-bf16.gguf";
    try {
        write_fixture(path);
        check_loader(path);
        check_stream(path);
        std::remove(path.c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        std::remove(path.c_str());
        return 1;
    }
}
