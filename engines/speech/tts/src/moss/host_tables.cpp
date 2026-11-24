#include "moss/host_tables.h"

#include "gguf_stream.h"

#include <stdexcept>

namespace tts_cpp::moss::detail {

HostTables::HostTables(const char * owner) : owner_(owner) {}

HostTables::~HostTables() {
    release();
}

void HostTables::release() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    buffer_ = nullptr;
}

ggml_backend_buffer_t HostTables::buffer() const {
    return buffer_;
}

void HostTables::fail(const std::string & message) const {
    throw std::runtime_error(owner_ + ": " + message);
}

void HostTables::allocate(ggml_context * tables) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    buffer_ = cpu ? ggml_backend_alloc_ctx_tensors_from_buft(tables, ggml_backend_dev_buffer_type(cpu)) : nullptr;
    if (!buffer_) {
        fail("host table allocation failed");
    }
}

void HostTables::stream(ggml_context * tables, ::tts_cpp::detail::gguf_stream_reader & reader) const {
    for (auto * t = ggml_get_first_tensor(tables); t; t = ggml_get_next_tensor(tables, t)) {
        if (!reader.to_host(ggml_get_name(t), t->data, ggml_nbytes(t))) {
            fail(std::string("failed to load tensor: ") + ggml_get_name(t));
        }
    }
}

void HostTables::read(ggml_context * tables, ::tts_cpp::detail::gguf_stream_reader & reader) {
    allocate(tables);
    stream(tables, reader);
}

} // namespace tts_cpp::moss::detail
