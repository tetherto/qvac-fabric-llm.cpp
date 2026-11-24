#pragma once

#include "ggml-backend.h"

#include <string>

namespace qvac_gguf {
class gguf_stream_reader;
}

namespace tts_cpp::moss::detail {

class HostTables {
public:
    explicit HostTables(const char * owner);
    ~HostTables();
    HostTables(const HostTables &) = delete;
    HostTables & operator=(const HostTables &) = delete;

    void read(ggml_context * tables, ::qvac_gguf::gguf_stream_reader & reader);
    void release();
    ggml_backend_buffer_t buffer() const;

private:
    void allocate(ggml_context * tables);
    void stream(ggml_context * tables, ::qvac_gguf::gguf_stream_reader & reader) const;
    [[noreturn]] void fail(const std::string & message) const;

    std::string owner_;
    ggml_backend_buffer_t buffer_ = nullptr;
};

} // namespace tts_cpp::moss::detail
