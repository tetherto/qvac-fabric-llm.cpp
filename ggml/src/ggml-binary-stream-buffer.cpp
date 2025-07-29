#include "ggml-binary-stream-buffer.h"

ggml_binary_stream_buffer::ggml_binary_stream_buffer(std::vector<uint8_t> && _data) : data(std::move(_data)) {
    // Cast uint8_t* to char* for basic_streambuf<char> - this is safe since both are 1-byte types
    char * start = reinterpret_cast<char *>(data.data());
    setg(start, start, start + data.size());
}

ggml_binary_stream_buffer::int_type ggml_binary_stream_buffer::underflow() {
    if (gptr() < egptr()) {
        return traits_type::to_int_type(*gptr());
    }
    return traits_type::eof();
}

std::streamsize ggml_binary_stream_buffer::xsgetn(char_type * s, std::streamsize n) {
    std::streamsize available = egptr() - gptr();
    std::streamsize to_read   = std::min(n, available);
    if (to_read > 0) {
        std::memcpy(s, gptr(), to_read);
        setg(eback(), gptr() + to_read, egptr());
    }
    return to_read;
}

ggml_binary_stream_buffer::pos_type ggml_binary_stream_buffer::seekoff(off_type                off,
                                                                        std::ios_base::seekdir  dir,
                                                                        std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    char_type * new_pos = nullptr;
    if (dir == std::ios_base::beg) {
        new_pos = eback() + off;
    } else if (dir == std::ios_base::cur) {
        new_pos = gptr() + off;
    } else if (dir == std::ios_base::end) {
        new_pos = egptr() + off;
    }
    if (new_pos >= eback() && new_pos <= egptr()) {
        setg(eback(), new_pos, egptr());
        return new_pos - eback();
    }
    return pos_type(off_type(-1));
}

ggml_binary_stream_buffer::pos_type ggml_binary_stream_buffer::seekpos(pos_type pos, std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    char_type * new_pos = eback() + pos;
    if (new_pos >= eback() && new_pos <= egptr()) {
        setg(eback(), new_pos, egptr());
        return pos;
    }
    return pos_type(off_type(-1));
}
