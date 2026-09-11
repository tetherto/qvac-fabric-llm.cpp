#include "llama-io.h"

#include "ggml-backend.h"

#include <vector>

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read_to(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read_to(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}

// qvac: convenience wrapper that consumes `size` bytes from the stream and
// writes them into `tensor` at the given byte offset via the ggml backend.
void llama_io_read_i::read_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    const uint8_t * src = read(size);
    ggml_backend_tensor_set(tensor, src, offset, size);
}
