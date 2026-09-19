#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    // qvac: upstream b9341 made `tensor` const since IO only reads from it.
    virtual void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    // qvac: upstream b9341 redesigned the read interface: `read(size)` exposes a
    // pointer into the underlying buffer for zero-copy paths, while `read_to`
    // copies into caller-owned memory.
    virtual const uint8_t * read(size_t size) = 0;
    virtual void            read_to(void * dst, size_t size) = 0;

    // qvac: kv-cache + recurrent-memory serialization paths still call
    // `io.read(dst, size)` everywhere; route to read_to to avoid churn.
    void read(void * dst, size_t size) { read_to(dst, size); }

    // qvac: kv-cache/memory state-deserialization callers still rely on the
    // tensor convenience that upstream b9341 dropped. The base implements it on
    // top of read(); the host/device readers (added upstream after b9341)
    // override it to batch the backend copies, so it must stay virtual.
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size);

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
