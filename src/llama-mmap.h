#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <cstdio>
#include "uint8-buff-stream.h"
#include <future>
#include <string>
#include <map>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    virtual ~llama_file() = default;

    virtual size_t tell() const = 0;
    virtual size_t size() const = 0;
    virtual int file_id() const = 0;

    virtual void seek(size_t offset, int whence) const = 0;

    virtual void read_raw(void * ptr, size_t len) = 0;
    virtual void read_raw_unsafe(void * ptr, size_t len) { read_raw(ptr, len); }
    virtual void read_aligned_chunk(void * dest, size_t size) { read_raw(dest, size); }
    virtual uint32_t read_u32() = 0;

    virtual void write_raw(const void * ptr, size_t len) const = 0;
    virtual void write_u32(uint32_t val) const = 0;

    virtual size_t read_alignment() const { return 1; }
    virtual bool has_direct_io() const { return false; }

    /// @return contiguous in-memory contents, or nullptr if not memory-backed
    virtual const void * data_ptr() const { return nullptr; }
};

struct llama_file_disk : public llama_file {
    llama_file_disk(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file_disk(FILE * file);
    ~llama_file_disk() override;

    size_t tell() const override;
    size_t size() const override;
    int file_id() const override;

    void seek(size_t offset, int whence) const override;

    void read_raw(void * ptr, size_t len) override;
    void read_raw_unsafe(void * ptr, size_t len) override;
    void read_aligned_chunk(void * dest, size_t size) override;
    uint32_t read_u32() override;

    void write_raw(const void * ptr, size_t len) const override;
    void write_u32(uint32_t val) const override;

    size_t read_alignment() const override;
    bool has_direct_io() const override;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

template <bool Writable> struct llama_file_buffer : public llama_file {
    /// @note Use char for the streambuf because not all platforms support uint8_t specialization (e.g. MacOS or newer NDKs)
    ///       from C++17 there are guarantees that make safe to access binary data from char
    llama_file_buffer(std::unique_ptr<std::basic_streambuf<char>> && streambuf);

    ~llama_file_buffer() override;

    size_t tell() const override;
    size_t size() const override;

    /// @return -1 to indicate this is not a real file descriptor
    int file_id() const override;

    void seek(size_t offset, int whence) const override;

    void     read_raw(void * ptr, size_t len) override;
    uint32_t read_u32() override;

    /// @throw std::runtime_error if the buffer is read-only
    void write_raw(const void * ptr, size_t len) const override;

    /// @throw std::runtime_error if the buffer is read-only
    void write_u32(uint32_t val) const override;

    const void * data_ptr() const override;

    std::unique_ptr<std::basic_streambuf<char>> streambuf;
};

template <bool Writable> struct llama_future_file_buffer {
    /// @brief A file buffer object whose operations will block
    /// until the given promise key is set with a file buffer.
    /// @param promise_key The key to use for the promise (e.g. a file path).
    /// @param context The context to use for the promise, used to distinguish same promise key (e.g. for a same file opened twice).
    llama_future_file_buffer(const std::string & promise_key, const std::string & context);

    // Delete copy constructor and copy assignment operator
    llama_future_file_buffer(const llama_future_file_buffer &)             = delete;
    llama_future_file_buffer & operator=(const llama_future_file_buffer &) = delete;

    llama_future_file_buffer(llama_future_file_buffer && other) noexcept;
    llama_future_file_buffer & operator=(llama_future_file_buffer && other) noexcept;

    ~llama_future_file_buffer();

    /// @brief Sets the given key and context with a file buffer so that
    /// operations can resume/start.
    static bool fulfill_promise(const std::string & promise_key, const std::string & context,
                                std::unique_ptr<llama_file_buffer<Writable>> && value);

    /// @brief Waits for future buffer or obtains current if already
    /// fulfilled and moves the future contents outside the registry.
    std::unique_ptr<llama_file_buffer<Writable>> extract() const;

  private:
    typename std::map<std::string, std::promise<std::unique_ptr<llama_file_buffer<Writable>>>>::iterator
                                                                      file_buffer_promise_iterator;
    mutable std::future<std::unique_ptr<llama_file_buffer<Writable>>> file_buffer_future;
    mutable std::unique_ptr<llama_file_buffer<Writable>>              file_buffer;
};

// Type aliases for convenience
using llama_file_buffer_ro = llama_file_buffer<false>;
using llama_file_buffer_rw = llama_file_buffer<true>;
using llama_future_file_buffer_ro = llama_future_file_buffer<false>;
using llama_future_file_buffer_rw = llama_future_file_buffer<true>;

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
