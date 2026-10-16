#include "llama-mmap.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>

enum class fault { none, interrupted, invalid, bad_address, partial_invalid, io_error };
static thread_local fault read_fault = fault::none;
static thread_local unsigned read_calls = 0;
static std::atomic<size_t> largest_allocation{0};
static std::atomic<size_t> allocation_count{0};

extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __real_read(int, void *, size_t);
extern "C" int __real_posix_memalign(void **, size_t, size_t);
extern "C" ssize_t __wrap_pread(int, void *, size_t, off_t);
extern "C" ssize_t __wrap_read(int, void *, size_t);
extern "C" int __wrap_posix_memalign(void **, size_t, size_t);

static bool inject_fault(int fd, size_t & count) {
    if (!(fcntl(fd, F_GETFL) & O_DIRECT)) {
        return false;
    }
    unsigned call = read_calls++;
    switch (read_fault) {
        case fault::none: return false;
        case fault::interrupted:
            if (call != 0) {
                return false;
            }
            errno = EINTR;
            break;
        case fault::partial_invalid:
            if (call == 0) {
                count = std::min<size_t>(count, 4096);
                return false;
            }
            errno = EINVAL;
            break;
        case fault::invalid:     errno = EINVAL; break;
        case fault::bad_address: errno = EFAULT; break;
        case fault::io_error:    errno = EIO;    break;
    }
    return true;
}

extern "C" ssize_t __wrap_pread(int fd, void * ptr, size_t count, off_t offset) {
    return inject_fault(fd, count) ? -1 : __real_pread(fd, ptr, count, offset);
}

extern "C" ssize_t __wrap_read(int fd, void * ptr, size_t count) {
    return inject_fault(fd, count) ? -1 : __real_read(fd, ptr, count);
}

extern "C" int __wrap_posix_memalign(void ** ptr, size_t alignment, size_t size) {
    allocation_count.fetch_add(1);
    size_t previous = largest_allocation.load();
    while (previous < size && !largest_allocation.compare_exchange_weak(previous, size)) {}
    return __real_posix_memalign(ptr, alignment, size);
}

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct fixture {
    char path[64] = "test-mmap-direct-io-XXXXXX";
    std::vector<uint8_t> bytes;

    fixture() : bytes(16 * 1024 * 1024 + 37) {
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i % 251);
        }
        int fd = mkstemp(path);
        check(fd >= 0, "mkstemp failed");
        FILE * fp = fdopen(fd, "wb");
        check(fp != nullptr, "fdopen failed");
        size_t written = fwrite(bytes.data(), 1, bytes.size(), fp);
        fclose(fp);
        check(written == bytes.size(), "fixture write failed");
    }

    ~fixture() { unlink(path); }
};

static void test_serial(const fixture & f) {
    std::vector<std::unique_ptr<llama_file_disk>> files;
    std::vector<uint8_t> result(f.bytes.size() - 7);
    size_t first_allocations = 0;
    for (int i = 0; i < 8; ++i) {
        auto file = std::make_unique<llama_file_disk>(f.path, "rb", true);
        file->seek(7, SEEK_SET);
        file->read_raw(result.data(), result.size());
        check(std::equal(result.begin(), result.end(), f.bytes.begin() + 7), "chunked data mismatch");
        check(file->tell() == f.bytes.size(), "incorrect position after chunked read");
        if (i == 0) {
            first_allocations = allocation_count;
        } else {
            check(allocation_count == first_allocations, "each shard allocated another bounce buffer");
        }
        files.push_back(std::move(file));
    }
    check(largest_allocation <= 8 * 1024 * 1024, "bounce buffer exceeded its size limit");

    auto & file = *files.front();
    file.seek(3, SEEK_SET);
    uint32_t expected[2];
    memcpy(expected, f.bytes.data() + 3, sizeof(expected));
    check(file.read_u32() == expected[0], "first unaligned read mismatch");
    check(file.read_u32() == expected[1], "sequential unaligned read mismatch");
    bool rejected = false;
    try {
        file.seek(f.bytes.size() - 1, SEEK_SET);
        file.read_raw(result.data(), 2);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "read past EOF was accepted");

    for (fault failure : {fault::invalid, fault::bad_address, fault::partial_invalid}) {
        llama_file_disk fallback(f.path, "rb", true);
        read_fault = failure;
        read_calls = 0;
        fallback.seek(7, SEEK_SET);
        fallback.read_raw(result.data(), result.size());
        read_fault = fault::none;
        check(std::equal(result.begin(), result.end(), f.bytes.begin() + 7), "serial fallback data mismatch");
    }
}

static void test_parallel(const fixture & f) {
    llama_file_disk file(f.path, "rb", true);
    const int fd = file.file_id();
    file.seek(123, SEEK_SET);
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    for (fault failure : {fault::none, fault::interrupted, fault::invalid, fault::bad_address, fault::partial_invalid}) {
        threads.emplace_back([&, failure] {
            try {
                void * ptr = nullptr;
                const size_t alignment = file.read_alignment();
                const size_t count = 4 * alignment;
                check(posix_memalign(&ptr, alignment, count) == 0, "aligned allocation failed");
                std::unique_ptr<void, decltype(&free)> buffer(ptr, &free);
                read_fault = failure;
                read_calls = 0;
                size_t got = file.read_raw_unsafe_at(ptr, count, alignment);
                check(got == count, "short positional read");
                check(memcmp(ptr, f.bytes.data() + alignment, count) == 0, "positional data mismatch");

                read_fault = fault::none;
                size_t tail = (f.bytes.size() - 1) & ~(alignment - 1);
                got = file.read_raw_unsafe_at(ptr, alignment, tail);
                check(got == f.bytes.size() - tail, "incorrect EOF byte count");
                check(memcmp(ptr, f.bytes.data() + tail, got) == 0, "EOF data mismatch");

                read_fault = fault::io_error;
                bool rejected = false;
                try {
                    file.read_raw_unsafe_at(ptr, alignment, 0);
                } catch (const std::runtime_error &) {
                    rejected = true;
                }
                check(rejected, "EIO was swallowed");
            } catch (const std::exception & e) {
                fprintf(stderr, "%s\n", e.what());
                failed = true;
            }
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    check(!failed, "parallel reads failed");
    check(file.file_id() == fd && file.has_direct_io(), "fallback replaced the shared descriptor");
    check(file.tell() == 123, "positional read changed the shared cursor");
}

int main() {
    try {
        fixture f;
        llama_file_disk probe(f.path, "rb", true);
        if (!probe.has_direct_io()) {
            fprintf(stderr, "SKIP: filesystem does not support direct IO\n");
            return 77;
        }
        test_serial(f);
        test_parallel(f);
        fprintf(stderr, "PASS: bounded staging, sequential reads, concurrent fallback and EOF\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
