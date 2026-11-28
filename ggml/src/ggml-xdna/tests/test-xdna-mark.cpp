// The pattern that tells a write the array made from one it did not
// (xdna_buffer_mark, xdna_buffer_mark_bulk, xdna_buffer_wait_written), on plain
// host memory with a thread standing in for the array's DMA.
// Host-only: no device is opened.

#include "../xdna-runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static constexpr uint32_t FILLER      = 0x12345678u;
static constexpr size_t   BUFFER_SIZE = 1 << 16;
static constexpr size_t   LINE_BYTES = 64;

namespace {

// A buffer of host memory with the fields the mark protocol reads
struct host_buffer {
    xdna_buffer buf;
    uint32_t *  words = nullptr;

    host_buffer() {
        words = (uint32_t *) std::aligned_alloc(LINE_BYTES, BUFFER_SIZE);
        std::fill(words, words + BUFFER_SIZE / sizeof(uint32_t), FILLER);
        buf.data  = words;
        buf.bytes = BUFFER_SIZE;
    }

    ~host_buffer() { std::free(words); }
};

struct mark_case {
    size_t offset;
    size_t bytes;
};

// Whole words, a head and a tail that are not 16-byte aligned, a range inside one cache line, and one that
// starts at a line's last word
const mark_case MARK_CASES[] = {
    { 0, 4096 }, { 4, 4096 }, { 20, 8196 }, { 36, 28 }, { 60, 8 }, { 64, 64 }, { 100, 4 }, { 8192, 16384 },
};

using mark_fn = void (*)(xdna_buffer *, size_t, size_t);

bool poisoned_exactly(const host_buffer & hb, size_t offset, size_t bytes) {
    for (size_t i = 0; i < BUFFER_SIZE / sizeof(uint32_t); i++) {
        const size_t at     = i * sizeof(uint32_t);
        const bool   inside = at >= offset && at < offset + bytes;
        if (hb.words[i] != (inside ? XDNA_POISON_F32 : FILLER)) {
            return false;
        }
    }
    return true;
}

int check_mark(const char * name, mark_fn mark) {
    int failures = 0;
    for (const mark_case & c : MARK_CASES) {
        host_buffer hb;
        mark(&hb.buf, c.bytes, c.offset);
        const bool ok = poisoned_exactly(hb, c.offset, c.bytes);
        std::printf("%s: %s offset %zu bytes %zu\n", ok ? "ok  " : "FAIL", name, c.offset, c.bytes);
        failures += ok ? 0 : 1;
    }
    return failures;
}

void write_words(host_buffer & hb, size_t offset, size_t bytes, size_t skip_line_at) {
    for (size_t at = offset; at < offset + bytes; at += sizeof(uint32_t)) {
        const bool lost = skip_line_at != SIZE_MAX && at / LINE_BYTES == skip_line_at / LINE_BYTES;
        if (!lost) {
            hb.words[at / sizeof(uint32_t)] = 0x3F800000u;
        }
    }
}

// A range the array wrote after a delay is returned in place
int check_waits_for_a_late_write(mark_fn mark) {
    host_buffer hb;
    mark(&hb.buf, 8192, 128);
    std::thread array([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        write_words(hb, 128, 8192, SIZE_MAX);
    });
    const uint8_t * got = xdna_buffer_wait_written(&hb.buf, 8192, 128);
    array.join();
    const bool ok = got == (const uint8_t *) hb.words + 128;
    std::printf("%s: a write that lands after the first look is waited for\n", ok ? "ok  " : "FAIL");
    return ok ? 0 : 1;
}

// A range the array never wrote, and one it wrote but for one cache line: the lost write is reported, not read
int check_reports_a_lost_write(mark_fn mark) {
    int failures = 0;
    {
        host_buffer hb;
        mark(&hb.buf, 8192, 0);
        const bool ok = xdna_buffer_wait_written(&hb.buf, 8192, 0) == nullptr;
        std::printf("%s: a range never written is reported\n", ok ? "ok  " : "FAIL");
        failures += ok ? 0 : 1;
    }
    {
        host_buffer hb;
        mark(&hb.buf, 8192, 0);
        write_words(hb, 0, 8192, 4096);
        const bool ok = xdna_buffer_wait_written(&hb.buf, 8192, 0) == nullptr;
        std::printf("%s: one line of the range never written is reported\n", ok ? "ok  " : "FAIL");
        failures += ok ? 0 : 1;
    }
    return failures;
}

}  // namespace

int main(void) {
    int failures = 0;
    failures += check_mark("mark", xdna_buffer_mark);
    failures += check_mark("mark_bulk", xdna_buffer_mark_bulk);
    failures += check_waits_for_a_late_write(xdna_buffer_mark_bulk);
    failures += check_reports_a_lost_write(xdna_buffer_mark_bulk);
    if (failures) {
        std::printf("test-xdna-mark: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-mark: all passed\n");
    return 0;
}
