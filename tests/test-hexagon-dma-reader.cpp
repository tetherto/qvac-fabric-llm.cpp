#undef NDEBUG
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#define HTP_DMA_H
#define HTP_TENSOR_H
#define MIN(a, b) std::min<uint32_t>((a), (b))

struct htp_tensor { uint64_t data; uint32_t size; };
static bool htp_tensor_is_extended(const htp_tensor * tensor) { return tensor->data >> 32; }

struct dma_queue {
    uint64_t base;
    std::vector<uint8_t> source;
    size_t calls = 0;
    size_t largest = 0;
};
struct dma_data { uint8_t * dst; uint64_t src; };
static dma_data dma_make_data(uint8_t * dst, uint64_t src) { return {dst, src}; }
static void dma_queue_copy_rows(dma_queue * queue, dma_data data, size_t, size_t, size_t bytes, size_t rows) {
    assert(rows == 1);
    assert(data.src >= queue->base);
    const uint64_t offset = data.src - queue->base;
    assert(offset <= queue->source.size() && bytes <= queue->source.size() - offset);
    std::memcpy(data.dst, queue->source.data() + offset, bytes);
    ++queue->calls;
    queue->largest = std::max(queue->largest, bytes);
}

#include "../ggml/src/ggml-hexagon/htp/htp-dma-reader.h"

static void check_read(htp_dma_reader & reader, dma_queue & queue, uint32_t offset, uint32_t bytes) {
    const void * data = htp_dma_reader_view(&reader, offset, bytes);
    assert(std::memcmp(data, queue.source.data() + offset, bytes) == 0);
}

static void test_window_edges(htp_dma_reader & reader, dma_queue & queue) {
    check_read(reader, queue, 0, 4);
    const size_t calls = queue.calls;
    check_read(reader, queue, 128, 128);
    assert(queue.calls == calls);
    check_read(reader, queue, HTP_DMA_READER_BYTES - 2, 4);
    check_read(reader, queue, 2 * HTP_DMA_READER_BYTES + 1, 256);
    check_read(reader, queue, 127, 1152);
    check_read(reader, queue, queue.source.size() - 2, 2);
    assert(queue.largest <= HTP_DMA_READER_BYTES);
}

static void test_operation_reset(const htp_tensor & tensor, dma_queue & queue, uint8_t * scratch) {
    auto reader = htp_dma_reader_init(&tensor, &queue, scratch);
    check_read(reader, queue, 0, 128);
    queue.source[0] ^= 0xff;
    const size_t calls = queue.calls;
    reader = htp_dma_reader_init(&tensor, &queue, scratch);
    check_read(reader, queue, 0, 128);
    assert(queue.calls == calls + 1);
}

int main() {
    dma_queue queue {UINT64_C(0x1234567800), std::vector<uint8_t>(3 * HTP_DMA_READER_BYTES + 6)};
    for (size_t i = 0; i < queue.source.size(); ++i) {
        queue.source[i] = static_cast<uint8_t>(i * 31 + i / 251);
    }
    const htp_tensor tensor {queue.base, static_cast<uint32_t>(queue.source.size())};
    alignas(128) std::array<uint8_t, HTP_DMA_READER_BYTES> scratch;
    auto reader = htp_dma_reader_init(&tensor, &queue, scratch.data());
    test_window_edges(reader, queue);
    test_operation_reset(tensor, queue, scratch.data());
}
