#ifndef HTP_DMA_READER_H
#define HTP_DMA_READER_H

#include <assert.h>
#include "dma-queue.h"
#include "htp-tensor.h"

enum { HTP_DMA_READER_BYTES = 4096, HTP_DMA_READER_ALIGNMENT = 128 };

struct htp_dma_reader {
    const struct htp_tensor * tensor;
    dma_queue * queue;
    uint8_t * window;
    uint32_t begin;
    uint32_t size;
};

static inline struct htp_dma_reader htp_dma_reader_init(const struct htp_tensor * tensor, dma_queue * queue,
                                                        uint8_t * window) {
    struct htp_dma_reader reader = { tensor, queue, window, 0, 0 };
    return reader;
}

static inline const void * htp_dma_reader_view(struct htp_dma_reader * reader, uint32_t offset, uint32_t size) {
    const struct htp_tensor * tensor = reader->tensor;
    assert(offset <= tensor->size && size <= tensor->size - offset);
    if (!htp_tensor_is_extended(tensor)) {
        return (const uint8_t *) (uintptr_t) tensor->data + offset;
    }
    assert(size <= HTP_DMA_READER_BYTES - HTP_DMA_READER_ALIGNMENT);
    if (offset < reader->begin || offset - reader->begin > reader->size ||
        size > reader->size - (offset - reader->begin)) {
        reader->begin = offset & ~(HTP_DMA_READER_ALIGNMENT - 1U);
        reader->size = MIN(HTP_DMA_READER_BYTES, tensor->size - reader->begin);
        dma_queue_copy_rows(reader->queue, dma_make_data(reader->window, tensor->data + reader->begin),
                            reader->size, reader->size, reader->size, 1);
    }
    return reader->window + offset - reader->begin;
}

#endif
