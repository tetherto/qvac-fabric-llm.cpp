#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "llama-coreml.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int hidden = 512;
static constexpr int dimension = 768;
static constexpr int vocabulary = 262144;
static constexpr int lengths[] = {32, 48, 64, 128, 256, 512};

static void set_error(char * error, size_t size, const char * message) {
    if (error && size) {
        snprintf(error, size, "%s", message);
    }
}

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::runtime_error ml_error(NSError * error) {
    return std::runtime_error(error ? error.localizedDescription.UTF8String : "Core ML operation failed");
}

struct coreml_function {
    MLModel * model = nil;
    MLDictionaryFeatureProvider * input = nil;
    std::vector<ggml_fp16_t> embeds;
    std::vector<ggml_fp16_t> mask;
    std::vector<ggml_fp16_t> positions;
    std::vector<ggml_fp16_t> pool;
    std::array<ggml_fp16_t, hidden> padding_row = {};
    bool padding_ready = false;
    int length = 0;
    int slots = 1;
    bool packed = false;
    bool raw = false;

    MLMultiArray * array(NSArray<NSNumber *> * shape, std::vector<ggml_fp16_t> & data) {
        NSMutableArray<NSNumber *> * strides = [NSMutableArray arrayWithCapacity:shape.count];
        size_t stride = 1;
        for (NSInteger i = shape.count - 1; i >= 0; --i) {
            [strides insertObject:@(stride) atIndex:0];
            stride *= shape[i].unsignedLongLongValue;
        }
        check(stride == data.size(), "invalid input buffer size");
        NSError * error = nil;
        MLMultiArray * result = [[MLMultiArray alloc] initWithDataPointer:data.data() shape:shape
                dataType:MLMultiArrayDataTypeFloat16 strides:strides deallocator:nil error:&error];
        if (!result) { throw ml_error(error); }
        return result;
    }

    void load(NSURL * url, int size, bool packing, MLModel * shared_model = nil, bool raw_output = false) {
        input = nil;
        length = size;
        packed = packing;
        raw = raw_output;
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        if (@available(macOS 15.0, iOS 18.0, *)) {
            config.functionName = packed ? @"pack_256" : [NSString stringWithFormat:@"embed_%d", length];
        }
        NSError * error = nil;
        model = shared_model ? shared_model : [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (!model) { throw ml_error(error); }
        if (packed) {
            MLMultiArrayConstraint * constraint = model.modelDescription.inputDescriptionsByName[@"pool"].multiArrayConstraint;
            check(constraint && constraint.shape.count == 2 && constraint.shape[1].intValue == length,
                    "unexpected packed pooling shape");
            slots = constraint.shape[0].intValue;
            check(slots == 8 || slots == 16, "packed model must have 8 or 16 slots");
        }
        if (raw) {
            auto constraint = model.modelDescription.outputDescriptionsByName[@"embedding_raw"].multiArrayConstraint;
            check(constraint && [constraint.shape isEqualToArray:@[@(slots), @(dimension)]],
                    "sidecar must provide embedding_raw outputs; re-export the model");
            check(constraint.dataType == MLMultiArrayDataTypeFloat32, "embedding_raw must use float32 output");
        }
        embeds.resize(length * hidden);
        mask.resize(packed ? length * length : length);
        NSMutableDictionary<NSString *, MLFeatureValue *> * features = [NSMutableDictionary dictionary];
        features[@"inputs_embeds"] = [MLFeatureValue featureValueWithMultiArray:array(@[@1, @(length), @(hidden)], embeds)];
        if (packed) {
            positions.resize(length);
            pool.resize(slots * length);
            features[@"attention_bias"] = [MLFeatureValue featureValueWithMultiArray:array(@[@1, @1, @(length), @(length)], mask)];
            features[@"positions"] = [MLFeatureValue featureValueWithMultiArray:array(@[@(length), @1], positions)];
            features[@"pool"] = [MLFeatureValue featureValueWithMultiArray:array(@[@(slots), @(length)], pool)];
        } else {
            features[@"attention_mask"] = [MLFeatureValue featureValueWithMultiArray:array(@[@1, @(length)], mask)];
        }
        auto descriptions = model.modelDescription.inputDescriptionsByName;
        check(descriptions.count == features.count, "unexpected Core ML input contract");
        for (NSString * name in features) {
            MLMultiArrayConstraint * constraint = descriptions[name].multiArrayConstraint;
            check(constraint && constraint.dataType == MLMultiArrayDataTypeFloat16 &&
                    [constraint.shape isEqualToArray:features[name].multiArrayValue.shape], "unexpected Core ML input shape or type");
        }
        input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
        if (!input) { throw ml_error(error); }
    }

    void write_tokens(const uint16_t * table, const int32_t * tokens, int count, int offset) {
        const float scale = sqrtf(hidden);
        for (int row = 0; row < count; ++row) {
            const size_t source = size_t(tokens[row]) * hidden;
            const size_t target = size_t(offset + row) * hidden;
            for (int col = 0; col < hidden; ++col) {
                uint32_t bits = uint32_t(table[source + col]) << 16;
                float value;
                memcpy(&value, &bits, sizeof(value));
                embeds[target + col] = ggml_fp32_to_fp16(value * scale);
            }
        }
    }

    void prepare(const uint16_t * table, const llama_coreml_embedding_sequence * sequences, int count) {
        // Padding uses token 0 and is excluded from pooling.
        if (!padding_ready) {
            int32_t token = 0;
            write_tokens(table, &token, 1, 0);
            std::copy(embeds.begin(), embeds.begin() + hidden, padding_row.begin());
            padding_ready = true;
        }
        for (int row = 0; row < length; ++row) {
            std::copy(padding_row.begin(), padding_row.end(), embeds.begin() + row * hidden);
        }
        std::fill(mask.begin(), mask.end(), ggml_fp32_to_fp16(packed ? -10000.0f : 0.0f));
        std::fill(positions.begin(), positions.end(), 0);
        std::fill(pool.begin(), pool.end(), 0);
        int offset = 0;
        for (int slot = 0; slot < count; ++slot) {
            const int n = sequences[slot].n_tokens;
            write_tokens(table, sequences[slot].tokens, n, offset);
            for (int row = offset; row < offset + n; ++row) {
                if (packed) {
                    for (int col = offset; col < offset + n; ++col) {
                        mask[row * length + col] = 0;
                    }
                    positions[row] = ggml_fp32_to_fp16(float(row - offset));
                    pool[slot * length + row] = ggml_fp32_to_fp16(1.0f / n);
                } else {
                    mask[row] = ggml_fp32_to_fp16(1.0f);
                }
            }
            offset += n;
        }
        if (packed) {
            // Padding rows attend to themselves to keep softmax finite.
            for (int row = offset; row < length; ++row) { mask[row * length + row] = 0; }
        }
    }

    void extract(id<MLFeatureProvider> result, int count, float * output) {
        MLMultiArray * embedding = [result featureValueForName:raw ? @"embedding_raw" : @"embedding"].multiArrayValue;
        check(embedding && ([embedding.shape isEqualToArray:@[@(slots), @(dimension)]] ||
                (packed && [embedding.shape isEqualToArray:@[@1, @(slots), @(dimension)]])),
                "unexpected Core ML output shape");
        check(embedding.dataType == MLMultiArrayDataTypeFloat16 || embedding.dataType == MLMultiArrayDataTypeFloat32,
                "unexpected Core ML output type");
        const size_t row_stride = embedding.strides[embedding.strides.count - 2].unsignedLongLongValue;
        const size_t col_stride = embedding.strides.lastObject.unsignedLongLongValue;
        __block bool valid = true;
        [embedding getBytesWithHandler:^(const void * bytes, NSInteger size) {
            const size_t element_size = embedding.dataType == MLMultiArrayDataTypeFloat16 ? 2 : 4;
            const size_t elements = size > 0 ? size_t(size) / element_size : 0;
            if (!elements || (count > 1 && row_stride > (elements - 1) / size_t(count - 1))) { valid = false; return; }
            const size_t last_row = size_t(count - 1) * row_stride;
            if (col_stride > (elements - 1 - last_row) / (dimension - 1)) { valid = false; return; }
            for (int row = 0; row < count; ++row) {
                for (int col = 0; col < dimension; ++col) {
                    const size_t index = row * row_stride + col * col_stride;
                    float value = embedding.dataType == MLMultiArrayDataTypeFloat16
                            ? ggml_fp16_to_fp32(((const ggml_fp16_t *) bytes)[index]) : ((const float *) bytes)[index];
                    if (!std::isfinite(value)) { valid = false; }
                    output[row * dimension + col] = value;
                }
            }
        }];
        check(valid, "invalid or non-finite Core ML output");
    }

    void run(const uint16_t * table, const llama_coreml_embedding_sequence * sequences, int count, float * output,
             llama_coreml_embedding_stats & stats) {
        const auto preparation_begin = std::chrono::steady_clock::now();
        prepare(table, sequences, count);
        const auto prediction_begin = std::chrono::steady_clock::now();
        NSError * error = nil;
        id<MLFeatureProvider> result = [model predictionFromFeatures:input error:&error];
        const auto output_begin = std::chrono::steady_clock::now();
        if (!result) { throw ml_error(error); }
        extract(result, count, output);
        const auto output_end = std::chrono::steady_clock::now();
        ++stats.calls;
        stats.preparation_ms += std::chrono::duration<double, std::milli>(prediction_begin - preparation_begin).count();
        stats.prediction_ms += std::chrono::duration<double, std::milli>(output_begin - prediction_begin).count();
        stats.output_ms += std::chrono::duration<double, std::milli>(output_end - output_begin).count();
    }
};

struct coreml_async_slot {
    coreml_function functions[7];
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    bool pending = false;
    std::string failure;
    llama_coreml_embedding_stats stats = {};

    void wait() {
        if (pending) {
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
            pending = false;
        }
    }
};

struct coreml_async_drain {
    std::array<coreml_async_slot, 8> & slots;
    ~coreml_async_drain() { for (auto & slot : slots) { slot.wait(); } }
};

struct llama_coreml_embedding {
    NSURL * url = nil;
    NSData * table = nil;
    coreml_function single[6];
    coreml_function packed;
    std::mutex mutex;
    int max_inflight = 1;
    bool raw = false;
    std::array<coreml_async_slot, 8> async_slots;
    llama_coreml_embedding_stats stats = {};
};

static llama_coreml_embedding * load_embedding(const char * directory, char * error, size_t error_size, bool raw) {
    set_error(error, error_size, "");
    @autoreleasepool {
        try {
            check(directory && directory[0], "model directory is required");
            if (@available(macOS 15.0, iOS 18.0, *)) {} else {
                throw std::runtime_error("requires macOS 15 or iOS 18");
            }
            NSString * path = [NSString stringWithUTF8String:directory];
            check(path != nil, "invalid UTF-8 directory");
            auto ctx = std::make_unique<llama_coreml_embedding>();
            ctx->raw = raw;
            NSError * failure = nil;
            ctx->table = [NSData dataWithContentsOfFile:[path stringByAppendingPathComponent:@"embeddings.bf16"]
                    options:NSDataReadingMappedAlways error:&failure];
            check(ctx->table && ctx->table.length == size_t(vocabulary) * hidden * 2, "missing or invalid embeddings.bf16");
            NSURL * compiled = [NSURL fileURLWithPath:[path stringByAppendingPathComponent:@"EmbeddingGemma2Text.mlmodelc"]];
            if (![[NSFileManager defaultManager] fileExistsAtPath:compiled.path]) {
                NSURL * package = [NSURL fileURLWithPath:[path stringByAppendingPathComponent:@"EmbeddingGemma2Text.mlpackage"]];
                NSURL * temporary = [MLModel compileModelAtURL:package error:&failure];
                if (!temporary) { throw ml_error(failure); }
                NSFileManager * files = [NSFileManager defaultManager];
                NSURL * staging = [NSURL fileURLWithPath:[compiled.path stringByAppendingFormat:@".%@.tmp", NSUUID.UUID.UUIDString]];
                try {
                    if (![files copyItemAtURL:temporary toURL:staging error:&failure]) {
                        throw ml_error(failure);
                    }
                    if (rename(staging.path.fileSystemRepresentation, compiled.path.fileSystemRepresentation) != 0) {
                        const int code = errno;
                        check((code == EEXIST || code == ENOTEMPTY) && [files fileExistsAtPath:compiled.path], strerror(code));
                    }
                } catch (...) {
                    [files removeItemAtURL:staging error:nullptr];
                    [files removeItemAtURL:temporary error:nullptr];
                    throw;
                }
                [files removeItemAtURL:staging error:nullptr];
                [files removeItemAtURL:temporary error:nullptr];
            }
            ctx->url = compiled;
            if (raw) {
                for (int i = 0; i < 6; ++i) {
                    ctx->single[i].load(ctx->url, lengths[i], false, nil, true);
                }
                ctx->packed.load(ctx->url, 256, true, nil, true);
            }
            return ctx.release();
        } catch (const std::exception & failure) {
            set_error(error, error_size, failure.what());
            return nullptr;
        }
    }
}

llama_coreml_embedding * llama_coreml_embedding_load(const char * directory, char * error, size_t error_size) {
    return load_embedding(directory, error, error_size, false);
}

llama_coreml_embedding * llama_coreml_embedding_load_raw(const char * directory, char * error, size_t error_size) {
    return load_embedding(directory, error, error_size, true);
}

void llama_coreml_embedding_free(llama_coreml_embedding * ctx) {
    delete ctx;
}

int32_t llama_coreml_embedding_set_max_inflight(llama_coreml_embedding * ctx, int32_t max_inflight) {
    if (!ctx || max_inflight < 1 || max_inflight > 8) { return -1; }
    try {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->max_inflight = max_inflight;
        return 0;
    } catch (const std::exception &) {
        return -1;
    }
}

int32_t llama_coreml_embedding_get_stats(llama_coreml_embedding * ctx, llama_coreml_embedding_stats * stats, bool reset) {
    if (!ctx || !stats) { return -1; }
    try {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        *stats = ctx->stats;
        if (reset) { ctx->stats = {}; }
        return 0;
    } catch (const std::exception &) {
        return -1;
    }
}

int32_t llama_coreml_embedding_encode(llama_coreml_embedding * ctx,
        const llama_coreml_embedding_sequence * sequences, int32_t n_sequences, bool pack,
        float * output, size_t output_capacity, char * error, size_t error_size) {
    set_error(error, error_size, "");
    @autoreleasepool {
        try {
            check(ctx && sequences && output && n_sequences > 0, "invalid embedding arguments");
            check(size_t(n_sequences) <= output_capacity / dimension, "output buffer is too small");
            for (int i = 0; i < n_sequences; ++i) {
                check(sequences[i].tokens && sequences[i].n_tokens > 0 && sequences[i].n_tokens <= 512,
                        "each sequence must contain 1..512 tokens");
                for (int j = 0; j < sequences[i].n_tokens; ++j) {
                    check(sequences[i].tokens[j] >= 0 && sequences[i].tokens[j] < vocabulary, "token ID is out of range");
                }
            }
            std::lock_guard<std::mutex> lock(ctx->mutex);
            coreml_async_drain drain{ctx->async_slots};
            for (auto & slot : ctx->async_slots) { slot.failure.clear(); slot.stats = {}; }
            const uint16_t * table = (const uint16_t *) ctx->table.bytes;
            int job = 0;
            for (int i = 0; i < n_sequences;) {
                int count = 1;
                int used = sequences[i].n_tokens;
                if (pack && i + 1 < n_sequences && used + sequences[i + 1].n_tokens <= 256) {
                    if (!ctx->packed.input) { ctx->packed.load(ctx->url, 256, true, nil, ctx->raw); }
                    while (i + count < n_sequences && count < ctx->packed.slots && used + sequences[i + count].n_tokens <= 256) {
                        used += sequences[i + count++].n_tokens;
                    }
                }
                coreml_function * function;
                int function_index = 6;
                if (count > 1) {
                    function = &ctx->packed;
                    if (!function->input) { function->load(ctx->url, 256, true, nil, ctx->raw); }
                } else {
                    int bucket = 0;
                    while (lengths[bucket] < used) { ++bucket; }
                    function_index = bucket;
                    function = &ctx->single[bucket];
                    if (!function->input) { function->load(ctx->url, lengths[bucket], false, nil, ctx->raw); }
                }
                if (ctx->max_inflight == 1) {
                    function->run(table, sequences + i, count, output + size_t(i) * dimension, ctx->stats);
                } else {
                    coreml_async_slot * slot = &ctx->async_slots[job++ % ctx->max_inflight];
                    slot->wait();
                    check(slot->failure.empty(), slot->failure.c_str());
                    coreml_function * buffer = &slot->functions[function_index];
                    if (!buffer->input) { buffer->load(ctx->url, function->length, function->packed, function->model, ctx->raw); }
                    const auto preparation_begin = std::chrono::steady_clock::now();
                    buffer->prepare(table, sequences + i, count);
                    const auto prediction_begin = std::chrono::steady_clock::now();
                    float * destination = output + size_t(i) * dimension;
                    slot->pending = true;
                    dispatch_semaphore_t completion = slot->semaphore;
                    // Each pending request owns its buffers until completion.
                    if (@available(macOS 15.0, iOS 18.0, *)) {
                        [function->model predictionFromFeatures:buffer->input completionHandler:^(id<MLFeatureProvider> result, NSError * failure) {
                            @autoreleasepool {
                                const auto output_begin = std::chrono::steady_clock::now();
                                try {
                                    if (!result) { throw ml_error(failure); }
                                    buffer->extract(result, count, destination);
                                    const auto output_end = std::chrono::steady_clock::now();
                                    ++slot->stats.calls;
                                    slot->stats.preparation_ms += std::chrono::duration<double, std::milli>(prediction_begin - preparation_begin).count();
                                    slot->stats.prediction_ms += std::chrono::duration<double, std::milli>(output_begin - prediction_begin).count();
                                    slot->stats.output_ms += std::chrono::duration<double, std::milli>(output_end - output_begin).count();
                                } catch (const std::exception & error) {
                                    slot->failure = error.what();
                                } catch (...) {
                                    slot->failure = "Core ML callback failed";
                                }
                            }
                            dispatch_semaphore_signal(completion);
                        }];
                    } else {
                        slot->pending = false;
                        throw std::runtime_error("asynchronous prediction requires macOS 15 or iOS 18");
                    }
                }
                i += count;
            }
            for (auto & slot : ctx->async_slots) {
                slot.wait();
                check(slot.failure.empty(), slot.failure.c_str());
                ctx->stats.calls += slot.stats.calls;
                ctx->stats.preparation_ms += slot.stats.preparation_ms;
                ctx->stats.prediction_ms += slot.stats.prediction_ms;
                ctx->stats.output_ms += slot.stats.output_ms;
            }
            return 0;
        } catch (const std::exception & failure) {
            set_error(error, error_size, failure.what());
            return -1;
        }
    }
}
