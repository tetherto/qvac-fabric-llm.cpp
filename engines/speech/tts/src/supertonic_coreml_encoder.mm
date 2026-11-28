#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "supertonic_coreml_encoder.h"

#include <cmath>
#include <cstdlib>
#include <mutex>

namespace {
bool shape_is(MLMultiArrayConstraint * c, NSArray<NSNumber *> * shape, MLMultiArrayDataType type) {
    return c != nil && c.dataType == type && [c.shape isEqualToArray:shape];
}
MLMultiArray * array(NSArray<NSNumber *> * shape, MLMultiArrayDataType type) {
    return [[MLMultiArray alloc] initWithShape:shape dataType:type error:nil];
}
}

struct supertonic_coreml_encoder_context {
    MLModel * model = nil;
    MLMultiArray * ids = nil;
    MLMultiArray * length = nil;
    MLMultiArray * style = nil;
    MLMultiArray * output = nil;
    MLDictionaryFeatureProvider * provider = nil;
    MLPredictionOptions * options = nil;
    int window = 0;
    int64_t vocabulary = 0;
    const char * label = "coreml-all";
    std::mutex mutex;
};

supertonic_coreml_encoder_context * supertonic_coreml_encoder_init(const char * path, int64_t vocabulary) {
    @autoreleasepool {
        auto * ctx = new supertonic_coreml_encoder_context;
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        const char * env = std::getenv("SUPERTONIC_COREML_ENCODER_COMPUTE_UNITS");
        if (!env) env = std::getenv("SUPERTONIC_COREML_COMPUTE_UNITS");
        NSString * units = env ? [NSString stringWithUTF8String:env] : @"";
        // ALL selected the GPU for this graph in local profiling. Prefer the
        // ANE, which gave lower encoder latency in local profiling.
        config.computeUnits = env ? MLComputeUnitsAll : MLComputeUnitsCPUAndNeuralEngine;
        ctx->label = env ? "coreml-all" : "coreml-ane";
        if ([units isEqualToString:@"cpu_only"]) {
            config.computeUnits = MLComputeUnitsCPUOnly; ctx->label = "coreml-cpu";
        } else if ([units isEqualToString:@"cpu_and_gpu"]) {
            config.computeUnits = MLComputeUnitsCPUAndGPU; ctx->label = "coreml-gpu";
        } else if ([units isEqualToString:@"cpu_and_ane"]) {
            config.computeUnits = MLComputeUnitsCPUAndNeuralEngine; ctx->label = "coreml-ane";
        }
        NSError * error = nil;
        ctx->model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]]
                                      configuration:config error:&error];
        auto inputs = ctx->model.modelDescription.inputDescriptionsByName;
        auto outputs = ctx->model.modelDescription.outputDescriptionsByName;
        MLMultiArrayConstraint * ids = inputs[@"text_ids"].multiArrayConstraint;
        const int window = ids.shape.count == 2 ? ids.shape[1].intValue : 0;
        bool valid = ctx->model != nil && error == nil && vocabulary > 0 && window > 0 &&
            inputs.count == 3 && outputs.count == 1 &&
            shape_is(ids, @[@1, @(window)], MLMultiArrayDataTypeInt32) &&
            shape_is(inputs[@"text_length"].multiArrayConstraint, @[@1], MLMultiArrayDataTypeInt32) &&
            shape_is(inputs[@"style_ttl"].multiArrayConstraint, @[@1, @50, @256], MLMultiArrayDataTypeFloat32) &&
            shape_is(outputs[@"text_emb"].multiArrayConstraint, @[@1, @256, @(window)], MLMultiArrayDataTypeFloat32);
        if (!valid) { delete ctx; return nullptr; }
        ctx->window = window;
        ctx->vocabulary = vocabulary;
        ctx->ids = array(@[@1, @(window)], MLMultiArrayDataTypeInt32);
        ctx->length = array(@[@1], MLMultiArrayDataTypeInt32);
        ctx->style = array(@[@1, @50, @256], MLMultiArrayDataTypeFloat32);
        ctx->output = array(@[@1, @256, @(window)], MLMultiArrayDataTypeFloat32);
        if (!ctx->ids || !ctx->length || !ctx->style || !ctx->output) { delete ctx; return nullptr; }
        ctx->provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:
            @{@"text_ids": ctx->ids, @"text_length": ctx->length, @"style_ttl": ctx->style} error:&error];
        if (!ctx->provider || error) { delete ctx; return nullptr; }
        ctx->options = [[MLPredictionOptions alloc] init];
        ctx->options.outputBackings = @{@"text_emb": ctx->output};
        return ctx;
    }
}

void supertonic_coreml_encoder_free(supertonic_coreml_encoder_context * ctx) { delete ctx; }

int supertonic_coreml_encoder_capacity(const supertonic_coreml_encoder_context * ctx) {
    return ctx ? ctx->window : 0;
}

const char * supertonic_coreml_encoder_backend_label(const supertonic_coreml_encoder_context * ctx) {
    return ctx ? ctx->label : "ggml";
}

bool supertonic_coreml_encoder_encode(supertonic_coreml_encoder_context * ctx,
                                      const int64_t * ids, int length, const float * style,
                                      std::vector<float> & embedding) {
    if (!ctx || !ids || !style || length <= 0 || length > ctx->window) return false;
    for (int t = 0; t < length; ++t) if (ids[t] < 0 || ids[t] >= ctx->vocabulary || ids[t] > INT32_MAX) return false;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    @autoreleasepool {
        auto * dest = static_cast<int32_t *>(ctx->ids.dataPointer);
        const auto stride = ctx->ids.strides[1].longLongValue;
        for (int t = 0; t < ctx->window; ++t) dest[t * stride] = t < length ? (int32_t) ids[t] : 0;
        *static_cast<int32_t *>(ctx->length.dataPointer) = length;
        auto * sd = static_cast<float *>(ctx->style.dataPointer);
        for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c)
            sd[t * ctx->style.strides[1].longLongValue + c * ctx->style.strides[2].longLongValue] = style[t * 256 + c];
        NSError * error = nil;
        id<MLFeatureProvider> result = [ctx->model predictionFromFeatures:ctx->provider options:ctx->options error:&error];
        MLMultiArray * out = [result featureValueForName:@"text_emb"].multiArrayValue;
        if (error || !out || out.dataType != MLMultiArrayDataTypeFloat32 ||
            ![out.shape isEqualToArray:@[@1, @256, @(ctx->window)]]) return false;
        std::vector<float> values((size_t) length * 256);
        const auto * src = static_cast<const float *>(out.dataPointer);
        for (int c = 0; c < 256; ++c) for (int t = 0; t < length; ++t) {
            float value = src[c * out.strides[1].longLongValue + t * out.strides[2].longLongValue];
            if (!std::isfinite(value)) return false;
            values[(size_t) c * length + t] = value;
        }
        embedding.swap(values);
        return true;
    }
}
