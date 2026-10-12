#include "ggml-metal-coreml.h"
#include "ggml-metal-coreml-policy.h"
#include "ggml-metal-coreml-kernels.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#if !defined(__MAC_OS_X_VERSION_MAX_ALLOWED) || __MAC_OS_X_VERSION_MAX_ALLOWED < 150000
#error GGML_METAL_COREML requires the macOS 15 SDK or newer
#endif

namespace {
static_assert(QK8_0 == 32 && sizeof(block_q8_0) == 34 && offsetof(block_q8_0, qs) == 2, "update the CoreML packing kernel for the Q8_0 layout");

struct surface {
    CVPixelBufferRef pixel = nullptr;
    id<MTLTexture> texture = nil;
    MLMultiArray * array = nil;
    ~surface() {
        [array release];
        [texture release];
        if (pixel) { CFRelease(pixel); }
    }
    bool init(id<MTLDevice> device, int width, int height) API_AVAILABLE(macos(15.0)) {
        NSDictionary * attrs = @{ (id) kCVPixelBufferIOSurfacePropertiesKey: @{},
                                  (id) kCVPixelBufferMetalCompatibilityKey: @YES };
        if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent16Half,
                                (CFDictionaryRef) attrs, &pixel) != kCVReturnSuccess) {
            return false;
        }
        // The multiarray and Metal texture refer to the same IOSurface, including row padding.
        IOSurfaceRef io = CVPixelBufferGetIOSurface(pixel);
        if (!io) { return false; }
        MTLTextureDescriptor * desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                        width:width height:height mipmapped:NO];
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        texture = [device newTextureWithDescriptor:desc iosurface:io plane:0];
        array = [[MLMultiArray alloc] initWithPixelBuffer:pixel shape:@[@1, @1, @(height), @(width)]];
        return texture && array && array.strides.count == 4 &&
               array.strides[3].unsignedLongLongValue == 1 &&
               array.strides[2].unsignedLongLongValue * sizeof(uint16_t) == CVPixelBufferGetBytesPerRow(pixel);
    }
};

struct compiled_model {
    NSURL * url = nil;
    ~compiled_model() {
        @autoreleasepool {
            if (url) { [[NSFileManager defaultManager] removeItemAtURL:url error:nullptr]; }
            [url release];
        }
    }
};

struct program {
    std::shared_ptr<compiled_model> compiled;
    MLModel * model = nil;
    surface w, x, y;
    std::string failure;
    ~program() { [model release]; }
};

bool env_on(const char * name) {
    const char * value = getenv(name);
    return value && strcmp(value, "1") == 0;
}

struct plan_result {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    bool found = false;
    ~plan_result() { if (done) { dispatch_release(done); } }
};

bool is_matmul_operation(NSString * name) {
    if (!name) { return false; }
    return [name isEqualToString:@"matmul"] ||
           [name rangeOfString:@"\\Aios[0-9]+\\.matmul\\z" options:NSRegularExpressionSearch].location != NSNotFound;
}

bool plan_uses_ane(std::shared_ptr<compiled_model> compiled, MLModelConfiguration * config) API_AVAILABLE(macos(15.0)) {
    auto result = std::make_shared<plan_result>();
    if (!result->done) { return false; }
    [MLComputePlan loadContentsOfURL:compiled->url configuration:config completionHandler:^(MLComputePlan * plan, NSError * error) {
        @autoreleasepool {
            @try {
                if (!error && plan.modelStructure.program) {
                    MLModelStructureProgramFunction * function = plan.modelStructure.program.functions[@"main"];
                    for (MLModelStructureProgramOperation * op in function.block.operations) {
                        if (is_matmul_operation(op.operatorName)) {
                            id device = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
                            result->found = [device isKindOfClass:[MLNeuralEngineComputeDevice class]];
                            if (!result->found) { break; }
                        }
                    }
                }
            } @catch (NSException * exception) {
                result->found = false;
                GGML_UNUSED(exception);
            } @finally {
                // Keep the file and completion state alive even after a timeout and cache eviction.
                GGML_UNUSED(compiled);
                dispatch_semaphore_signal(result->done);
            }
        }
    }];
    return dispatch_semaphore_wait(result->done, dispatch_time(DISPATCH_TIME_NOW, 30*NSEC_PER_SEC)) == 0 && result->found;
}

bool matches(MLFeatureDescription * desc, NSArray<NSNumber *> * shape) API_AVAILABLE(macos(15.0)) {
    return desc.type == MLFeatureTypeMultiArray &&
           desc.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat16 &&
           [desc.multiArrayConstraint.shape isEqualToArray:shape];
}

bool overlaps(ggml_metal_buffer_id a, size_t asize, ggml_metal_buffer_id b, size_t bsize) {
    if (a.metal == b.metal) { return a.offs < b.offs + bsize && b.offs < a.offs + asize; }
    // Mapped tensors can have distinct Metal objects for overlapping host memory.
    if ([(id<MTLBuffer>) a.metal storageMode] == MTLStorageModePrivate ||
        [(id<MTLBuffer>) b.metal storageMode] == MTLStorageModePrivate) { return false; }
    const void * abase = [(id<MTLBuffer>) a.metal contents];
    const void * bbase = [(id<MTLBuffer>) b.metal contents];
    if (!abase || !bbase) { return false; }
    const uintptr_t ap = (uintptr_t) abase + a.offs;
    const uintptr_t bp = (uintptr_t) bbase + b.offs;
    return ap <= bp ? bp - ap < asize : ap - bp < bsize;
}

bool valid_buffer(ggml_metal_buffer_id bid, const ggml_tensor * t, id<MTLDevice> device) {
    id<MTLBuffer> buffer = (id<MTLBuffer>) bid.metal;
    const size_t alignment = t->type == GGML_TYPE_Q8_0 ? 1 : ggml_type_size(t->type);
    return buffer && buffer.device == device && bid.offs % alignment == 0 &&
           bid.offs <= buffer.length && ggml_nbytes(t) <= buffer.length - bid.offs;
}

bool complete(id<MTLCommandBuffer> cb) {
    if (!cb) { return false; }
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted;
}

void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLTexture> texture) {
    [enc setComputePipelineState:pipeline];
    [enc setTexture:texture atIndex:0];
    [enc dispatchThreads:MTLSizeMake(texture.width, texture.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
}
} // namespace

struct ggml_metal_coreml {
    ggml_metal_device_t dev = nullptr;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pack_q8 = nil, pack_f16 = nil, pack_f32 = nil, unpack = nil;
    id<MTLBuffer> bad = nil;
    bool enabled = false;
    bool debug = false;
    std::string directory;
    std::mutex mutex;
    std::map<std::array<int64_t, 3>, std::unique_ptr<program>> cache;
    ~ggml_metal_coreml() {
        [pack_q8 release]; [pack_f16 release]; [pack_f32 release]; [unpack release]; [bad release];
    }
};

void ggml_metal_coreml_log(ggml_metal_coreml_t ctx, const ggml_tensor * op, const char * reason) {
    if (ctx && ctx->debug && op->op == GGML_OP_MUL_MAT) {
        GGML_LOG_INFO("coreml: route=Metal tensor=%s m=%lld n=%lld k=%lld reason=%s\n", op->name,
                      (long long) op->ne[0], (long long) op->ne[1], (long long) (op->src[0] ? op->src[0]->ne[0] : 0), reason);
    }
}

ggml_metal_coreml_t ggml_metal_coreml_init(ggml_metal_device_t dev) try {
    auto ctx = std::unique_ptr<ggml_metal_coreml>(new ggml_metal_coreml);
    ctx->dev = dev;
    ctx->debug = env_on("GGML_METAL_COREML_DEBUG");
    ctx->device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);
    ctx->queue = (id<MTLCommandQueue>) ggml_metal_device_get_queue(dev);
    const char * dir = getenv("GGML_METAL_COREML_MODEL_DIR");
    if (dir) { ctx->directory = dir; }
    if (!env_on("GGML_METAL_COREML") || ctx->directory.empty()) { return ctx.release(); }
    @autoreleasepool {
        @try {
            if (@available(macOS 15.0, *)) {
#if defined(__aarch64__) || defined(__arm64__)
                if (ctx->device.hasUnifiedMemory) {
                    for (id device in [MLModel availableComputeDevices]) {
                        if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) { ctx->enabled = true; }
                    }
                }
#endif
            }
        } @catch (NSException * exception) {
            ctx->enabled = false;
            GGML_UNUSED(exception);
        }
    }
    return ctx.release();
} catch (const std::exception &) {
    return nullptr;
}

ggml_metal_buffer_id ggml_metal_coreml_buffer_id(const ggml_tensor * t) {
    if (!t || !t->data) { return {nullptr, 0}; }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (!ggml_backend_buffer_is_metal(buffer) || !buffer->context) { return {nullptr, 0}; }
    return ggml_metal_buffer_get_id((ggml_metal_buffer_t) buffer->context, t);
}

void ggml_metal_coreml_free(ggml_metal_coreml_t ctx) { delete ctx; }
bool ggml_metal_coreml_enabled(ggml_metal_coreml_t ctx) { return ctx && ctx->enabled; }
bool ggml_metal_coreml_candidate(const ggml_tensor * op) { return ggml_metal_coreml_node_supported(op); }

static bool pipelines(ggml_metal_coreml_t ctx) {
    if (ctx->bad) { return true; }
    NSError * error = nil;
    MTLCompileOptions * options = [[[MTLCompileOptions alloc] init] autorelease];
    options.fastMathEnabled = NO;
    id<MTLLibrary> lib = [[ctx->device newLibraryWithSource:[NSString stringWithUTF8String:ggml_metal_coreml_source]
                                                 options:options error:&error] autorelease];
    if (!lib) { return false; }
    auto make = [&](NSString * name, id<MTLComputePipelineState> & pipeline) {
        id<MTLFunction> function = [[lib newFunctionWithName:name] autorelease];
        id<MTLComputePipelineState> replacement = function ? [ctx->device newComputePipelineStateWithFunction:function error:&error] : nil;
        [pipeline release];
        pipeline = replacement;
        if (pipeline && pipeline.maxTotalThreadsPerThreadgroup < 128) {
            [pipeline release];
            pipeline = nil;
        }
    };
    make(@"pack_q8", ctx->pack_q8); make(@"pack_f16", ctx->pack_f16);
    make(@"pack_f32", ctx->pack_f32); make(@"unpack_f32", ctx->unpack);
    if (!ctx->pack_q8 || !ctx->pack_f16 || !ctx->pack_f32 || !ctx->unpack) { return false; }
    ctx->bad = [ctx->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    return ctx->bad != nil;
}

static program * load_program(ggml_metal_coreml_t ctx, int64_t m, int64_t n, int64_t k) API_AVAILABLE(macos(15.0)) {
    const std::array<int64_t, 3> key = {m, n, k};
    auto it = ctx->cache.find(key);
    if (it != ctx->cache.end()) { return it->second.get(); }
    // Four shapes bound scratch to 512 MiB. Inputs are repacked on every call.
    if (ctx->cache.size() >= 4) { ctx->cache.erase(ctx->cache.begin()); }
    auto value = std::unique_ptr<program>(new program);
    program * p = value.get();
    p->failure = "model-load-or-layout";
    ctx->cache.emplace(key, std::move(value));
    NSString * name = [NSString stringWithFormat:@"q8-fp16-v1-%lld-%lld-%lld.mlpackage", (long long) m, (long long) n, (long long) k];
    NSString * path = [[NSString stringWithUTF8String:ctx->directory.c_str()] stringByAppendingPathComponent:name];
    NSError * error = nil;
    p->compiled = std::make_shared<compiled_model>();
    p->compiled->url = [[MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&error] retain];
    if (!p->compiled->url) { return p; }
    MLModelConfiguration * config = [[[MLModelConfiguration alloc] init] autorelease];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    if (!plan_uses_ane(p->compiled, config)) {
        p->failure = "compute-plan-does-not-prefer-ANE";
    } else {
        p->model = [[MLModel modelWithContentsOfURL:p->compiled->url configuration:config error:&error] retain];
    }
    if (!p->model) { return p; }
    MLModelDescription * desc = p->model.modelDescription;
    if (!matches(desc.inputDescriptionsByName[@"w"], @[@1, @1, @(m), @(k)]) ||
        !matches(desc.inputDescriptionsByName[@"x"], @[@1, @1, @(n), @(k)]) ||
        !matches(desc.outputDescriptionsByName[@"y"], @[@1, @1, @(n), @(m)]) ||
        !p->w.init(ctx->device, (int) k, (int) m) || !p->x.init(ctx->device, (int) k, (int) n) ||
        !p->y.init(ctx->device, (int) m, (int) n)) { return p; }
    p->failure.clear();
    return p;
}

static bool compute(ggml_metal_coreml_t ctx, ggml_tensor * op, const char * & reason) API_AVAILABLE(macos(15.0)) {
    if (env_on("GGML_METAL_COREML_FORCE_FAILURE")) { reason = "injected-failure"; return false; }
    const auto w = ggml_metal_coreml_buffer_id(op->src[0]), x = ggml_metal_coreml_buffer_id(op->src[1]), y = ggml_metal_coreml_buffer_id(op);
    if (!valid_buffer(w, op->src[0], ctx->device) || !valid_buffer(x, op->src[1], ctx->device) ||
        !valid_buffer(y, op, ctx->device)) { reason = "buffer"; return false; }
    if (overlaps(y, ggml_nbytes(op), w, ggml_nbytes(op->src[0])) ||
        overlaps(y, ggml_nbytes(op), x, ggml_nbytes(op->src[1]))) { reason = "aliased-output"; return false; }
    if (!pipelines(ctx)) { reason = "packing-pipeline"; return false; }
    program * p = load_program(ctx, op->ne[0], op->ne[1], op->src[0]->ne[0]);
    if (!p->failure.empty()) { reason = p->failure.c_str(); return false; }
    ggml_metal_device_rsets_keep_alive(ctx->dev);
    *(uint32_t *) ctx->bad.contents = 0;
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) { reason = "packing-encoder"; return false; }
    [enc setBuffer:ctx->bad offset:0 atIndex:1];
    [enc setBuffer:(id<MTLBuffer>) w.metal offset:w.offs atIndex:0];
    dispatch(enc, ctx->pack_q8, p->w.texture);
    [enc setBuffer:(id<MTLBuffer>) x.metal offset:x.offs atIndex:0];
    dispatch(enc, op->src[1]->type == GGML_TYPE_F32 ? ctx->pack_f32 : ctx->pack_f16, p->x.texture);
    [enc endEncoding];
    if (!complete(cb) || *(uint32_t *) ctx->bad.contents) { reason = "packing-or-fp16-range"; return false; }
    const char * fault = getenv("GGML_METAL_COREML_FORCE_FAILURE");
    if (fault && strcmp(fault, "prediction") == 0) { reason = "injected-prediction-failure"; return false; }

    NSError * error = nil;
    MLDictionaryFeatureProvider * inputs = [[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        @"w": [MLFeatureValue featureValueWithMultiArray:p->w.array],
        @"x": [MLFeatureValue featureValueWithMultiArray:p->x.array]} error:&error] autorelease];
    if (!inputs) {
        if (ctx->debug && error) { GGML_LOG_WARN("coreml: input-provider error: %s\n", error.description.UTF8String); }
        reason = "input-provider"; return false;
    }
    MLPredictionOptions * options = [[[MLPredictionOptions alloc] init] autorelease];
    options.outputBackings = @{ @"y": p->y.array };
    id<MLFeatureProvider> result = [p->model predictionFromFeatures:inputs options:options error:&error];
    if (!result || error) {
        if (ctx->debug && error) { GGML_LOG_WARN("coreml: prediction error: %s\n", error.description.UTF8String); }
        reason = "prediction-error"; return false;
    }
    MLMultiArray * output = [result featureValueForName:@"y"].multiArrayValue;
    if (!output) { reason = "output-missing"; return false; }
    if (output.dataType != MLMultiArrayDataTypeFloat16) { reason = "output-type"; return false; }
    if (![output.shape isEqualToArray:p->y.array.shape]) { reason = "output-shape"; return false; }
    if (![output.strides isEqualToArray:p->y.array.strides]) { reason = "output-strides"; return false; }
    // Compare backing objects, not dataPointer: accessing it locks the pixel buffer
    // and prevents reuse of this output backing by a later prediction.
    if (output != p->y.array) { reason = "output-backing-not-used"; return false; }
    // Check before writing the GGML destination so all prediction failures can fall back.
    if (CVPixelBufferLockBaseAddress(p->y.pixel, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
        reason = "output-lock"; return false;
    }
    bool finite = true;
    if (!CVPixelBufferGetBaseAddress(p->y.pixel)) { finite = false; }
    for (int64_t row = 0; row < op->ne[1] && finite; ++row) {
        const auto * values = (const uint16_t *) ((const char *) CVPixelBufferGetBaseAddress(p->y.pixel) + row*CVPixelBufferGetBytesPerRow(p->y.pixel));
        for (int64_t col = 0; col < op->ne[0]; ++col) {
            if ((values[col] & 0x7c00) == 0x7c00) { finite = false; break; }
        }
    }
    CVPixelBufferUnlockBaseAddress(p->y.pixel, kCVPixelBufferLock_ReadOnly);
    if (!finite) { reason = "nonfinite-output"; return false; }
    ggml_metal_device_rsets_keep_alive(ctx->dev);
    cb = [ctx->queue commandBuffer];
    enc = [cb computeCommandEncoder];
    if (!enc) { reason = "output-encoder"; return false; }
    [enc setBuffer:(id<MTLBuffer>) y.metal offset:y.offs atIndex:0];
    dispatch(enc, ctx->unpack, p->y.texture);
    [enc endEncoding];
    if (!complete(cb)) { reason = "output-command-buffer"; return false; }
    return true;
}

bool ggml_metal_coreml_compute(ggml_metal_coreml_t ctx, ggml_tensor * op) {
    if (!ggml_metal_coreml_enabled(ctx) || !ggml_metal_coreml_candidate(op)) { return false; }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    @autoreleasepool {
        const char * reason = "os-version";
        bool ok = false;
        try {
            @try {
                if (@available(macOS 15.0, *)) { ok = compute(ctx, op, reason); }
            } @catch (NSException * exception) {
                reason = "CoreML-exception";
                GGML_UNUSED(exception);
            }
        } catch (const std::exception &) { reason = "allocation-or-runtime-exception"; }
        if (ok) {
            if (ctx->debug) {
                GGML_LOG_INFO("coreml: route=ANE-plan tensor=%s m=%lld n=%lld k=%lld (CoreML CPUAndNeuralEngine; placement not measured)\n",
                              op->name, (long long) op->ne[0], (long long) op->ne[1], (long long) op->src[0]->ne[0]);
            }
        } else {
            ggml_metal_coreml_log(ctx, op, reason);
        }
        return ok;
    }
}
