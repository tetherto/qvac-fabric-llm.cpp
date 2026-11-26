#if !defined(__APPLE__)
#error "codec-synth.mm is Apple-only; compile it solely under TTS_CPP_USE_COREML"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "audio8/coreml/codec-synth.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

float decode_float16(uint16_t h) {
    __fp16 v;
    std::memcpy(&v, &h, sizeof(v));
    return (float) v;
}

uint16_t encode_float16(float f) {
    __fp16 v = (__fp16) f;
    uint16_t out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

struct placement {
    MLComputeUnits units;
    const char *   label;
};

constexpr placement CPU_PLACEMENT = {MLComputeUnitsCPUOnly, "coreml-cpu"};
constexpr placement GPU_PLACEMENT = {MLComputeUnitsCPUAndGPU, "coreml-gpu"};
constexpr placement ANE_PLACEMENT = {MLComputeUnitsCPUAndNeuralEngine, "coreml-ane"};
constexpr placement ALL_PLACEMENT = {MLComputeUnitsAll, "coreml-all"};

placement requested_placement() {
    const char * env = std::getenv("AUDIO8_COREML_COMPUTE_UNITS");
    const std::string requested = env != nullptr ? env : "";
    if (requested == "cpu_only") return CPU_PLACEMENT;
    if (requested == "cpu_and_gpu") return GPU_PLACEMENT;
    if (requested == "cpu_and_ane") return ANE_PLACEMENT;
    return ALL_PLACEMENT;
}

MLModel * load_model(const char * path_mlmodelc, std::string * label) {
    NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
    if (path == nil) return nil;
    const placement where = requested_placement();
    *label = where.label;
    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = where.units;
    NSError * err   = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                        configuration:config
                                                error:&err];
    return (model == nil || err != nil) ? nil : model;
}

NSString * sole_multiarray_feature(NSDictionary<NSString *, MLFeatureDescription *> * features) {
    NSString * found = nil;
    for (NSString * name in features) {
        if (features[name].type != MLFeatureTypeMultiArray) continue;
        if (found != nil) return nil;
        found = name;
    }
    return found;
}

std::vector<int64_t> dims_of(NSArray<NSNumber *> * shape) {
    std::vector<int64_t> dims;
    for (NSNumber * n in shape) dims.push_back(n.longLongValue);
    return dims;
}

bool supported_data_type(MLMultiArrayDataType dt) {
    return dt == MLMultiArrayDataTypeFloat32 || dt == MLMultiArrayDataTypeFloat16;
}

bool shape_is(const std::vector<int64_t> & dims, int64_t channels, int64_t length) {
    if (dims.size() < 2) return false;
    for (std::size_t i = 0; i + 2 < dims.size(); ++i) {
        if (dims[i] != 1) return false;
    }
    return dims[dims.size() - 2] == channels && dims[dims.size() - 1] == length;
}

int64_t validate_model_interface(MLModel * model, int64_t latent_dim, int64_t frame_size,
                                 NSString ** in_name, NSString ** out_name) {
    *in_name  = sole_multiarray_feature(model.modelDescription.inputDescriptionsByName);
    *out_name = sole_multiarray_feature(model.modelDescription.outputDescriptionsByName);
    if (*in_name == nil || *out_name == nil) return 0;

    MLMultiArrayConstraint * in_constraint =
        model.modelDescription.inputDescriptionsByName[*in_name].multiArrayConstraint;
    MLMultiArrayConstraint * out_constraint =
        model.modelDescription.outputDescriptionsByName[*out_name].multiArrayConstraint;
    if (in_constraint == nil || out_constraint == nil) return 0;
    if (!supported_data_type(in_constraint.dataType) ||
        !supported_data_type(out_constraint.dataType)) return 0;

    const std::vector<int64_t> in_dims  = dims_of(in_constraint.shape);
    const std::vector<int64_t> out_dims = dims_of(out_constraint.shape);
    if (in_dims.size() < 2 || in_dims.back() <= 0) return 0;
    const int64_t window = in_dims.back();
    if (!shape_is(in_dims, latent_dim, window)) return 0;
    if (!shape_is(out_dims, 1, window * frame_size)) return 0;
    return window;
}

struct CachedIo {
    MLMultiArray                * input    = nil;
    MLDictionaryFeatureProvider * provider = nil;
    MLMultiArray                * output   = nil;
    MLPredictionOptions         * options  = nil;
};

bool create_cached_io(MLModel * model, NSString * in_name, NSString * out_name, CachedIo * io) {
    MLMultiArrayConstraint * in_constraint =
        model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;
    MLMultiArrayConstraint * out_constraint =
        model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint;

    NSError * err = nil;
    io->input = [[MLMultiArray alloc] initWithShape:in_constraint.shape
                                           dataType:in_constraint.dataType
                                              error:&err];
    if (io->input == nil || err != nil) return false;
    io->provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ in_name : io->input } error:&err];
    if (io->provider == nil || err != nil) return false;
    io->output = [[MLMultiArray alloc] initWithShape:out_constraint.shape
                                            dataType:out_constraint.dataType
                                               error:&err];
    if (io->output == nil || err != nil) return false;
    io->options = [[MLPredictionOptions alloc] init];
    io->options.outputBackings = @{ out_name : io->output };
    return true;
}

struct ArrayView {
    void *  base;
    int64_t stride_c;
    int64_t stride_t;
    bool    f32;
};

ArrayView view_of(MLMultiArray * arr) {
    return {arr.dataPointer,
            arr.strides[arr.strides.count - 2].longLongValue,
            arr.strides[arr.strides.count - 1].longLongValue,
            arr.dataType == MLMultiArrayDataTypeFloat32};
}

void fill_post_array(MLMultiArray * arr, const float * post, int64_t latent_dim, int64_t window) {
    const ArrayView v = view_of(arr);
    for (int64_t c = 0; c < latent_dim; ++c) {
        const int64_t row = c * v.stride_c;
        if (v.f32) {
            for (int64_t t = 0; t < window; ++t)
                ((float *) v.base)[row + t * v.stride_t] = post[t * latent_dim + c];
        } else {
            for (int64_t t = 0; t < window; ++t)
                ((uint16_t *) v.base)[row + t * v.stride_t] =
                    encode_float16(post[t * latent_dim + c]);
        }
    }
}

void copy_pcm_array(MLMultiArray * arr, float * pcm, int64_t n_samples) {
    const ArrayView v = view_of(arr);
    if (v.f32) {
        for (int64_t t = 0; t < n_samples; ++t)
            pcm[t] = ((const float *) v.base)[t * v.stride_t];
    } else {
        for (int64_t t = 0; t < n_samples; ++t)
            pcm[t] = decode_float16(((const uint16_t *) v.base)[t * v.stride_t]);
    }
}

}  // namespace

struct audio8_coreml_codec_context {
    const void * model              = nullptr;  // CFBridgingRetain'd MLModel *
    const void * input_array        = nullptr;  // retained MLMultiArray *
    const void * input_provider     = nullptr;  // retained MLDictionaryFeatureProvider *
    const void * output_array       = nullptr;  // retained MLMultiArray *
    const void * prediction_options = nullptr;  // retained MLPredictionOptions *
    std::string  output_name;
    std::string  label;
    int64_t      latent_dim = 0;
    int64_t      frame_size = 0;
    int64_t      window     = 0;
    std::mutex   mutex;
};

namespace {

audio8_coreml_codec_context * create_context(MLModel * model, const CachedIo & io,
                                             NSString * out_name, std::string label,
                                             int64_t latent_dim, int64_t frame_size,
                                             int64_t window) {
    auto * ctx = new audio8_coreml_codec_context();
    ctx->model              = CFBridgingRetain(model);
    ctx->input_array        = CFBridgingRetain(io.input);
    ctx->input_provider     = CFBridgingRetain(io.provider);
    ctx->output_array       = CFBridgingRetain(io.output);
    ctx->prediction_options = CFBridgingRetain(io.options);
    ctx->output_name        = out_name.UTF8String;
    ctx->label              = std::move(label);
    ctx->latent_dim         = latent_dim;
    ctx->frame_size         = frame_size;
    ctx->window             = window;
    return ctx;
}

}  // namespace

struct audio8_coreml_codec_context * audio8_coreml_codec_init(const char * path_mlmodelc,
                                                              int64_t      latent_dim,
                                                              int64_t      frame_size) {
    if (path_mlmodelc == nullptr || latent_dim <= 0 || frame_size <= 0) return nullptr;

    @autoreleasepool {
        std::string label;
        MLModel * model = load_model(path_mlmodelc, &label);
        if (model == nil) return nullptr;

        NSString * in_name  = nil;
        NSString * out_name = nil;
        const int64_t window =
            validate_model_interface(model, latent_dim, frame_size, &in_name, &out_name);
        if (window <= 0) return nullptr;

        CachedIo io;
        if (!create_cached_io(model, in_name, out_name, &io)) return nullptr;
        return create_context(model, io, out_name, std::move(label), latent_dim, frame_size,
                              window);
    }
}

void audio8_coreml_codec_free(struct audio8_coreml_codec_context * ctx) {
    if (ctx == nullptr) return;
    if (ctx->prediction_options != nullptr) CFRelease(ctx->prediction_options);
    if (ctx->output_array != nullptr) CFRelease(ctx->output_array);
    if (ctx->input_provider != nullptr) CFRelease(ctx->input_provider);
    if (ctx->input_array != nullptr) CFRelease(ctx->input_array);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t audio8_coreml_codec_window_frames(const struct audio8_coreml_codec_context * ctx) {
    return ctx != nullptr ? ctx->window : 0;
}

int audio8_coreml_codec_synthesize(struct audio8_coreml_codec_context * ctx,
                                   const float * post,
                                   float       * pcm) {
    if (ctx == nullptr || ctx->model == nullptr || post == nullptr || pcm == nullptr) return 1;

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        fill_post_array((__bridge MLMultiArray *) ctx->input_array, post, ctx->latent_dim,
                        ctx->window);

        NSError * err = nil;
        MLModel * model = (__bridge MLModel *) ctx->model;
        MLDictionaryFeatureProvider * provider =
            (__bridge MLDictionaryFeatureProvider *) ctx->input_provider;
        MLPredictionOptions * options = (__bridge MLPredictionOptions *) ctx->prediction_options;
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider options:options error:&err];
        if (result == nil || err != nil) return 2;

        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];
        MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
        if (out_arr == nil) return 3;
        copy_pcm_array(out_arr, pcm, ctx->window * ctx->frame_size);
        return 0;
    }
}

const char * audio8_coreml_codec_backend_label(const struct audio8_coreml_codec_context * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}
