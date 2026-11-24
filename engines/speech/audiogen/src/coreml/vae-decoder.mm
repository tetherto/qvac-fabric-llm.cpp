#if !defined(__APPLE__)
#error "vae-decoder.mm is Apple-only; compile it solely under AUDIOGEN_USE_COREML"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "vae-decoder.h"

#include "vae_encode_windows.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

using tts_cpp::acestep::VAE_LATENT_CHANNELS;
using tts_cpp::acestep::VAE_PCM_CHANNELS;
using tts_cpp::acestep::VAE_ENCODER_UPSAMPLE;

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

MLComputeUnits requested_compute_units(std::string * label) {
    const char * env = std::getenv("ACESTEP_COREML_COMPUTE_UNITS");
    const std::string requested = env != nullptr ? env : "";
    if (requested == "cpu_only") {
        *label = "coreml-cpu";
        return MLComputeUnitsCPUOnly;
    }
    if (requested == "cpu_and_gpu") {
        *label = "coreml-gpu";
        return MLComputeUnitsCPUAndGPU;
    }
    if (requested == "cpu_and_ane") {
        *label = "coreml-ane";
        return MLComputeUnitsCPUAndNeuralEngine;
    }
    *label = "coreml-all";
    return MLComputeUnitsAll;
}

MLModel * load_model(const char * path_mlmodelc, std::string * label) {
    NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
    if (path == nil) return nil;
    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = requested_compute_units(label);
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

bool shape_is(const std::vector<int64_t> & dims, int64_t channels, int64_t frames) {
    if (dims.size() < 2) return false;
    for (std::size_t i = 0; i + 2 < dims.size(); ++i) {
        if (dims[i] != 1) return false;
    }
    return dims[dims.size() - 2] == channels && dims[dims.size() - 1] == frames;
}

int64_t validate_model_interface(MLModel * model, NSString ** in_name, NSString ** out_name) {
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
    const int64_t frames = in_dims.back();
    if (!shape_is(in_dims, VAE_LATENT_CHANNELS, frames)) return 0;
    if (!shape_is(out_dims, VAE_PCM_CHANNELS, frames * VAE_ENCODER_UPSAMPLE)) return 0;
    return frames;
}

struct CachedIo {
    MLMultiArray                * input   = nil;
    MLDictionaryFeatureProvider * provider = nil;
    MLMultiArray                * output  = nil;
    MLPredictionOptions         * options = nil;
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

void fill_latent_array(MLMultiArray * arr, const float * latent, int64_t frames) {
    const ArrayView v = view_of(arr);
    for (int64_t c = 0; c < VAE_LATENT_CHANNELS; ++c) {
        const int64_t row = c * v.stride_c;
        if (v.f32) {
            for (int64_t t = 0; t < frames; ++t)
                ((float *) v.base)[row + t * v.stride_t] = latent[t * VAE_LATENT_CHANNELS + c];
        } else {
            for (int64_t t = 0; t < frames; ++t)
                ((uint16_t *) v.base)[row + t * v.stride_t] =
                    encode_float16(latent[t * VAE_LATENT_CHANNELS + c]);
        }
    }
}

void copy_pcm_array(MLMultiArray * arr, float * pcm, int64_t t_audio) {
    const ArrayView v = view_of(arr);
    for (int64_t c = 0; c < VAE_PCM_CHANNELS; ++c) {
        const int64_t row = c * v.stride_c;
        if (v.f32) {
            for (int64_t t = 0; t < t_audio; ++t)
                pcm[t * VAE_PCM_CHANNELS + c] = ((const float *) v.base)[row + t * v.stride_t];
        } else {
            for (int64_t t = 0; t < t_audio; ++t)
                pcm[t * VAE_PCM_CHANNELS + c] =
                    decode_float16(((const uint16_t *) v.base)[row + t * v.stride_t]);
        }
    }
}

}  // namespace

struct acestep_coreml_vae_context {
    const void * model              = nullptr;  // CFBridgingRetain'd MLModel *
    const void * input_array        = nullptr;  // retained MLMultiArray *
    const void * input_provider     = nullptr;  // retained MLDictionaryFeatureProvider *
    const void * output_array       = nullptr;  // retained MLMultiArray *
    const void * prediction_options = nullptr;  // retained MLPredictionOptions *
    std::string  output_name;
    std::string  label;
    int64_t      window_frames = 0;
    std::mutex   mutex;
};

namespace {

acestep_coreml_vae_context * create_context(MLModel * model, const CachedIo & io,
                                            NSString * out_name, std::string label,
                                            int64_t window_frames) {
    auto * ctx = new acestep_coreml_vae_context();
    ctx->model              = CFBridgingRetain(model);
    ctx->input_array        = CFBridgingRetain(io.input);
    ctx->input_provider     = CFBridgingRetain(io.provider);
    ctx->output_array       = CFBridgingRetain(io.output);
    ctx->prediction_options = CFBridgingRetain(io.options);
    ctx->output_name        = out_name.UTF8String;
    ctx->label              = std::move(label);
    ctx->window_frames      = window_frames;
    return ctx;
}

}  // namespace

struct acestep_coreml_vae_context * acestep_coreml_vae_init(const char * path_mlmodelc) {
    if (path_mlmodelc == nullptr) return nullptr;

    @autoreleasepool {
        std::string label;
        MLModel * model = load_model(path_mlmodelc, &label);
        if (model == nil) return nullptr;

        NSString * in_name  = nil;
        NSString * out_name = nil;
        const int64_t window_frames = validate_model_interface(model, &in_name, &out_name);
        if (window_frames <= 0) return nullptr;

        CachedIo io;
        if (!create_cached_io(model, in_name, out_name, &io)) return nullptr;
        return create_context(model, io, out_name, std::move(label), window_frames);
    }
}

void acestep_coreml_vae_free(struct acestep_coreml_vae_context * ctx) {
    if (ctx == nullptr) return;
    if (ctx->prediction_options != nullptr) CFRelease(ctx->prediction_options);
    if (ctx->output_array != nullptr) CFRelease(ctx->output_array);
    if (ctx->input_provider != nullptr) CFRelease(ctx->input_provider);
    if (ctx->input_array != nullptr) CFRelease(ctx->input_array);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t acestep_coreml_vae_window_frames(const struct acestep_coreml_vae_context * ctx) {
    return ctx != nullptr ? ctx->window_frames : 0;
}

int acestep_coreml_vae_decode(struct acestep_coreml_vae_context * ctx,
                              const float * latent,
                              float       * pcm) {
    if (ctx == nullptr || ctx->model == nullptr || latent == nullptr || pcm == nullptr) return 1;

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        fill_latent_array((__bridge MLMultiArray *) ctx->input_array, latent, ctx->window_frames);

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
        copy_pcm_array(out_arr, pcm, ctx->window_frames * VAE_ENCODER_UPSAMPLE);
        return 0;
    }
}

const char * acestep_coreml_vae_backend_label(const struct acestep_coreml_vae_context * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}
