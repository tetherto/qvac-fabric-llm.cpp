#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "supertonic_coreml_vector.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

namespace {
struct Bucket {
    std::string path;
    MLModel * model = nil;
    NSMutableDictionary<NSString *, MLMultiArray *> * arrays = nil;
    MLDictionaryFeatureProvider * provider = nil;
    MLMultiArray * output = nil;
    MLPredictionOptions * options = nil;
    int capacity = 0, text_capacity = 0;
    bool failed = false;
};

bool same_shape(NSArray<NSNumber *> * a, NSArray<NSNumber *> * b) { return [a isEqualToArray:b]; }

// Host-created arrays are normally contiguous, but Core ML outputs can have
// padding/strides even when an output backing was offered.
bool contiguous(MLMultiArray * a) {
    int64_t stride = 1;
    for (NSInteger i = a.shape.count - 1; i >= 0; --i) {
        if (a.strides[i].longLongValue != stride) return false;
        stride *= a.shape[i].longLongValue;
    }
    return true;
}
int64_t offset(MLMultiArray * a, int64_t index) {
    int64_t out = 0;
    for (NSInteger i = a.shape.count - 1; i >= 0; --i) {
        const int64_t dim = a.shape[i].longLongValue;
        out += (index % dim) * a.strides[i].longLongValue;
        index /= dim;
    }
    return out;
}
void write(MLMultiArray * a, const float * src) {
    auto * dst = static_cast<float *>(a.dataPointer);
    if (contiguous(a)) std::memcpy(dst, src, (size_t) a.count * sizeof(float));
    else for (int64_t i = 0; i < a.count; ++i) dst[offset(a, i)] = src[i];
}
void read(MLMultiArray * a, std::vector<float> & dst) {
    dst.resize(a.count);
    const auto * src = static_cast<const float *>(a.dataPointer);
    if (contiguous(a)) std::memcpy(dst.data(), src, dst.size() * sizeof(float));
    else for (int64_t i = 0; i < a.count; ++i) dst[i] = src[offset(a, i)];
}
void write_padded(MLMultiArray * a, const float * src, int channels, int length, int capacity) {
    std::vector<float> padded((size_t) channels * capacity, 0);
    for (int c = 0; c < channels; ++c)
        std::copy(src + (size_t) c * length, src + (size_t) (c + 1) * length, padded.data() + (size_t) c * capacity);
    write(a, padded.data());
}
}

struct supertonic_coreml_vector_context {
    std::map<int, std::unique_ptr<Bucket>> buckets;
    MLModelConfiguration * config = nil;
    std::string label = "coreml-gpu";
    int weight_bits = 0;
    bool automatic_units = true;
    int heads = 0;
    float cond_scale = 0, uncond_scale = 0;
    std::mutex mutex;
};

namespace {
bool load_bucket(supertonic_coreml_vector_context * ctx, Bucket & b) {
    if (b.failed) return false;
    if (b.model) return true;
    b.failed = true;
    NSError * error = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:b.path.c_str()]]
                                      configuration:ctx->config error:&error];
    if (!model || error) return false;
    NSDictionary * meta = model.modelDescription.metadata[MLModelCreatorDefinedKey];
    if (![meta[@"supertonic.vector_protocol"] isEqualToString:@"2"] ||
        [meta[@"supertonic.vector_heads"] intValue] != ctx->heads ||
        !meta[@"supertonic.cfg_cond_scale"] || !meta[@"supertonic.cfg_uncond_scale"] ||
        std::abs([meta[@"supertonic.cfg_cond_scale"] floatValue] - ctx->cond_scale) > 1e-5 ||
        std::abs([meta[@"supertonic.cfg_uncond_scale"] floatValue] - ctx->uncond_scale) > 1e-5) return false;
    const int bits = [meta[@"supertonic.weight_bits"] intValue];
    if (bits != 32 && bits != 16) return false;
    if (ctx->weight_bits && ctx->weight_bits != bits) return false;
    const MLComputeUnits desired = bits == 32 ? MLComputeUnitsCPUAndGPU : MLComputeUnitsCPUAndNeuralEngine;
    if (ctx->automatic_units && ctx->config.computeUnits != desired) {
        ctx->config.computeUnits = desired;
        ctx->label = bits == 32 ? "coreml-gpu" : "coreml-ane";
        model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:b.path.c_str()]]
                                 configuration:ctx->config error:&error];
        if (!model || error) return false;
    }
    auto inputs = model.modelDescription.inputDescriptionsByName;
    auto outputs = model.modelDescription.outputDescriptionsByName;
    auto text_shape = inputs[@"text_emb"].multiArrayConstraint.shape;
    b.text_capacity = text_shape.count == 3 ? text_shape[2].intValue : 0;
    if (b.text_capacity <= 0 || b.text_capacity > 4096 || inputs.count != 10 || outputs.count != 1) return false;
    NSDictionary<NSString *, NSArray<NSNumber *> *> * shapes = @{
        @"noisy_latent": @[@1, @144, @(b.capacity)], @"text_emb": @[@1, @256, @(b.text_capacity)],
        @"style_ttl": @[@1, @50, @256], @"latent_mask": @[@1, @1, @(b.capacity)],
        @"text_mask": @[@1, @1, @(b.text_capacity)], @"time_emb": @[@1, @64],
        @"cos_q": @[@1, @1, @(b.capacity), @32], @"sin_q": @[@1, @1, @(b.capacity), @32],
        @"cos_k": @[@1, @1, @(b.text_capacity), @32], @"sin_k": @[@1, @1, @(b.text_capacity), @32]};
    b.arrays = [[NSMutableDictionary alloc] init];
    for (NSString * name in shapes) {
        auto constraint = inputs[name].multiArrayConstraint;
        if (!constraint || constraint.dataType != MLMultiArrayDataTypeFloat32 || !same_shape(constraint.shape, shapes[name])) return false;
        MLMultiArray * array = [[MLMultiArray alloc] initWithShape:shapes[name] dataType:MLMultiArrayDataTypeFloat32 error:&error];
        if (!array || error) return false;
        b.arrays[name] = array;
    }
    const int batch = ctx->uncond_scale != 0 ? 2 : 1;
    auto out = outputs[@"velocity"].multiArrayConstraint;
    if (!out || out.dataType != MLMultiArrayDataTypeFloat32 || !same_shape(out.shape, @[@(batch), @144, @(b.capacity)])) return false;
    b.output = [[MLMultiArray alloc] initWithShape:out.shape dataType:out.dataType error:&error];
    if (!b.output || error) return false;
    b.provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:b.arrays error:&error];
    if (!b.provider || error) return false;
    b.options = [[MLPredictionOptions alloc] init];
    b.options.outputBackings = @{@"velocity": b.output};
    b.model = model;
    ctx->weight_bits = bits;
    b.failed = false;
    return true;
}
}

supertonic_coreml_vector_context * supertonic_coreml_vector_init(
        const std::vector<std::pair<int, std::string>> & paths, int heads, float cond_scale, float uncond_scale) {
    if (paths.empty()) return nullptr;
    @autoreleasepool {
        auto ctx = std::make_unique<supertonic_coreml_vector_context>();
        ctx->heads = heads; ctx->cond_scale = cond_scale; ctx->uncond_scale = uncond_scale;
        ctx->config = [[MLModelConfiguration alloc] init];
        ctx->config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        const char * env = std::getenv("SUPERTONIC_COREML_VECTOR_COMPUTE_UNITS");
        if (!env) env = std::getenv("SUPERTONIC_COREML_COMPUTE_UNITS");
        ctx->automatic_units = env == nullptr;
        std::string units = env ? env : "cpu_and_ane";
        if (units == "cpu_only") { ctx->config.computeUnits = MLComputeUnitsCPUOnly; ctx->label = "coreml-cpu"; }
        else if (units == "cpu_and_gpu") { ctx->config.computeUnits = MLComputeUnitsCPUAndGPU; ctx->label = "coreml-gpu"; }
        else if (units == "cpu_and_ane") { ctx->config.computeUnits = MLComputeUnitsCPUAndNeuralEngine; ctx->label = "coreml-ane"; }
        else if (units != "cpu_and_ane") { ctx->config.computeUnits = MLComputeUnitsAll; ctx->label = "coreml-all"; }
        for (const auto & path : paths) {
            if (path.first <= 0 || path.first > 4096) continue;
            auto b = std::make_unique<Bucket>();
            b->capacity = path.first; b->path = path.second;
            ctx->buckets.emplace(path.first, std::move(b));
        }
        // Load only one bucket at construction. Larger variants load lazily.
        for (auto & entry : ctx->buckets) if (load_bucket(ctx.get(), *entry.second)) return ctx.release();
        return nullptr;
    }
}
void supertonic_coreml_vector_free(supertonic_coreml_vector_context * ctx) { delete ctx; }
const char * supertonic_coreml_vector_backend_label(const supertonic_coreml_vector_context * ctx) {
    return ctx ? ctx->label.c_str() : "ggml";
}
int supertonic_coreml_vector_weight_bits(const supertonic_coreml_vector_context * ctx) {
    return ctx ? ctx->weight_bits : 0;
}

bool supertonic_coreml_vector_run(supertonic_coreml_vector_context * ctx,
        const float * latent, int length, const float * text, int text_length,
        const float * style, const float * latent_mask, const float * time_embeddings,
        const float * theta, int total_steps, int count, std::vector<float> & output) {
    if (!ctx || !latent || !text || !style || !latent_mask || !time_embeddings || !theta ||
        length <= 0 || text_length <= 0 || total_steps <= 0 || count <= 0 || count > total_steps) return false;
    // Existing callers may supply general masks. This sidecar supports the
    // production all-valid unpadded input only; other masks retain ggml math.
    for (int i = 0; i < length; ++i) if (latent_mask[i] != 1.0f) return false;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    @autoreleasepool {
        auto selected = ctx->buckets.lower_bound(length);
        while (selected != ctx->buckets.end() &&
               (!load_bucket(ctx, *selected->second) || text_length > selected->second->text_capacity)) ++selected;
        if (selected == ctx->buckets.end()) return false;
        auto & b = *selected->second;
        write_padded(b.arrays[@"noisy_latent"], latent, 144, length, b.capacity);
        write_padded(b.arrays[@"text_emb"], text, 256, text_length, b.text_capacity);
        write(b.arrays[@"style_ttl"], style);
        std::vector<float> mask(b.capacity, 0), text_mask(b.text_capacity, 0);
        std::fill(mask.begin(), mask.begin()+length, 1);
        std::fill(text_mask.begin(), text_mask.begin()+text_length, 1);
        write(b.arrays[@"latent_mask"], mask.data()); write(b.arrays[@"text_mask"], text_mask.data());
        auto rotary = [&](int capacity, int actual, NSString * cos_name, NSString * sin_name) {
            std::vector<float> cs((size_t) capacity * 32), sn(cs.size());
            for (int t = 0; t < capacity; ++t) for (int d = 0; d < 32; ++d) {
                const float angle = (float) t / actual * theta[d];
                cs[t*32+d] = std::cos(angle); sn[t*32+d] = std::sin(angle);
            }
            write(b.arrays[cos_name], cs.data()); write(b.arrays[sin_name], sn.data());
        };
        rotary(b.capacity, length, @"cos_q", @"sin_q");
        rotary(b.text_capacity, text_length, @"cos_k", @"sin_k");
        std::vector<float> values;
        const size_t plane = (size_t) 144 * b.capacity;
        std::vector<float> state(plane, 0);
        for (int c = 0; c < 144; ++c)
            std::copy_n(latent + (size_t) c * length, length, state.data() + (size_t) c * b.capacity);
        const int batch = ctx->uncond_scale != 0 ? 2 : 1;
        for (int step = 0; step < count; ++step) {
            write(b.arrays[@"time_emb"], time_embeddings + step * 64);
            NSError * error = nil;
            auto prediction = [b.model predictionFromFeatures:b.provider options:b.options error:&error];
            MLMultiArray * out = [prediction featureValueForName:@"velocity"].multiArrayValue;
            if (!out || error || out.dataType != MLMultiArrayDataTypeFloat32 ||
                !same_shape(out.shape, @[@(batch), @144, @(b.capacity)])) { b.failed = true; return false; }
            read(out, values);
            for (float v : values) if (!std::isfinite(v)) { b.failed = true; return false; }
            for (int c = 0; c < 144; ++c) for (int t = 0; t < length; ++t) {
                const size_t i = (size_t) c * b.capacity + t;
                const float velocity = batch == 2 ? ctx->cond_scale * values[i] - ctx->uncond_scale * values[plane + i] : values[i];
                state[i] += velocity / total_steps;
                if (!std::isfinite(state[i])) { b.failed = true; return false; }
            }
            if (step + 1 < count) write(b.arrays[@"noisy_latent"], state.data());
        }
        std::vector<float> trimmed((size_t) 144 * length);
        for (int c = 0; c < 144; ++c)
            std::copy_n(state.data() + (size_t) c * b.capacity, length, trimmed.data() + (size_t) c * length);
        output.swap(trimmed);
        return true;
    }
}
