#include "ffn-coreml-common.h"
#include "ffn-coreml-hook.h"
#include "ffn-coreml-otf.h"
#include "ggml-metal.h"
#include "llama-ext.h"

#import <Metal/Metal.h>

#include <cstdlib>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>

static bool profile_enabled = false;
static bool direct_input = false;
static bool gpu_resident = false;

// FFN_COREML_INGRAPH: the Metal share of each FFN runs as ordinary graph nodes on views of the model's own weights, so
// it needs no weight copy and no per-layer buffers; only the ANE share goes through custom nodes
static const bool ingraph_mode = std::getenv("FFN_COREML_INGRAPH") != nullptr;
// rows of the Core ML programs; the in-graph path cuts larger microbatches into chunks of this size
static const int ingraph_tokens = 512;

// FFN_COREML_ASYNC: the resident path hands each ANE call to a serial worker instead of blocking the graph thread.
// The GPU signals ane_ready when the FFN input is written; the worker waits for it, predicts, then signals ane_done,
// which the GPU waits for before the join. Both counters advance together, one value per call.
struct ane_async {
    bool enabled = false;
    id<MTLSharedEvent> ready = nil;
    id<MTLSharedEvent> done = nil;
    dispatch_queue_t queue = nil;
    uint64_t seq = 0;

    static ane_async & get() {
        static ane_async state = [] {
            ane_async s;
            s.enabled = std::getenv("FFN_COREML_ASYNC") != nullptr;
            if (s.enabled) {
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                s.ready = [device newSharedEvent];
                s.done = [device newSharedEvent];
                s.queue = dispatch_queue_create("ffn-coreml-ane", DISPATCH_QUEUE_SERIAL);
            }
            return s;
        }();
        return state;
    }
};

struct stage_profile {
    double ms[10] = {};
    double callback_ms = 0;
    std::chrono::steady_clock::time_point last;

    void begin() {
        if (profile_enabled) { last = std::chrono::steady_clock::now(); }
    }

    void end(int stage) {
        if (!profile_enabled) { return; }
        auto now = std::chrono::steady_clock::now();
        ms[stage] += std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
    }
};

static MLMultiArray * decoder_output(id<MLFeatureProvider> result, int tokens = 512) {
    MLMultiArray * y = [result featureValueForName:@"y"].multiArrayValue;
    require(y && y.dataType == MLMultiArrayDataTypeFloat32 && y.shape.count == 2 && y.shape[0].intValue == tokens && y.shape[1].intValue == 5120, "unexpected decoder FFN output");
    return y;
}

static MLModel * decoder_load_compiled(NSURL * url) {
    NSError * failure = nil;
    MLModelConfiguration * configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel * model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&failure];
    require(model, failure ? failure.localizedDescription.UTF8String : "decoder Core ML load failed");
    return model;
}

// FFN_COREML_OTF=<Metal channels>x<packages>: write the sidecar from the GGUF at load time (ffn-coreml-otf.h) instead
// of reading prepared assets. Each package takes an equal share of the remaining channels; its template comes from
// FFN_COREML_TEMPLATES (int8-c<channels>), compiled models are cached under FFN_COREML_CACHE (default ~/Library/Caches/ffn-coreml)
struct otf_config {
    int metal = 0, packages = 0;
    otf_template tmpl;
    std::filesystem::path cache;
    double export_s = 0, compile_s = 0, load_s = 0;
    int created = 0;

    bool enabled() const { return packages > 0; }

    void init(const std::string & gguf) {
        const char * value = std::getenv("FFN_COREML_OTF");
        if (!value) { return; }
        require(sscanf(value, "%dx%d", &metal, &packages) == 2 && metal > 0 && metal % 256 == 0 && packages >= 1 && packages <= 2 &&
                (17408 - metal) % packages == 0, "FFN_COREML_OTF must be <Metal channels>x<1 or 2 packages>");
        const char * templates = std::getenv("FFN_COREML_TEMPLATES");
        require(templates, "FFN_COREML_OTF needs FFN_COREML_TEMPLATES");
        tmpl.load(std::filesystem::path(templates) / ("int8-c" + std::to_string((17408 - metal) / packages)));
        const char * root = std::getenv("FFN_COREML_CACHE");
        cache = otf_cache_dir(root ? std::filesystem::path(root) : std::filesystem::path(std::getenv("HOME")) / "Library/Caches/ffn-coreml", gguf, tmpl);
        printf("ffn_coreml_otf metal_channels=%d packages=%d template=%s cache=%s\n", metal, packages, tmpl.dir.c_str(), cache.c_str());
    }

    void load(const std::string & gguf, int layer, MLModel * __strong & first, MLModel * __strong & second) {
        otf_layer weights;
        bool read = false;
        auto get = [&]() -> const otf_layer & {
            if (!read) { weights.read(gguf, layer); read = true; }
            return weights;
        };
        for (int p = 0; p < packages; ++p) {
            double e = 0, c = 0;
            NSURL * url = otf_compiled(tmpl, cache, layer, metal + p * tmpl.channels, get, e, c);
            const auto t0 = std::chrono::steady_clock::now();
            (p == 0 ? first : second) = decoder_load_compiled(url);
            load_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            export_s += e;
            compile_s += c;
            created += e > 0;
        }
    }
};

static otf_config otf;

static MLModel * decoder_load(const std::string & assets, const char * name, int layer, temporary_model & compiled) {
    NSError * failure = nil;
    NSString * path = [NSString stringWithFormat:@"%s/%s.mlpackage", assets.c_str(), name];
    NSURL * model_url = nil;
    if (std::getenv("FFN_COREML_CACHE_COMPILED")) {
        NSString * cache_path = [[NSString stringWithFormat:@"%s/%s.mlmodelc", assets.c_str(), name] stringByResolvingSymlinksInPath];
        model_url = [NSURL fileURLWithPath:cache_path];
        NSFileManager * files = [NSFileManager defaultManager];
        const bool create_cache = ![files fileExistsAtPath:cache_path];
        if (create_cache) {
            compiled.url = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&failure];
            require(compiled.url, failure ? failure.localizedDescription.UTF8String : "decoder Core ML compilation failed");
            NSURL * staging = [NSURL fileURLWithPath:[cache_path stringByAppendingFormat:@".%@.tmp", NSUUID.UUID.UUIDString]];
            if (![files copyItemAtURL:compiled.url toURL:staging error:&failure] || ![files moveItemAtURL:staging toURL:model_url error:&failure]) {
                [files removeItemAtURL:staging error:nullptr];
                require(false, failure ? failure.localizedDescription.UTF8String : "compiled cache creation failed");
            }
        }
        printf("decoder_compiled_cache_layer=%d created=%d path=%s\n", layer, int(create_cache), cache_path.UTF8String);
    } else {
        compiled.url = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&failure];
        require(compiled.url, failure ? failure.localizedDescription.UTF8String : "decoder Core ML compilation failed");
        model_url = compiled.url;
    }
    return decoder_load_compiled(model_url);
}

struct decoder_ffn;

// custom op userdata: stage -1 runs the whole FFN, 0 starts it and 1 finishes it (two-half microbatches);
// 2 and 3 start and finish the ANE share of an in-graph FFN (FFN_COREML_INGRAPH)
struct ffn_call {
    decoder_ffn * runtime;
    int stage;
};

struct decoder_ffn {
    ffn_call call[5] = {{this, -1}, {this, 0}, {this, 1}, {this, 2}, {this, 3}};
    bool ingraph = false;
    int metal_channels = 0;
    // in-graph: Core ML output backings for each start-node buffer they were created for, and the async values of
    // started chunks, finished in order
    std::map<const void *, std::pair<MLPredictionOptions *, MLPredictionOptions *>> outputs;
    std::deque<uint64_t> started;
    // async value of the started call, finished by the next stage 1 call
    uint64_t pending = 0;
    // Core ML inputs bound to the Metal buffers they were created for
    std::map<const void *, MLDictionaryFeatureProvider *> inputs;
    temporary_model compiled{nil};
    temporary_model compiled2{nil};
    MLModel * model = nil;
    // optional second suffix package, run at the same time on the other ANE
    MLModel * model2 = nil;
    MLMultiArray * input = nil;
    MLDictionaryFeatureProvider * features = nil;
    metal_ffn prefix;
    metal_join join;
    std::vector<float> x, partial, residual;
    float inverse_scale = 1;
    uint64_t calls = 0;
    bool full_metal = false;
    // Core ML writes its partials straight into the Metal join inputs
    bool backed = false;
    MLPredictionOptions * options = nil;
    MLPredictionOptions * options2 = nil;
    uint64_t backing_fallbacks = 0;
    stage_profile profile;
    std::string error;
    std::mutex error_lock;

    void fail(const std::string & message) {
        std::lock_guard<std::mutex> lock(error_lock);
        if (error.empty()) { error = message; }
    }

    void predict(bool resident = false) {
        if (!direct_input && !resident) { memcpy(input.dataPointer, x.data(), x.size() * sizeof(float)); }
        profile.end(3);
        __block id<MLFeatureProvider> second = nil;
        __block NSError * second_failure = nil;
        dispatch_group_t group = dispatch_group_create();
        if (model2) {
            MLModel * other = model2;
            MLDictionaryFeatureProvider * inputs = features;
            dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                @autoreleasepool {
                    NSError * failure = nil;
                    second = [other predictionFromFeatures:inputs error:&failure];
                    second_failure = failure;
                }
            });
        }
        NSError * failure = nil;
        id<MLFeatureProvider> result = [model predictionFromFeatures:features error:&failure];
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        profile.end(4);
        require(result, failure ? failure.localizedDescription.UTF8String : "Core ML prediction failed");
        require(!model2 || second, second_failure ? second_failure.localizedDescription.UTF8String : "second Core ML prediction failed");
        MLMultiArray * y = decoder_output(result);
        MLMultiArray * y2 = model2 ? decoder_output(second) : nil;
        const float * values = static_cast<const float *>(y.dataPointer);
        const float * values2 = y2 ? static_cast<const float *>(y2.dataPointer) : nullptr;
        size_t row_stride = y.strides[0].unsignedLongLongValue, column_stride = y.strides[1].unsignedLongLongValue;
        size_t row_stride2 = y2 ? y2.strides[0].unsignedLongLongValue : 0, column_stride2 = y2 ? y2.strides[1].unsignedLongLongValue : 0;
        for (int t = 0; t < 512; ++t) {
            for (int h = 0; h < 5120; ++h) {
                float value = values[t * row_stride + h * column_stride];
                if (values2) { value += values2[t * row_stride2 + h * column_stride2]; }
                value *= inverse_scale;
                if (!std::isfinite(value)) { throw std::runtime_error("non-finite integrated FFN output"); }
                partial[t * 5120 + h] = value;
            }
        }
    }

    void adopt(id<MLFeatureProvider> result, ggml_tensor * target) { adopt_into(result, static_cast<float *>(target->data)); }

    void adopt_into(id<MLFeatureProvider> result, float * out, int tokens = 512) {
        MLMultiArray * y = decoder_output(result, tokens);
        if (y.dataPointer == out) { return; }
        ++backing_fallbacks;
        const float * values = static_cast<const float *>(y.dataPointer);
        size_t row_stride = y.strides[0].unsignedLongLongValue, column_stride = y.strides[1].unsignedLongLongValue;
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < 5120; ++h) { out[t * 5120 + h] = values[t * row_stride + h * column_stride]; }
        }
    }

    // both packages at once (the two ANEs), outputs into out (and the next ingraph_tokens x 5120 for the second package)
    void predict_into(MLDictionaryFeatureProvider * inputs, MLPredictionOptions * first, MLPredictionOptions * second_options, float * out) {
        __block id<MLFeatureProvider> second = nil;
        __block NSError * second_failure = nil;
        dispatch_group_t group = dispatch_group_create();
        if (model2) {
            MLModel * other = model2;
            dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                @autoreleasepool {
                    NSError * failure = nil;
                    second = [other predictionFromFeatures:inputs options:second_options error:&failure];
                    second_failure = failure;
                }
            });
        }
        NSError * failure = nil;
        id<MLFeatureProvider> result = [model predictionFromFeatures:inputs options:first error:&failure];
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        require(result, failure ? failure.localizedDescription.UTF8String : "Core ML prediction failed");
        require(!model2 || second, second_failure ? second_failure.localizedDescription.UTF8String : "second Core ML prediction failed");
        adopt_into(result, out, ingraph_tokens);
        if (model2) { adopt_into(second, out + ingraph_tokens * 5120, ingraph_tokens); }
    }

    void predict_backed(MLDictionaryFeatureProvider * inputs, bool timed = true) {
        __block id<MLFeatureProvider> second = nil;
        __block NSError * second_failure = nil;
        dispatch_group_t group = dispatch_group_create();
        if (model2) {
            MLModel * other = model2;
            MLPredictionOptions * other_options = options2;
            dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                @autoreleasepool {
                    NSError * failure = nil;
                    second = [other predictionFromFeatures:inputs options:other_options error:&failure];
                    second_failure = failure;
                }
            });
        }
        NSError * failure = nil;
        id<MLFeatureProvider> result = [model predictionFromFeatures:inputs options:options error:&failure];
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        if (timed) { profile.end(4); }
        require(result, failure ? failure.localizedDescription.UTF8String : "Core ML prediction failed");
        require(!model2 || second, second_failure ? second_failure.localizedDescription.UTF8String : "second Core ML prediction failed");
        adopt(result, join.ane);
        if (model2) { adopt(second, join.ane2); }
    }

    void execute(float * output, bool inputs_ready = false) {
        if (!inputs_ready) {
            ggml_backend_tensor_set(prefix.input, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_tensor_set(prefix.residual, residual.data(), 0, residual.size() * sizeof(float));
        }
        profile.end(1);
        if (full_metal) {
            require(ggml_backend_graph_compute(prefix.backend, prefix.graph) == GGML_STATUS_SUCCESS, "reference Metal FFN failed");
            profile.end(2);
            ggml_backend_tensor_get(prefix.output, output, 0, x.size() * sizeof(float));
            profile.end(9);
            ++calls;
            return;
        }
        require(ggml_backend_graph_compute_async(prefix.backend, prefix.graph) == GGML_STATUS_SUCCESS, "decoder Metal prefix failed");
        profile.end(2);
        predict();
        profile.end(5);
        ggml_backend_synchronize(prefix.backend);
        profile.end(6);
        ggml_backend_tensor_copy(prefix.output, join.gpu);
        ggml_backend_tensor_set(join.ane, partial.data(), 0, partial.size() * sizeof(float));
        profile.end(7);
        require(ggml_backend_graph_compute(join.backend, join.graph) == GGML_STATUS_SUCCESS, "Metal join failed");
        profile.end(8);
        ggml_backend_tensor_get(join.output, output, 0, x.size() * sizeof(float));
        profile.end(9);
        ++calls;
    }

    void init(const std::string & gguf, const std::string & assets, int layer, bool reference_only) {
        full_metal = reference_only;
        x.resize(5120 * 512);
        partial.resize(x.size());
        residual.resize(x.size());
        if (full_metal) {
            prefix.init(gguf, assets, layer, 512, false, 0, true, direct_input);
            std::vector<float> warm(x.size());
            profile.begin();
            execute(warm.data());
            calls = 0;
            return;
        }
        NSError * failure = nil;
        int gpu_channels = 0, ane_channels = 0, ane2_channels = 0;
        double scale = 1;
        if (otf.enabled()) {
            otf.load(gguf, layer, model, model2);
            gpu_channels = otf.metal;
            ane_channels = otf.tmpl.channels;
            ane2_channels = model2 ? otf.tmpl.channels : 0;
        } else {
            model = decoder_load(assets, "ffn-512", layer, compiled);
            NSDictionary * metadata = model.modelDescription.metadata[MLModelCreatorDefinedKey];
            gpu_channels = [metadata[@"channel_start"] intValue];
            ane_channels = [metadata[@"channel_count"] intValue];
            scale = [metadata[@"output_scale"] doubleValue];
            require([metadata[@"source_layer"] intValue] == layer, "wrong integrated FFN package");
            NSFileManager * files = [NSFileManager defaultManager];
            NSString * second = [NSString stringWithFormat:@"%s/ffn-512-1", assets.c_str()];
            if ([files fileExistsAtPath:[second stringByAppendingString:@".mlpackage"]] ||
                (std::getenv("FFN_COREML_CACHE_COMPILED") && [files fileExistsAtPath:[[second stringByAppendingString:@".mlmodelc"] stringByResolvingSymlinksInPath]])) {
                model2 = decoder_load(assets, "ffn-512-1", layer, compiled2);
                NSDictionary * metadata2 = model2.modelDescription.metadata[MLModelCreatorDefinedKey];
                ane2_channels = [metadata2[@"channel_count"] intValue];
                require([metadata2[@"source_layer"] intValue] == layer && [metadata2[@"channel_start"] intValue] == gpu_channels + ane_channels &&
                        [metadata2[@"output_scale"] doubleValue] == scale && ane2_channels > 0, "second FFN package must follow the first");
            }
        }
        require(gpu_channels > 0 && gpu_channels % 256 == 0 && ane_channels > 0 && gpu_channels + ane_channels + ane2_channels == 17408, "wrong integrated FFN package");
        require(scale == 1 || scale == 16, "unsupported integrated output scale");
        inverse_scale = 1.0 / scale;
        metal_channels = gpu_channels;
        if (ingraph_mode) {
            require(gpu_resident && ane_async::get().enabled && scale == 1, "FFN_COREML_INGRAPH needs FFN_COREML_GPU_RESIDENT, FFN_COREML_ASYNC and output scale 1");
            ingraph = true;
            // load the ANE programs now rather than during the first prompt; the pool frees the warm-up arrays
            // (about 30 MB per layer) here instead of at the end of the caller's pool
            @autoreleasepool {
                MLMultiArray * zeros = [[MLMultiArray alloc] initWithShape:@[@(ingraph_tokens), @5120] dataType:MLMultiArrayDataTypeFloat32 error:&failure];
                require(zeros, "warm-up input allocation failed");
                memset(zeros.dataPointer, 0, size_t(ingraph_tokens) * 5120 * sizeof(float));
                MLDictionaryFeatureProvider * warm = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": zeros} error:&failure];
                require(warm && [model predictionFromFeatures:warm error:&failure] && (!model2 || [model2 predictionFromFeatures:warm error:&failure]), "warm-up prediction failed");
            }
            printf("decoder_layer=%d in_graph=1 metal_channels=%d coreml_channels=%d coreml2_channels=%d\n", layer, gpu_channels, ane_channels, ane2_channels);
            return;
        }
        prefix.init(gguf, assets, layer, 512, false, gpu_channels, true, direct_input);
        if (direct_input) {
            input = [[MLMultiArray alloc] initWithDataPointer:prefix.input->data shape:@[@512, @5120] dataType:MLMultiArrayDataTypeFloat32 strides:@[@5120, @1] deallocator:nil error:&failure];
        } else {
            input = [[MLMultiArray alloc] initWithShape:@[@512, @5120] dataType:MLMultiArrayDataTypeFloat32 error:&failure];
        }
        require(input, "integrated FFN input allocation failed");
        features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": input} error:&failure];
        require(features, "integrated FFN provider failed");
        printf("decoder_layer=%d metal_channels=%d coreml_channels=%d coreml2_channels=%d output_scale=%.0f\n", layer, gpu_channels, ane_channels, ane2_channels, scale);
        // on by default for the resident path; FFN_COREML_OUTPUT_BACKING=0 keeps the CPU copy
        const char * backing_env = std::getenv("FFN_COREML_OUTPUT_BACKING");
        backed = gpu_resident && scale == 1 && !(backing_env && strcmp(backing_env, "0") == 0);
        join.init(prefix.backend, 5120, 512, backed && model2);
        if (backed) {
            auto backing = [&](ggml_tensor * target) {
                require(ggml_backend_metal_tensor_is_shared(target), "output backing requires shared Metal storage");
                NSError * failure = nil;
                MLMultiArray * array = [[MLMultiArray alloc] initWithDataPointer:target->data shape:@[@512, @5120] dataType:MLMultiArrayDataTypeFloat32 strides:@[@5120, @1] deallocator:nil error:&failure];
                require(array, "output backing allocation failed");
                MLPredictionOptions * result = [[MLPredictionOptions alloc] init];
                result.outputBackings = @{@"y": array};
                return result;
            };
            options = backing(join.ane);
            if (model2) { options2 = backing(join.ane2); }
        }
        std::vector<float> warm(x.size());
        profile.begin();
        execute(warm.data());
        calls = 0;
    }
};

struct decoder_state {
    std::map<int, std::unique_ptr<decoder_ffn>> layers;
    bool metal_reference = false;
    int batch_threads = 20;

    uint64_t calls() const {
        uint64_t count = 0;
        for (const auto & entry : layers) { count += entry.second->calls; }
        return count;
    }

    void check() const {
        for (const auto & entry : layers) {
            std::lock_guard<std::mutex> lock(entry.second->error_lock);
            require(entry.second->error.empty(), entry.second->error);
        }
    }
};

// layers listed in <assets>/layers.txt; with FFN_COREML_OTF and no list, every layer
static void load_layers(decoder_state & state, const std::string & gguf, const std::string & assets) {
    std::vector<int> layers;
    std::ifstream selected(std::filesystem::path(assets) / "layers.txt");
    int layer;
    while (selected >> layer) { layers.push_back(layer); }
    if (layers.empty() && otf.enabled()) {
        for (layer = 0; layer < 64; ++layer) { layers.push_back(layer); }
    }
    require(!layers.empty(), "no selected layers in layers.txt");
    const auto start = std::chrono::steady_clock::now();
    for (int l : layers) {
        require(l >= 0 && l < 64 && !state.layers.count(l), "invalid selected layer");
        auto runtime = std::make_unique<decoder_ffn>();
        runtime->init(gguf, (std::filesystem::path(assets) / ("layer" + std::to_string(l))).string(), l, state.metal_reference);
        state.layers.emplace(l, std::move(runtime));
    }
    if (otf.enabled()) {
        printf("ffn_coreml_otf layers=%zu packages_created=%d export_s=%.1f compile_s=%.1f load_s=%.1f setup_s=%.1f\n", layers.size(), otf.created,
               otf.export_s, otf.compile_s, otf.load_s, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        fflush(stdout);
    }
}

static void decoder_compute(ggml_tensor * dst, const ggml_tensor * src, const ggml_tensor * residual, int ith, int nth, void * userdata) {
    if (ith != 0) { return; }
    auto & runtime = *static_cast<ffn_call *>(userdata)->runtime;
    runtime.profile.begin();
    auto callback_start = runtime.profile.last;
    @autoreleasepool {
        try {
            require(nth >= 1 && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src->nb[0] == sizeof(float) && ggml_is_contiguous(dst), "unexpected CPU FFN tensor layout");
            require(residual->type == GGML_TYPE_F32 && residual->nb[0] == sizeof(float), "unexpected residual layout");
            float * input_data = direct_input ? static_cast<float *>(runtime.prefix.input->data) : runtime.x.data();
            float * residual_data = direct_input ? static_cast<float *>(runtime.prefix.residual->data) : runtime.residual.data();
            for (int t = 0; t < 512; ++t) {
                memcpy(input_data + t * 5120, static_cast<const char *>(src->data) + t * src->nb[1], 5120 * sizeof(float));
                memcpy(residual_data + t * 5120, static_cast<const char *>(residual->data) + t * residual->nb[1], 5120 * sizeof(float));
            }
            runtime.profile.end(0);
            runtime.execute(static_cast<float *>(dst->data), direct_input);
            if (profile_enabled) { runtime.profile.callback_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - callback_start).count(); }
        } catch (const std::exception & failure) {
            runtime.error = failure.what();
            std::fill_n(static_cast<float *>(dst->data), ggml_nelements(dst), NAN);
        }
    }
}

static void bind_resident_tensor(ggml_tensor * target, const ggml_tensor * source) {
    require(source->buffer && source->data && ggml_is_contiguous(source) && ggml_are_same_shape(target, source) && target->type == source->type, "invalid resident tensor binding");
    target->buffer = source->buffer;
    target->data = source->data;
}

// signal that the FFN input is written, queue the Metal share, and hand the ANE call to the worker
static void ane_start(ggml_backend_t backend, decoder_ffn & runtime, const ggml_tensor * src, const ggml_tensor * residual) {
    auto & async = ane_async::get();
    require(async.enabled && runtime.backed && !runtime.full_metal, "split FFN calls need FFN_COREML_ASYNC and output backing");
    require(ggml_backend_metal_tensor_is_shared(src), "Core ML input must use shared Metal storage");
    const uint64_t value = ++async.seq;
    runtime.pending = value;
    ggml_backend_metal_shared_event(backend, (__bridge void *) async.ready, value, false);
    bind_resident_tensor(runtime.prefix.input, src);
    bind_resident_tensor(runtime.prefix.residual, residual);
    require(ggml_backend_graph_compute_async(backend, runtime.prefix.graph) == GGML_STATUS_SUCCESS, "resident Metal prefix failed");
    __strong MLDictionaryFeatureProvider * & inputs = runtime.inputs[src->data];
    if (!inputs) {
        NSError * failure = nil;
        MLMultiArray * x = [[MLMultiArray alloc] initWithDataPointer:src->data shape:@[@512, @5120] dataType:MLMultiArrayDataTypeFloat32 strides:@[@5120, @1] deallocator:nil error:&failure];
        require(x, "resident Core ML input binding failed");
        inputs = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:&failure];
        require(inputs, "resident Core ML provider failed");
    }
    decoder_ffn * worker = &runtime;
    MLDictionaryFeatureProvider * features = inputs;
    id<MTLSharedEvent> ready = async.ready, done = async.done;
    dispatch_async(async.queue, ^{
        @autoreleasepool {
            if (![ready waitUntilSignaledValue:value timeoutMS:60000]) {
                worker->fail("timed out waiting for the FFN input");
            } else {
                try { worker->predict_backed(features, false); } catch (const std::exception & failure) { worker->fail(failure.what()); }
            }
            done.signaledValue = value;
        }
    });
}

// make the GPU wait for the ANE call, then add the three parts into dst
static void ane_finish(ggml_backend_t backend, decoder_ffn & runtime, ggml_tensor * dst) {
    require(runtime.pending != 0, "FFN finish without a started call");
    ggml_backend_metal_shared_event(backend, (__bridge void *) ane_async::get().done, runtime.pending, true);
    runtime.pending = 0;
    bind_resident_tensor(runtime.join.gpu, runtime.prefix.output);
    bind_resident_tensor(runtime.join.output, dst);
    require(ggml_backend_graph_compute_async(backend, runtime.join.graph) == GGML_STATUS_SUCCESS, "resident Metal join failed");
    ++runtime.calls;
}

static bool decoder_resident_compute(ggml_backend_t backend, ggml_tensor * dst, void * userdata) {
    auto & call = *static_cast<ffn_call *>(userdata);
    auto & runtime = *call.runtime;
    runtime.profile.begin();
    const auto start = runtime.profile.last;
    @autoreleasepool {
        try {
            const auto * src = dst->src[0];
            const auto * residual = dst->src[1];
            require(src->data != dst->data && residual->data != dst->data, "resident FFN output aliases input");
            auto & async = ane_async::get();
            if (call.stage == 0) { ane_start(backend, runtime, src, residual); return true; }
            if (call.stage == 1) { ane_finish(backend, runtime, dst); return true; }
            if (async.enabled && runtime.backed && !runtime.full_metal) {
                ane_start(backend, runtime, src, residual);
                ane_finish(backend, runtime, dst);
                return true;
            }
            if (!runtime.full_metal) { ggml_backend_synchronize(backend); }
            runtime.profile.end(0);
            bind_resident_tensor(runtime.prefix.input, src);
            bind_resident_tensor(runtime.prefix.residual, residual);
            if (runtime.full_metal) { bind_resident_tensor(runtime.prefix.output, dst); }
            runtime.profile.end(1);
            require(ggml_backend_graph_compute_async(backend, runtime.prefix.graph) == GGML_STATUS_SUCCESS, "resident Metal prefix failed");
            runtime.profile.end(2);
            if (!runtime.full_metal) {
                require(ggml_backend_metal_tensor_is_shared(src), "Core ML input must use shared Metal storage");
                if (runtime.input.dataPointer != src->data) {
                    NSError * failure = nil;
                    runtime.input = [[MLMultiArray alloc] initWithDataPointer:src->data shape:@[@512, @5120] dataType:MLMultiArrayDataTypeFloat32 strides:@[@5120, @1] deallocator:nil error:&failure];
                    require(runtime.input, "resident Core ML input binding failed");
                    runtime.features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": runtime.input} error:&failure];
                    require(runtime.features, "resident Core ML provider failed");
                }
                if (runtime.backed) { runtime.predict_backed(runtime.features); } else { runtime.predict(true); }
                runtime.profile.end(5);
                bind_resident_tensor(runtime.join.gpu, runtime.prefix.output);
                bind_resident_tensor(runtime.join.output, dst);
                if (!runtime.backed) { ggml_backend_tensor_set(runtime.join.ane, runtime.partial.data(), 0, runtime.partial.size() * sizeof(float)); }
                runtime.profile.end(7);
                require(ggml_backend_graph_compute_async(backend, runtime.join.graph) == GGML_STATUS_SUCCESS, "resident Metal join failed");
                runtime.profile.end(8);
            }
            ++runtime.calls;
            if (profile_enabled) { runtime.profile.callback_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
            return true;
        } catch (const std::exception & failure) {
            runtime.error = failure.what();
            ggml_backend_synchronize(backend);
            fprintf(stderr, "resident FFN failed: %s\n", failure.what());
            return false;
        }
    }
}

static ggml_tensor * decoder_build(ggml_context * ctx, ggml_tensor * input, ggml_tensor * residual, int layer, void * userdata) {
    auto & state = *static_cast<decoder_state *>(userdata);
    auto found = state.layers.find(layer);
    if (found == state.layers.end() || found->second->ingraph || input->type != GGML_TYPE_F32 || input->ne[0] != 5120 || input->ne[2] != 1 || input->ne[3] != 1) { return nullptr; }
    // a partial microbatch of at least FFN_COREML_PAD_MIN rows is padded with zero rows to the 512 rows of the
    // Core ML programs; smaller ones stay on Metal
    static const int pad_min = std::getenv("FFN_COREML_PAD_MIN") ? std::atoi(std::getenv("FFN_COREML_PAD_MIN")) : 256;
    const int rows = input->ne[1];
    if (rows == 512) { return ggml_map_custom2(ctx, input, residual, decoder_compute, 1, &found->second->call[0]); }
    if (!gpu_resident || rows > 1024 || (rows < 512 && rows < pad_min)) { return nullptr; }
    auto padded_call = [&](ggml_tensor * x, ggml_tensor * r) {
        const int n = x->ne[1];
        auto * out = ggml_map_custom2(ctx, ggml_pad(ctx, x, 0, 512 - n, 0, 0), ggml_pad(ctx, r, 0, 512 - n, 0, 0), decoder_compute, 1, &found->second->call[0]);
        return ggml_view_2d(ctx, out, 5120, n, out->nb[1], 0);
    };
    if (rows < 512) { return padded_call(input, residual); }
    // 513..1023 rows (a partial 1024-token microbatch): one full call and one padded call
    auto rows_of = [&](ggml_tensor * x, int first, int n) { return ggml_view_2d(ctx, x, 5120, n, x->nb[1], first * x->nb[1]); };
    auto * head = ggml_map_custom2(ctx, rows_of(input, 0, 512), rows_of(residual, 0, 512), decoder_compute, 1, &found->second->call[0]);
    return ggml_concat(ctx, head, padded_call(rows_of(input, 512, rows - 512), rows_of(residual, 512, rows - 512)), 1);
}

// two-half microbatches (LLAMA_QWEN35_HALVES): stage 0 starts a half's FFN, stage 1 finishes it.
// FFN_COREML_HALVES_METAL keeps the FFN on Metal to check the half-batch graph against the normal one
static ggml_tensor * decoder_split_build(ggml_context * ctx, ggml_tensor * input, ggml_tensor * residual, int layer, int stage, void * userdata) {
    static const bool metal = std::getenv("FFN_COREML_HALVES_METAL") != nullptr;
    auto & state = *static_cast<decoder_state *>(userdata);
    auto found = state.layers.find(layer);
    if (metal || found == state.layers.end() || found->second->ingraph || !gpu_resident || !ane_async::get().enabled || !found->second->backed) { return nullptr; }
    if (stage == 1) {
        // a half that started on Metal finishes there too
        return input->op == GGML_OP_MAP_CUSTOM2 ? ggml_map_custom2(ctx, input, input, decoder_compute, 1, &found->second->call[2]) : nullptr;
    }
    // a short second half is padded to the 512 rows of the Core ML programs, or stays on Metal below FFN_COREML_PAD_MIN
    static const int pad_min = std::getenv("FFN_COREML_PAD_MIN") ? std::atoi(std::getenv("FFN_COREML_PAD_MIN")) : 256;
    const int rows = input->ne[1];
    require(input->type == GGML_TYPE_F32 && input->ne[0] == 5120 && rows <= 512, "split FFN half must be at most 512 x 5120");
    if (rows < pad_min) { return nullptr; }
    if (rows < 512) {
        input = ggml_pad(ctx, input, 0, 512 - rows, 0, 0);
        residual = ggml_pad(ctx, residual, 0, 512 - rows, 0, 0);
    }
    return ggml_map_custom2(ctx, input, residual, decoder_compute, 1, &found->second->call[1]);
}

// In-graph FFN (FFN_COREML_INGRAPH). start hands the input to the ANEs, whose partial outputs land in its buffer; the
// Metal share runs as normal nodes meanwhile; finish makes the GPU wait for the ANEs and sums their partials. finish
// also takes the input, so the allocator keeps it until the ANEs have read it
static void ingraph_cpu(ggml_tensor *, int, int, void *) {
    GGML_ABORT("the Core ML FFN sidecar runs only on Metal");
}

static int pad_min() {
    static const int value = std::getenv("FFN_COREML_PAD_MIN") ? std::atoi(std::getenv("FFN_COREML_PAD_MIN")) : 256;
    return value;
}

// A microbatch goes to the ANEs in chunks of the programs' 512 rows. The last chunk is padded with
// zero rows, or below FFN_COREML_PAD_MIN rows its ANE share is computed on Metal from views of the remaining channels
static ggml_tensor * ingraph_build(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * input, ggml_tensor * residual,
                                   ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, int layer, void * userdata) {
    auto & state = *static_cast<decoder_state *>(userdata);
    auto found = state.layers.find(layer);
    if (found == state.layers.end() || !found->second->ingraph || input->type != GGML_TYPE_F32 || input->ne[0] != 5120 ||
        input->ne[2] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input)) {
        return nullptr;
    }
    auto & runtime = *found->second;
    const int t = ingraph_tokens;
    const int rows = input->ne[1];
    if (rows < pad_min()) { return nullptr; }
    const int64_t m = runtime.metal_channels;
    const int chunks = rows / t + (rows % t >= pad_min() ? 1 : 0);

    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> started;
    for (int c = 0; c < chunks; ++c) {
        const int n = std::min(t, rows - c * t);
        ggml_tensor * x = ggml_view_2d(ctx, input, 5120, n, input->nb[1], c * t * input->nb[1]);
        if (n < t) { x = ggml_pad(ctx, x, 0, t - n, 0, 0); }
        ggml_tensor * start = ggml_custom_4d(ctx, GGML_TYPE_F32, 5120, t, runtime.model2 ? 2 : 1, 1, &x, 1, ingraph_cpu, 1, &runtime.call[3]);
        ggml_build_forward_expand(gf, start);
        started.emplace_back(start, x);
    }

    // Metal share, channels [0, m), of every row
    ggml_tensor * g = ggml_mul_mat(ctx, ggml_view_2d(ctx, gate, gate->ne[0], m, gate->nb[1], 0), input);
    ggml_tensor * u = ggml_mul_mat(ctx, ggml_view_2d(ctx, up, up->ne[0], m, up->nb[1], 0), input);
    ggml_tensor * d = ggml_mul_mat(ctx, ggml_view_2d(ctx, down, m, down->ne[1], down->nb[1], 0), ggml_swiglu_split(ctx, g, u));

    ggml_tensor * ane = nullptr;
    for (int c = 0; c < chunks; ++c) {
        ggml_tensor * args[2] = {started[c].first, started[c].second};
        ggml_tensor * part = ggml_custom_4d(ctx, GGML_TYPE_F32, 5120, t, 1, 1, args, 2, ingraph_cpu, 1, &runtime.call[4]);
        const int n = std::min(t, rows - c * t);
        if (n < t) { part = ggml_view_2d(ctx, part, 5120, n, part->nb[1], 0); }
        ane = ane ? ggml_concat(ctx, ane, part, 1) : part;
    }
    if (chunks * t < rows) {
        // the rows after the last chunk: channels [m, end) on Metal too
        const int64_t first = int64_t(chunks) * t, n = rows - first, rest = gate->ne[1] - m;
        ggml_tensor * x = ggml_view_2d(ctx, input, 5120, n, input->nb[1], first * input->nb[1]);
        ggml_tensor * gt = ggml_mul_mat(ctx, ggml_view_2d(ctx, gate, gate->ne[0], rest, gate->nb[1], m * gate->nb[1]), x);
        ggml_tensor * ut = ggml_mul_mat(ctx, ggml_view_2d(ctx, up, up->ne[0], rest, up->nb[1], m * up->nb[1]), x);
        ggml_tensor * dt = ggml_mul_mat(ctx, ggml_view_2d(ctx, down, rest, down->ne[1], down->nb[1], ggml_row_size(down->type, m)), ggml_swiglu_split(ctx, gt, ut));
        ane = ane ? ggml_concat(ctx, ane, dt, 1) : dt;
    }
    return ggml_add(ctx, ggml_add(ctx, d, residual), ane);
}

static void bind_data(ggml_tensor * target, ggml_backend_buffer_t buffer, void * data) {
    target->buffer = buffer;
    target->data = data;
}

static bool ingraph_execute(ggml_backend_t backend, ggml_tensor * dst, void * userdata) {
    auto & call = *static_cast<ffn_call *>(userdata);
    auto & runtime = *call.runtime;
    auto & async = ane_async::get();
    @autoreleasepool {
        try {
            if (call.stage == 2) {
                const ggml_tensor * x = dst->src[0];
                require(ggml_backend_metal_tensor_is_shared(x) && ggml_backend_metal_tensor_is_shared(dst), "in-graph FFN buffers must use shared Metal storage");
                const uint64_t value = ++async.seq;
                runtime.started.push_back(value);
                ggml_backend_metal_shared_event(backend, (__bridge void *) async.ready, value, false);
                __strong MLDictionaryFeatureProvider * & inputs = runtime.inputs[x->data];
                if (!inputs) {
                    NSError * failure = nil;
                    MLMultiArray * array = [[MLMultiArray alloc] initWithDataPointer:x->data shape:@[@(ingraph_tokens), @5120] dataType:MLMultiArrayDataTypeFloat32 strides:@[@5120, @1] deallocator:nil error:&failure];
                    require(array, "in-graph Core ML input binding failed");
                    inputs = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": array} error:&failure];
                    require(inputs, "in-graph Core ML provider failed");
                }
                auto & backings = runtime.outputs[dst->data];
                if (!backings.first) {
                    auto backing = [&](float * data) {
                        NSError * failure = nil;
                        MLMultiArray * array = [[MLMultiArray alloc] initWithDataPointer:data shape:@[@(ingraph_tokens), @5120] dataType:MLMultiArrayDataTypeFloat32 strides:@[@5120, @1] deallocator:nil error:&failure];
                        require(array, "in-graph output backing failed");
                        MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
                        options.outputBackings = @{@"y": array};
                        return options;
                    };
                    backings.first = backing(static_cast<float *>(dst->data));
                    if (runtime.model2) { backings.second = backing(static_cast<float *>(dst->data) + ingraph_tokens * 5120); }
                }
                decoder_ffn * worker = &runtime;
                MLDictionaryFeatureProvider * features = inputs;
                MLPredictionOptions * first = backings.first, * second = backings.second;
                float * out = static_cast<float *>(dst->data);
                id<MTLSharedEvent> ready = async.ready, done = async.done;
                dispatch_async(async.queue, ^{
                    @autoreleasepool {
                        if (![ready waitUntilSignaledValue:value timeoutMS:60000]) {
                            worker->fail("timed out waiting for the FFN input");
                        } else {
                            try { worker->predict_into(features, first, second, out); } catch (const std::exception & failure) { worker->fail(failure.what()); }
                        }
                        done.signaledValue = value;
                    }
                });
                return true;
            }
            // chunks finish in the order they started
            require(call.stage == 3 && !runtime.started.empty(), "in-graph FFN finish without a started call");
            ggml_backend_metal_shared_event(backend, (__bridge void *) async.done, runtime.started.front(), true);
            runtime.started.pop_front();
            // one join graph for every layer: dst = first partial + second partial (zeros with one package)
            static metal_join * join = nullptr;
            if (!join) {
                join = new metal_join;
                join->init(backend, 5120, ingraph_tokens);
            }
            const ggml_tensor * start = dst->src[0];
            bind_data(join->gpu, start->buffer, start->data);
            if (runtime.model2) { bind_data(join->ane, start->buffer, static_cast<float *>(start->data) + ingraph_tokens * 5120); }
            bind_data(join->output, dst->buffer, dst->data);
            require(ggml_backend_graph_compute_async(backend, join->graph) == GGML_STATUS_SUCCESS, "in-graph FFN join failed");
            ++runtime.calls;
            return true;
        } catch (const std::exception & failure) {
            runtime.fail(failure.what());
            fprintf(stderr, "in-graph FFN failed: %s\n", failure.what());
            return false;
        }
    }
}

static void set_builders(bool split, decoder_state * state) {
    static const bool halves = std::getenv("LLAMA_QWEN35_HALVES") != nullptr;
    llama_ffn_coreml_set_builder(split ? decoder_build : nullptr, split ? state : nullptr);
    // the all-Metal reference (quality-metal) keeps the normal graph
    llama_ffn_coreml_set_split_builder(split && halves && !state->metal_reference ? decoder_split_build : nullptr, split ? state : nullptr);
    llama_ffn_coreml_set_graph_builder(split && ingraph_mode && !state->metal_reference ? ingraph_build : nullptr, split ? state : nullptr);
}

// prompt tokens per llama_decode call and microbatch (FFN_COREML_UBATCH, 512 or 1024)
static int chunk_tokens() {
    static const int chunk = std::getenv("FFN_COREML_UBATCH") ? std::atoi(std::getenv("FFN_COREML_UBATCH")) : 512;
    return chunk;
}

// Core ML calls per offloaded layer for a prompt: one per microbatch, or with FFN_COREML_INGRAPH one per chunk
// (a last chunk below FFN_COREML_PAD_MIN rows stays on Metal)
static uint64_t offload_calls(int length) {
    if (!ingraph_mode) { return length / 512; }
    uint64_t calls = 0;
    for (int offset = 0; offset < length; offset += chunk_tokens()) {
        const int n = std::min(chunk_tokens(), length - offset);
        calls += n / ingraph_tokens + (n % ingraph_tokens >= pad_min() ? 1 : 0);
    }
    return calls;
}

static llama_context * make_context(llama_model * model, int context, decoder_state & state, bool split) {
    set_builders(split, &state);
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = chunk_tokens();
    cp.n_threads = 20;
    cp.n_threads_batch = state.batch_threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx = llama_init_from_model(model, cp);
    require(ctx, "could not create decoder benchmark context");
    return ctx;
}

static std::vector<llama_token> prompt_tokens(llama_model * model, const std::string & path, int needed) {
    std::ifstream input(path);
    require(bool(input), "could not read benchmark prompt");
    std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    require(!source.empty(), "empty benchmark prompt");
    std::string text = source;
    auto * vocab = llama_model_get_vocab(model);
    for (;;) {
        require(text.size() < (1u << 22), "benchmark prompt exceeded 4 MiB");
        int count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, false);
        require(count < 0, "could not count prompt tokens");
        if (-count >= needed) {
            std::vector<llama_token> tokens(-count);
            require(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, false) >= needed, "tokenization failed");
            tokens.resize(needed);
            return tokens;
        }
        text += source;
    }
}

static llama_token greedy(llama_context * ctx, int vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits, "missing decoder logits");
    return std::max_element(logits, logits + vocab) - logits;
}

static double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static void request(llama_context * ctx, decoder_state & state, const std::vector<llama_token> & prompt, int length, int generated,
                    const std::string & mode, int repetition, int vocab, const std::filesystem::path & output) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
    uint64_t before = state.calls();
    for (auto & entry : state.layers) { entry.second->profile = {}; }
    auto start = std::chrono::steady_clock::now();
    for (int offset = 0; offset < length; offset += chunk_tokens()) {
        int n = std::min(chunk_tokens(), length - offset);
        require(llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(prompt.data()) + offset, n)) == 0, "prompt evaluation failed");
    }
    llama_synchronize(ctx);
    state.check();
    double pp_ms = milliseconds(start);
    uint64_t after_prompt = state.calls();
    std::vector<llama_token> result;
    result.push_back(greedy(ctx, vocab));
    double ttft_ms = milliseconds(start);
    const bool decode_diagnostics = std::getenv("FFN_COREML_DECODE_DIAGNOSTICS") != nullptr;
    std::vector<double> submit_times, wait_times, sample_times;
    if (decode_diagnostics) {
        submit_times.reserve(generated - 1);
        wait_times.reserve(generated - 1);
        sample_times.reserve(generated - 1);
    }
    if (std::getenv("FFN_COREML_LAYOUT")) { setenv("FFN_COREML_TRACE_DECODE", "1", 1); }
    const auto perf_before_decode = llama_perf_context(ctx);
    auto decode_start = std::chrono::steady_clock::now();
    double transition_ms = 0;
    if (const char * value = std::getenv("FFN_COREML_DECODE_PLAN")) {
        const int reserve_tokens = std::stoi(value);
        require(reserve_tokens == 1 || reserve_tokens == 512, "decode plan must use 1 or 512 tokens");
        set_builders(false, nullptr);
        auto * graph = llama_graph_reserve(ctx, reserve_tokens, 1, reserve_tokens);
        set_builders(mode == "split" || mode == "hook-metal", &state);
        require(graph, "could not reserve decode allocation plan");
        transition_ms = milliseconds(decode_start);
    }
    for (int i = 1; i < generated; ++i) {
        llama_token previous = result.back();
        const auto submit_start = decode_diagnostics ? std::chrono::steady_clock::now() : decode_start;
        require(llama_decode(ctx, llama_batch_get_one(&previous, 1)) == 0, "generation failed");
        const auto wait_start = decode_diagnostics ? std::chrono::steady_clock::now() : decode_start;
        llama_synchronize(ctx);
        const auto sample_start = decode_diagnostics ? std::chrono::steady_clock::now() : decode_start;
        state.check();
        result.push_back(greedy(ctx, vocab));
        if (decode_diagnostics) {
            submit_times.push_back(std::chrono::duration<double, std::milli>(wait_start - submit_start).count());
            wait_times.push_back(std::chrono::duration<double, std::milli>(sample_start - wait_start).count());
            sample_times.push_back(milliseconds(sample_start));
        }
    }
    double decode_ms = milliseconds(decode_start), total_ms = milliseconds(start);
    unsetenv("FFN_COREML_TRACE_DECODE");
    const auto perf_after_decode = llama_perf_context(ctx);
    printf("decode_graphs repetition=%d prompt_tokens=%d steps=%d reused=%d transition_ms=%.6f\n", repetition, length, generated - 1, perf_after_decode.n_reused - perf_before_decode.n_reused, transition_ms);
    if (decode_diagnostics) {
        for (size_t i = 0; i < submit_times.size(); ++i) {
            printf("decode_step repetition=%d prompt_tokens=%d step=%zu submit_ms=%.6f wait_ms=%.6f sample_ms=%.6f\n", repetition, length, i + 1, submit_times[i], wait_times[i], sample_times[i]);
        }
    }
    uint64_t decode_calls = state.calls() - after_prompt;
    require(decode_calls == 0, "Core ML unexpectedly ran during token decoding");
    uint64_t expected = (mode == "split" || mode == "hook-metal") && !std::getenv("FFN_COREML_HALVES_METAL") ? state.layers.size() * offload_calls(length) : 0;
    require(after_prompt - before == expected, "unexpected number of integrated FFN calls");
    printf("request mode=%s prompt_tokens=%d generated=%d repetition=%d pp_ms=%.6f pp_tps=%.6f ttft_ms=%.6f decode_ms=%.6f decode_tps=%.6f total_ms=%.6f output_tps=%.6f offload_calls=%llu decode_offload_calls=%llu\n",
           mode.c_str(), length, generated, repetition, pp_ms, length * 1000.0 / pp_ms, ttft_ms, decode_ms,
           (generated - 1) * 1000.0 / decode_ms, total_ms, generated * 1000.0 / total_ms,
           (unsigned long long) (after_prompt - before), (unsigned long long) decode_calls);
    if (profile_enabled) {
        const char * stages[] = {"callback_copy", "metal_input_copy", "metal_submit_or_compute", "coreml_input_copy", "coreml_predict", "coreml_output", "metal_wait", "join_copy", "join_compute", "output_copy"};
        for (const auto & entry : state.layers) {
            const auto & profile = entry.second->profile;
            printf("profile mode=%s prompt_tokens=%d repetition=%d layer=%d callback_ms=%.6f", mode.c_str(), length, repetition, entry.first, profile.callback_ms);
            for (int i = 0; i < 10; ++i) { printf(" %s_ms=%.6f", stages[i], profile.ms[i]); }
            printf("\n");
        }
    }
    fflush(stdout);
    if (repetition >= 0) {
        std::ofstream tokens(output / (mode + "-p" + std::to_string(length) + "-r" + std::to_string(repetition) + ".tokens"));
        for (auto token : result) { tokens << token << '\n'; }
        require(bool(tokens), "could not save generated token IDs");
    }
}

static void quality(llama_model * model, decoder_state & state, const std::vector<llama_token> & prompt, int length, int vocab) {
    auto * baseline = make_context(model, length + 64, state, false);
    auto * split = make_context(model, length + 64, state, true);
    double kl_sum = 0, kl_max = 0, nll_metal = 0, nll_split = 0;
    int agreement = 0;
    std::vector<float> reference(size_t(chunk_tokens()) * vocab);
    std::vector<int8_t> logits(chunk_tokens(), 1);
    uint64_t before = state.calls();
    for (int offset = 0; offset < length; offset += chunk_tokens()) {
        int n = std::min(chunk_tokens(), length - offset);
        auto batch = llama_batch_get_one(const_cast<llama_token *>(prompt.data()) + offset, n);
        batch.logits = logits.data();
        set_builders(false, nullptr);
        require(llama_decode(baseline, batch) == 0, "quality baseline failed");
        llama_synchronize(baseline);
        memcpy(reference.data(), llama_get_logits(baseline), size_t(n) * vocab * sizeof(float));
        set_builders(true, &state);
        require(llama_decode(split, batch) == 0, "quality split failed");
        llama_synchronize(split);
        state.check();
        const float * actual = llama_get_logits(split);
        for (int row = 0; row < n; ++row) {
            const float * a = reference.data() + size_t(row) * vocab;
            const float * b = actual + size_t(row) * vocab;
            float ma = *std::max_element(a, a + vocab), mb = *std::max_element(b, b + vocab);
            double za = 0, zb = 0, delta = 0;
            for (int j = 0; j < vocab; ++j) {
                if (!std::isfinite(a[j]) || !std::isfinite(b[j])) { throw std::runtime_error("non-finite quality logits"); }
                double ea = std::exp(double(a[j]) - ma);
                za += ea;
                zb += std::exp(double(b[j]) - mb);
                delta += ea * (double(a[j]) - b[j]);
            }
            double la = ma + std::log(za), lb = mb + std::log(zb);
            double kl = std::max(0.0, delta / za + lb - la);
            kl_sum += kl;
            kl_max = std::max(kl_max, kl);
            agreement += std::max_element(a, a + vocab) - a == std::max_element(b, b + vocab) - b;
            auto label = prompt[offset + row + 1];
            nll_metal += la - a[label];
            nll_split += lb - b[label];
        }
    }
    require(std::getenv("FFN_COREML_HALVES_METAL") || state.calls() - before == state.layers.size() * offload_calls(length), "quality did not use intended offload layers");
    bool screen_pass = kl_sum / length <= 0.001 && std::exp((nll_split - nll_metal) / length) <= 1.01;
    printf("quality prompt_tokens=%d mean_kl=%.9f max_kl=%.9f top1_agreement=%.9f metal_nll=%.9f split_nll=%.9f metal_ppl=%.9f split_ppl=%.9f offload_calls=%llu screen_pass=%d reference_custom=%d\n",
           length, kl_sum / length, kl_max, double(agreement) / length, nll_metal / length, nll_split / length,
           std::exp(nll_metal / length), std::exp(nll_split / length), (unsigned long long) (state.calls() - before), int(screen_pass), int(state.metal_reference));
    fflush(stdout);
    llama_free(split);
    llama_free(baseline);
    set_builders(false, nullptr);
}

int main(int argc, char ** argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s model.gguf assets-root metal|resident|resident-metal|split|quality|hook-metal|quality-metal|quality-self prompt-file lengths generated repetitions output-directory\n", argv[0]);
        return 1;
    }
    @autoreleasepool {
        try {
            gpu_resident = std::getenv("FFN_COREML_GPU_RESIDENT") != nullptr;
            if (gpu_resident) {
                ggml_backend_metal_set_custom2(decoder_compute, decoder_resident_compute);
                ggml_backend_metal_set_custom(ingraph_cpu, ingraph_execute);
            }
            printf("gpu_resident=%d\n", int(gpu_resident));
            if (gpu_resident) { printf("resident_profile_stages=producer_wait,bind,prefix_submit,input_bind,predict,output_convert,unused,join_input,join_submit,unused\n"); }
            direct_input = std::getenv("FFN_COREML_DIRECT_INPUT") != nullptr;
            printf("direct_input=%d\n", int(direct_input));
            profile_enabled = std::getenv("FFN_COREML_PROFILE") != nullptr;
            printf("profile_enabled=%d\n", int(profile_enabled));
            std::string mode = argv[3];
            require(mode == "metal" || mode == "resident" || mode == "resident-metal" || mode == "split" || mode == "quality" || mode == "hook-metal" || mode == "quality-metal" || mode == "quality-self", "invalid decoder mode");
            int generated = std::stoi(argv[6]), reps = std::stoi(argv[7]);
            require(generated >= 2 && generated <= 512 && reps >= 1 && reps <= 10, "invalid repetition or generation count");
            std::vector<int> lengths;
            std::stringstream list(argv[5]);
            std::string item;
            while (std::getline(list, item, ',')) {
                int n = std::stoi(item);
                // the in-graph sidecar also takes partial microbatches
                require(n >= 512 && n <= 8192 && (ingraph_mode || n % chunk_tokens() == 0), "prompt length must be a multiple of the microbatch in 512..8192");
                lengths.push_back(n);
            }
            require(!lengths.empty(), "no prompt lengths");
            std::filesystem::path output(argv[8]);
            std::filesystem::create_directories(output);
            llama_backend_init();
            decoder_state state;
            if (const char * value = std::getenv("FFN_COREML_THREADS_BATCH")) {
                state.batch_threads = std::stoi(value);
                require(state.batch_threads >= 1 && state.batch_threads <= 20, "batch threads must be in 1..20");
            }
            printf("decoder_mode=%s batch_threads=%d decode_threads=20\n", mode.c_str(), state.batch_threads);
            state.metal_reference = mode == "hook-metal" || mode == "quality-metal" || mode == "resident-metal";
            if (mode != "metal" && mode != "quality-self") {
                otf.init(argv[1]);
                load_layers(state, argv[1], argv[2]);
            }
            auto * device = ggml_backend_dev_by_name("MTL0");
            require(device, "MTL0 unavailable");
            ggml_backend_dev_t devices[] = {device, nullptr};
            auto mp = llama_model_default_params();
            mp.n_gpu_layers = 99;
            mp.devices = devices;
            auto * model = llama_model_load_from_file(argv[1], mp);
            require(model, "decoder model load failed");
            int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            auto prompt = prompt_tokens(model, argv[4], *std::max_element(lengths.begin(), lengths.end()) + 1);
            for (int length : lengths) {
                std::ofstream input(output / ("prompt-" + std::to_string(length) + ".tokens"));
                for (int i = 0; i < length; ++i) { input << prompt[i] << '\n'; }
                require(bool(input), "could not save benchmark inputs");
                if (mode == "quality" || mode == "quality-metal" || mode == "quality-self") { quality(model, state, prompt, length, vocab); continue; }
                auto * ctx = make_context(model, length + generated + 64, state, mode == "split" || mode == "hook-metal");
                request(ctx, state, prompt, length, std::min(generated, 16), mode, -1, vocab, output);
                for (int rep = 0; rep < reps; ++rep) { request(ctx, state, prompt, length, generated, mode, rep, vocab, output); }
                llama_free(ctx);
                set_builders(false, nullptr);
            }
            llama_model_free(model);
            uint64_t fallbacks = 0;
            for (const auto & entry : state.layers) { fallbacks += entry.second->backing_fallbacks; }
            printf("output_backing_fallbacks=%llu\n", (unsigned long long) fallbacks);
            state.layers.clear();
            if (gpu_resident) {
                ggml_backend_metal_set_custom2(nullptr, nullptr);
                ggml_backend_metal_set_custom(nullptr, nullptr);
            }
            llama_backend_free();
        } catch (const std::exception & failure) {
            fprintf(stderr, "%s\n", failure.what());
            return 1;
        }
    }
    return 0;
}
