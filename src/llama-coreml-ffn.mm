#include "llama-coreml-ffn-impl.h"
#include "llama-coreml-ffn-export.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-metal.h"

#import <CoreML/CoreML.h>
#import <Metal/Metal.h>

#include <deque>
#include <memory>
#include <mutex>

#include "llama-coreml-ffn-templates.inc"

namespace {
constexpr int hidden = 5120;
constexpr int channels = 17408;
constexpr int rows = 512;
std::mutex registry_mutex;
size_t registered_contexts = 0;

void require(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string failure_message(NSError * failure, const char * fallback) {
    return failure ? failure.localizedDescription.UTF8String : fallback;
}

struct ffn_layer;
struct ffn_call {
    ffn_layer * layer;
    bool finish;
};

struct ffn_layer {
    llama_coreml_ffn * owner;
    MLModel * model[2] = {nil, nil};
    ffn_call start{this, false};
    ffn_call finish{this, true};
    std::deque<uint64_t> started;
};

void coreml_op_cpu(ggml_tensor *, int, int, void *) {
    GGML_ABORT("Core ML FFN custom operations require Metal");
}

MLMultiArray * bind_array(float * data, NSError ** failure) {
    return [[MLMultiArray alloc] initWithDataPointer:data shape:@[@(rows), @(hidden)]
        dataType:MLMultiArrayDataTypeFloat32 strides:@[@(hidden), @1] deallocator:nil error:failure];
}

void predict(MLModel * model, MLDictionaryFeatureProvider * features, float * output) {
    NSError * failure = nil;
    MLMultiArray * backing = bind_array(output, &failure);
    require(backing, failure_message(failure, "Core ML output binding failed"));
    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    options.outputBackings = @{@"y": backing};
    id<MLFeatureProvider> result = [model predictionFromFeatures:features options:options error:&failure];
    require(result, failure_message(failure, "Core ML FFN prediction failed"));
    MLMultiArray * y = [result featureValueForName:@"y"].multiArrayValue;
    require(y && y.dataType == MLMultiArrayDataTypeFloat32 && y.shape.count == 2 &&
        y.shape[0].intValue == rows && y.shape[1].intValue == hidden, "invalid Core ML FFN output");
    if (y.dataPointer != output) {
        const float * data = static_cast<const float *>(y.dataPointer);
        const size_t s0 = y.strides[0].unsignedLongLongValue;
        const size_t s1 = y.strides[1].unsignedLongLongValue;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < hidden; ++c) { output[r * hidden + c] = data[r * s0 + c * s1]; }
        }
    }
}

bool execute(ggml_backend_t backend, ggml_tensor * dst, void * userdata);
} // namespace

struct llama_coreml_ffn {
    int metal_channels;
    int packages;
    int min_tokens;
    std::vector<std::unique_ptr<ffn_layer>> layers;
    id<MTLSharedEvent> ready = nil;
    id<MTLSharedEvent> done = nil;
    dispatch_queue_t queue = nil;
    uint64_t sequence = 0;
    uint64_t checked_sequence = 0;
    std::atomic<bool> failed{false};
    bool registered = false;
    std::mutex error_mutex;
    std::string error;

    ~llama_coreml_ffn() {
        if (queue) { dispatch_sync(queue, ^{}); }
        if (registered) {
            std::lock_guard<std::mutex> lock(registry_mutex);
            if (--registered_contexts == 0) { ggml_backend_metal_set_custom_serial(coreml_op_cpu, nullptr); }
        }
    }

    void fail(const std::string & message) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (error.empty()) { error = message; }
        failed.store(true);
    }
};

namespace {
bool execute(ggml_backend_t backend, ggml_tensor * dst, void * userdata) {
    auto & call = *static_cast<ffn_call *>(userdata);
    auto & layer = *call.layer;
    auto & state = *layer.owner;
    @autoreleasepool {
        try {
            if (call.finish) {
                require(!layer.started.empty(), "Core ML FFN finish without start");
                ggml_backend_metal_shared_event(backend, (__bridge void *) state.done, layer.started.front(), true);
                layer.started.pop_front();
                return true;
            }
            auto * x = dst->src[0];
            require(ggml_backend_metal_tensor_is_shared(x) && ggml_backend_metal_tensor_is_shared(dst),
                "Core ML FFN requires shared Metal tensors");
            NSError * failure = nil;
            MLMultiArray * array = bind_array(static_cast<float *>(x->data), &failure);
            require(array, failure_message(failure, "Core ML input binding failed"));
            MLDictionaryFeatureProvider * features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": array} error:&failure];
            require(features, failure_message(failure, "Core ML input provider failed"));
            const uint64_t value = ++state.sequence;
            layer.started.push_back(value);
            ggml_backend_metal_shared_event(backend, (__bridge void *) state.ready, value, false);
            ffn_layer * worker = &layer;
            llama_coreml_ffn * owner = &state;
            float * output = static_cast<float *>(dst->data);
            dispatch_async(state.queue, ^{
                @autoreleasepool {
                    try {
                        require([owner->ready waitUntilSignaledValue:value timeoutMS:60000], "Core ML FFN input wait timed out");
                        if (owner->packages == 2) {
                            __block std::string second_error;
                            dispatch_group_t group = dispatch_group_create();
                            dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                                @autoreleasepool {
                                    try { predict(worker->model[1], features, output + rows * hidden); }
                                    catch (const std::exception & e) { second_error = e.what(); }
                                }
                            });
                            std::string first_error;
                            try { predict(worker->model[0], features, output); }
                            catch (const std::exception & e) { first_error = e.what(); }
                            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
                            require(first_error.empty(), first_error);
                            require(second_error.empty(), second_error);
                        } else {
                            predict(worker->model[0], features, output);
                        }
                    } catch (const std::exception & e) { owner->fail(e.what()); }
                    owner->done.signaledValue = value;
                }
            });
            return true;
        } catch (const std::exception & e) {
            state.fail(e.what());
            return false;
        }
    }
}

otf_template builtin_template(int count) {
    otf_template result;
    result.dir = "int8-c" + std::to_string(count);
    const unsigned char * program;
    const unsigned char * manifest;
    const unsigned char * layout;
    size_t program_size, manifest_size, layout_size;
#define FFN_TEMPLATE(C) \
    program = ffn_##C##_model_mlmodel; program_size = sizeof(ffn_##C##_model_mlmodel); \
    manifest = ffn_##C##_Manifest_json; manifest_size = sizeof(ffn_##C##_Manifest_json); \
    layout = ffn_##C##_layout_txt; layout_size = sizeof(ffn_##C##_layout_txt)
    if (count == 4608) { FFN_TEMPLATE(4608); }
    else { require(count == 6144, "unsupported Core ML FFN partition"); FFN_TEMPLATE(6144); }
#undef FFN_TEMPLATE
    result.program.assign(program, program + program_size);
    result.manifest.assign(reinterpret_cast<const char *>(manifest), manifest_size);
    result.layout_text.assign(reinterpret_cast<const char *>(layout), layout_size);
    std::istringstream input(result.layout_text);
    result.parse(input);
    require(result.channels == count && result.hidden == hidden, "invalid bundled FFN template");
    return result;
}
} // namespace

llama_coreml_ffn * llama_coreml_ffn_load(const llama_model & model, const llama_coreml_ffn_params & params) {
    @autoreleasepool {
        require(params.cache_dir && params.cache_dir[0], "cache_dir is required");
        require(!model.source_path.empty(), "Core ML FFN requires a model loaded from a single-file GGUF on disk");
        const auto & model_path = model.source_path;
        require(std::filesystem::file_size(model_path) == model.source_size &&
            std::filesystem::last_write_time(model_path).time_since_epoch().count() == model.source_mtime,
            "GGUF changed since model loading");
        require(params.n_packages >= 1 && params.n_packages <= 2 && params.metal_channels > 0 &&
            params.metal_channels < channels && params.metal_channels % 256 == 0 &&
            (channels - params.metal_channels) % params.n_packages == 0 && params.min_tokens >= 2 && params.min_tokens <= rows,
            "invalid Core ML FFN configuration");
        require(model.arch == LLM_ARCH_QWEN35 && model.hparams.n_embd == hidden && model.hparams.n_layer() == 64 &&
            !model.hparams.no_alloc && model.hdmd.rot.empty() && model.hdmd.inv.empty(), "Core ML FFN requires an unrotated Qwen3.8-27B model");
        for (int i = 0; i < 64; ++i) {
            const auto & layer = model.layers[i];
            require(model.dev_layer(i) && std::string(ggml_backend_dev_name(model.dev_layer(i))) == "MTL0",
                "Core ML FFN requires the default Metal device for every layer");
            for (auto * weight : {layer.ffn_gate, layer.ffn_up, layer.ffn_down}) {
                require(model.prec_policy.prec_acc.count(weight) == 0 && model.prec_policy.prec_src1.count(weight) == 0,
                    "Core ML FFN does not support per-weight precision overrides");
            }
            require(model.hparams.n_ff(i) == channels && !layer.ffn_gate_s && !layer.ffn_up_s && !layer.ffn_down_s &&
                layer.ffn_gate && layer.ffn_up && layer.ffn_down &&
                ggml_backend_metal_tensor_is_shared(layer.ffn_gate) && ggml_backend_metal_tensor_is_shared(layer.ffn_up) &&
                ggml_backend_metal_tensor_is_shared(layer.ffn_down), "all FFN weights must use shared Metal storage without external scales");
        }
        const auto tmpl = builtin_template((channels - params.metal_channels) / params.n_packages);
        const auto cache = otf_cache_dir(params.cache_dir, model_path, tmpl);
        auto state = std::make_unique<llama_coreml_ffn>();
        state->metal_channels = params.metal_channels;
        state->packages = params.n_packages;
        state->min_tokens = params.min_tokens;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device, "no Metal device");
        state->ready = [device newSharedEvent];
        state->done = [device newSharedEvent];
        state->queue = dispatch_queue_create("llama-coreml-ffn", DISPATCH_QUEUE_SERIAL);
        require(state->ready && state->done && state->queue, "could not create Core ML FFN synchronization objects");
        MLModelConfiguration * configuration = [[MLModelConfiguration alloc] init];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        for (int i = 0; i < 64; ++i) {
            @autoreleasepool {
                auto layer = std::make_unique<ffn_layer>();
                layer->owner = state.get();
                otf_layer weights;
                bool read = false;
                for (int p = 0; p < state->packages; ++p) {
                    double exported, compiled;
                    NSURL * url = otf_compiled(tmpl, cache, i, state->metal_channels + p * tmpl.channels, [&]() -> const otf_layer & {
                        if (!read) { weights.read(model_path, i); read = true; }
                        return weights;
                    }, exported, compiled);
                    NSError * failure = nil;
                    layer->model[p] = [MLModel modelWithContentsOfURL:url configuration:configuration error:&failure];
                    require(layer->model[p], failure_message(failure, "Core ML FFN load failed"));
                    MLMultiArray * zeros = [[MLMultiArray alloc] initWithShape:@[@(rows), @(hidden)] dataType:MLMultiArrayDataTypeFloat32 error:&failure];
                    require(zeros, "Core ML FFN warm-up input allocation failed");
                    memset(zeros.dataPointer, 0, rows * hidden * sizeof(float));
                    MLDictionaryFeatureProvider * features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": zeros} error:&failure];
                    require(features && [layer->model[p] predictionFromFeatures:features error:&failure],
                        failure_message(failure, "Core ML FFN warm-up failed"));
                }
                state->layers.push_back(std::move(layer));
            }
        }
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            ggml_backend_metal_set_custom_serial(coreml_op_cpu, execute);
            ++registered_contexts;
            state->registered = true;
        }
        LLAMA_LOG_INFO("Core ML FFN: %zu layers, %d Metal channels, %d packages; cache %s\n",
            state->layers.size(), state->metal_channels, state->packages, cache.c_str());
        return state.release();
    }
}

void llama_coreml_ffn_free(llama_coreml_ffn * runtime) { delete runtime; }

bool llama_coreml_ffn_check(llama_coreml_ffn * runtime) {
    if (!runtime) { return true; }
    if (runtime->sequence == runtime->checked_sequence && !runtime->failed.load()) { return true; }
    dispatch_sync(runtime->queue, ^{});
    runtime->checked_sequence = runtime->sequence;
    std::lock_guard<std::mutex> lock(runtime->error_mutex);
    if (runtime->error.empty()) { return true; }
    LLAMA_LOG_ERROR("Core ML FFN: %s\n", runtime->error.c_str());
    return false;
}

ggml_tensor * llama_coreml_ffn_build(llama_coreml_ffn * runtime, ggml_context * ctx, ggml_cgraph * graph,
        ggml_tensor * input, ggml_tensor * residual, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, int il) {
    if (!runtime || il < 0 || size_t(il) >= runtime->layers.size() || input->type != GGML_TYPE_F32 || input->ne[0] != hidden ||
        input->ne[1] < runtime->min_tokens || input->ne[2] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input)) { return nullptr; }
    const int64_t m = runtime->metal_channels;
    const int64_t n = input->ne[1];
    const int64_t chunks = n / rows + (n % rows >= runtime->min_tokens ? 1 : 0);
    auto & layer = *runtime->layers[il];
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> started;
    for (int64_t c = 0; c < chunks; ++c) {
        const int64_t count = std::min<int64_t>(rows, n - c * rows);
        auto * x = ggml_view_2d(ctx, input, hidden, count, input->nb[1], c * rows * input->nb[1]);
        if (count < rows) { x = ggml_pad(ctx, x, 0, rows - count, 0, 0); }
        auto * start = ggml_custom_4d(ctx, GGML_TYPE_F32, hidden, rows, runtime->packages, 1, &x, 1, coreml_op_cpu, 1, &layer.start);
        ggml_build_forward_expand(graph, start);
        started.emplace_back(start, x);
    }
    auto * g = ggml_mul_mat(ctx, ggml_view_2d(ctx, gate, gate->ne[0], m, gate->nb[1], 0), input);
    auto * u = ggml_mul_mat(ctx, ggml_view_2d(ctx, up, up->ne[0], m, up->nb[1], 0), input);
    auto * gpu = ggml_mul_mat(ctx, ggml_view_2d(ctx, down, m, down->ne[1], down->nb[1], 0), ggml_swiglu_split(ctx, g, u));
    ggml_build_forward_expand(graph, gpu);
    ggml_tensor * ane = nullptr;
    for (int64_t c = 0; c < chunks; ++c) {
        // The identity op waits on the GPU and retains the input until the worker has read it.
        ggml_tensor * dependencies[] = {started[c].second, gpu};
        auto * finish = ggml_custom_inplace(ctx, started[c].first, dependencies, 2, coreml_op_cpu, 1, &layer.finish);
        auto * partial = ggml_view_2d(ctx, finish, hidden, rows, finish->nb[1], 0);
        if (runtime->packages == 2) {
            partial = ggml_add(ctx, partial, ggml_view_2d(ctx, finish, hidden, rows, finish->nb[1], finish->nb[2]));
        }
        const int64_t count = std::min<int64_t>(rows, n - c * rows);
        if (count < rows) { partial = ggml_view_2d(ctx, partial, hidden, count, partial->nb[1], 0); }
        ane = ane ? ggml_concat(ctx, ane, partial, 1) : partial;
    }
    if (chunks * rows < n) {
        const int64_t first = chunks * rows, count = n - first, rest = channels - m;
        auto * x = ggml_view_2d(ctx, input, hidden, count, input->nb[1], first * input->nb[1]);
        auto * gt = ggml_mul_mat(ctx, ggml_view_2d(ctx, gate, hidden, rest, gate->nb[1], m * gate->nb[1]), x);
        auto * ut = ggml_mul_mat(ctx, ggml_view_2d(ctx, up, hidden, rest, up->nb[1], m * up->nb[1]), x);
        auto * dt = ggml_mul_mat(ctx, ggml_view_2d(ctx, down, rest, hidden, down->nb[1], ggml_row_size(down->type, m)), ggml_swiglu_split(ctx, gt, ut));
        ane = ane ? ggml_concat(ctx, ane, dt, 1) : dt;
    }
    return ggml_add(ctx, ggml_add(ctx, gpu, residual), ane);
}
