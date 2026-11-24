#if !__has_feature(objc_arc)
#error "parakeet-diarizer.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-diarizer.h"
#include "coreml/coreml_diarizers.h"
#include "coreml/parakeet_coreml_load_error.h"
#include "coreml/parakeet_coreml_sidecars.h"
#include "coreml/parakeet_coreml_units.h"
#include "parakeet_coreml_path.h"
#include "parakeet_log.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <dirent.h>
#include <limits>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char * kValidMaskName = "valid_mask";
constexpr int kTrailingDims = 2;

enum DiarizerStatus {
    kOk = 0,
    kBadArguments = 1,
    kBadSlice = 2,
    kPredictionFailed = 3,
    kMissingOutput = 4,
};

struct Binding {
    std::string name;
    const void * array = nullptr;
    int64_t rows = 0;
    int64_t columns = 0;
};

MLMultiArray * bridged(const void * array) {
    return (__bridge MLMultiArray *) array;
}

bool trailing_rows_and_columns(NSArray<NSNumber *> * shape, int64_t * rows, int64_t * columns) {
    if (shape.count == 0) return false;
    if (shape.count == 1) {
        *rows = shape[0].longLongValue;
        *columns = 1;
        return *rows > 0;
    }
    *rows = shape[shape.count - kTrailingDims].longLongValue;
    *columns = shape[shape.count - 1].longLongValue;
    return *rows > 0 && *columns > 0;
}

bool is_contiguous(MLMultiArray * array, int64_t columns) {
    NSArray<NSNumber *> * strides = array.strides;
    if (strides.count == 0 || strides[strides.count - 1].longLongValue != 1) return false;
    return strides.count == 1 || strides[strides.count - kTrailingDims].longLongValue == columns;
}

// A framework failure while binding (an array or provider Core ML could not
// allocate), as opposed to a schema the wrapper cannot bind. Set on the
// loading thread; read once the bind returns.
thread_local parakeet_coreml_load_error t_bind_failure{};
thread_local bool t_bind_failed = false;

void record_bind_failure(NSError * error) {
    t_bind_failed = true;
    parakeet_coreml_record_load_error(error, &t_bind_failure);
}

MLMultiArray * float_array(MLMultiArrayConstraint * constraint) {
    if (constraint == nil || constraint.dataType != MLMultiArrayDataTypeFloat32) return nil;
    NSError * error = nil;
    MLMultiArray * array = [[MLMultiArray alloc] initWithShape:constraint.shape
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&error];
    if (array == nil || error != nil) {
        record_bind_failure(error);
        return nil;
    }
    return array;
}

bool bind_array(MLMultiArrayConstraint * constraint, NSString * name, Binding & binding,
                MLMultiArray ** array) {
    *array = float_array(constraint);
    if (*array == nil) return false;
    if (!trailing_rows_and_columns(constraint.shape, &binding.rows, &binding.columns)) return false;
    if (!is_contiguous(*array, binding.columns)) return false;
    binding.name = name.UTF8String;
    binding.array = CFBridgingRetain(*array);
    return true;
}

void release_bindings(std::vector<Binding> & bindings) {
    for (Binding & binding : bindings) {
        if (binding.array != nullptr) CFRelease(binding.array);
        binding.array = nullptr;
    }
    bindings.clear();
}

const parakeet_coreml_rows * find_input(const parakeet_coreml_rows * inputs, int count,
                                        const std::string & name) {
    for (int index = 0; index < count; ++index) {
        if (inputs[index].name != nullptr && name == inputs[index].name) return &inputs[index];
    }
    return nullptr;
}

const Binding * find_binding(const std::vector<Binding> & bindings, const char * name) {
    for (const Binding & binding : bindings) {
        if (name != nullptr && binding.name == name) return &binding;
    }
    return nullptr;
}

bool slice_fits(int64_t row_offset, int64_t rows, int64_t columns, const Binding & binding) {
    return row_offset >= 0 && rows >= 0 && row_offset + rows <= binding.rows &&
           columns == binding.columns;
}

bool fill_input(const Binding & binding, const parakeet_coreml_rows * slice) {
    if (slice == nullptr) return false;
    float * destination = (float *) bridged(binding.array).dataPointer;
    const int64_t total = binding.rows * binding.columns;
    if (slice->rows == 0) {
        std::fill(destination, destination + total, 0.0f);
        return true;
    }
    if (slice->data == nullptr ||
        !slice_fits(slice->row_offset, slice->rows, slice->columns, binding)) return false;
    const int64_t begin = slice->row_offset * binding.columns;
    const int64_t end = begin + slice->rows * binding.columns;
    std::fill(destination, destination + begin, 0.0f);
    std::memcpy(destination + begin, slice->data, (size_t) (end - begin) * sizeof(float));
    std::fill(destination + end, destination + total, 0.0f);
    return true;
}

void fill_mask(const Binding & mask, int64_t valid_frames) {
    float * destination = (float *) bridged(mask.array).dataPointer;
    std::fill(destination, destination + mask.rows, 0.0f);
    std::fill(destination, destination + valid_frames, 1.0f);
}

void copy_row(const float * line, int64_t column_stride, int64_t columns, float * destination) {
    for (int64_t column = 0; column < columns; ++column) {
        destination[column] = line[column * column_stride];
    }
}

bool copy_rows(MLMultiArray * array, const parakeet_coreml_rows_out & output) {
    if (array == nil || array.dataType != MLMultiArrayDataTypeFloat32) return false;
    NSArray<NSNumber *> * strides = array.strides;
    const int64_t row_stride = strides.count >= kTrailingDims
        ? strides[strides.count - kTrailingDims].longLongValue : 1;
    const int64_t column_stride = strides[strides.count - 1].longLongValue;
    const float * source = (const float *) array.dataPointer;
    for (int64_t row = 0; row < output.rows; ++row) {
        copy_row(source + (output.row_offset + row) * row_stride, column_stride,
                 output.columns, output.data + row * output.columns);
    }
    return true;
}

}  // namespace

struct parakeet_coreml_diarizer {
    const void * model = nullptr;
    const void * provider = nullptr;
    const void * options = nullptr;
    std::vector<Binding> inputs;
    std::vector<Binding> outputs;
    Binding mask;
    std::string label;
    std::mutex mutex;
};

namespace {

bool bind_inputs(parakeet_coreml_diarizer * ctx, MLModel * model,
                 NSMutableDictionary<NSString *, MLMultiArray *> * features) {
    NSDictionary<NSString *, MLFeatureDescription *> * descriptions =
        model.modelDescription.inputDescriptionsByName;
    for (NSString * name in descriptions) {
        Binding binding;
        MLMultiArray * array = nil;
        if (!bind_array(descriptions[name].multiArrayConstraint, name, binding, &array)) return false;
        features[name] = array;
        if (binding.name == kValidMaskName) {
            ctx->mask = binding;
        } else {
            ctx->inputs.push_back(binding);
        }
    }
    return ctx->mask.array != nullptr;
}

bool inputs_share_capacity(const parakeet_coreml_diarizer * ctx) {
    for (const Binding & binding : ctx->inputs) {
        if (binding.rows != ctx->mask.rows) return false;
    }
    return true;
}

bool bind_outputs(parakeet_coreml_diarizer * ctx, MLModel * model,
                  NSMutableDictionary<NSString *, id> * backings) {
    NSDictionary<NSString *, MLFeatureDescription *> * descriptions =
        model.modelDescription.outputDescriptionsByName;
    for (NSString * name in descriptions) {
        Binding binding;
        MLMultiArray * array = nil;
        if (!bind_array(descriptions[name].multiArrayConstraint, name, binding, &array)) return false;
        backings[name] = array;
        ctx->outputs.push_back(binding);
    }
    return !ctx->outputs.empty();
}

bool bind_model(parakeet_coreml_diarizer * ctx, MLModel * model) {
    NSMutableDictionary<NSString *, MLMultiArray *> * features = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, id> * backings = [NSMutableDictionary dictionary];
    if (!bind_inputs(ctx, model, features) || !inputs_share_capacity(ctx)) return false;
    if (!bind_outputs(ctx, model, backings)) return false;
    NSError * error = nil;
    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    if (provider == nil || error != nil) {
        record_bind_failure(error);
        return false;
    }
    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    options.outputBackings = backings;
    ctx->provider = CFBridgingRetain(provider);
    ctx->options = CFBridgingRetain(options);
    return true;
}

bool inputs_bound(const parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows * inputs,
                  int count) {
    for (int index = 0; index < count; ++index) {
        if (find_binding(ctx->inputs, inputs[index].name) == nullptr) return false;
    }
    return true;
}

bool fill_inputs(parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows * inputs, int count) {
    for (const Binding & binding : ctx->inputs) {
        if (!fill_input(binding, find_input(inputs, count, binding.name))) return false;
    }
    return true;
}

bool outputs_fit(const parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows_out * outputs,
                 int count) {
    for (int index = 0; index < count; ++index) {
        const Binding * binding = find_binding(ctx->outputs, outputs[index].name);
        if (binding == nullptr || outputs[index].data == nullptr ||
            !slice_fits(outputs[index].row_offset, outputs[index].rows, outputs[index].columns,
                        *binding)) return false;
    }
    return true;
}

bool read_outputs(id<MLFeatureProvider> result, const parakeet_coreml_rows_out * outputs,
                  int count) {
    for (int index = 0; index < count; ++index) {
        NSString * name = [NSString stringWithUTF8String:outputs[index].name];
        if (!copy_rows([result featureValueForName:name].multiArrayValue, outputs[index])) return false;
    }
    return true;
}

}  // namespace

struct parakeet_coreml_diarizer * parakeet_coreml_diarizer_init(const char * path_mlmodelc,
                                                                const char * units_env) {
    return parakeet_coreml_diarizer_init_with_units(
        path_mlmodelc, units_env != nullptr ? std::getenv(units_env) : nullptr);
}

struct parakeet_coreml_diarizer * parakeet_coreml_diarizer_init_with_units(const char * path_mlmodelc,
                                                                           const char * units) {
    return parakeet_coreml_diarizer_init_with_units_error(path_mlmodelc, units, nullptr);
}

struct parakeet_coreml_diarizer * parakeet_coreml_diarizer_init_with_units_error(
        const char * path_mlmodelc, const char * units, parakeet_coreml_load_error * error) {
    if (error != nullptr) *error = parakeet_coreml_load_error{};
    if (path_mlmodelc == nullptr) return nullptr;
    @autoreleasepool {
        std::string label;
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = parakeet_coreml_units_for(
            units, /*prefer_neural_engine=*/true, &label);
        NSError * load_error = nil;
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path_mlmodelc]];
        MLModel * model = [MLModel modelWithContentsOfURL:url configuration:config error:&load_error];
        if (model == nil || load_error != nil) {
            parakeet_coreml_record_load_error(load_error, error);
            return nullptr;
        }
        auto * ctx = new parakeet_coreml_diarizer();
        ctx->model = CFBridgingRetain(model);
        ctx->label = label;
        t_bind_failed = false;
        if (!bind_model(ctx, model)) {
            parakeet_coreml_diarizer_free(ctx);
            if (error != nullptr && t_bind_failed) {
                *error = t_bind_failure;  // Core ML could not allocate: keep its error
            } else if (error != nullptr) {
                error->incompatible = 1;  // a schema the wrapper cannot bind
                std::snprintf(error->message, sizeof(error->message), "%s",
                              "unexpected diarizer inputs or outputs");
            }
            return nullptr;
        }
        return ctx;
    }
}

void parakeet_coreml_diarizer_free(struct parakeet_coreml_diarizer * ctx) {
    if (ctx == nullptr) return;
    release_bindings(ctx->inputs);
    release_bindings(ctx->outputs);
    if (ctx->mask.array != nullptr) CFRelease(ctx->mask.array);
    if (ctx->options != nullptr) CFRelease(ctx->options);
    if (ctx->provider != nullptr) CFRelease(ctx->provider);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t parakeet_coreml_diarizer_capacity(const struct parakeet_coreml_diarizer * ctx) {
    return ctx != nullptr ? ctx->mask.rows : 0;
}

int parakeet_coreml_diarizer_run(struct parakeet_coreml_diarizer * ctx,
                                 int64_t valid_frames,
                                 const struct parakeet_coreml_rows * inputs,
                                 int n_inputs,
                                 const struct parakeet_coreml_rows_out * outputs,
                                 int n_outputs) {
    if (ctx == nullptr || valid_frames <= 0 || valid_frames > ctx->mask.rows ||
        n_inputs < 0 || n_outputs <= 0 || outputs == nullptr) return kBadArguments;
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        if (!inputs_bound(ctx, inputs, n_inputs) || !fill_inputs(ctx, inputs, n_inputs) ||
            !outputs_fit(ctx, outputs, n_outputs)) {
            return kBadSlice;
        }
        fill_mask(ctx->mask, valid_frames);
        NSError * error = nil;
        MLModel * model = (__bridge MLModel *) ctx->model;
        id<MLFeatureProvider> result =
            [model predictionFromFeatures:(__bridge MLDictionaryFeatureProvider *) ctx->provider
                                  options:(__bridge MLPredictionOptions *) ctx->options
                                    error:&error];
        if (result == nil || error != nil) return kPredictionFailed;
        return read_outputs(result, outputs, n_outputs) ? kOk : kMissingOutput;
    }
}

const char * parakeet_coreml_diarizer_label(const struct parakeet_coreml_diarizer * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}

namespace parakeet {
namespace {

constexpr char kPathSeparator = '/';
constexpr const char * kCurrentDirectory = ".";

std::string directory_of(const std::string & path) {
    const std::size_t slash = path.find_last_of(kPathSeparator);
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

bool is_directory(const std::string & path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode);
}

std::vector<int> listed_capacities(DIR * listing, const std::string & directory,
                                   const std::string & gguf_path) {
    std::vector<int> capacities;
    while (const dirent * entry = ::readdir(listing)) {
        const std::string candidate = directory + entry->d_name;
        const int capacity = coreml_diarizer_sidecar_capacity(gguf_path, candidate);
        if (capacity > 0 && is_directory(candidate)) capacities.push_back(capacity);
    }
    return capacities;
}

std::vector<int> scan_capacities(const std::string & gguf_path) {
    const std::string directory = directory_of(gguf_path);
    DIR * listing = ::opendir(directory.empty() ? kCurrentDirectory : directory.c_str());
    if (listing == nullptr) return {};
    std::vector<int> capacities = listed_capacities(listing, directory, gguf_path);
    ::closedir(listing);
    std::sort(capacities.begin(), capacities.end());
    capacities.erase(std::unique(capacities.begin(), capacities.end()), capacities.end());
    return capacities;
}

constexpr const char * kWeightsFile = "/weights/weight.bin";

uint64_t file_bytes(const std::string & path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0 ? static_cast<uint64_t>(status.st_size) : 0;
}

CoremlDiarizerBundle bundle_of(const std::string & gguf_path, int capacity) {
    return {capacity, file_bytes(coreml_diarizer_sidecar_path(gguf_path, capacity) + kWeightsFile)};
}

}

bool coreml_diarizers_disabled() {
    return std::getenv(kCoremlDisableEnv) != nullptr ||
           std::getenv(kCoremlDiarizerDisableEnv) != nullptr;
}

std::vector<CoremlDiarizerBundle> list_coreml_diarizer_bundles(const std::string & gguf_path) {
    const std::vector<int> capacities = scan_capacities(gguf_path);
    std::vector<CoremlDiarizerBundle> bundles(capacities.size());
    std::transform(capacities.begin(), capacities.end(), bundles.begin(),
                   [&](int capacity) { return bundle_of(gguf_path, capacity); });
    return bundles;
}

namespace {

// Runs on the runtime's load lane with the resolved units: never reads
// the environment. The capacity the bundle reports must match its file name.
// A generated bundle (request.tmpl set) loads under its cache entry's lease.
pcm::LoadResult load_diarizer(const pcm::LoadRequest & request) {
    pcm::LoadResult result;
    if (!pcm::generation::begin_load(request, result)) return result;
    // sidecar_geometry(diarizer, {capacity})
    const int64_t capacity = request.geometry.size() == 2 ? request.geometry[1] : 0;
    parakeet_coreml_load_error error{};
    parakeet_coreml_diarizer * context = parakeet_coreml_diarizer_init_with_units_error(
        request.artifact_path.c_str(), pcm::to_string(request.units), &error);
    if (context != nullptr && parakeet_coreml_diarizer_capacity(context) != capacity) {
        // A verified incompatibility: the bundle is not the capacity its name says.
        error = parakeet_coreml_load_error{};
        error.incompatible = 1;
        std::snprintf(error.message, sizeof(error.message), "capacity %lld, expected %lld",
                      (long long) parakeet_coreml_diarizer_capacity(context), (long long) capacity);
        parakeet_coreml_diarizer_free(context);
        context = nullptr;
    }
    if (context == nullptr) {
        result.error = sidecar_load_error("Core ML diarizer", error);
        PARAKEET_LOG_WARN("parakeet: failed to load Core ML diarizer at '%s' (%s); using ggml\n",
                          request.artifact_path.c_str(), result.error.message.c_str());
        // The framework refused a generated bundle (or it is not the expected
        // capacity): end_load quarantines it, unless the error is a resource
        // one, and the slot prepares it again. The error's class is unchanged.
        result.rejected = request.tmpl != nullptr;
        pcm::generation::end_load(request, result);
        return result;
    }
    PARAKEET_LOG_INFO("parakeet: Core ML diarizer loaded from '%s'\n", request.artifact_path.c_str());
    result.model = std::shared_ptr<void>(context, [](void * p) {
        parakeet_coreml_diarizer_free(static_cast<parakeet_coreml_diarizer *>(p));
    });
    pcm::generation::end_load(request, result);
    return result;
}

}  // namespace

struct CoremlDiarizers::State {
    State(const std::string & gguf_path, std::vector<DiarizerCapacity> capacities, CoremlSidecars & sidecars,
          const pcm::SourceStat & source)
        : slots(gguf_path, std::move(capacities), sidecars, &load_diarizer, source) {}
    DiarizerSlots slots;
    std::mutex mutex;  // label and call stats
    const std::atomic<bool> * cancel = nullptr;  // the owning engine's
    std::string label;
    CoremlDiarizerCallStats stats;
};

CoremlDiarizers::CoremlDiarizers(const std::string & gguf_path, std::vector<DiarizerCapacity> capacities,
                                 CoremlSidecars & sidecars, const pcm::SourceStat & source)
    : state_(std::make_unique<State>(gguf_path, std::move(capacities), sidecars, source)) {
    for (const std::shared_ptr<pcm::Slot> & slot : state_->slots.status_only()) {
        const pcm::SlotStatus status = slot->status();
        PARAKEET_LOG_INFO("parakeet: no Core ML diarizer for %s is generated (%s)\n", status.stage.c_str(),
                          pcm::state_label(status.state, status.reason).c_str());
    }
}

CoremlDiarizers::~CoremlDiarizers() = default;

void CoremlDiarizers::set_cancel(const std::atomic<bool> * cancel) { state_->cancel = cancel; }

bool CoremlDiarizers::covers(int frames) { return state_->slots.covers(frames); }

bool CoremlDiarizers::prepare(int frames) { return state_->slots.prepare(frames); }

int CoremlDiarizers::run(int valid_frames,
                         const parakeet_coreml_rows * inputs, int n_inputs,
                         const parakeet_coreml_rows_out * outputs, int n_outputs) {
    const bool strict = state_->slots.strict();
    std::string label;
    int served = 0;
    const DiarizerPass pass = state_->slots.run(valid_frames, state_->cancel, [&](pcm::Handle & model, int capacity) {
        auto * context = model->as<parakeet_coreml_diarizer>();
        int status = 0;
        {
            auto prediction = model->lock_for_prediction();
            status = parakeet_coreml_diarizer_run(context, valid_frames, inputs, n_inputs, outputs, n_outputs);
        }
        if (status != 0) {
            PARAKEET_LOG_WARN("parakeet: Core ML diarizer (%d frames) failed (rc=%d); "
                              "quarantining it%s\n", capacity, status, strict ? "" : " and using ggml");
        } else {
            label = parakeet_coreml_diarizer_label(context);
            served = capacity;
        }
        return status;
    });
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (pass.not_ready) ++state_->stats.not_ready;
    if (pass.status == 0) {
        state_->label = label;
        ++state_->stats.coreml;
        state_->stats.label = label;
        state_->stats.capacity = served;
    }
    return pass.status;
}

std::string CoremlDiarizers::label() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->label.empty() ? parakeet_coreml_diarizer_label(nullptr) : state_->label;
}

std::vector<int> CoremlDiarizers::capacities() const { return state_->slots.capacities(); }

void CoremlDiarizers::reset_call_stats() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->stats = CoremlDiarizerCallStats{};
}

CoremlDiarizerCallStats CoremlDiarizers::call_stats() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->stats;
}

const std::vector<CoremlDiarizerTemplate> & nemotron_diarizer_templates() {
    static const std::vector<CoremlDiarizerTemplate> templates = {
        {358, "nemotron-diar-f358"}, {745, "nemotron-diar-f745"}, {1125, "nemotron-diar-f1125"}};
    return templates;
}

const std::vector<CoremlDiarizerTemplate> & sortformer_diarizer_templates() {
    static const std::vector<CoremlDiarizerTemplate> templates = {{410, "sortformer-diar-f410"}};
    return templates;
}

const std::vector<CoremlDiarizerTemplate> & diarizer_templates_for(ParakeetModelType type,
                                                                   const std::string & variant) {
    static const std::vector<CoremlDiarizerTemplate> none;
    if (type == ParakeetModelType::NEMOTRON_DIARIZATION) return nemotron_diarizer_templates();
    if (type == ParakeetModelType::SORTFORMER && variant == "sortformer-streaming-v2.1-aosc") {
        return sortformer_diarizer_templates();
    }
    return none;
}

std::unique_ptr<CoremlDiarizers> discover_coreml_diarizers(const std::string & gguf_path,
                                                           CoremlSidecars & sidecars,
                                                           const std::vector<CoremlDiarizerTemplate> & templates,
                                                           const pcm::SourceStat & source) {
    // A library without the generator embeds no templates.
    std::vector<int> template_frames;
    std::vector<std::string> template_ids;
    if (pcm::generation::built()) {
        for (const CoremlDiarizerTemplate & t : templates) {
            template_frames.push_back(t.frames);
            template_ids.emplace_back(t.template_id);
        }
    }
    // The candidates are read only while the slots are registered.
    const std::vector<pcm::generation::Candidate> probed =
        probe_diarizer_templates(sidecars, gguf_path, template_ids);
    std::vector<DiarizerCapacity> capacities =
        diarizer_capacities(scan_capacities(gguf_path), template_frames, probed);
    if (capacities.empty()) return nullptr;
    return std::make_unique<CoremlDiarizers>(gguf_path, std::move(capacities), sidecars, source);
}

}
