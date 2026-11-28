#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <TargetConditionals.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif
#import <mach/mach.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kRepeats = 5;
constexpr double kMilliseconds = 1000.0;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count() * kMilliseconds;
}

void require(bool condition, NSString * message) {
    if (!condition) throw std::runtime_error(message.UTF8String);
}

NSDictionary * memory_metrics() {
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count);
    return @{@"footprint_bytes": @(info.phys_footprint),
             @"peak_footprint_bytes": @(info.ledger_phys_footprint_peak),
             @"thermal": @(NSProcessInfo.processInfo.thermalState)};
}

void save_json(id value, NSString * path) {
    NSError * error = nil;
    NSData * data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];
    require(data && [data writeToFile:path options:NSDataWritingAtomic error:&error], error.description ?: @"JSON write failed");
}

NSDictionary * read_manifest(NSString * path) {
    NSData * data = [NSData dataWithContentsOfFile:path];
    require(data != nil, @"Missing fixture manifest");
    NSError * error = nil;
    id manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require(manifest != nil && error == nil, @"Invalid fixture manifest JSON");
    require([manifest isKindOfClass:NSDictionary.class], @"Fixture manifest root must be an object");
    id fixtures = manifest[@"fixtures"];
    require([fixtures isKindOfClass:NSArray.class] && [fixtures count] > 0,
            @"Fixture manifest must contain a nonempty fixtures array");
    return manifest;
}

MLModelConfiguration * configuration(NSString * function) {
    MLModelConfiguration * config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    if (function) config.functionName = function;
    return config;
}

MLModel * load_model(NSString * path, NSString * function, NSMutableDictionary * record) {
    NSError * error = nil;
    const auto started = Clock::now();
    MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                    configuration:configuration(function) error:&error];
    require(model != nil, error.description ?: @"Model load failed");
    record[@"load_ms"] = @(milliseconds(started));
    return model;
}

void count_plan(MLComputePlan * plan, MLModelStructureProgramBlock * block, NSMutableDictionary * counts) {
    for (MLModelStructureProgramOperation * operation in block.operations) {
        MLComputePlanDeviceUsage * usage = [plan computeDeviceUsageForMLProgramOperation:operation];
        NSString * device = usage ? NSStringFromClass([(NSObject *)usage.preferredComputeDevice class]) : @"None";
        counts[device] = @([counts[device] integerValue] + 1);
        for (MLModelStructureProgramBlock * nested in operation.blocks) count_plan(plan, nested, counts);
    }
}

NSDictionary * compute_plan(NSString * path, NSString * function) {
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    NSMutableDictionary * result = [NSMutableDictionary new];
    [MLComputePlan loadContentsOfURL:[NSURL fileURLWithPath:path] configuration:configuration(function)
                 completionHandler:^(MLComputePlan * plan, NSError * error) {
        if (error) result[@"error"] = error.description;
        else count_plan(plan, plan.modelStructure.program.functions[function ?: @"main"].block, result);
        dispatch_semaphore_signal(finished);
    }];
    dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
    return result;
}

id<MLFeatureProvider> input_features(NSDictionary * fixture, NSString * directory, int scale) {
    NSMutableDictionary * inputs = [NSMutableDictionary new];
    NSDictionary * shapes = fixture[@"shapes"];
    for (NSString * name in shapes) {
        NSError * error = nil;
        MLMultiArray * array = [[MLMultiArray alloc] initWithShape:shapes[name] dataType:MLMultiArrayDataTypeFloat32 error:&error];
        NSData * bytes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];
        require(array && bytes.length == array.count * sizeof(float), @"Invalid input fixture");
        std::memcpy(array.dataPointer, bytes.bytes, bytes.length);
        inputs[name] = [MLFeatureValue featureValueWithMultiArray:array];
    }
    if (scale >= 0) {
        MLMultiArray * array = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @1] dataType:MLMultiArrayDataTypeFloat32 error:nil];
        ((float *)array.dataPointer)[0] = scale;
        inputs[@"adapter_scale"] = [MLFeatureValue featureValueWithMultiArray:array];
    }
    return [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:nil];
}

float array_value(MLMultiArray * array, NSInteger index) {
    if (array.dataType == MLMultiArrayDataTypeFloat16) return ((const __fp16 *)array.dataPointer)[index];
    if (array.dataType == MLMultiArrayDataTypeFloat32) return ((const float *)array.dataPointer)[index];
    throw std::runtime_error("Unexpected output precision");
}

double compare_output(MLMultiArray * output, NSDictionary * fixture, NSString * directory,
                      NSString * route, NSString * output_path) {
    const NSInteger frames = [fixture[@"valid_encoder_frames"] integerValue];
    require(output.shape.count == 4 && [output.shape[0] integerValue] == 1 &&
            [output.shape[2] integerValue] == 1 && [output.shape[3] integerValue] >= frames,
            @"Unexpected output geometry");
    const NSInteger channels = [output.shape[1] integerValue];
    NSData * reference = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:[route stringByAppendingString:@".bin"]]];
    require(reference.length == frames * channels * sizeof(float), @"Reference shape mismatch");
    const float * want = (const float *)reference.bytes;
    std::vector<float> values(frames * channels);
    double numerator = 0.0, denominator = 0.0;
    for (NSInteger frame = 0; frame < frames; ++frame) {
        for (NSInteger channel = 0; channel < channels; ++channel) {
            const NSInteger index = frame * channels + channel;
            const float value = array_value(output, channel * [output.strides[1] integerValue] + frame * [output.strides[3] integerValue]);
            require(std::isfinite(value), @"Non-finite compiled output");
            values[index] = value;
            numerator += (value - want[index]) * (value - want[index]);
            denominator += want[index] * want[index];
        }
    }
    require([[NSData dataWithBytes:values.data() length:values.size() * sizeof(float)] writeToFile:output_path atomically:YES], @"Output write failed");
    return std::sqrt(numerator / std::max(denominator, 1e-30));
}

NSDictionary * predict_fixture(MLModel * model, NSDictionary * fixture, NSString * directory,
                               NSString * route, bool unified, NSString * output_path) {
    const int scale = unified ? ([route isEqual:@"other"] ? 1 : 0) : -1;
    id<MLFeatureProvider> inputs = input_features(fixture, directory, scale);
    NSError * error = nil;
    auto started = Clock::now();
    id<MLFeatureProvider> result = [model predictionFromFeatures:inputs error:&error];
    require(result != nil, error.description ?: @"Prediction failed");
    const double first_ms = milliseconds(started);
    std::vector<double> times;
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        started = Clock::now();
        require([model predictionFromFeatures:inputs error:&error] != nil, error.description ?: @"Prediction failed");
        times.push_back(milliseconds(started));
    }
    std::sort(times.begin(), times.end());
    NSString * key = unified && [route isEqual:@"gate"] ? @"gate_features" : @"encoder_out";
    MLMultiArray * output = [result featureValueForName:key].multiArrayValue;
    require(output != nil, @"Missing output feature");
    return @{@"fixture": fixture[@"name"], @"first_prediction_ms": @(first_ms),
             @"warm_median_ms": @(times[kRepeats / 2]),
             @"relative_l2_to_fp32": @(compare_output(output, fixture, directory, route, output_path))};
}

NSArray * routes_for_kind(NSString * kind) {
    if ([kind isEqual:@"load-en"]) return @[@"en"];
    if ([kind isEqual:@"load-other"]) return @[@"other"];
    if ([kind isEqual:@"load-auto-en"]) return @[@"gate", @"en"];
    if ([kind isEqual:@"load-auto-other"]) return @[@"gate", @"other"];
    NSArray * kinds = @[@"separate", @"shared", @"unified", @"load-eager"];
    require([kinds containsObject:kind], @"Unknown probe kind");
    return @[@"en", @"other", @"gate"];
}

NSDictionary * probe_route(NSString * models, NSString * kind, NSString * route,
                           NSDictionary * manifest, NSString * fixture_root,
                           NSString * output, NSMutableArray * retained) {
    const bool loading = [kind hasPrefix:@"load-"];
    NSString * name = loading || [kind isEqual:@"separate"] ? route : kind;
    NSString * path = [models stringByAppendingPathComponent:[name stringByAppendingString:@".mlmodelc"]];
    NSString * function = [kind isEqual:@"shared"] ? route : nil;
    NSMutableDictionary * record = [@{@"kind": kind, @"route": route} mutableCopy];
    MLModel * model = load_model(path, function, record);
    if (loading) [retained addObject:model];
    else {
        NSMutableArray * predictions = [NSMutableArray new];
        for (NSDictionary * fixture in manifest[@"fixtures"]) {
            NSString * filename = [NSString stringWithFormat:@"%@-%@.bin", route, fixture[@"name"]];
            [predictions addObject:predict_fixture(model, fixture,
                [fixture_root stringByAppendingPathComponent:fixture[@"name"]], route,
                [kind isEqual:@"unified"], [output stringByAppendingPathComponent:filename])];
        }
        record[@"predictions"] = predictions;
        record[@"placement"] = compute_plan(path, function);
    }
    record[@"memory"] = memory_metrics();
    NSLog(@"ROUTE_PROBE %@ %@ %@", kind, route, record[@"load_ms"]);
    return record;
}

void run_probe() {
    @autoreleasepool {
        NSArray<NSString *> * arguments = NSProcessInfo.processInfo.arguments;
        require(arguments.count == 4 || arguments.count == 5,
                @"Usage: route-probe MODEL_DIRECTORY KIND OUTPUT_NAME [FIXTURE_DIRECTORY]");
        NSString * models = [arguments[1] isAbsolutePath] ? arguments[1] :
            [NSBundle.mainBundle.bundlePath stringByAppendingPathComponent:arguments[1]];
        NSString * kind = arguments[2];
        require(arguments[3].length > 0 && ![arguments[3] isEqual:@"."] && ![arguments[3] isEqual:@".."] &&
                [arguments[3].lastPathComponent isEqual:arguments[3]], @"Output name must be simple");
        NSString * documents = NSProcessInfo.processInfo.environment[@"QVAC_ROUTE_OUTPUT_DIR"] ?:
            NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
        NSString * output = [documents stringByAppendingPathComponent:arguments[3]];
        [[NSFileManager defaultManager] createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil];
        NSMutableArray * records = [NSMutableArray new];
        NSMutableArray * retained = [NSMutableArray new];
        const bool loading = [kind hasPrefix:@"load-"];
        NSString * fixture_root = arguments.count == 5 ? arguments[4] :
            [NSBundle.mainBundle.bundlePath stringByAppendingPathComponent:@"fixtures"];
        NSDictionary * manifest = loading ? nil : read_manifest([fixture_root stringByAppendingPathComponent:@"manifest.json"]);
        for (NSString * route in routes_for_kind(kind)) {
            @autoreleasepool {
                NSDictionary * record = probe_route(models, kind, route, manifest, fixture_root, output, retained);
                [records addObject:record];
                save_json(@{@"model_directory": arguments[1], @"records": records, @"memory": memory_metrics()},
                          [output stringByAppendingPathComponent:@"results.json"]);
            }
        }
    }
}
} // namespace

#if TARGET_OS_IPHONE
@interface RouteProbeDelegate : UIResponder <UIApplicationDelegate>
@property(strong, nonatomic) UIWindow * window;
@end

@implementation RouteProbeDelegate
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)options {
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    self.window.rootViewController = [UIViewController new];
    [self.window makeKeyAndVisible];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        try { run_probe(); std::exit(0); }
        catch (const std::exception & error) { std::fprintf(stderr, "ROUTE_PROBE_ERROR %s\n", error.what()); std::exit(1); }
    });
    return YES;
}
@end

int main(int argc, char ** argv) {
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(RouteProbeDelegate.class)); }
}
#else
int main() {
    try { run_probe(); return 0; }
    catch (const std::exception & error) { std::fprintf(stderr, "ROUTE_PROBE_ERROR %s\n", error.what()); return 1; }
}
#endif
