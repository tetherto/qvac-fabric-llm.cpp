// Per-call time of one FFN slice with FP32 vs FP16 input/output, both with output backings, alone and two concurrently.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

static MLModel * load(NSString * path) {
    MLModelConfiguration * cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSError * e = nil;
    MLModel * m = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] configuration:cfg error:&e];
    if (!m) { fprintf(stderr, "load failed: %s\n", e.localizedDescription.UTF8String); exit(1); }
    return m;
}

static double median_ms(const std::function<void()> & f) {
    for (int i = 0; i < 5; ++i) f();
    std::vector<double> v;
    for (int i = 0; i < 40; ++i) {
        auto s = std::chrono::steady_clock::now(); f();
        v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct runner {
    MLModel * model; MLDictionaryFeatureProvider * features; MLPredictionOptions * options; MLMultiArray * out;
    runner(NSString * path, MLMultiArrayDataType type) : model(load(path)) {
        MLMultiArray * x = [[MLMultiArray alloc] initWithShape:@[@512, @5120] dataType:type error:nil];
        for (int i = 0; i < 512 * 5120; ++i) { x[i] = @(float(i % 97) / 97.0f - 0.5f); }
        features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:nil];
        out = [[MLMultiArray alloc] initWithShape:@[@512, @5120] dataType:type error:nil];
        options = [MLPredictionOptions new];
        options.outputBackings = @{@"y": out};
    }
    void run() {
        @autoreleasepool {
            NSError * e = nil;
            id<MLFeatureProvider> r = [model predictionFromFeatures:features options:options error:&e];
            if (!r) { fprintf(stderr, "predict failed\n"); exit(1); }
            if ([r featureValueForName:@"y"].multiArrayValue.dataPointer != out.dataPointer) { fprintf(stderr, "backing ignored\n"); }
        }
    }
};

int main(int argc, char ** argv) {
    @autoreleasepool {
        dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
        for (int k = 0; k < 2; ++k) {
            const bool fp16 = k == 1;
            NSString * path = [NSString stringWithUTF8String:argv[1 + k]];
            MLMultiArrayDataType type = fp16 ? MLMultiArrayDataTypeFloat16 : MLMultiArrayDataTypeFloat32;
            runner a(path, type), b(path, type);
            double one = median_ms([&] { a.run(); });
            double two = median_ms([&] {
                dispatch_group_t g = dispatch_group_create();
                dispatch_group_async(g, q, ^{ b.run(); });
                a.run();
                dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
            });
            printf("io=%s single_ms=%.3f concurrent_pair_ms=%.3f\n", fp16 ? "fp16" : "fp32", one, two);
        }
    }
    return 0;
}
