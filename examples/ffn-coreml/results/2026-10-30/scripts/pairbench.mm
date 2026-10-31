// Concurrent pair of different FFN slice models on the two ANEs: one call of T tokens, with output backings.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

struct slice {
    MLModel * model; MLDictionaryFeatureProvider * features; MLPredictionOptions * options; MLMultiArray * out;
    slice(const char * path, int tokens) {
        MLModelConfiguration * cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError * e = nil;
        model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] configuration:cfg error:&e];
        if (!model) { fprintf(stderr, "load failed %s\n", path); exit(1); }
        MLMultiArray * x = [[MLMultiArray alloc] initWithShape:@[@(tokens), @5120] dataType:MLMultiArrayDataTypeFloat32 error:nil];
        float * xd = (float *) x.dataPointer;
        for (long i = 0; i < (long) tokens * 5120; ++i) xd[i] = float(i % 97) / 97.0f - 0.5f;
        features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:nil];
        out = [[MLMultiArray alloc] initWithShape:@[@(tokens), @5120] dataType:MLMultiArrayDataTypeFloat32 error:nil];
        options = [MLPredictionOptions new];
        options.outputBackings = @{@"y": out};
    }
    void run() {
        @autoreleasepool {
            NSError * e = nil;
            id<MLFeatureProvider> r = [model predictionFromFeatures:features options:options error:&e];
            if (!r) { fprintf(stderr, "predict failed: %s\n", e.localizedDescription.UTF8String); exit(1); }
        }
    }
};

int main(int argc, char ** argv) {
    @autoreleasepool {
        const int tokens = atoi(argv[3]);
        slice a(argv[1], tokens), b(argv[2], tokens);
        dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
        auto pair = [&] {
            dispatch_group_t g = dispatch_group_create();
            dispatch_group_async(g, q, ^{ b.run(); });
            a.run();
            dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
        };
        for (int i = 0; i < 5; ++i) pair();
        std::vector<double> v;
        for (int i = 0; i < 30; ++i) {
            auto s = std::chrono::steady_clock::now(); pair();
            v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
        }
        std::sort(v.begin(), v.end());
        printf("tokens=%d concurrent_pair_ms=%.3f per_512_tokens_ms=%.3f\n", tokens, v[v.size() / 2], v[v.size() / 2] * 512.0 / tokens);
    }
    return 0;
}
