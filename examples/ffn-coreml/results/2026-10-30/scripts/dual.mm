#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

static MLModel * load(NSString * package) {
    NSError * e = nil;
    NSURL * c = [MLModel compileModelAtURL:[NSURL fileURLWithPath:package] error:&e];
    if (!c) { fprintf(stderr, "compile %s\n", e.localizedDescription.UTF8String); exit(1); }
    MLModelConfiguration * cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel * m = [MLModel modelWithContentsOfURL:c configuration:cfg error:&e];
    if (!m) { fprintf(stderr, "load %s\n", e.localizedDescription.UTF8String); exit(1); }
    return m;
}

struct runner {
    MLModel * model; MLDictionaryFeatureProvider * features; std::vector<float> y;
    runner(MLModel * m, MLMultiArray * x) : model(m), y(512 * 5120) {
        features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:nil];
    }
    void run() {
        @autoreleasepool {
            NSError * e = nil;
            id<MLFeatureProvider> r = [model predictionFromFeatures:features error:&e];
            if (!r) { fprintf(stderr, "predict %s\n", e.localizedDescription.UTF8String); exit(1); }
            MLMultiArray * out = [r featureValueForName:@"y"].multiArrayValue;
            const float * d = (const float *) out.dataPointer;
            size_t rs = out.strides[0].unsignedLongValue, cs = out.strides[1].unsignedLongValue;
            for (int t = 0; t < 512; ++t) for (int h = 0; h < 5120; ++h) y[t * 5120 + h] = d[t * rs + h * cs];
        }
    }
};

template <class F> static void time_it(const char * name, int reps, F f) {
    for (int i = 0; i < 3; ++i) f();
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) {
        auto s = std::chrono::steady_clock::now(); f();
        v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
    }
    std::sort(v.begin(), v.end());
    printf("%-28s median_ms=%.3f min_ms=%.3f\n", name, v[v.size() / 2], v[0]);
    fflush(stdout);
}

static void parallel(std::vector<runner *> rs) {
    dispatch_group_t g = dispatch_group_create();
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    for (auto * r : rs) dispatch_group_async(g, q, ^{ r->run(); });
    dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        NSString * full = [NSString stringWithUTF8String:argv[1]], * a = [NSString stringWithUTF8String:argv[2]], * b = [NSString stringWithUTF8String:argv[3]];
        int reps = 30;
        MLMultiArray * x = [[MLMultiArray alloc] initWithShape:@[@512, @5120] dataType:MLMultiArrayDataTypeFloat32 error:nil];
        std::mt19937 rng(11259); std::normal_distribution<float> n(0, 1);
        float * xd = (float *) x.dataPointer;
        for (int i = 0; i < 512 * 5120; ++i) xd[i] = n(rng);
        runner rf(load(full), x), rf2(load(full), x), ra(load(a), x), rb(load(b), x);
        time_it("full6144", reps, [&] { rf.run(); });
        time_it("half_a3072", reps, [&] { ra.run(); });
        time_it("half_b3072", reps, [&] { rb.run(); });
        time_it("halves_sequential", reps, [&] { ra.run(); rb.run(); });
        time_it("halves_parallel", reps, [&] { parallel({&ra, &rb}); });
        time_it("full_x2_parallel", reps, [&] { parallel({&rf, &rf2}); });
        double err = 0, ref = 0;
        for (size_t i = 0; i < rf.y.size(); ++i) { double d = rf.y[i] - (ra.y[i] + rb.y[i]); err += d * d; ref += double(rf.y[i]) * rf.y[i]; }
        printf("halves_vs_full relative_l2=%.6f\n", std::sqrt(err / ref));
    }
    return 0;
}
