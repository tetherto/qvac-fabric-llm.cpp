// Per-call ANE latency when cycling through N loaded models: N separate packages vs N functions of one package.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

static NSURL * compiled(NSString * package) {
    NSString * cache = [[package stringByDeletingPathExtension] stringByAppendingString:@".mlmodelc"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:cache]) {
        NSError * e = nil;
        NSURL * tmp = [MLModel compileModelAtURL:[NSURL fileURLWithPath:package] error:&e];
        if (!tmp) { fprintf(stderr, "compile failed %s\n", e.localizedDescription.UTF8String); exit(1); }
        [[NSFileManager defaultManager] moveItemAtURL:tmp toURL:[NSURL fileURLWithPath:cache] error:&e];
    }
    return [NSURL fileURLWithPath:cache];
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        NSString * root = [NSString stringWithUTF8String:argv[1]];
        const std::string mode = argv[2];
        const int n = atoi(argv[3]);
        NSURL * multi = mode == "multi" ? compiled([root stringByAppendingPathComponent:@"multi.mlpackage"]) : nil;
        std::vector<MLModel *> models;
        for (int i = 0; i < n; ++i) {
            MLModelConfiguration * cfg = [MLModelConfiguration new];
            cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            NSURL * url = multi;
            if (multi) { cfg.functionName = [NSString stringWithFormat:@"f%d", i]; }
            else { url = compiled([root stringByAppendingPathComponent:[NSString stringWithFormat:@"singles/m%d.mlpackage", i]]); }
            NSError * e = nil;
            MLModel * m = [MLModel modelWithContentsOfURL:url configuration:cfg error:&e];
            if (!m) { fprintf(stderr, "load %d failed: %s\n", i, e.localizedDescription.UTF8String); return 1; }
            models.push_back(m);
        }
        MLMultiArray * x = [[MLMultiArray alloc] initWithShape:@[@512, @5120] dataType:MLMultiArrayDataTypeFloat32 error:nil];
        float * xd = (float *) x.dataPointer;
        for (int i = 0; i < 512 * 5120; ++i) xd[i] = float(i % 97) / 97.0f - 0.5f;
        MLDictionaryFeatureProvider * features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:nil];
        std::vector<double> times;
        for (int round = 0; round < 6; ++round) {
            for (auto * m : models) {
                @autoreleasepool {
                    auto s = std::chrono::steady_clock::now();
                    NSError * e = nil;
                    if (![m predictionFromFeatures:features error:&e]) { fprintf(stderr, "predict failed\n"); return 1; }
                    if (round >= 2) { times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count()); }
                }
            }
        }
        std::sort(times.begin(), times.end());
        printf("mode=%s n=%d per_call_median_ms=%.3f p90_ms=%.3f\n", mode.c_str(), n, times[times.size() / 2], times[times.size() * 9 / 10]);
    }
    return 0;
}
