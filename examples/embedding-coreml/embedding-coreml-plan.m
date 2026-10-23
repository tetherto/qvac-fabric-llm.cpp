#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <stdio.h>

static void count_block(MLComputePlan * plan, MLModelStructureProgramBlock * block, int * total, int * ane, int * unknown) {
    for (MLModelStructureProgramOperation * operation in block.operations) {
        if ([operation.operatorName isEqualToString:@"const"] || [operation.operatorName hasPrefix:@"constexpr_"]) { continue; }
        ++*total;
        MLComputePlanDeviceUsage * usage = [plan computeDeviceUsageForMLProgramOperation:operation];
        if (!usage) { ++*unknown; }
        if ([usage.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]]) { ++*ane; }
        for (MLModelStructureProgramBlock * nested in operation.blocks) {
            count_block(plan, nested, total, ane, unknown);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s EmbeddingGemma2Text.mlmodelc\n", argv[0]);
        return 1;
    }
    if (@available(macOS 15.0, iOS 18.0, *)) {} else { return 1; }
    @autoreleasepool {
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        for (NSString * name in @[@"embed_32", @"embed_48", @"embed_64", @"embed_128", @"embed_256", @"embed_512", @"pack_256"]) {
            MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            config.functionName = name;
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            __block BOOL valid = NO;
            [MLComputePlan loadContentsOfURL:url configuration:config completionHandler:^(MLComputePlan * plan, NSError * error) {
                MLModelStructureProgramFunction * function = plan.modelStructure.program.functions[name];
                if (!function) {
                    fprintf(stderr, "%s: %s\n", name.UTF8String, error ? error.localizedDescription.UTF8String : "function missing from compute plan");
                } else {
                    int total = 0, ane = 0, unknown = 0;
                    count_block(plan, function.block, &total, &ane, &unknown);
                    printf("%s operations=%d preferred_ane=%d unknown=%d\n", name.UTF8String, total, ane, unknown);
                    fflush(stdout);
                    valid = total > 0 && unknown == 0;
                }
                dispatch_semaphore_signal(done);
            }];
            dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
            if (!valid) { return 1; }
        }
    }
    return 0;
}
