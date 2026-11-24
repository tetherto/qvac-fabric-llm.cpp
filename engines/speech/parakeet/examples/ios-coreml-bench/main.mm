#import <UIKit/UIKit.h>
#import <CoreML/CoreML.h>
#import <mach/mach.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>
#include <sys/resource.h>
#include "parakeet/cli.h"

int transcribepsy_eu_coreml_test(int argc, char ** argv);

static void count_plan(MLComputePlan * plan, MLModelStructureProgramBlock * block,
                        NSMutableDictionary * counts) {
    for (MLModelStructureProgramOperation * operation in block.operations) {
        MLComputePlanDeviceUsage * usage = [plan computeDeviceUsageForMLProgramOperation:operation];
        NSString * kind = usage ? NSStringFromClass([(NSObject *) usage.preferredComputeDevice class]) : @"None";
        counts[kind] = @([counts[kind] integerValue] + 1);
        for (MLModelStructureProgramBlock * nested in operation.blocks) count_plan(plan, nested, counts);
    }
}

static int print_compute_plan(NSString * path, NSString * name) {
    const bool flexible = [name containsString:@"flexible"];
    const bool precise_gpu = [name containsString:@"fp32"];
    MLModelConfiguration * configuration = [MLModelConfiguration new];
    configuration.computeUnits = flexible ? MLComputeUnitsCPUOnly : precise_gpu ? MLComputeUnitsCPUAndGPU : MLComputeUnitsCPUAndNeuralEngine;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block int result = 0;
    [MLComputePlan loadContentsOfURL:[NSURL fileURLWithPath:path] configuration:configuration
        completionHandler:^(MLComputePlan * plan, NSError * error) {
            if (!plan || error) {
                NSLog(@"PLAN_ERROR %@ %@", path, error);
                result = 1;
            } else {
                NSMutableDictionary * counts = [NSMutableDictionary new];
                count_plan(plan, plan.modelStructure.program.functions[@"main"].block, counts);
                NSDictionary * record = @{@"model":name, @"compute_units":flexible ? @"cpu_only" : precise_gpu ? @"cpu_and_gpu" : @"cpu_and_ane", @"counts":counts};
                NSData * json = [NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];
                std::fprintf(stderr, "PHONE_PLAN %.*s\n", (int) json.length, (const char *) json.bytes);
            }
            dispatch_semaphore_signal(done);
        }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    return result;
}

static int print_compute_plans(const char * directory) {
    NSArray * names = @[@"transcribepsy-250m-eu-encoder", @"transcribepsy-250m-eu-encoder-other",
        @"transcribepsy-250m-eu-encoder-gate", @"transcribepsy-250m-eu-decoder",
        @"transcribepsy-250m-eu-encoder-flexible", @"transcribepsy-250m-eu-encoder-other-flexible",
        @"transcribepsy-250m-eu-encoder-gate-flexible",
        @"transcribepsy-250m-eu-encoder-fp32-1500", @"transcribepsy-250m-eu-encoder-other-fp32-1500",
        @"transcribepsy-250m-eu-encoder-fp32-6008", @"transcribepsy-250m-eu-encoder-other-fp32-6008"];
    for (NSString * name in names) {
        NSString * path = [[NSString stringWithUTF8String:directory]
            stringByAppendingPathComponent:[name stringByAppendingString:@".mlmodelc"]];
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) continue;
        if (int result = print_compute_plan(path, name)) return result;
    }
    return 0;
}

static std::vector<std::string> prepare_arguments() {
    std::vector<std::string> arguments;
    for (NSString * argument in NSProcessInfo.processInfo.arguments) arguments.emplace_back(argument.UTF8String);
    NSString * documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    for (size_t index = 1; index + 1 < arguments.size(); ++index) {
        if (arguments[index] == "--bench-json") {
            NSString * filename = [NSString stringWithUTF8String:arguments[index + 1].c_str()].lastPathComponent;
            arguments[index + 1] = std::string(documents.UTF8String) + "/" + filename.UTF8String;
        }
    }
    return arguments;
}

static int run_command(std::vector<std::string> arguments) {
    if (arguments.size() > 1 && arguments[1] == "--status") return 0;
    if (arguments.size() > 1 && arguments[1] == "--compute-plan")
        return print_compute_plans(arguments.size() > 2 ? arguments[2].c_str() : "models");
    const bool api = arguments.size() > 1 && arguments[1] == "--api-check";
    if (api) arguments.erase(arguments.begin() + 1);
    std::vector<char *> argv;
    for (auto & argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    return api ? transcribepsy_eu_coreml_test((int) arguments.size(), argv.data())
               : parakeet_cli_main((int) arguments.size(), argv.data());
}

static void report_metrics(int result, double elapsed_ms) {
    rusage usage = {};
    getrusage(RUSAGE_SELF, &usage);
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count);
    const int64_t peak_footprint = count >= TASK_VM_INFO_REV3_COUNT ? info.ledger_phys_footprint_peak : 0;
    std::fprintf(stderr, "PHONE_METRICS rc=%d elapsed_ms=%.3f peak_rss_bytes=%ld footprint_bytes=%llu peak_footprint_bytes=%lld thermal=%ld\n",
        result, elapsed_ms, usage.ru_maxrss, info.phys_footprint, (long long) peak_footprint,
        (long) NSProcessInfo.processInfo.thermalState);
}

static void run_benchmark() {
    @autoreleasepool {
        chdir(NSBundle.mainBundle.bundlePath.UTF8String);
        const auto started = std::chrono::steady_clock::now();
        int result = 1;
        try { result = run_command(prepare_arguments()); }
        catch (const std::exception & error) { std::fprintf(stderr, "PHONE_ERROR %s\n", error.what()); }
        report_metrics(result, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
        std::fflush(stdout);
        std::fflush(stderr);
        std::exit(result);
    }
}

@interface PhoneDelegate : UIResponder <UIApplicationDelegate>
@property(strong, nonatomic) UIWindow * window;
@end
@implementation PhoneDelegate
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)options {
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    self.window.rootViewController = [UIViewController new];
    self.window.backgroundColor = UIColor.whiteColor;
    [self.window makeKeyAndVisible];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ run_benchmark(); });
    return YES;
}
@end

int main(int argc, char ** argv) {
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(PhoneDelegate.class)); }
}
