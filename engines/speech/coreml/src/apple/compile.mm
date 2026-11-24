// On-device compile of a generated .mlpackage (step 6 of the prepare
// pipeline). C and C++ APIs plus Objective-C messages only: no @interface,
// @implementation or categories in coreml, because the Objective-C
// runtime has one flat class namespace per process and every engine library
// carries its own copy of this file.

#include "../compile.h"
#include "../device.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

namespace QVAC_COREML_NS {
namespace apple {

namespace {

std::string ns_string(NSString * s) { return s ? std::string(s.UTF8String ?: "") : std::string(); }

Error ns_error(NSError * e, const std::string & what) {
    if (!e) return Error::transient(what + ": unknown error");
    return Error::transient(what + ": " + ns_string(e.localizedDescription), ns_string(e.domain),
                            static_cast<int64_t>(e.code));
}

struct Footprint {
    uint64_t current = 0;
    uint64_t peak = 0;
};

Footprint footprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    Footprint f;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        f.current = info.phys_footprint;
        if (count >= TASK_VM_INFO_REV3_COUNT && info.ledger_phys_footprint_peak > 0) {
            f.peak = static_cast<uint64_t>(info.ledger_phys_footprint_peak);
        }
    }
    return f;
}

// Samples the physical footprint every millisecond while it lives. The
// process peak ledger catches spikes shorter than a sample when it rose past
// its previous value; the samples catch growth below an earlier, higher peak.
class PeakSampler {
public:
    PeakSampler() : base_(footprint()), max_(base_.current) {
        th_ = std::thread([this] {
            std::unique_lock<std::mutex> lk(mu_);
            while (!stop_) {
                max_ = std::max(max_, footprint().current);
                cv_.wait_for(lk, std::chrono::milliseconds(1));
            }
        });
    }
    ~PeakSampler() { finish(); }

    uint64_t finish() {
        if (th_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                stop_ = true;
            }
            cv_.notify_all();
            th_.join();
        }
        const Footprint end = footprint();
        uint64_t top = std::max(max_, end.current);
        if (end.peak > base_.peak) top = std::max(top, end.peak);
        return top > base_.current ? top - base_.current : 0;
    }

private:
    Footprint base_;
    uint64_t max_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread th_;
};

std::string read_head(const std::string & path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    std::string s(n, '\0');
    in.read(s.data(), static_cast<std::streamsize>(n));
    s.resize(static_cast<size_t>(std::max<std::streamsize>(in.gcount(), 0)));
    return s;
}

}  // namespace

std::string coremlc_version(std::string_view head) {
    static constexpr std::string_view key = "\"coremlc-version\"";
    const size_t k = head.find(key);
    if (k == std::string_view::npos) return {};
    const size_t open = head.find('"', k + key.size());
    if (open == std::string_view::npos) return {};
    const size_t close = head.find('"', open + 1);
    if (close == std::string_view::npos) return {};
    return std::string(head.substr(open + 1, close - open - 1));
}

Error move_bundle(const std::string & from, const std::string & to, bool force_copy) {
    if (!force_copy) {
        if (std::rename(from.c_str(), to.c_str()) == 0) return {};
        if (errno != EXDEV) {
            const int e = errno;
            return Error::transient("move " + from + " -> " + to + ": " + std::strerror(e), "NSPOSIXErrorDomain", e);
        }
    }
    @autoreleasepool {
        NSFileManager * fm = NSFileManager.defaultManager;
        NSError * err = nil;
        if (![fm copyItemAtPath:@(from.c_str()) toPath:@(to.c_str()) error:&err]) {
            [fm removeItemAtPath:@(to.c_str()) error:nil];
            return ns_error(err, "copy " + from + " -> " + to);
        }
        [fm removeItemAtPath:@(from.c_str()) error:nil];
    }
    return {};
}

cache::CompileResult compile_package(const std::string & mlpackage, const std::string & mlmodelc_out) {
    cache::CompileResult r;
    r.os_build = detail::device_os_build();
    @autoreleasepool {
        NSURL * in = [NSURL fileURLWithPath:@(mlpackage.c_str()) isDirectory:YES];
        if (![NSFileManager.defaultManager fileExistsAtPath:in.path]) {
            r.error = Error::transient("compile: " + mlpackage + " does not exist", "NSPOSIXErrorDomain", ENOENT);
            r.stage = failed::Stage::prepare;
            return r;
        }
        __block NSURL * out = nil;
        __block NSError * err = nil;
        PeakSampler sampler;
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [MLModel compileModelAtURL:in
                 completionHandler:^(NSURL * u, NSError * e) {
                   out = u;
                   err = e;
                   dispatch_semaphore_signal(done);
                 }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        r.compile_peak_bytes = sampler.finish();
        if (!out) {
            r.error = ns_error(err, "compile " + mlpackage);
            return r;
        }
        r.tmp_output = ns_string(out.path);
    }
    r.error = move_bundle(r.tmp_output, mlmodelc_out, false);
    if (!r.error.ok()) {
        r.stage = failed::Stage::prepare;
        @autoreleasepool {
            [NSFileManager.defaultManager removeItemAtPath:@(r.tmp_output.c_str()) error:nil];
        }
        return r;
    }
    r.coremlc = coremlc_version(read_head(mlmodelc_out + "/model.mil", 4096));
    return r;
}

}  // namespace apple
}  // namespace QVAC_COREML_NS
