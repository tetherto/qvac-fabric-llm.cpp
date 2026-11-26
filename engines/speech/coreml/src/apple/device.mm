// Device facts on Apple platforms. C and C++ APIs plus Objective-C messages
// only: no @interface, @implementation or categories in coreml, because
// the Objective-C runtime has one flat class namespace per process and every
// engine library carries its own copy of this file.

#include "../device.h"

#import <Foundation/Foundation.h>

#include <CoreFoundation/CoreFoundation.h>
#include <TargetConditionals.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <cstdio>
#include <string>

#if TARGET_OS_OSX
#include <Security/Security.h>
#endif
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif

namespace QVAC_COREML_NS {
namespace detail {

namespace {

std::string sysctl_string(const char * name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
    std::string out(len, '\0');
    if (sysctlbyname(name, out.data(), &len, nullptr, 0) != 0) return {};
    while (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

std::string cf_string(CFStringRef s) {
    if (!s) return {};
    if (const char * fast = CFStringGetCStringPtr(s, kCFStringEncodingUTF8)) return fast;
    const CFIndex max = CFStringGetMaximumSizeForEncoding(CFStringGetLength(s), kCFStringEncodingUTF8) + 1;
    std::string out(static_cast<size_t>(max), '\0');
    if (!CFStringGetCString(s, out.data(), max, kCFStringEncodingUTF8)) return {};
    out.resize(std::char_traits<char>::length(out.c_str()));
    return out;
}

#if TARGET_OS_OSX
std::string code_signing_identifier() {
    SecCodeRef code = nullptr;
    if (SecCodeCopySelf(kSecCSDefaultFlags, &code) != errSecSuccess || !code) return {};
    CFDictionaryRef info = nullptr;
    std::string id;
    if (SecCodeCopySigningInformation(reinterpret_cast<SecStaticCodeRef>(code), kSecCSSigningInformation, &info) ==
            errSecSuccess &&
        info) {
        const void * v = CFDictionaryGetValue(info, kSecCodeInfoIdentifier);
        if (v && CFGetTypeID(v) == CFStringGetTypeID()) id = cf_string(static_cast<CFStringRef>(v));
        CFRelease(info);
    }
    CFRelease(code);
    return id;
}
#endif

}  // namespace

uint64_t device_available_memory() {
#if TARGET_OS_IPHONE
    return static_cast<uint64_t>(os_proc_available_memory());
#else
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) page = 4096;
    const uint64_t pages = (uint64_t) vm.free_count + vm.inactive_count + vm.purgeable_count + vm.speculative_count;
    return pages * (uint64_t) page;
#endif
}

bool device_low_power() {
    @autoreleasepool {
        if (@available(macOS 12.0, iOS 9.0, *)) {
            return [[NSProcessInfo processInfo] isLowPowerModeEnabled] ? true : false;
        }
    }
    return false;
}

std::string device_host_id() {
#if TARGET_OS_OSX
    std::string id = code_signing_identifier();
    if (!id.empty()) return id;
#endif
    if (CFBundleRef bundle = CFBundleGetMainBundle()) {
        std::string id = cf_string(CFBundleGetIdentifier(bundle));
        if (!id.empty()) return id;
    }
    const char * name = getprogname();
    return name ? name : "unknown";
}

std::string device_os_build() { return sysctl_string("kern.osversion"); }

OsVersion device_os_version() {
    OsVersion v;
    @autoreleasepool {
        const NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
        v.major = static_cast<int>(os.majorVersion);
        v.minor = static_cast<int>(os.minorVersion);
        v.patch = static_cast<int>(os.patchVersion);
    }
    return v;
}

Platform device_platform() {
#if TARGET_OS_OSX
    return Platform::macos;
#elif TARGET_OS_IPHONE
    return Platform::ios;
#else
    return Platform::other;
#endif
}

}  // namespace detail
}  // namespace QVAC_COREML_NS
