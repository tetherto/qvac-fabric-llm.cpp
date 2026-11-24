// Device facts without Objective-C: Linux test builds and Apple test builds
// that do not enable OBJCXX. Low-power mode is unknown here (false).

#include "device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <TargetConditionals.h>
#endif
#include <sys/utsname.h>
#include <unistd.h>

namespace QVAC_COREML_NS {
namespace detail {

#if defined(__APPLE__)
namespace {
std::string sysctl_string(const char * name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
    std::string out(len, '\0');
    if (sysctlbyname(name, out.data(), &len, nullptr, 0) != 0) return {};
    while (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}
}  // namespace
#endif

uint64_t device_available_memory() {
#if defined(__APPLE__)
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
#else
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            kb = std::strtoull(line + 13, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return kb * 1024;
#endif
}

bool device_low_power() { return false; }

std::string device_host_id() {
#if defined(__APPLE__)
    const char * name = getprogname();
    return name ? name : "unknown";
#elif defined(__GLIBC__)
    return program_invocation_short_name ? program_invocation_short_name : "unknown";
#else
    return "unknown";
#endif
}

std::string device_os_build() {
#if defined(__APPLE__)
    return sysctl_string("kern.osversion");
#else
    struct utsname u{};
    if (uname(&u) != 0) return {};
    return std::string(u.sysname) + "-" + u.release;
#endif
}

OsVersion device_os_version() {
    OsVersion v;
#if defined(__APPLE__)
    const std::string s = sysctl_string("kern.osproductversion");
    if (!s.empty()) std::sscanf(s.c_str(), "%d.%d.%d", &v.major, &v.minor, &v.patch);
#endif
    return v;
}

Platform device_platform() {
#if defined(__APPLE__) && TARGET_OS_OSX
    return Platform::macos;
#elif defined(__APPLE__) && TARGET_OS_IPHONE
    return Platform::ios;
#else
    return Platform::other;
#endif
}

}  // namespace detail
}  // namespace QVAC_COREML_NS
