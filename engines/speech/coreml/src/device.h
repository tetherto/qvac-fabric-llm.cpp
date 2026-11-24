#pragma once

// Device facts the scheduler needs. Implemented by src/apple/device.mm on Apple
// builds with Objective-C++ enabled, and by src/device_posix.cpp elsewhere
// (Linux test builds, and Apple test builds without OBJCXX).

#include "qvac_coreml/options.h"

#include <cstdint>
#include <string>

namespace QVAC_COREML_NS {
namespace detail {

// macOS: free + inactive + purgeable + speculative pages (host_statistics64);
// iOS: os_proc_available_memory(); Linux: MemAvailable.
uint64_t device_available_memory();
// NSProcessInfo.isLowPowerModeEnabled; false where unknown.
bool device_low_power();
// Code-signing identifier, else bundle identifier, else the program name.
std::string device_host_id();
// kern.osversion (e.g. "24G419"); uname release elsewhere.
std::string device_os_build();
OsVersion device_os_version();
Platform device_platform();

enum class PinState {
    untried,
    pinned,      // dlopen(path, RTLD_NOLOAD | RTLD_NODELETE) took a reference
    main_image,  // the runtime lives in the main executable: never unloaded
    failed,      // a loadable image that could not be pinned
};

// Pins the library image that contains this runtime (dladdr on a runtime
// symbol, then dlopen(path, RTLD_NOLOAD | RTLD_NODELETE)), so machine code and
// callbacks outlive any dlclose. Idempotent. It enters the dynamic loader, so
// it is never called with Runtime::mu held.
PinState pin_image();
// The cached result of pin_image(); never enters the loader.
PinState pin_state();

}  // namespace detail
}  // namespace QVAC_COREML_NS
