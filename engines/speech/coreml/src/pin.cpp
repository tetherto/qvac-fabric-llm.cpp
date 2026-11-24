// Image pin: keeps this image mapped while it has work, so a dlclose cannot
// unmap code a worker still runs. Pure POSIX: dladdr and
// RTLD_NOLOAD | RTLD_NODELETE exist in the macOS and iOS SDKs (dlfcn.h) and in
// glibc and bionic, so the dlclose test runs on the Linux lanes as well.

#include "device.h"

#include <atomic>
#include <cstdint>

#include <dlfcn.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif

namespace QVAC_COREML_NS {
namespace detail {

namespace {

std::atomic<int> g_pin_state{static_cast<int>(PinState::untried)};

// The main executable is never unloaded, so it needs no pin.
bool in_main_executable(const Dl_info & info, const void * addr) {
#if defined(__APPLE__)
    (void) addr;
    return info.dli_fbase == static_cast<const void *>(_dyld_get_image_header(0));
#else
    (void) info;
    // dl_iterate_phdr visits the main program first.
    struct Probe {
        uintptr_t addr;
        bool first;
        bool in_main;
    } probe{reinterpret_cast<uintptr_t>(addr), true, false};
    dl_iterate_phdr(
        [](struct dl_phdr_info * i, size_t, void * data) -> int {
            auto * p = static_cast<Probe *>(data);
            if (!p->first) return 1;
            p->first = false;
            for (int k = 0; k < i->dlpi_phnum; ++k) {
                const auto & ph = i->dlpi_phdr[k];
                if (ph.p_type != PT_LOAD) continue;
                const uintptr_t lo = static_cast<uintptr_t>(i->dlpi_addr + ph.p_vaddr);
                if (p->addr >= lo && p->addr < lo + ph.p_memsz) p->in_main = true;
            }
            return 1;
        },
        &probe);
    return probe.in_main;
#endif
}

}  // namespace

PinState pin_image() {
    const int state = g_pin_state.load();
    if (state != static_cast<int>(PinState::untried)) return static_cast<PinState>(state);
    Dl_info info{};
    void * self = reinterpret_cast<void *>(&pin_image);
    PinState result = PinState::failed;
    if (dladdr(self, &info) != 0 && info.dli_fname != nullptr) {
        // The returned handle is leaked on purpose: the image stays referenced
        // and, with RTLD_NODELETE, mapped for the life of the process.
        if (dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE) != nullptr) {
            result = PinState::pinned;
        } else if (in_main_executable(info, self)) {
            result = PinState::main_image;
        }
    }
    int expected = static_cast<int>(PinState::untried);
    g_pin_state.compare_exchange_strong(expected, static_cast<int>(result));
    return static_cast<PinState>(g_pin_state.load());
}

PinState pin_state() { return static_cast<PinState>(g_pin_state.load()); }

}  // namespace detail
}  // namespace QVAC_COREML_NS
