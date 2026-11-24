#include "tts-cpp/coreml_exit.h"

#if defined(QVAC_COREML_NS)
#include "qvac_coreml/manager.h"

#include <chrono>
#endif

namespace tts_cpp {

bool coreml_arm_exit_guard() {
#if defined(QVAC_COREML_NS)
    return QVAC_COREML_NS::arm_exit_guard();
#else
    return false;
#endif
}

bool coreml_shutdown_for_exit(int64_t timeout_ms) {
#if defined(QVAC_COREML_NS)
    return QVAC_COREML_NS::shutdown_for_exit(std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms));
#else
    (void) timeout_ms;
    return true;
#endif
}

void coreml_set_automatic_exit_guard(bool on) {
#if defined(QVAC_COREML_NS)
    QVAC_COREML_NS::set_automatic_exit_guard(on);
#else
    (void) on;
#endif
}

}  // namespace tts_cpp
