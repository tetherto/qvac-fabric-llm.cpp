// test-coreml-link: tts-cpp, qvac-parakeet and audiogen
// in one binary, together with every engine's copy of the Core ML runtime.
// The link itself is the main assertion (a symbol that escaped its namespace,
// or an Objective-C class, is a duplicate); at run time each copy works and
// each has its own per-image manager, and each engine's re-exported exit hook
// reaches its own runtime copy (it arms on the main thread).

#include "qvac_coreml_link_probes.h"

#include "audiogen-cpp/coreml_exit.h"
#include "parakeet/coreml_exit.h"
#include "tts-cpp/coreml_exit.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#define QVAC_DECLARE_PROBE(ns)          \
    namespace ns {                      \
    int link_probe();                   \
    const void * link_probe_identity(); \
    }
QVAC_COREML_LINK_PROBES(QVAC_DECLARE_PROBE)
#undef QVAC_DECLARE_PROBE

int main() {
    int failures = 0;
    std::vector<const void *> ids;
#define QVAC_RUN_PROBE(ns)                                                  \
    {                                                                       \
        const int rc = ns::link_probe();                                    \
        std::printf("%s %s (rc=%d)\n", rc == 0 ? "ok  " : "FAIL", #ns, rc); \
        if (rc != 0) ++failures;                                            \
        ids.push_back(ns::link_probe_identity());                           \
    }
    QVAC_COREML_LINK_PROBES(QVAC_RUN_PROBE)
#undef QVAC_RUN_PROBE
    std::vector<const void *> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        std::printf("FAIL two engines share one Core ML manager\n");
        ++failures;
    }
    // Core ML is on in all three libraries here: a hook that does not reach
    // its runtime returns false.
    const struct {
        const char * name;
        bool (*arm)();
    } hooks[] = {{"tts_cpp::coreml_arm_exit_guard", &tts_cpp::coreml_arm_exit_guard},
                 {"parakeet::coreml_arm_exit_guard", &parakeet::coreml_arm_exit_guard},
                 {"tts_cpp::acestep::coreml_arm_exit_guard", &tts_cpp::acestep::coreml_arm_exit_guard}};
    for (const auto & h : hooks) {
        const bool armed = h.arm();
        std::printf("%s %s() on the main thread returned %s\n", armed ? "ok  " : "FAIL", h.name,
                    armed ? "true" : "false");
        if (!armed) ++failures;
    }
    std::printf("%s: %zu runtime copies in one image\n", failures == 0 ? "PASS" : "FAIL", ids.size());
    return failures == 0 ? 0 : 1;
}
