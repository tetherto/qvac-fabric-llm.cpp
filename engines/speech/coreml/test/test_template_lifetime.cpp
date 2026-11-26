// An embedded template outlives exit-time static destructors: prepare jobs
// on the runtime's threads read it through raw pointers and are never joined,
// so a job still running while the process exits must find it intact.
//
// The atexit handler is registered before the first templates::embedded()
// call, so it runs after any function-local static that call constructs has
// been destroyed. It checks that the template is still alive and that
// embedded() still answers, and exits with status 1 otherwise.

#include "templates.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unistd.h>

namespace tpl = QVAC_COREML_NS::templates;

namespace {

constexpr const char * kId = "toy-stateless";
std::weak_ptr<const QVAC_COREML_NS::recipe::Template> g_template;
const void * g_address = nullptr;

void check_at_exit() {
    if (g_template.expired()) {
        std::fprintf(stderr, "FAIL: the embedded template was destroyed before exit finished\n");
        std::fflush(stderr);
        ::_exit(1);
    }
    if (tpl::embedded(kId).get() != g_address) {
        std::fprintf(stderr, "FAIL: embedded() returned another template during exit\n");
        std::fflush(stderr);
        ::_exit(1);
    }
    std::printf("ok   the embedded template is alive during exit\n");
    std::fflush(stdout);
}

}  // namespace

int main() {
    std::atexit(&check_at_exit);
    std::shared_ptr<const QVAC_COREML_NS::recipe::Template> t = tpl::embedded(kId);
    if (!t) {
        std::fprintf(stderr, "FAIL: the toy library does not embed %s\n", kId);
        return 1;
    }
    g_template = t;
    g_address = t.get();
    t.reset();  // only the library holds it now, as during a prepare job
    if (g_template.expired()) {
        std::fprintf(stderr, "FAIL: the library does not keep the template\n");
        return 1;
    }
    std::printf("ok   the library keeps the template\n");
    return 0;
}
