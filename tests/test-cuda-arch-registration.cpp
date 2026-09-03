// Pins the CUDA arch-availability registration guard: devices with no compiled
// kernels for their compute capability are skipped in ggml_backend_cuda_reg(),
// and everything downstream must cope with the registry holding a subset.
//
// Regression fixture for QVAC-23763 / tetherto/qvac#4171. Before the guard, a
// card below the compiled floor enumerated, won backend selection over Vulkan,
// and then aborted at the first kernel launch instead of falling back.
//
// Skips with 77 when no CUDA registry is present, so it runs everywhere.

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

// GGML_CUDA_NAME, and so the registry name, is "ROCm" on HIP and "MUSA" on MUSA.
// The same source builds all three, and the guard is compiled into all three, so
// matching only "CUDA" would make those builds skip with hardware present.
static ggml_backend_reg_t find_cuda_reg() {
    static const char * const names[] = { "CUDA", "ROCm", "MUSA" };
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        for (const char * name : names) {
            if (strcmp(ggml_backend_reg_name(reg), name) == 0) {
                return reg;
            }
        }
    }
    return nullptr;
}

// virtual devices get "-v<n>" appended to the physical card's bus id
static std::string physical_key(const char * device_id) {
    std::string id = device_id ? device_id : "";
    const size_t v = id.rfind("-v");
    if (v != std::string::npos && v + 2 < id.size()) {
        bool digits = true;
        for (size_t i = v + 2; i < id.size(); i++) {
            digits = digits && id[i] >= '0' && id[i] <= '9';
        }
        if (digits) {
            id.resize(v);
        }
    }
    return id;
}

static void print_archs(ggml_backend_reg_t reg) {
    auto get_features = (ggml_backend_get_features_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (get_features == nullptr) {
        printf("compiled archs: unavailable\n");
        return;
    }
    for (ggml_backend_feature * f = get_features(reg); f && f->name; f++) {
        if (strcmp(f->name, "ARCHS") == 0) {
            printf("compiled archs: %s\n", f->value);
            return;
        }
    }
    printf("compiled archs: not reported\n");
}

int main() {
    ggml_backend_load_all();

    ggml_backend_reg_t cuda = find_cuda_reg();
    if (cuda == nullptr) {
        printf("no CUDA registry, skipping\n");
        return 77;
    }

    print_archs(cuda);

    const size_t n_cuda  = ggml_backend_reg_dev_count(cuda);
    const size_t n_total = ggml_backend_dev_count();
    printf("reg=CUDA devices=%zu total_devices=%zu\n", n_cuda, n_total);

    int fails = 0;

    // The whole point of skipping at registration: a host whose every CUDA
    // device is uncovered must still have somewhere to run.
    //
    // Checking n_total == 0 would be dead code - the CPU backend registers
    // unconditionally, so ggml_backend_dev_count() is never 0 and the assertion
    // could not fail however badly registration broke. Look for a device that
    // can actually take the work instead.
    if (n_cuda == 0) {
        bool have_fallback = false;
        for (size_t i = 0; i < n_total; i++) {
            // 'enum' tag required: the accessor function shadows the enum name.
            const enum ggml_backend_dev_type t = ggml_backend_dev_type(ggml_backend_dev_get(i));
            if (t == GGML_BACKEND_DEVICE_TYPE_CPU || t == GGML_BACKEND_DEVICE_TYPE_GPU) {
                have_fallback = true;
                break;
            }
        }
        if (!have_fallback) {
            fprintf(stderr, "FAIL: every CUDA device was skipped and no CPU or GPU device remains\n");
            fails++;
        } else {
            // Deliberately not phrased as "all devices were skipped". An empty
            // registry is also what a machine with no NVIDIA card at all
            // produces, and nothing here can tell the two apart: skipped
            // devices leave the registry entirely and the unfiltered count is
            // not reachable from the public API. Passing this is not evidence
            // the guard fired. See the note at the end of this file.
            printf("CUDA registry is empty (no covered device, or no device at all); "
                   "a non-CUDA fallback device is present\n");
        }
    }

    std::map<std::string, size_t> per_card;

    for (size_t i = 0; i < n_cuda; i++) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(cuda, i);

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        printf("  device %zu: name=%s id=%s desc=%s\n",
            i, props.name, props.device_id ? props.device_id : "(none)", props.description);

        per_card[physical_key(props.device_id)]++;

        // A subset registry means raw CUDA ids no longer index it. If any site
        // still resolves by index, the buffer type comes back null or bound to
        // the wrong device.
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        if (buft == nullptr) {
            fprintf(stderr, "FAIL: device %zu (%s) has a null buffer type\n", i, props.name);
            fails++;
            continue;
        }
        if (ggml_backend_buft_get_device(buft) != dev) {
            fprintf(stderr, "FAIL: device %zu (%s) buffer type is bound to a different device\n",
                i, props.name);
            fails++;
        }

        // The other index-resolving site, and the riskier one. Before the fix,
        // ggml_backend_cuda_init() range-checked against the UNFILTERED device
        // count and then indexed the filtered registry, so on a host with a
        // skipped device 0 and a kept device 1 it passed the check and aborted
        // inside ggml_backend_cuda_reg_get_device's GGML_ASSERT. A null buffer
        // type would not have caught that.
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            fprintf(stderr, "FAIL: device %zu (%s) failed to initialise a backend\n", i, props.name);
            fails++;
        } else {
            if (ggml_backend_get_device(backend) != dev) {
                fprintf(stderr, "FAIL: device %zu (%s) backend is bound to a different device\n",
                    i, props.name);
                fails++;
            }
            ggml_backend_free(backend);
        }
    }

    // GGML_CUDA_DEVICES splits each physical card into virtual devices. They all
    // read the same properties, so they share a compute capability and must be
    // skipped or kept as a group - never split. Round-robin assignment allows a
    // difference of one between cards.
    //
    // Know what this does NOT catch, because the shape is weaker than it looks.
    // Skipped devices leave the registry entirely, so only survivors are
    // countable, and the check is a spread over those. With one physical card
    // the map has a single key and lo == hi unconditionally, so it cannot fail -
    // and one card is the ordinary dev and CI configuration. With two cards the
    // round-robin tolerance of one is exactly the size of the smallest real
    // failure (one card losing a single virtual device), so that case passes
    // too. It catches a gross split, not a subtle one.
    //
    // Making this tight needs the skipped count to be observable rather than
    // inferred - e.g. a SKIPPED_DEVICES entry in ggml_backend_get_features, so
    // the test can assert n_cuda + skipped == ggml_backend_cuda_get_device_count()
    // and check each card against its full expected count. Worth doing when
    // there is CUDA hardware in CI to run it on; today both ctest cases skip
    // with 77 on every runner (build-cuda-ubuntu.yml builds in a GPU-less
    // container and runs no ctest step), so this fixture has no CI value yet
    // and is here for the manual runs on the QVAC-23763 hosts.
    if (!per_card.empty()) {
        size_t lo = SIZE_MAX, hi = 0;
        for (const auto & kv : per_card) {
            lo = kv.second < lo ? kv.second : lo;
            hi = kv.second > hi ? kv.second : hi;
        }
        printf("surviving cards=%zu virtual devices per card=%zu..%zu\n", per_card.size(), lo, hi);
        if (hi - lo > 1) {
            fprintf(stderr, "FAIL: one physical card was partially skipped (%zu..%zu per card)\n", lo, hi);
            fails++;
        }
    }

    if (fails > 0) {
        return 1;
    }

    printf("OK\n");
    return 0;
}
