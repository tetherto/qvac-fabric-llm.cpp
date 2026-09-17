#include "xdna-verify.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace {

struct verify_stat {
    double   sum_rel = 0.0;   // sum of per-call relative RMS
    double   worst   = 0.0;   // worst per-call relative RMS
    uint64_t calls   = 0;
};

struct verify_state {
    int level = 0;
    int layer = 0;
    std::mutex mutex;
    std::map<std::string, verify_stat> by_key;

    ~verify_state() {
        if (level <= 0 || by_key.empty()) {
            return;
        }
        fprintf(stderr, "xdna-verify: %-34s %8s %10s %10s\n", "op", "calls", "relRMS", "worst");
        for (const auto & kv : by_key) {
            fprintf(stderr, "xdna-verify: %-34s %8llu %9.3e %9.3e\n",
                    kv.first.c_str(), (unsigned long long) kv.second.calls,
                    kv.second.sum_rel / (double) kv.second.calls, kv.second.worst);
        }
    }
};

verify_state & state(void) {
    // verify_state holds a mutex, so it is configured in place rather than
    // returned by value from an initializer lambda.
    static verify_state st;
    static std::once_flag once;
    std::call_once(once, []() {
        const char * v = getenv("GGML_XDNA_VERIFY");
        st.level = v ? atoi(v) : 0;
        const char * l = getenv("GGML_XDNA_VERIFY_LAYER");
        st.layer = l ? atoi(l) : 0;
    });
    return st;
}

} // namespace

int xdna_verify_level(void) {
    return state().level;
}

int xdna_verify_layer(void) {
    return state().layer;
}

void xdna_verify_add(const char * key, double rel_rms) {
    verify_state & st = state();
    std::lock_guard<std::mutex> lock(st.mutex);
    verify_stat & s = st.by_key[key ? key : "?"];
    s.sum_rel += rel_rms;
    s.worst = std::max(s.worst, rel_rms);
    s.calls++;
}

double xdna_verify_rel(const float * got, const float * ref, size_t n) {
    double se = 0.0;
    double sr = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) got[i] - (double) ref[i];
        se += d * d;
        sr += (double) ref[i] * (double) ref[i];
    }
    return sr > 0.0 ? std::sqrt(se / sr) : 0.0;
}
