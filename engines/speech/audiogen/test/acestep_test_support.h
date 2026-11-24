#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_failures;                                                             \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                             \
    } while (0)

bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1);
    else       unsetenv(key);
#endif
}

std::string test_temp_dir() {
    for (const char * var : { "TMPDIR", "TEMP", "TMP" }) {
        const char * val = std::getenv(var);
        if (val && *val) {
            return val;
        }
    }
    return "/tmp";
}

}  // namespace
