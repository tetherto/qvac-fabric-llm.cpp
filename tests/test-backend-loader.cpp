// Backend loader checks that need no GPU. Stub modules named like CUDA modules
// are copied into a temporary directory and loaded through
// ggml_backend_load_all_from_path:
//   - a module that scores 0 is scored but never registered
//   - the best scoring module wins over a zero-score one
//   - a module reachable through two directory entries is scored once
//   - GGML_DISABLE_CUDA stops the cuda search entirely

#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

#ifdef _WIN32
static const char * const k_prefix = "qvac-ggml-cuda-";
static const char * const k_ext    = ".dll";

static void set_env(const char * name, const char * value) {
    _putenv_s(name, value ? value : "");
}
#else
static const char * const k_prefix = "libqvac-ggml-cuda-";
static const char * const k_ext    = ".so";

static void set_env(const char * name, const char * value) {
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
}
#endif

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

static ggml_backend_reg_t find_reg(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (strcmp(ggml_backend_reg_name(reg), name) == 0) {
            return reg;
        }
    }
    return nullptr;
}

static int count_regs(const char * name) {
    int n = 0;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        n += strcmp(ggml_backend_reg_name(ggml_backend_reg_get(i)), name) == 0 ? 1 : 0;
    }
    return n;
}

static int count_scores(const fs::path & log, const char * name) {
    std::ifstream in(log);
    std::string   line;
    int           n = 0;
    while (std::getline(in, line)) {
        n += line == name ? 1 : 0;
    }
    return n;
}

static void unload_all(const char * name) {
    while (ggml_backend_reg_t reg = find_reg(name)) {
        ggml_backend_unload(reg);
    }
}

int main() {
    std::error_code ec;
    const fs::path  dir =
        fs::temp_directory_path(ec) /
        ("ggml-loader-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path log = dir / "scores.log";
    fs::remove_all(dir, ec);
    if (!fs::create_directories(dir, ec)) {
        printf("cannot create %s, skipping\n", dir.string().c_str());
        return 77;
    }
    set_env("GGML_LOADER_TEST_LOG", log.string().c_str());
    set_env("GGML_DISABLE_CUDA", nullptr);

    const fs::path zero = dir / (std::string(k_prefix) + "zero" + k_ext);
    const fs::path ok   = dir / (std::string(k_prefix) + "ok" + k_ext);
    const fs::path link = dir / (std::string(k_prefix) + "ok-link" + k_ext);

    fs::copy_file(STUB_ZERO_PATH, zero, ec);
    check(!ec, "copy zero-score stub");

    ggml_backend_load_all_from_path(dir.string().c_str());
    check(count_scores(log, "STUB_ZERO") == 1, "zero-score module is scored");
    check(find_reg("STUB_ZERO") == nullptr, "zero-score module is not registered");

    fs::remove(log, ec);
    fs::copy_file(STUB_OK_PATH, ok, ec);
    check(!ec, "copy scoring stub");
    fs::create_symlink(ok, link, ec);
    const bool have_link = !ec;
    if (!have_link) {
        printf("symlinks unavailable, dedupe case skipped\n");
    }

    ggml_backend_load_all_from_path(dir.string().c_str());
    check(count_regs("STUB_OK") == 1, "best module is registered once");
    check(find_reg("STUB_ZERO") == nullptr, "zero-score module still not registered");
    if (have_link) {
        check(count_scores(log, "STUB_OK") == 1, "module behind two entries is scored once");
    }
    unload_all("STUB_OK");

    fs::remove(log, ec);
    set_env("GGML_DISABLE_CUDA", "1");
    ggml_backend_load_all_from_path(dir.string().c_str());
    set_env("GGML_DISABLE_CUDA", nullptr);
    check(find_reg("STUB_OK") == nullptr, "GGML_DISABLE_CUDA keeps the cuda search off");
    check(count_scores(log, "STUB_OK") == 0 && count_scores(log, "STUB_ZERO") == 0,
          "GGML_DISABLE_CUDA scores no cuda module");

    fs::remove_all(dir, ec);
    printf("%s\n", failures == 0 ? "PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}
