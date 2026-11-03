// Backend loader checks that need no GPU. Stub modules named like CUDA modules
// are copied into a temporary directory and loaded through
// ggml_backend_load_all_from_path:
//   - a module that scores 0 is scored but never registered
//   - the best scoring module wins over a zero-score one
//   - a module reachable through two directory entries is scored once
//   - GGML_DISABLE_CUDA stops the cuda search entirely
//   - a module logs through the logger set with ggml_log_set, also while it is scored
//   - outside Windows, with a static ggml-base: a module error line and its CONT lines also go to stderr,
//     and loading again keeps a logger set on the module
//   - on Windows, a cuda module's runtime DLL resolves from CUDA_PATH or CUDA_PATH_V*

#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

static const char * const k_prefix = "qvac-ggml-cuda-";
static const char * const k_ext    = ".dll";

// Both the CRT copy, read by getenv, and the process block, read by
// GetEnvironmentStrings, are updated.
static void set_env(const char * name, const char * value) {
    _putenv_s(name, value ? value : "");
    SetEnvironmentVariableA(name, value);
}
#else
#    include <dlfcn.h>
#    include <fcntl.h>
#    include <unistd.h>

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

struct log_line {
    ggml_log_level level;
    std::string    text;
};

static std::vector<log_line> log_lines;

static void capture_log(ggml_log_level level, const char * text, void *) {
    log_lines.push_back({ level, text });
}

static bool logged(ggml_log_level level, const std::string & text) {
    for (const auto & line : log_lines) {
        if (line.level == level && line.text == text) {
            return true;
        }
    }
    return false;
}

#ifndef _WIN32
static std::vector<std::string> module_lines;

static void module_log(ggml_log_level, const char * text, void *) {
    module_lines.emplace_back(text);
}

// Run fn with stderr sent to a file, and return what was written.
template <typename F> static std::string capture_stderr(const fs::path & file, F fn) {
    fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    const int fd    = open(file.string().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (saved < 0 || fd < 0) {
        return "<cannot capture stderr>";
    }
    dup2(fd, STDERR_FILENO);
    close(fd);
    fn();
    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    std::ifstream in(file);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
#endif

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
    ggml_log_set(capture_log, nullptr);

    const fs::path zero = dir / (std::string(k_prefix) + "zero" + k_ext);
    const fs::path ok   = dir / (std::string(k_prefix) + "ok" + k_ext);
    const fs::path link = dir / (std::string(k_prefix) + "ok-link" + k_ext);

    fs::copy_file(STUB_ZERO_PATH, zero, ec);
    check(!ec, "copy zero-score stub");

    ggml_backend_load_all_from_path(dir.string().c_str());
    check(count_scores(log, "STUB_ZERO") == 1, "zero-score module is scored");
    check(find_reg("STUB_ZERO") == nullptr, "zero-score module is not registered");
    check(logged(GGML_LOG_LEVEL_INFO, "STUB_ZERO: score\n"), "module log while scored reaches the logger");

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
    check(logged(GGML_LOG_LEVEL_DEBUG, "STUB_OK: init\n"), "module debug log reaches the logger");
    check(find_reg("STUB_ZERO") == nullptr, "zero-score module still not registered");
    if (have_link) {
        check(count_scores(log, "STUB_OK") == 1, "module behind two entries is scored once");
    }

#ifndef _WIN32
    {
        // The module's own logger, when it links its own ggml-base.
        using log_set_t      = void (*)(ggml_log_callback, void *);
        using log_internal_t = void (*)(ggml_log_level, const char *, ...);
        Dl_info            info{};
        ggml_backend_reg_t reg    = find_reg("STUB_OK");
        void *             handle = nullptr;
        if (reg != nullptr && dladdr(reg, &info) != 0 && info.dli_fname != nullptr) {
            handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
        }
        auto mod_log_set      = handle ? (log_set_t) dlsym(handle, "ggml_log_set") : nullptr;
        auto mod_log_internal = handle ? (log_internal_t) dlsym(handle, "ggml_log_internal") : nullptr;
        if (mod_log_set == nullptr || mod_log_internal == nullptr || mod_log_set == &ggml_log_set) {
            printf("module shares ggml-base, own-logger cases skipped\n");
        } else {
            const std::string err = capture_stderr(dir / "stderr.log", [&] {
                mod_log_internal(GGML_LOG_LEVEL_ERROR, "STUB_OK: error\n");
                mod_log_internal(GGML_LOG_LEVEL_CONT, "  error detail\n");
                mod_log_internal(GGML_LOG_LEVEL_INFO, "STUB_OK: info\n");
                mod_log_internal(GGML_LOG_LEVEL_CONT, ".");
            });
            check(logged(GGML_LOG_LEVEL_ERROR, "STUB_OK: error\n"), "module error reaches the logger");
            check(err == "STUB_OK: error\n  error detail\n",
                  "module error and its CONT lines also go to stderr, other lines do not");

            mod_log_set(module_log, nullptr);
            ggml_backend_load_all_from_path(dir.string().c_str());
            mod_log_internal(GGML_LOG_LEVEL_INFO, "STUB_OK: probe\n");
            check(!module_lines.empty() && module_lines.back() == "STUB_OK: probe\n",
                  "loading again keeps a logger set on the module");
            mod_log_set(nullptr, nullptr);
        }
        if (handle != nullptr) {
            dlclose(handle);
        }
    }
#endif
    unload_all("STUB_OK");

    fs::remove(log, ec);
    set_env("GGML_DISABLE_CUDA", "1");
    ggml_backend_load_all_from_path(dir.string().c_str());
    set_env("GGML_DISABLE_CUDA", nullptr);
    check(find_reg("STUB_OK") == nullptr, "GGML_DISABLE_CUDA keeps the cuda search off");
    check(count_scores(log, "STUB_OK") == 0 && count_scores(log, "STUB_ZERO") == 0,
          "GGML_DISABLE_CUDA scores no cuda module");

#ifdef STUB_DEP_PATH
    // The Windows loader does not search PATH, so a CUDA module's runtime DLLs are
    // found only through %CUDA_PATH%\bin\x64.
    const fs::path dep_dir  = dir / "dep";
    const fs::path cuda_dir = dir / "cuda";
    fs::create_directories(dep_dir, ec);
    fs::create_directories(cuda_dir / "bin" / "x64", ec);
    fs::copy_file(STUB_DEP_PATH, dep_dir / (std::string(k_prefix) + "dep" + k_ext), ec);
    check(!ec, "copy dependent stub");
    fs::copy_file(DEP_DLL_PATH, cuda_dir / "bin" / "x64" / fs::path(DEP_DLL_PATH).filename(), ec);
    check(!ec, "copy dependency into fake CUDA_PATH");

    set_env("CUDA_PATH", nullptr);
    ggml_backend_load_all_from_path(dep_dir.string().c_str());
    check(find_reg("STUB_DEP") == nullptr, "module with an unresolved runtime DLL is not registered");

    set_env("CUDA_PATH", dir.string().c_str());
    ggml_backend_load_all_from_path(dep_dir.string().c_str());
    check(find_reg("STUB_DEP") == nullptr, "CUDA_PATH without bin\\x64 is not used");

    set_env("CUDA_PATH", cuda_dir.string().c_str());
    ggml_backend_load_all_from_path(dep_dir.string().c_str());
    set_env("CUDA_PATH", nullptr);
    check(count_regs("STUB_DEP") == 1, "runtime DLL is resolved from CUDA_PATH\\bin\\x64");
    unload_all("STUB_DEP");

    // CUDA_PATH names only the last toolkit installed, which may be an older major.
    set_env("CUDA_PATH", dir.string().c_str());
    set_env("CUDA_PATH_V13_0", cuda_dir.string().c_str());
    ggml_backend_load_all_from_path(dep_dir.string().c_str());
    set_env("CUDA_PATH", nullptr);
    set_env("CUDA_PATH_V13_0", nullptr);
    check(count_regs("STUB_DEP") == 1, "runtime DLL is resolved from CUDA_PATH_V13_0 when CUDA_PATH has no bin\\x64");
    unload_all("STUB_DEP");
#endif

    ggml_log_set(nullptr, nullptr);
    fs::remove_all(dir, ec);
    printf("%s\n", failures == 0 ? "PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}
