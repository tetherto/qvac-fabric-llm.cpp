#include "ggml-backend-dl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

#    include <cwchar>
#    include <cwctype>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#include "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#include "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#include "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#include "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#include "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#include "ggml-et.h"
#endif

namespace fs = std::filesystem;

static std::string path_str(const fs::path & path) {
    try {
#if defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        return std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());
#else
        // C++17: u8string() returns std::string
        return path.u8string();
#endif
    } catch (...) {
        return std::string();
    }
}

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

// GGML_DISABLE_<NAME>, such as GGML_DISABLE_CUDA, keeps a backend from loading
// in static and DL builds alike.
static bool ggml_backend_disabled_by_env(const char * name) {
    std::string var = "GGML_DISABLE_";
    for (const char * c = name; *c != '\0'; ++c) {
        var += (char) std::toupper((unsigned char) *c);
    }
    if (getenv(var.c_str()) == nullptr) {
        return false;
    }
    GGML_LOG_DEBUG("%s backend disabled by %s environment variable\n", name, var.c_str());
    return true;
}

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        if (!ggml_backend_disabled_by_env("cuda")) {
            register_backend(ggml_backend_cuda_reg());
        }
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
        if (!ggml_backend_disabled_by_env("vulkan")) {
            register_backend(ggml_backend_vk_reg());
        }
#endif
#ifdef GGML_USE_WEBGPU
        register_backend(ggml_backend_webgpu_reg());
#endif
#ifdef GGML_USE_ZDNN
        register_backend(ggml_backend_zdnn_reg());
#endif
#ifdef GGML_USE_VIRTGPU_FRONTEND
        register_backend(ggml_backend_virtgpu_reg());
#endif

#ifdef GGML_USE_OPENCL
        register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_ZENDNN
        register_backend(ggml_backend_zendnn_reg());
#endif
#ifdef GGML_USE_HEXAGON
        register_backend(ggml_backend_hexagon_reg());
#endif
#ifdef GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_OPENVINO
        register_backend(ggml_backend_openvino_reg());
#endif
#ifdef GGML_USE_ET
        register_backend(ggml_backend_et_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        for (auto & entry : backends) {
            if (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        if (!reg) {
            return;
        }

        for (auto & entry : backends) {
            if (entry.reg == reg) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
        for (auto & dev : devices) {
            if (dev == device) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            return nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        return load_backend(path, silent, std::move(handle));
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent, dl_handle_ptr handle) {
        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } else {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            return nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        return reg;
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry & entry) { return entry.reg == reg; });

        if (it == backends.end()) {
            if (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            return;
        }

        if (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { return ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

size_t ggml_backend_reg_count() {
    return get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    return get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (striequals(ggml_backend_reg_name(reg), name)) {
            return reg;
        }
    }
    return nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    return get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    return get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (striequals(ggml_backend_dev_name(dev), name)) {
            return dev;
        }
    }
    return nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
    return get_reg().load_backend(path, false);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static fs::path get_executable_path() {
#if defined(__APPLE__)
    // get executable path
    std::vector<char> path;
    uint32_t size;
    while (true) {
        size = path.size();
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return base_path + "/";
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    if (len == 0) {
        return {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + L"\\";
#else
    return {};
#endif
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32
    return fs::u8path("qvac-ggml-");
#else
    return fs::u8path("libqvac-ggml-");
#endif
}

static fs::path backend_filename_extension() {
#ifdef _WIN32
    return fs::u8path(".dll");
#else
    return fs::u8path(".so");
#endif
}

#ifdef _WIN32
// The Windows loader does not search PATH, so the CUDA module's runtime DLLs are
// resolved from the user's CUDA install instead of being bundled. CUDA 13 keeps
// them in <toolkit>\bin\x64, which CUDA 12 installs do not have. CUDA_PATH names
// only the last toolkit installed, so every CUDA_PATH_V* toolkit is added too;
// the DLL names carry the CUDA major, so toolkits of different majors never
// shadow each other. The directories are searched only while the CUDA module
// loads; LOAD_LIBRARY_SEARCH_DEFAULT_DIRS in dl_load_library includes them.
class cuda_runtime_dll_directory {
  public:
    explicit cuda_runtime_dll_directory(bool enabled) {
        if (!enabled) {
            return;
        }
        std::vector<std::wstring> roots;
        if (const wchar_t * cuda_path = _wgetenv(L"CUDA_PATH"); cuda_path != nullptr && *cuda_path != L'\0') {
            roots.emplace_back(cuda_path);
        }
        if (wchar_t * env = GetEnvironmentStringsW(); env != nullptr) {
            static const std::wstring prefix = L"CUDA_PATH_V";
            for (const wchar_t * entry = env; *entry != L'\0'; entry += wcslen(entry) + 1) {
                const std::wstring var(entry);
                const size_t       eq = var.find(L'=');
                if (var.compare(0, prefix.size(), prefix) == 0 && eq != std::wstring::npos && eq + 1 < var.size()) {
                    roots.push_back(var.substr(eq + 1));
                }
            }
            FreeEnvironmentStringsW(env);
        }

        std::unordered_set<std::wstring> added;
        for (const auto & root : roots) {
            const fs::path  bin_dir = fs::path(root) / L"bin" / L"x64";
            std::error_code ec;
            std::wstring    key = bin_dir.lexically_normal().wstring();
            std::transform(key.begin(), key.end(), key.begin(), ::towlower);
            if (!fs::is_directory(bin_dir, ec) || !added.insert(key).second) {
                continue;
            }
            if (DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(bin_dir.c_str()); cookie != nullptr) {
                cookies.push_back(cookie);
            } else {
                GGML_LOG_INFO("%s: AddDllDirectory(%s) failed: %lu\n", __func__, path_str(bin_dir).c_str(),
                              GetLastError());
            }
        }
        if (cookies.empty()) {
            GGML_LOG_DEBUG(
                "%s: no CUDA toolkit with bin\\x64 in CUDA_PATH or CUDA_PATH_V*, "
                "CUDA backend needs a CUDA 13 install\n",
                __func__);
        }
    }

    ~cuda_runtime_dll_directory() {
        for (DLL_DIRECTORY_COOKIE cookie : cookies) {
            RemoveDllDirectory(cookie);
        }
    }

    cuda_runtime_dll_directory(const cuda_runtime_dll_directory &)             = delete;
    cuda_runtime_dll_directory & operator=(const cuda_runtime_dll_directory &) = delete;

  private:
    std::vector<DLL_DIRECTORY_COOKIE> cookies;
};
#endif

static ggml_backend_reg_t ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    if (ggml_backend_disabled_by_env(name)) {
        return nullptr;
    }
#ifdef _WIN32
    const cuda_runtime_dll_directory cuda_dlls(striequals(name, "cuda"));
#endif
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    if (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, and current directory outside Windows
        search_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        const fs::path cwd = fs::current_path(cwd_ec);
        if (cwd_ec) {
            GGML_LOG_DEBUG("%s: current_path() failure, error-message: %s\n", __func__, cwd_ec.message().c_str());
        } else {
            search_paths.push_back(cwd);
        }

        // Android does not require prepending path, the .apk will have embedded the dynamic .so, only the name is needed for dlopen
        // TODO add here prebuild/ search patch for Desktop platforms where we want to support dynamic loading
    } else {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    int best_score = 0;
    bool                                          found_candidate = false;
    fs::path best_path;
    dl_handle_ptr   best_handle;
    std::error_code ec;
    std::unordered_set<std::string> attempted_paths;
    std::vector<std::pair<fs::path, std::string>> load_failures;

    auto markAttempted = [&attempted_paths](const fs::path & path, bool name_only) {
        std::string key;
        if (name_only) {
            key = "name:" + path_str(path);
        } else {
            std::error_code path_ec;
            fs::path        canonical_path = fs::weakly_canonical(path, path_ec);
            if (path_ec) {
                path_ec.clear();
                canonical_path = fs::absolute(path, path_ec);
            }
            if (path_ec) {
                canonical_path = path;
            }
            key = "path:" + path_str(canonical_path.lexically_normal());
        }
#ifdef _WIN32
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
#endif
        return attempted_paths.insert(std::move(key)).second;
    };

    auto tryEntryWithScore = [&best_score, &best_path, &best_handle, &found_candidate, &markAttempted, &load_failures, silent, _func = __func__](
                                 const fs::path & entryPath, int scoreOffset = 1, bool name_only = false) {
        if (!markAttempted(entryPath, name_only)) {
            return;
        }
        dl_handle_ptr handle{ dl_load_library(entryPath) };
        if (!handle) {
            const char * error_ptr = dl_error();
            const std::string error = error_ptr != nullptr ? error_ptr : "unknown loader error";
            if (!silent) {
                GGML_LOG_DEBUG("%s: failed to load %s: %s\n", _func, path_str(entryPath).c_str(), error.c_str());
            }
            load_failures.emplace_back(entryPath, error);
        }
        if (handle) {
            // a module that failed to load must not hide a working one the platform loader finds
            found_candidate = found_candidate || !name_only;
            auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
            int  s        = 1;
            if (score_fn) {
                const int backend_score = score_fn();
                if (backend_score == 0) {
                    return;
                }
                s = backend_score + scoreOffset;
            }
#ifdef NDEBUG
            GGML_LOG_DEBUG("%s: %s score: %d\n", _func, path_str(entryPath).c_str(), s);
#endif
            if (s > best_score) {
                best_score = s;
                best_path  = entryPath;
                best_handle = std::move(handle);
            }
        }
    };

    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path, ec)) {
            if (ec) {
                GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(search_path).c_str(), ec.message().c_str());
            } else {
                GGML_LOG_DEBUG("%s: search path %s does not exist\n", __func__, path_str(search_path).c_str());
            }
            continue;
        }
        GGML_LOG_INFO("%s: searching for %s in %s\n", __func__, path_str(name_path).c_str(), path_str(search_path).c_str());
        std::vector<fs::path> candidates;
        std::error_code dir_ec;
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied, dir_ec);
        if (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(search_path).c_str(), dir_ec.message().c_str());
            continue;
        }
        for (const fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            const auto & entry = *dir_it;
            if (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                if (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    candidates.push_back(entry.path());
                }
            }
        }
        // A tie keeps the first filename. The arm64 CUDA 13 and CUDA 12 Jetson
        // modules ship disjoint archs and runtimes, so no device scores on both.
        std::sort(candidates.begin(), candidates.end());
        for (const auto & candidate : candidates) {
            tryEntryWithScore(candidate);
        }
    }

    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            if (std::error_code ec; fs::exists(path, ec)) {
                if (markAttempted(path, false)) {
                    return get_reg().load_backend(path, silent);
                }
            } else {
                if (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
    }

#ifndef _WIN32
    // Let the platform loader resolve libraries outside the explicit search paths,
    // only when no module there loaded. A module there that scored 0 was
    // rejected on purpose, and the loader could resolve to it or a stale copy.
    if (!found_candidate) {
        // From worst to best
        std::vector<fs::path> names = { name_path };
#    ifdef __ANDROID__
        if (strcmp(name, "cpu") == 0) {
            names.emplace_back("cpu-android_armv8.0_1");
            names.emplace_back("cpu-android_armv8.2_1");
            names.emplace_back("cpu-android_armv8.2_2");
            names.emplace_back("cpu-android_armv8.6_1");
        }
#    endif
        for (size_t scoreOffset = 0; scoreOffset < names.size(); ++scoreOffset) {
            const auto & loopNamePath = names[scoreOffset];
            // Try loading backend with just the library name, leave to dlopen path resolution.
            fs::path     filename     = backend_filename_prefix().native() + loopNamePath.native() +
                                backend_filename_extension().native();
            tryEntryWithScore(filename, 1 + scoreOffset, true);
        }
    }
#endif

    if (!best_handle) {
        if (!silent) {
            for (const auto & failure : load_failures) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(failure.first).c_str(), failure.second.c_str());
            }
        }
        return nullptr;
    }
    return get_reg().load_backend(best_path, silent, std::move(best_handle));
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

#ifdef __ANDROID__
namespace {
// Parses adreno version from gpu description or returns -1 if its not Adreno GPU or -3 if failed to parse the version
int adrenoVersion(const std::string & gpuDescription) {
    std::regex  adrenoRegex(R"((\d+))");
    std::smatch matches;
    if (gpuDescription.find("dreno") != std::string::npos && std::regex_search(gpuDescription, matches, adrenoRegex) && matches.size() > 1) {
        try {
            int adrenoVersion = std::stoi(matches[1].str());
            return adrenoVersion;
        } catch (std::invalid_argument & e) {
            GGML_LOG_ERROR("%s: failed to parse adreno version from %s: %s\n", __func__, gpuDescription.c_str(),
                           e.what());
            return -3;
        }
    }
    return -1;
}

// Returns smallest Adreno version among GPU devices or -1 if there is no adreno GPU
int minAdrenoVersion(ggml_backend_reg_t vulkanBackend) {
    if (!vulkanBackend) {
        return -2;
    }
    int minFoundVersion = std::numeric_limits<int>::max();
    for (size_t i = 0; i < vulkanBackend->iface.get_device_count(vulkanBackend); i++) {
        ggml_backend_dev_t dev = vulkanBackend->iface.get_device(vulkanBackend, i);
        if (!dev) {
            continue;
        }
        auto description = std::string(dev->iface.get_description(dev));
        GGML_LOG_INFO("%s: found device description: %s\n", __func__, description.c_str());
        int devAdrenoVersion = adrenoVersion(description);
        if (devAdrenoVersion > 0) {
            minFoundVersion = std::min(minFoundVersion, devAdrenoVersion);
        }
    }
    if (minFoundVersion < std::numeric_limits<int>::max()) {
        return minFoundVersion;
    }
    return -1;
}
}  // namespace
#endif

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef GGML_BACKEND_DL
    // Only attempt to dlopen backends when built with dynamic backend support
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu", silent, dir_path);

    bool useOpencl = true;

#ifdef __ANDROID__
    // Logic for buggy backends on Adreno GPUs
    // Use Vulkan backend to obtain GPU information
    ggml_backend_reg_t vulkanBackend           = ggml_backend_reg_by_name("vulkan");
    int                devicesMinAdrenoVersion = minAdrenoVersion(vulkanBackend);
    if (devicesMinAdrenoVersion <= 0) {
        GGML_LOG_INFO(
            "%s: no adreno GPU version found (%d) removing OpenCL backend (if any) to rely on Vulkan/cpu only\n",
            __func__, devicesMinAdrenoVersion);
        useOpencl = false;
    } else if (devicesMinAdrenoVersion > 700) {
        GGML_LOG_INFO("%s: Adreno GPU version %d found keeping OpenCL backend\n", __func__, devicesMinAdrenoVersion);
    } else if (devicesMinAdrenoVersion > 600) {
        GGML_LOG_INFO("%s: Adreno GPU version %d should rely on cpu only\n", __func__, devicesMinAdrenoVersion);
        if (vulkanBackend) {
            ggml_backend_unload(vulkanBackend);
            GGML_LOG_INFO("%s: Vulkan backend removed\n", __func__);
        }
        useOpencl = false;
    }
#endif

    if(useOpencl) {
        ggml_backend_load_best("opencl", silent, dir_path);
    }
    ggml_backend_load_best("hexagon", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
#else
    // When built without GGML_BACKEND_DL, backends are statically linked
    // No dynamic loading needed - avoids potential conflicts with system libraries
    GGML_UNUSED(dir_path);
#endif
}
