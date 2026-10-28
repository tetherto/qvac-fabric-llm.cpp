#include "../common/fit.h"
#include "../ggml/src/ggml-impl.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

struct logger_state {
    std::mutex mutex;
    std::condition_variable changed;
    bool probe_logging = false;
    bool release_probe = false;
    bool unrelated_info = false;
    bool probe_debug = false;
    bool restored_info = false;
};

static void log_callback(ggml_log_level level, const char * text, void * data) {
    auto & state = *static_cast<logger_state *>(data);
    std::unique_lock<std::mutex> lock(state.mutex);
    if (std::strstr(text, "fit-logger-missing.gguf") && !state.probe_logging) {
        state.probe_logging = true;
        lock.unlock();
        ggml_log_internal(GGML_LOG_LEVEL_INFO, "probe logging\n");
        lock.lock();
        state.changed.notify_all();
        state.changed.wait(lock, [&] { return state.release_probe; });
    }
    if (std::strcmp(text, "unrelated logging\n") == 0) {
        state.unrelated_info = level == GGML_LOG_LEVEL_INFO;
    }
    if (std::strcmp(text, "probe logging\n") == 0) {
        state.probe_debug = level == GGML_LOG_LEVEL_DEBUG;
    }
    if (std::strcmp(text, "probe finished\n") == 0) {
        state.restored_info = level == GGML_LOG_LEVEL_INFO;
    }
}

int main() {
    logger_state state;
    llama_log_set(log_callback, &state);
    std::thread probe([] {
        auto mparams = llama_model_default_params();
        auto cparams = llama_context_default_params();
        std::vector<ggml_backend_dev_t> devices;
        uint32_t layers = 0, context = 0, experts = 0;
        try {
            common_get_device_memory_data("fit-logger-missing.gguf", &mparams, &cparams,
                devices, layers, context, experts, GGML_LOG_LEVEL_ERROR);
        } catch (const std::exception &) {
        }
        ggml_log_internal(GGML_LOG_LEVEL_INFO, "probe finished\n");
    });

    bool entered;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        entered = state.changed.wait_for(lock, std::chrono::seconds(10), [&] { return state.probe_logging; });
    }
    ggml_log_callback callback;
    void * data;
    llama_log_get(&callback, &data);
    const bool unchanged = callback == log_callback && data == &state;
    ggml_log_internal(GGML_LOG_LEVEL_INFO, "unrelated logging\n");
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.release_probe = true;
    }
    state.changed.notify_all();
    probe.join();
    llama_log_get(&callback, &data);
    const bool restored = callback == log_callback && data == &state;
    llama_log_set(nullptr, nullptr);

    if (!entered || !unchanged || !state.unrelated_info || !restored || !state.probe_debug || !state.restored_info) {
        fprintf(stderr, "FAIL fit logger: entered=%d unchanged=%d unrelated_info=%d restored=%d probe_debug=%d restored_info=%d\n",
            entered, unchanged, state.unrelated_info, restored, state.probe_debug, state.restored_info);
        return 1;
    }
    return 0;
}
