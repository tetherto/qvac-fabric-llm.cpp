#include "json.h"
#include "log.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
struct callback_context {
    common_log *             log;
    bool                     reentered = false;
    std::vector<std::string> messages;
};

static void log_callback(ggml_log_level, const char * text, void * user_data) {
    callback_context * context = static_cast<callback_context *>(user_data);
    context->messages.emplace_back(text);
    if (!context->reentered) {
        context->reentered = true;
        common_log_add(context->log, GGML_LOG_LEVEL_INFO, "nested\n");
    }
}

int main() {
    common_log *     log     = common_log_init();
    callback_context context = { log, false, {} };
    common_log_set_file(log, "test-log-output.txt");
    common_log_set_callback(log, log_callback, &context);

    common_log_add(log, GGML_LOG_LEVEL_INFO, "first\n");
    if (context.messages != std::vector<std::string>({ "first\n", "nested\n" })) {
        return 1;
    }

    common_log_pause(log);
    common_log_add(log, GGML_LOG_LEVEL_INFO, "discarded\n");
    if (context.messages.size() != 2) {
        return 2;
    }

    common_log_resume(log);
    common_log_add(log, GGML_LOG_LEVEL_INFO, "second\n");
    common_log_set_jsonl(true);
    const common_json json = {
        { "value", 1 }
    };
    common_log_add_json(log, "entry", json);
    common_log_set_jsonl(false);
    common_log_flush(log);
    common_log_free(log);

    const std::vector<std::string> expected = {
        "first\n",
        "nested\n",
        "second\n",
        "{\"type\":\"entry\",\"data\":{\"value\":1}}",
    };
    if (context.messages != expected) {
        return 3;
    }

    std::ifstream     output("test-log-output.txt");
    const std::string contents((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>());
    const std::string expected_contents =
        "first\n"
        "nested\n"
        "second\n"
        "{\"type\":\"entry\",\"data\":{\"value\":1}}\n";
    return contents == expected_contents ? 0 : 4;
}
#else
int main() {
    const int n_thread = 8;

    std::thread threads[n_thread];
    for (int i = 0; i < n_thread; i++) {
        threads[i] = std::thread([i]() {
            const int n_msg = 1000;

            for (int j = 0; j < n_msg; j++) {
                const int log_type = std::rand() % 4;

                switch (log_type) {
                    case 0: LOG_INF("Thread %d: %d\n", i, j); break;
                    case 1: LOG_WRN("Thread %d: %d\n", i, j); break;
                    case 2: LOG_ERR("Thread %d: %d\n", i, j); break;
                    case 3: LOG_DBG("Thread %d: %d\n", i, j); break;
                    default:
                        break;
                }

                if (rand () % 10 < 5) {
                    common_log_set_timestamps(common_log_main(), rand() % 2);
                    common_log_set_prefix    (common_log_main(), rand() % 2);
                }
            }
        });
    }

    for (int i = 0; i < n_thread; i++) {
        threads[i].join();
    }

    common_log_flush(common_log_main());
    // We explicitly free the logger singleton to avoid hanging on Windows
    // related to timing issues of thread startup and DLL teardown
    common_log_free(common_log_main());
    return 0;
}
#endif
