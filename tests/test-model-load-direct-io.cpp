#include "llama-model-loader.h"
#include "ggml-backend-impl.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include <unistd.h>

static constexpr int n_tensors = 12;

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static std::string tensor_name(int i) { return "blk." + std::to_string(i) + ".weight"; }
static size_t tensor_elements(int i) { return 256 + 8 * i; }

struct fixture {
    std::vector<std::string> paths;

    fixture(int n_splits) {
        for (int split = 0; split < n_splits; ++split) {
            char path[] = "test-model-load-direct-io-XXXXXX";
            int fd = mkstemp(path);
            check(fd >= 0, "mkstemp failed");
            close(fd);
            paths.emplace_back(path);
            ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, false}));
            gguf_context_ptr meta(gguf_init_empty());
            gguf_set_val_str(meta.get(), "general.architecture", "llama");
            if (n_splits > 1) {
                gguf_set_val_u16(meta.get(), "split.no", split);
                gguf_set_val_u16(meta.get(), "split.count", n_splits);
                gguf_set_val_i32(meta.get(), "split.tensors.count", n_tensors);
            }
            for (int i = split * n_tensors / n_splits; i < (split + 1) * n_tensors / n_splits; ++i) {
                auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, tensor_elements(i));
                ggml_set_name(tensor, tensor_name(i).c_str());
                auto * data = static_cast<float *>(tensor->data);
                for (size_t j = 0; j < tensor_elements(i); ++j) {
                    data[j] = static_cast<float>(i * 1000 + j);
                }
                gguf_add_tensor(meta.get(), tensor);
            }
            check(gguf_write_to_file(meta.get(), path, false), "GGUF write failed");
        }
    }

    ~fixture() {
        for (const auto & path : paths) {
            unlink(path.c_str());
        }
    }
};

struct upload_state {
    std::vector<uint8_t> bytes;
    std::atomic<int> active{0};
    std::atomic<bool> overlap{false};
    std::mutex mutex;
    std::set<std::thread::id> threads;
};

static void set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto & state = *static_cast<upload_state *>(buffer->context);
    if (state.active.fetch_add(1) != 0) {
        state.overlap = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.threads.insert(std::this_thread::get_id());
    }
    state.active.fetch_sub(1);
}

static void test_load(fixture & f, int threads, bool validate) {
    setenv("LLAMA_LOAD_THREADS", std::to_string(threads).c_str(), 1);
    llama_model_loader loader(nullptr, nullptr, nullptr,
            load_input_variant::fname_load_input{f.paths.front(), f.paths}, nullptr,
            LLAMA_LOAD_MODE_DIRECT_IO, validate, false, false, nullptr, nullptr);
    loader.init_mappings(false, nullptr);
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * n_tensors, nullptr, true}));

    upload_state state;
    state.bytes.resize(loader.size_data);
    ggml_backend_buffer_type buft = {};
    buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "test-serial-upload"; };
    ggml_backend_buffer_i iface = {};
    iface.get_base = [](ggml_backend_buffer_t buffer) -> void * {
        return static_cast<upload_state *>(buffer->context)->bytes.data();
    };
    iface.set_tensor = set_tensor;
    ggml_backend_buffer_ptr buffer(ggml_backend_buffer_init(&buft, iface, &state, state.bytes.size()));
    size_t offset = 0;
    for (int i = 0; i < n_tensors; ++i) {
        auto * tensor = ggml_dup_tensor(ctx.get(), loader.get_tensor_meta(tensor_name(i).c_str()));
        ggml_set_name(tensor, tensor_name(i).c_str());
        check(ggml_backend_tensor_alloc(buffer.get(), tensor, state.bytes.data() + offset) == GGML_STATUS_SUCCESS,
              "tensor allocation failed");
        offset += ggml_nbytes(tensor);
    }
    llama_buf_map buffers = {{0, buffer.get()}};
    check(loader.load_all_data(loader.size_data, ctx.get(), buffers, nullptr, nullptr, nullptr), "load failed");
    check(!state.overlap, "backend uploads ran concurrently");
    if (threads > 1 && !validate) {
        check(state.threads.size() > 1, "parallel loader was not exercised");
    }
    for (int i = 0; i < n_tensors; ++i) {
        const auto * tensor = ggml_get_tensor(ctx.get(), tensor_name(i).c_str());
        const auto * data = static_cast<const float *>(tensor->data);
        for (size_t j = 0; j < tensor_elements(i); ++j) {
            check(data[j] == static_cast<float>(i * 1000 + j), "loaded tensor data mismatch");
        }
    }
}

int main() {
    try {
        for (int splits : {1, 2}) {
            fixture f(splits);
            llama_file_disk probe(f.paths.front().c_str(), "rb", true);
            if (!probe.has_direct_io()) {
                fprintf(stderr, "SKIP: filesystem does not support direct IO\n");
                return 77;
            }
            for (int threads : {1, 4, 32}) {
                test_load(f, threads, false);
            }
            test_load(f, 4, true);
        }
        fprintf(stderr, "PASS: direct-IO tensor data and serialized uploads for single and split GGUFs\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
