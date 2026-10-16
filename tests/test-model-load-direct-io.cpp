#include "llama-model-loader.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
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
    size_t first_elements;
    bool uniform;

    size_t elements(int i) const { return uniform || i == 0 ? first_elements : tensor_elements(i); }

    fixture(int n_splits, size_t first_elements = 256, bool uniform = false) : first_elements(first_elements), uniform(uniform) {
        for (int split = 0; split < n_splits; ++split) {
            char path[] = "test-model-load-direct-io-XXXXXX";
            int fd = mkstemp(path);
            check(fd >= 0, "mkstemp failed");
            close(fd);
            paths.emplace_back(path);
            ggml_context_ptr ctx(ggml_init({1024 * 1024 + first_elements * sizeof(float) * (uniform ? n_tensors : 1), nullptr, false}));
            gguf_context_ptr meta(gguf_init_empty());
            gguf_set_val_str(meta.get(), "general.architecture", "llama");
            if (n_splits > 1) {
                gguf_set_val_u16(meta.get(), "split.no", split);
                gguf_set_val_u16(meta.get(), "split.count", n_splits);
                gguf_set_val_i32(meta.get(), "split.tensors.count", n_tensors);
            }
            for (int i = split * n_tensors / n_splits; i < (split + 1) * n_tensors / n_splits; ++i) {
                auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, elements(i));
                ggml_set_name(tensor, tensor_name(i).c_str());
                auto * data = static_cast<float *>(tensor->data);
                for (size_t j = 0; j < elements(i); ++j) {
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

struct guarded_host_buffer {
    std::vector<uint8_t> bytes;
    uint8_t * base;
    size_t size;

    guarded_host_buffer(size_t size, size_t alignment) : bytes(size + 3 * alignment, 0xa5), size(size) {
        const uintptr_t aligned = (reinterpret_cast<uintptr_t>(bytes.data()) + alignment - 1) & ~(alignment - 1);
        base = reinterpret_cast<uint8_t *>(aligned) + 1;
    }

    bool guards_intact() const {
        return std::all_of(bytes.data(), static_cast<const uint8_t *>(base), [](uint8_t b) { return b == 0xa5; }) &&
               std::all_of(static_cast<const uint8_t *>(base) + size, bytes.data() + bytes.size(), [](uint8_t b) { return b == 0xa5; });
    }
};

struct async_upload_state {
    ggml_backend_device device = {};
    ggml_backend_buffer_type host_buft = {};
    ggml_backend_buffer_type_t device_buft;
    std::vector<std::unique_ptr<guarded_host_buffer>> host_buffers;
    size_t alignment;
    size_t uploads = 0;

    async_upload_state(ggml_backend_buffer_type_t buft, size_t alignment) : device_buft(buft), alignment(alignment) {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "test-async-upload"; };
        device.iface.get_props = [](ggml_backend_dev_t, ggml_backend_dev_props * props) {
            *props = {};
            props->caps.async = props->caps.host_buffer = props->caps.events = true;
        };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return static_cast<async_upload_state *>(dev->context)->device_buft;
        };
        device.iface.get_host_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<async_upload_state *>(dev->context)->host_buft;
        };
        device.iface.event_new = [](ggml_backend_dev_t dev) { return new ggml_backend_event{dev, nullptr}; };
        device.iface.event_free = [](ggml_backend_dev_t, ggml_backend_event_t event) { delete event; };
        device.iface.event_synchronize = [](ggml_backend_dev_t, ggml_backend_event_t) {};
        device.iface.init_backend = [](ggml_backend_dev_t dev, const char *) {
            auto * backend = new ggml_backend{};
            backend->device = dev;
            backend->context = dev->context;
            backend->iface.get_name = [](ggml_backend_t) { return "test-async-upload"; };
            backend->iface.free = [](ggml_backend_t backend) { delete backend; };
            backend->iface.synchronize = [](ggml_backend_t) {};
            backend->iface.event_record = [](ggml_backend_t, ggml_backend_event_t) {};
            backend->iface.set_tensor_async = [](ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
                auto & state = *static_cast<async_upload_state *>(backend->context);
                ++state.uploads;
                memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
            };
            return backend;
        };
        host_buft.context = this;
        host_buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "test-misaligned-host"; };
        host_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t size) {
            auto & state = *static_cast<async_upload_state *>(buft->context);
            state.host_buffers.emplace_back(std::make_unique<guarded_host_buffer>(size, state.alignment));
            ggml_backend_buffer_i iface = {};
            iface.get_base = [](ggml_backend_buffer_t buffer) -> void * {
                return static_cast<guarded_host_buffer *>(buffer->context)->base;
            };
            return ggml_backend_buffer_init(buft, iface, state.host_buffers.back().get(), size);
        };
    }
};

struct staging_usage {
    std::mutex mutex;
    std::map<std::thread::id, size_t> capacities;

    size_t retained_bytes() const {
        size_t total = 0;
        for (const auto & entry : capacities) {
            total += entry.second;
        }
        return total;
    }
};

struct tracked_file : llama_file_disk {
    staging_usage & usage;

    tracked_file(const char * path, staging_usage & usage) : llama_file_disk(path, "rb", true), usage(usage) {}

    size_t read_raw_unsafe_at(void * ptr, size_t len, size_t offset) const override {
        {
            std::lock_guard<std::mutex> lock(usage.mutex);
            auto & capacity = usage.capacities[std::this_thread::get_id()];
            capacity = std::max(capacity, len);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return llama_file_disk::read_raw_unsafe_at(ptr, len, offset);
    }
};

static void test_load(fixture & f, int threads, bool validate, bool async = false, bool track_staging = false) {
    setenv("LLAMA_LOAD_THREADS", std::to_string(threads).c_str(), 1);
    llama_model_loader loader(nullptr, nullptr, nullptr,
            load_input_variant::fname_load_input{f.paths.front(), f.paths}, nullptr,
            LLAMA_LOAD_MODE_DIRECT_IO, validate, false, false, nullptr, nullptr);
    loader.init_mappings(false, nullptr);
    staging_usage usage;
    if (track_staging) {
        for (size_t i = 0; i < f.paths.size(); ++i) {
            loader.files[i] = std::make_unique<tracked_file>(f.paths[i].c_str(), usage);
        }
    }
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * n_tensors, nullptr, true}));

    upload_state state;
    state.bytes.resize(loader.size_data);
    ggml_backend_buffer_type buft = {};
    buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "test-serial-upload"; };
    llama_file_disk probe(f.paths.front().c_str(), "rb", true);
    async_upload_state async_state(&buft, probe.read_alignment());
    if (async) {
        check(probe.read_alignment() > 1, "direct-IO alignment is required for this test");
        buft.device = &async_state.device;
    }
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
    if (track_staging) {
        fprintf(stderr, "staging: %zu workers retained %zu bytes\n", usage.capacities.size(), usage.retained_bytes());
        if (f.first_elements * sizeof(float) > 256 * 1024 * 1024) {
            check(usage.capacities.size() == 1, "oversized tensor did not limit staging to one worker");
        } else {
            check(usage.capacities.size() > 1, "staging test did not exercise parallel reads");
            check(usage.retained_bytes() <= 256 * 1024 * 1024, "parallel staging exceeded the shared memory budget");
        }
    }
    if (async) {
        check(async_state.uploads > n_tensors, "async loader did not split the large tensor into chunks");
        check(!async_state.host_buffers.empty(), "async staging buffers were not allocated");
        for (const auto & host : async_state.host_buffers) {
            check(host->guards_intact(), "async read exceeded the host-buffer allocation");
        }
    }
    check(!state.overlap, "backend uploads ran concurrently");
    if (threads > 1 && !validate && !track_staging) {
        check(state.threads.size() > 1, "parallel loader was not exercised");
    }
    for (int i = 0; i < n_tensors; ++i) {
        const auto * tensor = ggml_get_tensor(ctx.get(), tensor_name(i).c_str());
        const auto * data = static_cast<const float *>(tensor->data);
        for (size_t j = 0; j < f.elements(i); ++j) {
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
        fixture large(1, 20 * 1024 * 1024);
        test_load(large, 1, false, true);
        fixture parallel(1, 6 * 1024 * 1024, true);
        test_load(parallel, 32, false, false, true);
        fixture oversized(1, 68 * 1024 * 1024);
        test_load(oversized, 32, false, false, true);
        fprintf(stderr, "PASS: direct-IO tensor data, serialized uploads, misaligned async buffers and bounded parallel staging\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
