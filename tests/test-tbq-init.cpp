#include "ggml.h"
#include "ggml-impl.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static constexpr size_t thread_count = 8;
static constexpr int64_t row_size = 128;
static constexpr std::array<ggml_type, 8> types = {
    GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0_64, GGML_TYPE_TBQ4_0_64,
    GGML_TYPE_PQ3_0, GGML_TYPE_PQ4_0, GGML_TYPE_PQ3_0_64, GGML_TYPE_PQ4_0_64,
};

struct result {
    std::vector<uint8_t> quantized;
    std::array<float, row_size> restored;
};

static std::array<float, row_size> make_input() {
    std::array<float, row_size> input;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(float(i));
    }
    return input;
}

static result quantize(ggml_type type, const std::array<float, row_size> & input) {
    result output;
    output.quantized.resize(ggml_row_size(type, row_size));
    const auto * traits = ggml_get_type_traits(type);
    traits->from_float_ref(input.data(), output.quantized.data(), row_size);
    traits->to_float(output.quantized.data(), output.restored.data(), row_size);
    return output;
}

static void quantize_all(std::array<result, types.size()> & output,
                         const std::array<float, row_size> & input) {
    for (size_t i = 0; i < types.size(); ++i) {
        output[i] = quantize(types[i], input);
    }
}

static bool compare(const std::array<std::array<result, types.size()>, thread_count> & outputs,
                    const std::array<float, row_size> & input) {
    for (size_t i = 0; i < types.size(); ++i) {
        const auto expected = quantize(types[i], input);
        for (const auto & output : outputs) {
            if (output[i].quantized != expected.quantized ||
                std::memcmp(output[i].restored.data(), expected.restored.data(), sizeof(expected.restored)) != 0) {
                std::fprintf(stderr, "Concurrent initialization changed %s output\n", ggml_type_name(types[i]));
                return false;
            }
        }
    }
    return true;
}

static void run_workers(std::array<std::array<result, types.size()>, thread_count> & outputs,
                        const std::array<float, row_size> & input) {
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::array<std::thread, thread_count> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] {
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            quantize_all(outputs[i], input);
        });
    }
    while (ready.load() != thread_count) {
        std::this_thread::yield();
    }
    start.store(true);
    for (auto & worker : workers) {
        worker.join();
    }
}

int main() {
    ggml_context * ctx = ggml_init({0, nullptr, true});
    if (!ctx) {
        return 1;
    }
    const auto input = make_input();
    std::array<std::array<result, types.size()>, thread_count> outputs;
    run_workers(outputs, input);
    const bool passed = compare(outputs, input);
    ggml_free(ctx);
    return passed ? 0 : 1;
}
