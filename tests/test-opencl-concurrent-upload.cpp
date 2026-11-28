#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr const char * kOpenclRegName = "OpenCL";
constexpr int64_t      kCols          = 512;
constexpr int64_t      kRows          = 512;
constexpr int          kTensors       = 16;
constexpr int          kRounds        = 8;
constexpr uint32_t     kPatternPrime  = 2654435761u;
constexpr uint32_t     kPatternRange  = 2001;
constexpr float        kPatternScale  = 1000.0f;

const ggml_type kTypes[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_BF16 };

float pattern_value(int64_t index, uint32_t seed) {
    const uint32_t mixed = (uint32_t(index) + seed)*kPatternPrime;
    return float(int32_t(mixed % kPatternRange) - int32_t(kPatternRange/2)) / kPatternScale;
}

std::vector<float> make_pattern(uint32_t seed) {
    std::vector<float> values(kCols*kRows);
    for (int64_t i = 0; i < kCols*kRows; ++i) {
        values[i] = pattern_value(i, seed);
    }
    return values;
}

std::vector<uint8_t> encode(ggml_type type, const std::vector<float> & values) {
    std::vector<uint8_t> bytes(ggml_row_size(type, kCols)*kRows);
    if (type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(values.data(), (ggml_bf16_t *) bytes.data(), int64_t(values.size()));
        return bytes;
    }
    ggml_quantize_chunk(type, values.data(), bytes.data(), 0, kRows, kCols, nullptr);
    return bytes;
}

std::vector<std::vector<uint8_t>> make_payloads(ggml_type type, int round) {
    std::vector<std::vector<uint8_t>> payloads(kTensors);
    for (int i = 0; i < kTensors; ++i) {
        payloads[i] = encode(type, make_pattern(uint32_t(round*kTensors + i)));
    }
    return payloads;
}

std::vector<ggml_tensor *> new_tensors(ggml_context * ctx, ggml_type type) {
    std::vector<ggml_tensor *> tensors(kTensors);
    for (int i = 0; i < kTensors; ++i) {
        tensors[i] = ggml_new_tensor_2d(ctx, type, kCols, kRows);
    }
    return tensors;
}

void wait_for_start(const std::atomic<bool> & start) {
    while (!start.load()) {
        std::this_thread::yield();
    }
}

std::vector<std::thread> start_upload_threads(const std::vector<ggml_tensor *> & tensors,
                                              const std::vector<std::vector<uint8_t>> & payloads,
                                              const std::atomic<bool> & start) {
    std::vector<std::thread> threads;
    for (int i = 0; i < kTensors; ++i) {
        threads.emplace_back([&, i]() {
            wait_for_start(start);
            ggml_backend_tensor_set(tensors[i], payloads[i].data(), 0, payloads[i].size());
        });
    }
    return threads;
}

void join_all(std::vector<std::thread> & threads) {
    for (std::thread & thread : threads) {
        thread.join();
    }
}

void upload_concurrently(const std::vector<ggml_tensor *> & tensors, const std::vector<std::vector<uint8_t>> & payloads) {
    std::atomic<bool> start(false);
    std::vector<std::thread> threads = start_upload_threads(tensors, payloads, start);
    start.store(true);
    join_all(threads);
}

int count_mismatches(const std::vector<ggml_tensor *> & tensors, const std::vector<std::vector<uint8_t>> & payloads) {
    int mismatches = 0;
    std::vector<uint8_t> readback;
    for (int i = 0; i < kTensors; ++i) {
        readback.resize(payloads[i].size());
        ggml_backend_tensor_get(tensors[i], readback.data(), 0, readback.size());
        mismatches += std::memcmp(readback.data(), payloads[i].data(), readback.size()) != 0;
    }
    return mismatches;
}

int run_round(ggml_backend_t backend, ggml_type type, int round) {
    ggml_init_params params = { ggml_tensor_overhead()*kTensors, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    const std::vector<ggml_tensor *> tensors = new_tensors(ctx, type);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const std::vector<std::vector<uint8_t>> payloads = make_payloads(type, round);
    upload_concurrently(tensors, payloads);
    const int mismatches = count_mismatches(tensors, payloads);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return mismatches;
}

int run_rounds(ggml_backend_t backend, ggml_type type) {
    int mismatches = 0;
    for (int round = 0; round < kRounds; ++round) {
        mismatches += run_round(backend, type, round);
    }
    return mismatches;
}

bool test_backend(ggml_backend_t backend) {
    bool ok = true;
    for (ggml_type type : kTypes) {
        const int mismatches = run_rounds(backend, type);
        std::printf("  %-6s %d of %d tensors read back wrong\n", ggml_type_name(type), mismatches, kTensors*kRounds);
        ok = ok && mismatches == 0;
    }
    return ok;
}

bool is_opencl(ggml_backend_dev_t dev) {
    return std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), kOpenclRegName) == 0;
}

int test_opencl_devices(bool & all_ok) {
    int tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!is_opencl(dev)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        std::printf("%s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        all_ok = test_backend(backend) && all_ok;
        ggml_backend_free(backend);
        ++tested;
    }
    return tested;
}

} // namespace

int main() {
    ggml_backend_load_all();
    bool all_ok = true;
    if (test_opencl_devices(all_ok) == 0) {
        std::puts("no OpenCL device, skipping");
        return 0;
    }
    std::puts(all_ok ? "concurrent upload test passed" : "concurrent upload test FAILED");
    return all_ok ? 0 : 1;
}
