#pragma once

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void require(bool value, const std::string & message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static void read_at(const std::string & path, size_t offset, void * data, size_t size) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "could not open " + path);
    input.seekg(offset);
    input.read(static_cast<char *>(data), size);
    require(bool(input), "short read from " + path);
}

struct metal_ffn {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weights = nullptr;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_backend_buffer_t residual_buffer = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * residual = nullptr;
    int64_t channels = 0;
    void * host_input = nullptr;
    void * host_residual = nullptr;

    ~metal_ffn() {
        if (backend) { ggml_backend_synchronize(backend); }
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(input_buffer);
        ggml_backend_buffer_free(residual_buffer);
        ggml_backend_buffer_free(weights);
        ggml_free(graph_ctx);
        ggml_free(weight_ctx);
        ggml_backend_free(backend);
        free(host_input);
        free(host_residual);
    }

    void init(const std::string & gguf_path, const std::string & assets, int layer, int tokens, bool f16, int leading_channels = 0, bool add_residual = false, bool mapped_inputs = false) {
        require(!f16 || leading_channels == 0, "partial Metal weights require GGUF quantization");
        auto reg = ggml_backend_reg_by_name("MTL");
        require(reg && ggml_backend_reg_dev_count(reg) > 0, "Metal device unavailable");
        backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        require(backend, "Metal backend unavailable");
        ggml_context * header_ctx = nullptr;
        gguf_context * header = gguf_init_from_file(gguf_path.c_str(), {true, &header_ctx});
        require(header && header_ctx, "could not read GGUF header");
        weight_ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
        require(weight_ctx, "could not allocate weight context");
        ggml_tensor * w[3];
        int64_t ids[3];
        size_t source_row_bytes[3];
        const char * kinds[] = {"gate", "up", "down"};
        for (int i = 0; i < 3; ++i) {
            std::string name = "blk." + std::to_string(layer) + ".ffn_" + kinds[i] + ".weight";
            ids[i] = gguf_find_tensor(header, name.c_str());
            auto * source = ggml_get_tensor(header_ctx, name.c_str());
            require(ids[i] >= 0 && source && ggml_n_dims(source) == 2, "missing matrix " + name);
            require(source->ne[0] <= 32768 && source->ne[1] <= 32768, "matrix too large");
            int64_t ne0 = source->ne[0], ne1 = source->ne[1];
            source_row_bytes[i] = ggml_row_size(source->type, ne0);
            if (leading_channels) {
                require(leading_channels > 0 && leading_channels < (i == 2 ? ne0 : ne1), "invalid Metal channel split");
                if (i == 2) {
                    require(leading_channels % ggml_blck_size(source->type) == 0, "split is not quantization-block aligned");
                    ne0 = leading_channels;
                } else { ne1 = leading_channels; }
            }
            w[i] = ggml_new_tensor_2d(weight_ctx, f16 ? GGML_TYPE_F16 : source->type, ne0, ne1);
        }
        require(w[0]->ne[0] == w[1]->ne[0] && w[0]->ne[1] == w[1]->ne[1] &&
                w[2]->ne[0] == w[0]->ne[1] && w[2]->ne[1] == w[0]->ne[0], "incompatible SwiGLU dimensions");
        channels = w[0]->ne[1];
        weights = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
        require(weights, "could not allocate Metal weights");
        ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < 3; ++i) {
            std::vector<char> bytes(ggml_nbytes(w[i]));
            size_t offset = gguf_get_data_offset(header) + gguf_get_tensor_offset(header, ids[i]);
            if (leading_channels && i == 2) {
                size_t row_bytes = ggml_row_size(w[i]->type, w[i]->ne[0]);
                std::vector<char> full(source_row_bytes[i] * w[i]->ne[1]);
                read_at(gguf_path, offset, full.data(), full.size());
                for (int64_t row = 0; row < w[i]->ne[1]; ++row) {
                    memcpy(bytes.data() + row * row_bytes, full.data() + row * source_row_bytes[i], row_bytes);
                }
            } else {
                read_at(f16 ? assets + "/" + kinds[i] + ".f16" : gguf_path, f16 ? 0 : offset, bytes.data(), bytes.size());
            }
            ggml_backend_tensor_set(w[i], bytes.data(), 0, bytes.size());
        }
        gguf_free(header);
        ggml_free(header_ctx);
        graph_ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true});
        require(graph_ctx, "could not allocate graph context");
        input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, w[0]->ne[0], tokens);
        ggml_set_input(input);
        auto allocate_input = [&](ggml_tensor * tensor, void * & host) {
            size_t size = ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(backend), tensor);
            if (!mapped_inputs) { return ggml_backend_alloc_buffer(backend, size); }
            const size_t page = size_t(sysconf(_SC_PAGESIZE));
            size = (size + page - 1) / page * page;
            require(posix_memalign(&host, page, size) == 0, "could not allocate mapped input storage");
            return ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(backend), host, size, size);
        };
        input_buffer = allocate_input(input, host_input);
        require(input_buffer && ggml_backend_tensor_alloc(input_buffer, input, ggml_backend_buffer_get_base(input_buffer)) == GGML_STATUS_SUCCESS,
                "could not allocate persistent Metal input");
        if (add_residual) {
            residual = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, w[0]->ne[0], tokens);
            ggml_set_input(residual);
            residual_buffer = allocate_input(residual, host_residual);
            require(residual_buffer && ggml_backend_tensor_alloc(residual_buffer, residual, ggml_backend_buffer_get_base(residual_buffer)) == GGML_STATUS_SUCCESS, "could not allocate Metal residual input");
        }
        auto * gate = ggml_mul_mat(graph_ctx, w[0], input);
        auto * up = ggml_mul_mat(graph_ctx, w[1], input);
        output = ggml_mul_mat(graph_ctx, w[2], ggml_swiglu_split(graph_ctx, gate, up));
        if (residual) { output = ggml_add(graph_ctx, output, residual); }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(graph_ctx, 32, false);
        ggml_build_forward_expand(graph, output);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc, graph), "could not allocate Metal graph");
    }

    void predict(const std::vector<float> & x, std::vector<float> & y) {
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Metal compute failed");
        ggml_backend_tensor_get(output, y.data(), 0, y.size() * sizeof(float));
    }
};

static void measure(const char * name, int tokens, int reps, const std::function<void()> & run) {
    const char * gap = std::getenv("FFN_COREML_IDLE_MS");
    int idle_ms = gap ? std::stoi(gap) : 0;
    require(idle_ms >= 0 && idle_ms <= 10000, "idle gap must be in 0..10000 ms");
    for (int i = 0; i < 3; ++i) { run(); }
    std::vector<double> times;
    for (int i = 0; i < reps; ++i) {
        if (idle_ms) { std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms)); }
        auto start = std::chrono::steady_clock::now();
        run();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::sort(times.begin(), times.end());
    printf("backend=%s tokens=%d mean_ms=%.6f median_ms=%.6f tokens_per_second=%.3f idle_ms=%d\n",
            name, tokens, mean, times[times.size() / 2], tokens * 1000.0 / mean, idle_ms);
    fflush(stdout);
}

struct metal_boundary {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ~metal_boundary() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    void init(ggml_backend_t backend, int hidden, int tokens, const std::vector<float> & x) {
        ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
        require(ctx, "could not create boundary context");
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer && input->data != output->data, "could not allocate separate boundary buffers");
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    }
};

struct metal_join {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * gpu = nullptr;
    ggml_tensor * ane = nullptr;
    ggml_tensor * ane2 = nullptr;
    ggml_tensor * output = nullptr;

    ~metal_join() {
        if (backend) { ggml_backend_synchronize(backend); }
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(inputs);
        ggml_free(ctx);
    }

    // second: add a separate input for a second Core ML partial
    void init(ggml_backend_t device, int hidden, int tokens, bool second = false) {
        backend = device;
        ctx = ggml_init({ggml_tensor_overhead() * 5 + ggml_graph_overhead_custom(8, false), nullptr, true});
        require(ctx, "could not create Metal join context");
        gpu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        ane = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        ggml_set_input(gpu);
        ggml_set_input(ane);
        if (second) {
            ane2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
            ggml_set_input(ane2);
        }
        inputs = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(inputs, "could not allocate Metal join inputs");
        ggml_backend_buffer_clear(inputs, 0);
        output = ggml_add(ctx, gpu, ane);
        if (ane2) { output = ggml_add(ctx, output, ane2); }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(graph, output);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(alloc, graph), "could not allocate Metal join graph");
        require(output->data != gpu->data && output->data != ane->data && (!ane2 || output->data != ane2->data), "Metal join inputs alias output");
    }

    void run(ggml_tensor * prefix, const std::vector<float> & partial) {
        ggml_backend_tensor_copy(prefix, gpu);
        ggml_backend_tensor_set(ane, partial.data(), 0, partial.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Metal join failed");
    }
};

static bool compare(const char * name, const std::vector<float> & reference, const std::vector<float> & actual) {
    double dot = 0, r2 = 0, a2 = 0, err2 = 0;
    float max_error = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(actual[i]), "non-finite FFN output");
        double delta = double(reference[i]) - actual[i];
        dot += double(reference[i]) * actual[i];
        r2 += double(reference[i]) * reference[i];
        a2 += double(actual[i]) * actual[i];
        err2 += delta * delta;
        max_error = std::max(max_error, float(std::abs(delta)));
    }
    double cosine = dot / std::sqrt(r2 * a2);
    double relative_l2 = std::sqrt(err2 / r2);
    printf("accuracy=%s cosine=%.9f relative_l2=%.9f max_abs=%.9f\n", name, cosine, relative_l2, max_error);
    bool passed = cosine > 0.999 && relative_l2 < 0.02;
    printf("accuracy_pass=%s value=%d\n", name, int(passed));
    return passed;
}

static bool measure_split(const std::string & gguf, const std::string & assets, int layer, int tokens, int reps, int gpu_channels,
                          metal_ffn & full, std::vector<float> & x, const std::vector<float> & reference,
                          std::vector<float> & actual, const std::function<void()> & predict) {
    metal_ffn prefix;
    prefix.init(gguf, assets, layer, tokens, false, gpu_channels);
    ggml_backend_tensor_set(prefix.input, x.data(), 0, x.size() * sizeof(float));
    metal_join join;
    join.init(prefix.backend, x.size() / tokens, tokens);
    require(ggml_backend_graph_compute(prefix.backend, prefix.graph) == GGML_STATUS_SUCCESS, "Metal prefix warm-up failed");
    predict();
    measure("split-metal-full-co-resident", tokens, reps, [&]() {
        require(ggml_backend_graph_compute(full.backend, full.graph) == GGML_STATUS_SUCCESS, "co-resident Metal compute failed");
    });
    ggml_backend_tensor_get(full.output, actual.data(), 0, actual.size() * sizeof(float));
    require(compare("metal-co-resident-vs-reference", reference, actual), "co-resident Metal output changed");
    measure("split-metal-prefix", tokens, reps, [&]() {
        require(ggml_backend_graph_compute(prefix.backend, prefix.graph) == GGML_STATUS_SUCCESS, "Metal prefix failed");
    });
    measure("split-coreml-suffix", tokens, reps, predict);
    auto run = [&](bool parallel) {
        ggml_backend_synchronize(prefix.backend);
        ggml_backend_tensor_get(prefix.input, x.data(), 0, x.size() * sizeof(float));
        require(ggml_backend_graph_compute_async(prefix.backend, prefix.graph) == GGML_STATUS_SUCCESS, "Metal prefix submission failed");
        if (!parallel) { ggml_backend_synchronize(prefix.backend); }
        predict();
        ggml_backend_synchronize(prefix.backend);
        join.run(prefix.output, actual);
    };
    measure("split-serial", tokens, reps, [&]() { run(false); });
    ggml_backend_tensor_get(join.output, actual.data(), 0, actual.size() * sizeof(float));
    bool ok = compare("split-serial", reference, actual);
    const auto serial = actual;
    measure("split-parallel", tokens, reps, [&]() { run(true); });
    ggml_backend_tensor_get(join.output, actual.data(), 0, actual.size() * sizeof(float));
    if (!compare("split-parallel", reference, actual)) { ok = false; }
    bool identical = serial == actual;
    printf("concurrency_identity=%d\n", int(identical));
    if (!identical) { compare("split-serial-vs-parallel", serial, actual); }
    return ok && identical;
}

static void print_memory(const char * backend, const char * stage) {
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        printf("memory=%s stage=%s footprint_mib=%.3f resident_mib=%.3f\n",
                backend, stage, info.phys_footprint / 1048576.0, info.resident_size / 1048576.0);
    }
}

struct temporary_model {
    NSURL * url;
    ~temporary_model() { if (url) { [[NSFileManager defaultManager] removeItemAtURL:url error:nil]; } }
};

static void count_devices(MLComputePlan * plan, MLModelStructureProgramBlock * block, int & total, int & ane, int & gpu) {
    for (MLModelStructureProgramOperation * op in block.operations) {
        NSString * kind = [op.operatorName componentsSeparatedByString:@"."].lastObject;
        if ([kind isEqualToString:@"const"] || [kind hasPrefix:@"constexpr_"]) { continue; }
        ++total;
        auto * device = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
        if (std::getenv("FFN_COREML_PLAN_DETAIL")) {
            NSString * name = device ? NSStringFromClass([device class]) : @"none";
            printf("placement_op=%s preferred=%s\n", op.operatorName.UTF8String, name.UTF8String);
        }
        if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) { ++ane; }
        if ([device isKindOfClass:[MLGPUComputeDevice class]]) { ++gpu; }
        for (MLModelStructureProgramBlock * nested in op.blocks) { count_devices(plan, nested, total, ane, gpu); }
    }
}

static void report_placement(NSURL * url, MLModelConfiguration * config, const char * name) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:url configuration:config completionHandler:^(MLComputePlan * plan, NSError * plan_error) {
        int total = 0, ane = 0, gpu = 0;
        if (plan) {
            count_devices(plan, plan.modelStructure.program.functions[@"main"].block, total, ane, gpu);
            printf("placement=%s operations=%d preferred_ane=%d preferred_gpu=%d\n", name, total, ane, gpu);
        } else {
            fprintf(stderr, "placement=%s error=%s\n", name, plan_error ? plan_error.localizedDescription.UTF8String : "unknown plan error");
        }
        dispatch_semaphore_signal(done);
    }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
}
