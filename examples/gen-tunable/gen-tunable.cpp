#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t n_threads = std::thread::hardware_concurrency();
        // static RNG initialization (revisit if n_threads stops being constant)
        static std::vector<std::default_random_engine> generators = []() {
            std::random_device rd;
            std::vector<std::default_random_engine> vec;
            vec.reserve(n_threads);
            //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
            for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
            return vec;
        }();

        auto init_thread = [&](size_t ith, size_t start, size_t end) {
            std::uniform_real_distribution<float> distribution(min, max);
            auto & gen = generators[ith];
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        std::vector<std::future<void>> tasks;
        tasks.reserve(n_threads);
        for (size_t i = 0; i < n_threads; i++) {
            size_t start =     i*nels/n_threads;
            size_t end   = (i+1)*nels/n_threads;
            tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
        }
        for (auto & t : tasks) {
            t.get();
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

         // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
                    start * blck_size, end - start, blck_size, im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_threads = std::min<size_t>(std::thread::hardware_concurrency()/2,
                                                      std::max<size_t>(1, n_blocks / min_blocks_per_thread));
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start =     i*n_blocks/n_threads;
                size_t end   = (i+1)*n_blocks/n_threads;
                tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
            }
            for (auto & t : tasks) {
                t.get();
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific values are again not really meaningful.
        const size_t nbytes_half = ggml_nbytes(tensor)/2;
        ggml_backend_tensor_set(tensor, data.data(), 0*nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1*nbytes_half, nbytes_half);
    } else {
        GGML_ABORT("fatal error");
    }
}

// utils for printing the variables of the test cases

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_prec prec) {
    return prec == GGML_PREC_F32 ? "f32" : "def";
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:  return "avg";
        case GGML_OP_POOL_MAX:  return "max";
        default:                return std::to_string(pool);
    }
}

static std::string var_to_str(ggml_scale_mode mode) {
    switch (mode) {
        case GGML_SCALE_MODE_NEAREST:  return "nearest";
        case GGML_SCALE_MODE_BILINEAR: return "bilinear";
        default:                      return std::to_string(mode);
    }
}

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)
#define VARS_TO_STR13(a, b, c, d, e, f, g, h, i, j, k, l, m) VAR_TO_STR(a) + "," + VARS_TO_STR12(b, c, d, e, f, g, h, i, j, k, l, m)
#define VARS_TO_STR14(a, b, c, d, e, f, g, h, i, j, k, l, m, n) VAR_TO_STR(a) + "," + VARS_TO_STR13(b, c, d, e, f, g, h, i, j, k, l, m, n)
#define VARS_TO_STR15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) VAR_TO_STR(a) + "," + VARS_TO_STR14(b, c, d, e, f, g, h, i, j, k, l, m, n, o)
#define VARS_TO_STR16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) VAR_TO_STR(a) + "," + VARS_TO_STR15(b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum test_mode {
    MODE_PERF,
};

// Output format support similar to llama-bench
enum output_formats { CONSOLE, SQL, CSV };

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "console") {
        format = CONSOLE;
    } else {
        return false;
    }
    return true;
}

// Test result structure for SQL output
struct test_result {
    std::string backend_name;
    std::string op_name;
    std::string op_params;
    json        op_params_json;
    std::string test_time;
    std::string build_commit;
    bool        supported;
    bool        passed;
    std::string error_message;
    double      time_us;
    int         n_runs;

    test_result() {
        // Initialize with default values
        time_us        = 0.0;
        n_runs         = 0;
        supported      = false;
        passed         = false;

        // Set test time
        time_t t = time(NULL);
        char   buf[32];
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        // Set build info
        build_commit = ggml_commit();
    }

    test_result(const std::string & backend_name, const std::string & op_name,
                const std::string & op_params, const json & op_params_json,
                bool supported, bool passed, const std::string & error_message = "",
                double time_us = 0.0, int n_runs = 0) :
        backend_name(backend_name),
        op_name(op_name),
        op_params(op_params),
        op_params_json(op_params_json),
        supported(supported),
        passed(passed),
        error_message(error_message),
        time_us(time_us),
        n_runs(n_runs) {
        // Set test time
        time_t t = time(NULL);
        char   buf[32];
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        // Set build info
        build_commit = ggml_commit();
    }
};

// Printer classes for different output formats
enum class test_status_t { NOT_SUPPORTED, OK, FAIL };

struct testing_start_info {
    size_t device_count;

    testing_start_info() = default;

    testing_start_info(size_t device_count) : device_count(device_count) {}
};

struct backend_init_info {
    size_t      device_index;
    size_t      total_devices;
    std::string device_name;
    bool        skipped = false;
    std::string skip_reason;
    std::string description;
    size_t      memory_total_mb = 0;
    size_t      memory_free_mb  = 0;
    bool        has_memory_info = false;

    backend_init_info() = default;

    backend_init_info(size_t device_index, size_t total_devices, const std::string & device_name, bool skipped = false,
                      const std::string & skip_reason = "", const std::string & description = "",
                      size_t memory_total_mb = 0, size_t memory_free_mb = 0, bool has_memory_info = false) :
        device_index(device_index),
        total_devices(total_devices),
        device_name(device_name),
        skipped(skipped),
        skip_reason(skip_reason),
        description(description),
        memory_total_mb(memory_total_mb),
        memory_free_mb(memory_free_mb),
        has_memory_info(has_memory_info) {}
};

struct backend_status_info {
    std::string   backend_name;
    test_status_t status;

    backend_status_info() = default;

    backend_status_info(const std::string & backend_name, test_status_t status) :
        backend_name(backend_name),
        status(status) {}
};

struct overall_summary_info {
    size_t backends_passed;
    size_t backends_total;
    bool   all_passed;

    overall_summary_info() = default;

    overall_summary_info(size_t backends_passed, size_t backends_total, bool all_passed) :
        backends_passed(backends_passed),
        backends_total(backends_total),
        all_passed(all_passed) {}
};

struct printer {
    virtual ~printer() {}

    FILE * fout = stdout;

    virtual void print_header() {}

    virtual void print_footer() {}

    virtual void print_testing_start(const testing_start_info & info) { (void) info; }

    virtual void print_backend_init(const backend_init_info & info) { (void) info; }

    virtual void print_backend_status(const backend_status_info & info) { (void) info; }

    virtual void print_overall_summary(const overall_summary_info & info) { (void) info; }

    virtual void print_test_result(const test_result & result) = 0;

    virtual void print_tunable_config(const char *path_output_tunable_config) { (void) path_output_tunable_config; }

    std::vector<test_result> test_results;
};

struct console_printer : public printer {
    void print_test_result(const test_result & result) override {
        print_perf_console(result);
    }

    void print_backend_status(const backend_status_info & info) override {
        printf("  Backend %s: ", info.backend_name.c_str());
        if (info.status == test_status_t::OK) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_testing_start(const testing_start_info & info) override {
        printf("Testing %zu devices\n\n", info.device_count);
    }

    void print_backend_init(const backend_init_info & info) override {
        printf("Backend %zu/%zu: %s\n", info.device_index + 1, info.total_devices, info.device_name.c_str());

        if (info.skipped) {
            printf("  %s\n", info.skip_reason.c_str());
            return;
        }

        if (!info.description.empty()) {
            printf("  Device description: %s\n", info.description.c_str());
        }

        if (info.has_memory_info) {
            printf("  Device memory: %zu MB (%zu MB free)\n", info.memory_total_mb, info.memory_free_mb);
        }

        printf("\n");
    }

    void print_overall_summary(const overall_summary_info & info) override {
        printf("%zu/%zu backends passed\n", info.backends_passed, info.backends_total);
        if (info.all_passed) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_tunable_config(const char *path_output_tunable_config) override {
        std::map<std::string, test_result> best_results;
        for (const test_result &result : test_results) {
            test_result best_result;
            bool found = false;
            std::string key = result.op_name + result.op_params;

            if (best_results.find(key) != best_results.end()) {
                best_result = best_results[key];
                found = true;
            }

            if (result.passed) {
                if (found) {
                    if (result.time_us > 0) {
                        if (result.time_us < best_result.time_us) {
                            best_results[key] = result;
                        }
                    }
                } else {
                    best_results[key] = result;
                }
            }
        }

        // print

        json tunable_ops_j;
        json tunable_op_j;
        for (const auto &it : best_results) {
            json op_config_j;
            test_result result = it.second;

            op_config_j["constraints"] = result.op_params_json;
            op_config_j["backend"] = result.backend_name.c_str();

            if (tunable_op_j.contains(result.op_name)) {
                tunable_op_j[result.op_name].push_back(op_config_j);
            } else {
                std::vector<json> op_configs_j;
                op_configs_j.push_back(op_config_j);
                tunable_op_j[result.op_name] = op_configs_j;
            }

            if (tunable_ops_j.contains("tunable_ops")) {
                tunable_ops_j["tunable_ops"].update(tunable_op_j, true);
            } else {
                tunable_ops_j["tunable_ops"] = tunable_op_j;
            }
        }

        printf("Saving tunable config %s\n", path_output_tunable_config);
        std::ofstream out_tunable_config_file(path_output_tunable_config);
        out_tunable_config_file << tunable_ops_j.dump(2);
    }

  private:
    void print_perf_console(const test_result & result) {
        int len = printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (!result.supported) {
            printf("not supported\n");
            return;
        }

        // align while also leaving some margin for variations in parameters
        int align = 8;
        int last  = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        printf("%*s", last - len, "");

        printf("    %8d runs - %8.2f us/run\n", result.n_runs, result.time_us);
    }
};

static std::unique_ptr<printer> create_printer() {
    return std::make_unique<console_printer>();
}

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual json to_json() {
        return json();
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        return t;
    }

    // Checks an op against the test filter, which is a comma separated list of OP names or specific variations
    bool matches_filter(ggml_tensor * op, const char * op_names_filter) {
        if (op_names_filter) {
            const auto op_name = op_desc(op);
            const auto op_full_name = op_name + "(" + vars() + ")";
            std::string_view filter(op_names_filter);
            while (!filter.empty()) {
                auto comma_pos = filter.find_first_of(',');
                const auto lparen_pos = filter.find_first_of('(');
                if (lparen_pos < comma_pos) {
                    auto rparen_pos = filter.find_first_of(')');
                    comma_pos = filter.find_first_of(',', rparen_pos);
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_full_name) {
                        return true;
                    }
                } else {
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_name) {
                        return true;
                    }
                }
                filter = comma_pos != std::string_view::npos ? filter.substr(comma_pos + 1) : "";
            }
            return false;
        } else {
            return true;
        }
    }

    test_result eval_perf(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        ggml_tensor * out             = build_graph(ctx.get());
        if (!out) {
            return test_result();
        }

        std::string   current_op_name = op_desc(out);
        if (!matches_filter(out, op_names_filter)) {
            return test_result();
        }

        if (!ggml_backend_supports_op(backend, out)) {
            // Create test result for unsupported performance test
            test_result result(ggml_backend_name(backend), current_op_name, vars(), to_json(), false, false,
                               "not supported");

            output_printer->print_test_result(result);

            return result;
        }

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend)); // smart ptr

        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            return test_result();
        }

        // randomize tensors
        initialize_tensors(ctx.get());

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
            return test_result();
        }

        // determine number of runs
        int n_runs = 32;

        // duplicate the op
        for (int i = 1; i < n_runs; i++) {
            ggml_graph_add_node(gf, out);
        }

        // run
        int64_t total_time_us = 0;
        int total_runs = 0;
        do {
            int64_t start_time = ggml_time_us();
            ggml_status status = ggml_backend_graph_compute(backend, gf);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                return test_result();
            }
            int64_t end_time = ggml_time_us();

            total_time_us += end_time - start_time;
            total_runs++;
        } while (total_runs < n_runs);

        // Create test result
        double avg_time_us      = (double) total_time_us / total_runs;

        std::string backend_reg_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)));
        test_result result(backend_reg_name, current_op_name, vars(), to_json(), true, true, "", avg_time_us, total_runs);

        if (output_printer) {
            output_printer->print_test_result(result);
        }

        return result;
    }
};

struct test_unary_op : public test_case {
    ggml_op op;
    ggml_type type;
    std::vector<int64_t> ne;

    test_unary_op(ggml_op op,
            ggml_type type,
            const std::vector<int64_t> & ne)
        : op(op), type(type), ne(ne) {}

    std::string vars() override {
        return VARS_TO_STR6(op, type, ne[0], ne[1], ne[2], ne[3]);
    }

    json to_json() override {
        json out_j;
        std::vector<std::string> types;
        types.push_back(ggml_type_name(type));
        std::vector<std::vector<int64_t>> sizes;
        sizes.push_back(ne);
        out_j["types"] = types;
        out_j["sizes"] = sizes;
        return out_j;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = nullptr;
        switch (op) {
            case GGML_OP_DUP:
                out = ggml_dup(ctx, a);
                break;
            case GGML_OP_SQR:
                out = ggml_sqr(ctx, a);
                break;
            case GGML_OP_SQRT:
                out = ggml_sqrt(ctx, a);
                break;
            case GGML_OP_LOG:
                out = ggml_log(ctx, a);
                break;
            case GGML_OP_SIN:
                out = ggml_sin(ctx, a);
                break;
            case GGML_OP_COS:
                out = ggml_cos(ctx, a);
                break;
            case GGML_OP_SUM:
                out = ggml_sum(ctx, a);
                break;
            case GGML_OP_SUM_ROWS:
                out = ggml_sum_rows(ctx, a);
                break;
            case GGML_OP_MEAN:
                out = ggml_mean(ctx, a);
                break;
            case GGML_OP_ARGMAX:
                out = ggml_argmax(ctx, a);
                break;
            case GGML_OP_CONT:
                out = ggml_cont(ctx, a);
                break;
            case GGML_OP_RMS_NORM:
                out = ggml_rms_norm(ctx, a, 1e-6f);
                break;
            default:
                printf("  Unsupported op %s\n", ggml_op_name(op));
                break;
        }

        if (out)
            ggml_set_name(out, "out");

        return out;
    }
};

struct test_binary_op : public test_case {
    ggml_op op;
    ggml_type type_a;
    ggml_type type_b;
    std::vector<int64_t> ne_a;
    std::vector<int64_t> ne_b;

    test_binary_op(const ggml_op &op,
            ggml_type type_a, ggml_type type_b,
            const std::vector<int64_t> & ne_a, const std::vector<int64_t> & ne_b )
        : op(op), type_a(type_a), type_b(type_b), ne_a(ne_a), ne_b(ne_b) {}

    std::string vars() override {
        return VARS_TO_STR11(op, type_a, type_b,
                ne_a[0], ne_a[1], ne_a[2], ne_a[3],
                ne_b[0], ne_b[1], ne_b[2], ne_b[3]);
    }

    json to_json() override {
        json out_j;
        std::vector<std::string> types;
        types.push_back(ggml_type_name(type_a));
        types.push_back(ggml_type_name(type_b));
        std::vector<std::vector<int64_t>> sizes;
        sizes.push_back(ne_a);
        sizes.push_back(ne_b);
        out_j["types"] = types;
        out_j["sizes"] = sizes;
        return out_j;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type_a, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type_b, 4, ne_b.data());
        ggml_set_name(b, "b");

        ggml_tensor * out = nullptr;
        switch (op) {
            case GGML_OP_ADD:
                out = ggml_add(ctx, a, b);
                break;
            case GGML_OP_SUB:
                out = ggml_sub(ctx, a, b);
                break;
            case GGML_OP_MUL:
                out = ggml_mul(ctx, a, b);
                break;
            case GGML_OP_DIV:
                out = ggml_div(ctx, a, b);
                break;
            case GGML_OP_COUNT_EQUAL:
                out = ggml_count_equal(ctx, a, b);
                break;
            case GGML_OP_REPEAT:
                out = ggml_repeat(ctx, a, b);
                break;
            case GGML_OP_MUL_MAT:
                out = ggml_mul_mat(ctx, a, b);
                break;
            case GGML_OP_OUT_PROD:
                out = ggml_out_prod(ctx, a, b);
                break;
            // FIXME: GET_ROWS crashes on CPU backend, re-enable here once fixed
            //case GGML_OP_GET_ROWS:
            //    out = ggml_get_rows(ctx, a, b);
            //    break;
            case GGML_OP_CPY:
                out = ggml_cpy(ctx, a, b);
                break;
            case GGML_OP_RESHAPE:
                out = ggml_reshape(ctx, a, b);
                break;
            default:
                printf("  Unsupported op %s\n", ggml_op_name(op));
                break;
        }

        if (out)
            ggml_set_name(out, "out");

        return out;
    }
};

static std::vector<std::unique_ptr<test_case>> make_test_cases_from_json(const char *path_test_cases_file) {
    std::vector<std::unique_ptr<test_case>> test_cases;

    printf("Loading test cases from %s\n", path_test_cases_file);
    std::ifstream file(path_test_cases_file);
    if (!file) {
        printf("Failed to open file '%s' for reading\n", path_test_cases_file);
        return test_cases;
    }

    std::map<std::string, ggml_op> str_to_ggml_op_map;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        ggml_op op = static_cast<ggml_op>(i);
        str_to_ggml_op_map[ggml_op_name(op)] = op;
    }
    std::map<std::string, ggml_type> str_to_ggml_type_map;
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type t = static_cast<ggml_type>(i);
        str_to_ggml_type_map[ggml_type_name(t)] = t;
    }

    std::set<std::string> parsed_tests;
    json test_cases_config_j = json::parse(file);
    auto test_cases_ops_j = test_cases_config_j["ops"];
    for (auto& [_, op_j] : test_cases_ops_j.items()) {
        std::string op_name = op_j["name"];
        std::vector<std::string> types = op_j["types"];
        std::vector<std::vector<int64_t>> sizes = op_j["sizes"];

        GGML_ASSERT(types.size() == sizes.size());

        test_case *test_case = nullptr;
        ggml_op op = str_to_ggml_op_map[op_name];
        if (types.size() == 1) {
            ggml_type type = str_to_ggml_type_map[types[0]];
            test_case = new test_unary_op(op, type, sizes[0]);
        } else if (types.size() == 2) {
            ggml_type type_a = str_to_ggml_type_map[types[0]];
            ggml_type type_b = str_to_ggml_type_map[types[1]];
            test_case = new test_binary_op(op, type_a, type_b, sizes[0], sizes[1]);
        } else {
            // TODO: add support for loading more ops from config
            printf("Ignoring unsupported op %s\n", op_name.c_str());
            continue;
        }

        if (parsed_tests.find(test_case->vars()) != parsed_tests.end()) {
            delete test_case;
            continue;
        }
        parsed_tests.insert(test_case->vars());
        test_cases.emplace_back(test_case);
    }

    return test_cases;
}

static bool test_backend(ggml_backend_t backend, const char * op_names_filter, const char * params_filter,
                         printer * output_printer, std::vector<std::unique_ptr<test_case>> test_cases) {
    auto filter_test_cases = [](std::vector<std::unique_ptr<test_case>> & test_cases, const char * params_filter) {
        if (params_filter == nullptr) {
            return;
        }

        std::regex params_filter_regex(params_filter);

        for (auto it = test_cases.begin(); it != test_cases.end();) {
            if (!std::regex_search((*it)->vars(), params_filter_regex)) {
                it = test_cases.erase(it);
                continue;
            }

            it++;
        }
    };

    filter_test_cases(test_cases, params_filter);
    for (auto & test : test_cases) {
        test_result result = test->eval_perf(backend, op_names_filter, output_printer);
        output_printer->test_results.push_back(result);
    }
    return true;

    GGML_ABORT("fatal error");
}

static void usage(char ** argv) {
    printf("Usage: %s [-o <op,..>] [-b <backend>] [-p <params regex>]\n", argv[0]);
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc),\n");
    printf("    --ops-file json file defining ops to run\n");
    printf("    --output-tunable-config specifies the tunable config output file\n");
}

int main(int argc, char ** argv) {
    const char * op_names_filter = nullptr;
    const char * backend_filter = nullptr;
    const char * params_filter = nullptr;
    const char * path_ops_file = nullptr;
    const char * path_output_tunable_config = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_names_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                params_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--ops-file") == 0) {
            if (i + 1 < argc) {
                path_ops_file = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--output-tunable-config") == 0) {
            if (i + 1 < argc) {
                path_output_tunable_config = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else {
            usage(argv);
            return 1;
        }
    }

    if (path_ops_file == nullptr || path_output_tunable_config == nullptr) {
        if (path_ops_file)
            printf("missing path_ops_file\n");
        if (path_output_tunable_config)
            printf("missing path_output_tunable_config\n");
        usage(argv);
        return 1;
    }

    // load and enumerate backends
    ggml_backend_load_all();

    // Create printer for output format
    std::unique_ptr<printer> output_printer = create_printer();
    if (output_printer) {
        output_printer->print_header();
    }

    output_printer->print_testing_start(testing_start_info(ggml_backend_dev_count()));

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_dev_name(dev)) != 0) {
            output_printer->print_backend_init(
                backend_init_info(i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), true, "Skipping"));
            n_ok++;
            continue;
        }

        //if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        //    output_printer->print_backend_init(backend_init_info(
        //        i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), true, "Skipping CPU backend"));
        //    n_ok++;
        //    continue;
        //}

        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            // TODO: better value for n_threads
            ggml_backend_set_n_threads_fn(backend, std::thread::hardware_concurrency());
        }

        size_t free, total;  // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        output_printer->print_backend_init(backend_init_info(i, ggml_backend_dev_count(), ggml_backend_dev_name(dev),
                                                             false, "", ggml_backend_dev_description(dev),
                                                             total / 1024 / 1024, free / 1024 / 1024, true));

        std::vector<std::unique_ptr<test_case>> test_cases;
        // TODO: cache test cases (note the other test cases are created on test_backend itself and also not cached)
        test_cases = make_test_cases_from_json(path_ops_file);
        bool ok = test_backend(backend, op_names_filter, params_filter, output_printer.get(), std::move(test_cases));

        if (ok) {
            n_ok++;
        }
        output_printer->print_backend_status(
            backend_status_info(ggml_backend_name(backend), ok ? test_status_t::OK : test_status_t::FAIL));

        ggml_backend_free(backend);
    }

    ggml_quantize_free();

    if (output_printer) {
        output_printer->print_footer();
    }

    output_printer->print_overall_summary(
        overall_summary_info(n_ok, ggml_backend_dev_count(), n_ok == ggml_backend_dev_count()));

    output_printer->print_tunable_config(path_output_tunable_config);

    if (n_ok != ggml_backend_dev_count()) {
        return 1;
    }

    return 0;
}
