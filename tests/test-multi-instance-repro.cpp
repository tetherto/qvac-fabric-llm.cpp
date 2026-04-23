// Multi-instance memory-behaviour repro.
//
// model: Llama-3.2-1B-Instruct-Q4_0.gguf
// Usage:
//   ./test-multi-instance-repro -m /path/to/model.gguf -ngl 999 -c 1024 -n 1

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static long read_kb(const char * key) {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long val = -1;
    const size_t key_len = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, key_len) == 0) {
            sscanf(line + key_len, " %ld kB", &val);
            break;
        }
    }
    fclose(f);
    return val;
}

static void print_mem(const char * label) {
    long rss  = read_kb("VmRSS:");
    long size = read_kb("VmSize:");
    long peak = read_kb("VmPeak:");
    if (rss < 0) {
        printf("[MEM] %-36s (no /proc/self/status)\n", label);
        return;
    }
    printf("[MEM] %-36s RSS=%5ld MiB  VmSize=%5ld MiB  VmPeak=%5ld MiB\n",
           label, rss / 1024, size / 1024, peak / 1024);
    fflush(stdout);
}

static bool decode_once(llama_context * ctx, const llama_model * model) {
    const auto * vocab = llama_model_get_vocab(model);
    llama_token tok = llama_vocab_bos(vocab);
    if (tok == LLAMA_TOKEN_NULL) {
        // model has no BOS; pick any valid token
        tok = 0;
    }
    llama_batch batch = llama_batch_get_one(&tok, 1);
    return llama_decode(ctx, batch) == 0;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    print_mem("baseline");

    printf("\n=== Scenario 1: two simultaneous instances ===\n");
    {
        common_params p1 = params;
        common_params p2 = params;

        print_mem("S1 before load #1");
        common_init_result_ptr i1 = common_init_from_params(p1);
        if (!i1 || !i1->model() || !i1->context()) {
            LOG_ERR("S1: failed to load instance #1\n");
            return 1;
        }
        print_mem("S1 after  load #1");

        common_init_result_ptr i2 = common_init_from_params(p2);
        if (!i2 || !i2->model() || !i2->context()) {
            LOG_ERR("S1: failed to load instance #2\n");
            return 1;
        }
        print_mem("S1 after  load #2");

        if (!decode_once(i1->context(), i1->model())) {
            LOG_ERR("S1: decode #1 failed\n");
            return 1;
        }
        print_mem("S1 after  decode #1");
        if (!decode_once(i2->context(), i2->model())) {
            LOG_ERR("S1: decode #2 failed\n");
            return 1;
        }
        print_mem("S1 after  decode #2");

        i1.reset();
        print_mem("S1 after  unload #1");
        i2.reset();
        print_mem("S1 after  unload #2");
    }
    print_mem("S1 after  scope exit");

    printf("\n=== Scenario 2: 20 load/unload cycles ===\n");
    constexpr int NUM_CYCLES = 20;
    for (int i = 1; i <= NUM_CYCLES; ++i) {
        common_params p = params;
        char label[64];

        snprintf(label, sizeof(label), "S2 cycle %d before load", i);
        print_mem(label);

        common_init_result_ptr inst = common_init_from_params(p);
        if (!inst || !inst->model() || !inst->context()) {
            LOG_ERR("S2: cycle %d failed to load\n", i);
            return 1;
        }
        snprintf(label, sizeof(label), "S2 cycle %d after  load", i);
        print_mem(label);

        if (!decode_once(inst->context(), inst->model())) {
            LOG_ERR("S2: cycle %d decode failed\n", i);
            return 1;
        }
        snprintf(label, sizeof(label), "S2 cycle %d after  decode", i);
        print_mem(label);

        inst.reset();
        snprintf(label, sizeof(label), "S2 cycle %d after  unload", i);
        print_mem(label);
    }

    llama_backend_free();
    print_mem("after backend free");
    return 0;
}
