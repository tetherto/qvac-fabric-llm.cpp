// The host decode's pinned threads, and every thread's CPU mask after it.
//
//   test-xdna-affinity -m model.gguf
//
// The decode of a model wider than the fused layers runs on host threads pinned
// one a core (GGML_XDNA_HOST_DECODE_PIN), borrowing the caller's OpenMP team.
// The prompt, then 8 greedy tokens: a sampler thread has to see, while they
// decode, threads pinned one to each core of the start mask and to no other CPU,
// and after each decode every thread in /proc/self/task has to have the mask the
// main thread had at start. Skips
// (exit 0) without a model or an XDNA device, for a model the fused layers
// cover, with the pin or the host decode switched off, or with OpenMP binding.
#include "llama.h"

#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

static constexpr int  N_GEN         = 8;
static constexpr int  N_CTX         = 512;
static constexpr int  N_GPU_LAYERS  = 99;
static constexpr int  MAX_PROMPT    = 64;
static constexpr int  FUSED_WIDTH   = 1024;
static constexpr auto SAMPLE_PERIOD = std::chrono::milliseconds(1);

static const char * const SKIP_ENV[] = {
    "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "GGML_XDNA_HOST_DECODE_PIN", "GGML_XDNA_HOST_DECODE_THREADS",
};

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static const char * skip_env_set(void) {
    for (const char * name : SKIP_ENV) {
        if (getenv(name) != nullptr) {
            return name;
        }
    }
    return nullptr;
}

static llama_token greedy(llama_context * ctx, int nv) {
    const float * l = llama_get_logits_ith(ctx, -1);
    llama_token   b = 0;
    for (llama_token t = 1; t < nv; t++) {
        b = l[t] > l[b] ? t : b;
    }
    return b;
}

// The CPU masks of this process's threads; a thread that exits meanwhile is left out.
static std::vector<cpu_set_t> thread_masks(void) {
    std::vector<cpu_set_t> masks;
    for (const auto & task : std::filesystem::directory_iterator("/proc/self/task")) {
        cpu_set_t m;
        if (sched_getaffinity((pid_t) std::stol(task.path().filename().string()), sizeof(m), &m) == 0) {
            masks.push_back(m);
        }
    }
    return masks;
}

// The first CPU of each core in `m`, as the backend picks them (thread_siblings_list).
static std::set<int> core_cpus(const cpu_set_t & m) {
    std::set<int> firsts;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &m)) {
            continue;
        }
        const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list";
        FILE *            f    = std::fopen(path.c_str(), "r");
        int               first = -1;
        if (f && std::fscanf(f, "%d", &first) == 1 && CPU_ISSET(first, &m)) {
            firsts.insert(first);
        }
        if (f) {
            std::fclose(f);
        }
    }
    return firsts;
}

static int count_off(const std::vector<cpu_set_t> & masks, const cpu_set_t & want) {
    int off = 0;
    for (const cpu_set_t & m : masks) {
        off += CPU_EQUAL(&m, &want) ? 0 : 1;
    }
    return off;
}

// The CPUs of the threads pinned to exactly one.
static void add_pinned(const std::vector<cpu_set_t> & masks, std::set<int> & cpus) {
    for (const cpu_set_t & m : masks) {
        if (CPU_COUNT(&m) != 1) {
            continue;
        }
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if (CPU_ISSET(cpu, &m)) {
                cpus.insert(cpu);
            }
        }
    }
}

static void sample_until(const std::atomic<bool> & stop, std::set<int> & pinned, std::mutex & mu) {
    while (!stop.load()) {
        const std::vector<cpu_set_t> masks = thread_masks();
        {
            std::lock_guard<std::mutex> lock(mu);
            add_pinned(masks, pinned);
        }
        std::this_thread::sleep_for(SAMPLE_PERIOD);
    }
}

// Greedy single-token decodes; the threads off the start mask after each.
static int decode_and_check(llama_context * ctx, int nv, const cpu_set_t & want) {
    int off = 0;
    for (int i = 0; i < N_GEN; i++) {
        llama_token t = greedy(ctx, nv);
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            std::printf("decode %d failed\n", i);
            return off + 1;
        }
        off += count_off(thread_masks(), want);
    }
    return off;
}

static int run(llama_model * model, const cpu_set_t & start) {
    const llama_vocab *      v      = llama_model_get_vocab(model);
    const std::string        prompt = "The capital of France is";
    std::vector<llama_token> p(MAX_PROMPT);
    p.resize(llama_tokenize(v, prompt.c_str(), (int) prompt.size(), p.data(), (int) p.size(), true, false));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = N_CTX;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx || llama_decode(ctx, llama_batch_get_one(p.data(), (int) p.size())) != 0) {
        std::printf("no context or the prompt failed\n");
        llama_free(ctx);
        return 1;
    }
    std::atomic<bool> stop{ false };
    std::set<int>     pinned;
    std::mutex        mu;
    std::thread       sampler(sample_until, std::cref(stop), std::ref(pinned), std::ref(mu));
    const int         off = decode_and_check(ctx, llama_vocab_n_tokens(v), start);
    stop                  = true;
    sampler.join();
    llama_free(ctx);
    const std::set<int> want = core_cpus(start);
    std::printf("CPUs pinned while decoding: %zu (one a core of the start mask: %zu), threads off the start mask: %d\n",
                pinned.size(), want.size(), off);
    return !want.empty() && pinned == want && off == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
        }
    }
    cpu_set_t start;
    if (sched_getaffinity(0, sizeof(start), &start) != 0) {
        return 1;
    }
    if (const char * name = skip_env_set()) {
        std::printf("test-xdna-affinity: %s is set, skipping\n", name);
        return 0;
    }
    llama_backend_init();
    if (!model_path || !xdna_device()) {
        std::printf("test-xdna-affinity: no model or no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { xdna_device(), nullptr };
    mp.devices                 = devs;
    mp.n_gpu_layers            = N_GPU_LAYERS;
    llama_model * model        = llama_model_load_from_file(model_path, mp);
    int           rc           = model ? 0 : 1;
    if (model && llama_model_n_embd(model) == FUSED_WIDTH) {
        std::printf("test-xdna-affinity: the fused layers cover this model, skipping\n");
    } else if (model) {
        rc = run(model, start);
        std::printf("test-xdna-affinity: %s\n", rc == 0 ? "passed" : "FAILED");
    }
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
