#include "../../src/llama-coreml-impl.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static std::atomic<uint64_t> coreml_batches{0};

static void capture_log(ggml_log_level, const char * text, void *) {
    if (strstr(text, "Core ML encoded")) { ++coreml_batches; }
    fputs(text, stderr);
}

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static double cosine(const float * a, const float * b) {
    const double value = llama_coreml_cosine(a, b, LLAMA_COREML_EMBD_OUT);
    require(value > 0, "zero or non-finite embedding");
    return value;
}

static void check_normalized(const std::vector<float> & values) {
    for (size_t row = 0; row < values.size() / LLAMA_COREML_EMBD_OUT; ++row) {
        double squared = 0;
        for (int col = 0; col < LLAMA_COREML_EMBD_OUT; ++col) {
            double value = values[row * LLAMA_COREML_EMBD_OUT + col];
            require(std::isfinite(value), "non-finite embedding");
            squared += value * value;
        }
        require(std::abs(sqrt(squared) - 1.0) < 0.01, "embedding is not L2 normalized");
    }
}

static void test_planning() {
    for (int length = 1; length <= LLAMA_COREML_MAX_TOKENS; ++length) {
        llama_coreml_embedding_sequence sequence = {nullptr, length};
        const auto job = llama_coreml_plan_job(&sequence, 1, true, LLAMA_COREML_MAX_SLOTS);
        require(job.count == 1, "single sequence was packed");
        const int bucket = job.function_index;
        require(llama_coreml_lengths[bucket] >= length, "bucket is too small");
        require(bucket == 0 || llama_coreml_lengths[bucket - 1] < length, "bucket is not minimal");
    }
    std::vector<llama_coreml_embedding_sequence> sequences(17, {nullptr, 1});
    for (int slots : {LLAMA_COREML_MIN_SLOTS, LLAMA_COREML_MAX_SLOTS}) {
        const auto job = llama_coreml_plan_job(sequences.data(), sequences.size(), true, slots);
        require(job.count == slots && job.function_index == llama_coreml_n_lengths, "packing slot limit failed");
    }
    sequences[0].n_tokens = 128;
    sequences[1].n_tokens = 128;
    sequences[2].n_tokens = 1;
    require(llama_coreml_plan_job(sequences.data(), 3, true, 16).count == 2, "256-token budget failed");
    sequences[1].n_tokens = 129;
    require(llama_coreml_plan_job(sequences.data(), 3, true, 16).count == 1, "257-token pack was accepted");
    sequences[0].n_tokens = 1;
    require(llama_coreml_plan_job(sequences.data(), 3, false, 16).count == 1, "packing-disabled job was packed");
}

static void test_grouping() {
    llama_token ids[] = {10, 20, 11, 21};
    llama_pos positions[] = {0, 0, 1, 1};
    int32_t counts[] = {1, 1, 1, 1};
    llama_seq_id a[] = {7, 9}, b[] = {3};
    llama_seq_id * rows[] = {a, b, a, b};
    llama_ubatch batch = {};
    batch.n_tokens = 4;
    batch.n_pos = 1;
    batch.token = ids;
    batch.pos = positions;
    batch.n_seq_id = counts;
    batch.seq_id = rows;
    llama_coreml_tokens grouped;
    require(llama_coreml_group_tokens(batch, grouped), "interleaved IDs were rejected");
    require(grouped.at(7) == std::vector<int32_t>({10, 11}) && grouped.at(3) == std::vector<int32_t>({20, 21}), "sequence mapping failed");
    counts[0] = 2;
    require(!llama_coreml_group_tokens(batch, grouped), "shared row was accepted");
    counts[0] = 1;
    positions[0] = 1;
    require(!llama_coreml_group_tokens(batch, grouped), "position offset was accepted");
    positions[0] = 0;
    float embedding = 0;
    batch.embd = &embedding;
    require(!llama_coreml_group_tokens(batch, grouped), "supplied embeddings were accepted");
    batch.embd = nullptr;
    int8_t type = 0;
    batch.type = &type;
    require(!llama_coreml_group_tokens(batch, grouped), "mixed input was accepted");
    batch.type = nullptr;
    batch.n_pos = 2;
    require(!llama_coreml_group_tokens(batch, grouped), "multiple position inputs were accepted");
    batch.n_pos = 1;
    std::vector<llama_token> tokens(LLAMA_COREML_MAX_TOKENS + 1, 10);
    std::vector<llama_pos> pos(tokens.size());
    std::vector<int32_t> n_seq(tokens.size(), 1);
    std::vector<llama_seq_id *> seq(tokens.size(), a);
    for (size_t i = 0; i < pos.size(); ++i) { pos[i] = i; }
    batch.token = tokens.data();
    batch.pos = pos.data();
    batch.n_seq_id = n_seq.data();
    batch.seq_id = seq.data();
    batch.n_tokens = LLAMA_COREML_MAX_TOKENS;
    require(llama_coreml_group_tokens(batch, grouped), "512 tokens were rejected");
    ++batch.n_tokens;
    require(!llama_coreml_group_tokens(batch, grouped), "513 tokens were accepted");
}

static void test_context_contract() {
    llama_hparams hparams = {};
    hparams.rope_freq_base_train = 1000000.0f;
    hparams.rope_freq_scale_train = 1.0f;
    hparams.yarn_attn_factor = hparams.rope_attn_factor = 1.0f;
    llama_cparams defaults = {};
    defaults.embeddings = true;
    defaults.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    defaults.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    defaults.rope_freq_base = hparams.rope_freq_base_train;
    defaults.rope_freq_scale = hparams.rope_freq_scale_train;
    defaults.yarn_attn_factor = 1.0f;
    defaults.causal_attn = hparams.causal_attn;
    require(llama_coreml_context_compatible(defaults, hparams), "default context contract was rejected");
    for (int test = 0; test < 10; ++test) {
        auto params = defaults;
        switch (test) {
            case 0: params.rope_freq_base /= 2; break;
            case 1: params.rope_freq_scale /= 2; break;
            case 2: params.yarn_ext_factor = 1; break;
            case 3: params.yarn_attn_factor = 2; break;
            case 4: params.pooling_type = LLAMA_POOLING_TYPE_CLS; break;
            case 5: params.pooling_type = LLAMA_POOLING_TYPE_NONE; break;
            case 6: params.pooling_type = LLAMA_POOLING_TYPE_LAST; break;
            case 7: params.embeddings = false; break;
            case 8: params.training = true; break;
            case 9: params.causal_attn = !hparams.causal_attn; break;
        }
        require(!llama_coreml_context_compatible(params, hparams), "incompatible context contract was accepted");
    }
}

static int self_test() {
    try {
        require(llama_supports_coreml(), "Core ML build support is missing");
        require(!llama_set_coreml_model(nullptr, nullptr, 1), "null context was accepted");
        for (int value : {0, 9}) {
            require(!llama_coreml_valid_inflight(value), "invalid concurrency was accepted");
        }
        for (int value : {1, 8}) {
            require(llama_coreml_valid_inflight(value), "valid concurrency was rejected");
        }
        char error[1024] = {};
        require(!llama_coreml_embedding_load("", error, sizeof(error)) && error[0], "empty directory was accepted");
        require(!llama_coreml_embedding_load("/dev/null/embeddinggemma-coreml", error, sizeof(error)) && error[0], "missing directory was accepted");
        test_planning();
        test_grouping();
        test_context_contract();
        printf("coreml_contract=passed\n");
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

static void test_context_settings(llama_model * model, const llama_context_params & defaults, const char * directory) {
    for (int test = 0; test < 6; ++test) {
        auto params = defaults;
        switch (test) {
            case 0: params.rope_freq_base = 500000.0f; break;
            case 1: params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; break;
            case 2: params.pooling_type = LLAMA_POOLING_TYPE_CLS; break;
            case 3: params.pooling_type = LLAMA_POOLING_TYPE_NONE; break;
            case 4: params.pooling_type = LLAMA_POOLING_TYPE_LAST; break;
            case 5: params.yarn_attn_factor = 2.0f; break;
        }
        auto ctx = std::unique_ptr<llama_context, decltype(&llama_free)>(llama_init_from_model(model, params), llama_free);
        require(ctx != nullptr, "rejection test context failed");
        require(!llama_set_coreml_model(ctx.get(), directory, 1), "incompatible context settings were accepted");
        require(llama_set_coreml_model(ctx.get(), nullptr, 1), "null directory could not disable the sidecar");
    }
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        return self_test();
    }
    if (argc < 4 || argc > 10) {
        fprintf(stderr, "usage: %s model.gguf coreml-directory prefixed-texts.txt [repetitions=5] [cpu|metal=metal] [workers=1] [parallel=8] [batch=512] [max-inflight=1]\n", argv[0]);
        return 1;
    }
    llama_log_set(capture_log, nullptr);
    llama_backend_init();
    int status = 1;
    try {
        int repetitions = argc > 4 ? std::stoi(argv[4]) : 5;
        require(repetitions > 0 && repetitions <= 1000, "repetitions must be 1..1000");
        std::string backend = argc > 5 ? argv[5] : "metal";
        require(backend == "metal" || backend == "cpu", "backend must be cpu or metal");
        int workers = argc > 6 ? std::stoi(argv[6]) : 1;
        require(workers >= 1 && workers <= 8, "workers must be 1..8");
        int parallel = argc > 7 ? std::stoi(argv[7]) : 8;
        int batch_size = argc > 8 ? std::stoi(argv[8]) : 512;
        int max_inflight = argc > 9 ? std::stoi(argv[9]) : LLAMA_COREML_DEFAULT_INFLIGHT;
        require(llama_coreml_valid_inflight(max_inflight), "max-inflight must be 1..8");
        require(parallel >= 1 && parallel <= 64, "parallel must be 1..64");
        require(batch_size >= 32 && batch_size <= 2048, "batch must be 32..2048");
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = backend == "cpu" ? 0 : -1;
        auto model = std::unique_ptr<llama_model, decltype(&llama_model_free)>(
                llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "failed to load GGUF reference model");
        char architecture[64] = {};
        llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture));
        require(std::string(architecture) == "gemma-embedding2", "GGUF must be EmbeddingGemma 2");
        require(llama_model_n_embd_out(model.get()) == LLAMA_COREML_EMBD_OUT, "reference output dimension does not match the sidecar");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        require(llama_vocab_n_tokens(vocab) == LLAMA_COREML_VOCAB && llama_vocab_bos(vocab) == LLAMA_COREML_BOS && llama_vocab_eos(vocab) == LLAMA_COREML_EOS,
                "GGUF vocabulary does not match the Core ML model");
        std::ifstream file(argv[3]);
        require(file.is_open(), "failed to open input text file");
        std::vector<std::vector<int32_t>> tokens;
        std::string line;
        while (std::getline(file, line)) {
            require(line.size() <= 1024 * 1024, "input line exceeds 1 MiB");
            int32_t count = llama_tokenize(vocab, line.c_str(), int32_t(line.size()), nullptr, 0, false, false);
            require(count <= 0 && count != std::numeric_limits<int32_t>::min(), "tokenization failed");
            require(-count <= LLAMA_COREML_MAX_TOKENS - 2, "input exceeds 512 tokens; shorten it explicitly");
            tokens.emplace_back(size_t(-count) + 2);
            auto & sequence = tokens.back();
            count = llama_tokenize(vocab, line.c_str(), int32_t(line.size()), sequence.data() + 1, int32_t(sequence.size()) - 2, false, false);
            require(count == int32_t(sequence.size()) - 2, "tokenization failed");
            sequence.front() = llama_vocab_bos(vocab);
            sequence.back() = llama_vocab_eos(vocab);
            require(tokens.size() <= 100000, "input contains too many texts");
        }
        require(!tokens.empty(), "input text file is empty");
        for (const auto & sequence : tokens) {
            require(sequence.size() <= size_t(batch_size), "reference batch is smaller than an input sequence");
        }
        printf("coreml_directory=%s reference_parallel=%d reference_batch=%d max_inflight=%d\n", argv[2], parallel, batch_size, max_inflight);
        std::vector<llama_coreml_embedding_sequence> sequences;
        size_t total_tokens = 0;
        for (const auto & sequence : tokens) {
            sequences.push_back({sequence.data(), int32_t(sequence.size())});
            total_tokens += sequence.size();
        }
        char error[1024] = {};
        auto coreml = std::unique_ptr<llama_coreml_embedding, decltype(&llama_coreml_embedding_free)>(
                llama_coreml_embedding_load(argv[2], error, sizeof(error)), llama_coreml_embedding_free);
        require(coreml != nullptr, error);
        require(llama_coreml_embedding_set_max_inflight(coreml.get(), max_inflight) == 0, "failed to set concurrency");
        std::vector<float> single(tokens.size() * LLAMA_COREML_EMBD_OUT), packed(single.size()), reference(single.size());
        auto encode = [&](bool pack, std::vector<float> & output) {
            require(llama_coreml_embedding_encode(coreml.get(), sequences.data(), int32_t(sequences.size()), pack,
                    output.data(), output.size(), error, sizeof(error)) == 0, error);
        };
        // Warm every function used by this dataset before timing.
        encode(false, single);
        encode(true, packed);
        check_normalized(single);
        check_normalized(packed);
        double minimum_packed = 1;
        for (size_t i = 0; i < tokens.size(); ++i) {
            minimum_packed = std::min(minimum_packed, cosine(single.data() + i * LLAMA_COREML_EMBD_OUT, packed.data() + i * LLAMA_COREML_EMBD_OUT));
        }
        // Check errors through the public C boundary using a real loaded context.
        int32_t invalid = LLAMA_COREML_VOCAB;
        llama_coreml_embedding_sequence bad = {&invalid, 1};
        require(llama_coreml_embedding_encode(coreml.get(), &bad, 1, false, packed.data(), packed.size(), error, sizeof(error)) < 0,
                "out-of-range token was accepted");
        bad = {tokens[0].data(), 513};
        require(llama_coreml_embedding_encode(coreml.get(), &bad, 1, false, packed.data(), packed.size(), error, sizeof(error)) < 0,
                "oversized sequence was accepted");
        require(llama_coreml_embedding_encode(coreml.get(), sequences.data(), int32_t(sequences.size()), true,
                packed.data(), packed.size() - 1, error, sizeof(error)) < 0, "undersized output was accepted");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = batch_size;
        cp.n_seq_max = parallel;
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        cp.kv_unified = true;
        auto ctx = std::unique_ptr<llama_context, decltype(&llama_free)>(llama_init_from_model(model.get(), cp), llama_free);
        require(ctx != nullptr, "reference context failed");
        test_context_settings(model.get(), cp, argv[2]);
        require(!llama_set_coreml_model(ctx.get(), argv[2], 0), "zero concurrency was accepted");
        require(!llama_set_coreml_model(ctx.get(), argv[2], LLAMA_COREML_MAX_INFLIGHT + 1), "excess concurrency was accepted");
        const std::string missing = std::string(argv[2]) + "/embeddings.bf16/missing";
        require(!llama_set_coreml_model(ctx.get(), missing.c_str(), 1), "missing directory was accepted");
        auto run_api = [&](llama_context * target, bool decode = false, int32_t position_offset = 0, bool rotate_ids = false) {
            llama_batch batch = llama_batch_init(batch_size, 0, 1);
            for (size_t start = 0; start < tokens.size();) {
                batch.n_tokens = 0;
                size_t end = start;
                while (end < tokens.size() && end - start < size_t(parallel) && batch.n_tokens + tokens[end].size() <= size_t(batch_size)) {
                    for (size_t pos = 0; pos < tokens[end].size(); ++pos) {
                        int i = batch.n_tokens++;
                        batch.token[i] = tokens[end][pos];
                        batch.pos[i] = int32_t(pos) + position_offset;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = (int32_t(end - start) + (rotate_ids ? parallel / 2 : 0)) % parallel;
                        batch.logits[i] = true;
                    }
                    ++end;
                }
                if ((decode ? llama_decode(target, batch) : llama_encode(target, batch)) != 0) {
                    llama_batch_free(batch);
                    throw std::runtime_error("reference inference failed");
                }
                for (size_t i = start; i < end; ++i) {
                    const int32_t id = (int32_t(i - start) + (rotate_ids ? parallel / 2 : 0)) % parallel;
                    const float * out = llama_get_embeddings_seq(target, id);
                    if (!out) {
                        llama_batch_free(batch);
                        throw std::runtime_error("reference embedding missing");
                    }
                    std::copy(out, out + LLAMA_COREML_EMBD_OUT, reference.data() + i * LLAMA_COREML_EMBD_OUT);
                }
                start = end;
            }
            llama_batch_free(batch);
        };
        auto baseline = [&]() { run_api(ctx.get()); };
        baseline();
        const auto ggml_reference = reference;
        double minimum_reference = 1, mean_reference = 0, minimum_packed_reference = 1;
        for (size_t i = 0; i < tokens.size(); ++i) {
            double value = cosine(reference.data() + i * LLAMA_COREML_EMBD_OUT, single.data() + i * LLAMA_COREML_EMBD_OUT);
            minimum_reference = std::min(minimum_reference, value);
            mean_reference += value / tokens.size();
            minimum_packed_reference = std::min(minimum_packed_reference, cosine(reference.data() + i * LLAMA_COREML_EMBD_OUT, packed.data() + i * LLAMA_COREML_EMBD_OUT));
        }
        printf("texts=%zu tokens=%zu packed_vs_single_min_cos=%.8f reference_min_cos=%.8f reference_mean_cos=%.8f packed_reference_min_cos=%.8f\n",
                tokens.size(), total_tokens, minimum_packed, minimum_reference, mean_reference, minimum_packed_reference);
        require(minimum_reference >= llama_coreml_min_cosine, "Core ML output differs from GGUF reference");
        require(minimum_packed >= llama_coreml_min_cosine, "packed output differs from single output");
        require(minimum_packed_reference >= llama_coreml_min_cosine, "packed Core ML output differs from GGUF reference");
        auto measure = [&](const char * name, auto run) {
            llama_coreml_embedding_stats stats = {};
            require(llama_coreml_embedding_get_stats(coreml.get(), &stats, true) == 0, "failed to reset timing counters");
            auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < repetitions; ++i) { run(); }
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            printf("%s seconds=%.4f texts/s=%.2f tokens/s=%.2f\n", name, seconds,
                    tokens.size() * repetitions / seconds, total_tokens * repetitions / seconds);
            require(llama_coreml_embedding_get_stats(coreml.get(), &stats, false) == 0, "failed to read timing counters");
            if (stats.calls) {
                printf("%s calls=%llu preparation_ms=%.3f prediction_ms=%.3f output_ms=%.3f\n", name,
                        (unsigned long long) stats.calls, stats.preparation_ms, stats.prediction_ms, stats.output_ms);
            }
        };
        measure("coreml_single", [&]() { encode(false, single); });
        measure("coreml_packed", [&]() { encode(true, packed); });
        measure(backend.c_str(), baseline);
        if (llama_set_coreml_model(ctx.get(), argv[2], max_inflight)) {
            auto check_raw = [&](const std::vector<float> & expected) {
                double maximum_relative_error = 0;
                for (size_t row = 0; row < tokens.size(); ++row) {
                    double squared_error = 0, squared_reference = 0;
                    for (size_t col = 0; col < LLAMA_COREML_EMBD_OUT; ++col) {
                        const size_t index = row * LLAMA_COREML_EMBD_OUT + col;
                        const double delta = reference[index] - expected[index];
                        squared_error += delta * delta;
                        squared_reference += double(expected[index]) * expected[index];
                    }
                    require(cosine(reference.data() + row * LLAMA_COREML_EMBD_OUT, expected.data() + row * LLAMA_COREML_EMBD_OUT) >= llama_coreml_min_cosine,
                            "integrated embedding direction differs from ggml");
                    maximum_relative_error = std::max(maximum_relative_error, sqrt(squared_error / squared_reference));
                }
                require(maximum_relative_error < 0.03, "integrated embeddings lost their raw magnitude");
                return maximum_relative_error;
            };
            uint64_t before = coreml_batches.load();
            baseline();
            require(coreml_batches.load() > before, "llama_encode did not use Core ML");
            const double raw_error = check_raw(ggml_reference);
            before = coreml_batches.load();
            run_api(ctx.get(), true, 0, true);
            require(coreml_batches.load() > before, "llama_decode did not use Core ML");
            check_raw(ggml_reference);
            measure("coreml_integrated", baseline);
            check_raw(ggml_reference);
            before = coreml_batches.load();
            run_api(ctx.get(), false, 7);
            require(coreml_batches.load() == before, "position-offset batch did not fall back to ggml");
            const auto shifted = reference;
            require(llama_set_coreml_model(ctx.get(), nullptr, 1), "could not disable Core ML");
            run_api(ctx.get(), false, 7);
            check_raw(shifted);
            baseline();
            require(coreml_batches.load() == before, "disabled sidecar still executed");
            check_raw(ggml_reference);

            llama_context_params fallback_params = cp;
            fallback_params.n_ctx = fallback_params.n_batch = fallback_params.n_ubatch = 1024;
            fallback_params.n_seq_max = 2;
            auto fallback = std::unique_ptr<llama_context, decltype(&llama_free)>(
                    llama_init_from_model(model.get(), fallback_params), llama_free);
            require(fallback != nullptr, "fallback context failed");
            auto check_fallback = [&](llama_batch & batch, int n_sequences) {
                require(llama_set_coreml_model(fallback.get(), nullptr, 1), "could not disable fallback context sidecar");
                require(llama_encode(fallback.get(), batch) == 0, "ggml fallback reference failed");
                std::vector<float> expected;
                for (int id = 0; id < n_sequences; ++id) {
                    const float * out = llama_get_embeddings_seq(fallback.get(), id);
                    require(out != nullptr, "fallback reference missing");
                    expected.insert(expected.end(), out, out + LLAMA_COREML_EMBD_OUT);
                }
                require(llama_set_coreml_model(fallback.get(), argv[2], max_inflight), "fallback context sidecar failed");
                const uint64_t calls = coreml_batches.load();
                require(llama_decode(fallback.get(), batch) == 0, "unsupported batch fallback failed");
                require(coreml_batches.load() == calls, "unsupported batch used Core ML");
                for (int id = 0; id < n_sequences; ++id) {
                    const float * out = llama_get_embeddings_seq(fallback.get(), id);
                    require(out && cosine(out, expected.data() + id * LLAMA_COREML_EMBD_OUT) > 0.99999, "fallback output mismatch");
                }
            };
            llama_batch long_batch = llama_batch_init(513, 0, 2);
            long_batch.n_tokens = 513;
            for (int i = 0; i < 513; ++i) {
                long_batch.token[i] = tokens[0][i % tokens[0].size()];
                long_batch.pos[i] = i;
                long_batch.n_seq_id[i] = 1;
                long_batch.seq_id[i][0] = 0;
                long_batch.logits[i] = true;
            }
            check_fallback(long_batch, 1);
            long_batch.n_tokens = 16;
            for (int i = 0; i < 16; ++i) {
                long_batch.n_seq_id[i] = 2;
                long_batch.seq_id[i][1] = 1;
            }
            check_fallback(long_batch, 2);
            llama_batch_free(long_batch);
            llama_batch media = llama_batch_init(2, LLAMA_COREML_HIDDEN, 1);
            media.n_tokens = 2;
            for (int i = 0; i < 2; ++i) {
                for (int col = 0; col < LLAMA_COREML_HIDDEN; ++col) { media.embd[i * LLAMA_COREML_HIDDEN + col] = float(col % 7 - 3) * 0.1f; }
                media.pos[i] = i;
                media.n_seq_id[i] = 1;
                media.seq_id[i][0] = 0;
                media.logits[i] = true;
            }
            check_fallback(media, 1);
            llama_batch_free(media);
            printf("integrated_api=passed raw_relative_error_max=%.8f encode_decode=passed sequence_ids=passed fallback=passed\n", raw_error);
        } else {
            printf("integrated_api=unavailable model_requires_embedding_raw=1\n");
        }
        if (workers > 1) {
            using coreml_ptr = std::unique_ptr<llama_coreml_embedding, decltype(&llama_coreml_embedding_free)>;
            std::vector<coreml_ptr> contexts;
            workers = std::min(workers, int(tokens.size()));
            for (int i = 0; i < workers; ++i) {
                contexts.emplace_back(llama_coreml_embedding_load(argv[2], error, sizeof(error)), llama_coreml_embedding_free);
                require(contexts.back() != nullptr, error);
                require(llama_coreml_embedding_set_max_inflight(contexts.back().get(), max_inflight) == 0, "failed to set worker concurrency");
            }
            auto concurrent = [&]() {
                std::vector<std::thread> threads;
                std::vector<std::string> failures(workers);
                try {
                    for (int worker = 0; worker < workers; ++worker) {
                        threads.emplace_back([&, worker]() {
                            const size_t start = sequences.size() * worker / workers;
                            const size_t end = sequences.size() * (worker + 1) / workers;
                            char failure[1024] = {};
                            if (llama_coreml_embedding_encode(contexts[worker].get(), sequences.data() + start,
                                    int32_t(end - start), true, packed.data() + start * LLAMA_COREML_EMBD_OUT, (end - start) * LLAMA_COREML_EMBD_OUT,
                                    failure, sizeof(failure)) < 0) {
                                failures[worker] = failure;
                            }
                        });
                    }
                } catch (...) {
                    for (auto & thread : threads) { thread.join(); }
                    throw;
                }
                for (auto & thread : threads) { thread.join(); }
                for (const auto & failure : failures) { require(failure.empty(), failure.c_str()); }
            };
            concurrent();
            check_normalized(packed);
            for (size_t i = 0; i < tokens.size(); ++i) {
                require(cosine(single.data() + i * LLAMA_COREML_EMBD_OUT, packed.data() + i * LLAMA_COREML_EMBD_OUT) >= llama_coreml_min_cosine,
                        "concurrent packed output differs from single output");
            }
            std::string label = "coreml_packed_" + std::to_string(workers) + "_workers";
            measure(label.c_str(), concurrent);
        }
        status = 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "error: %s\n", error.what());
    }
    llama_backend_free();
    return status;
}
