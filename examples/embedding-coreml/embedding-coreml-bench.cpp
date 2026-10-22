#include "embedding-coreml.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static double cosine(const float * a, const float * b) {
    double dot = 0, aa = 0, bb = 0;
    for (int i = 0; i < 768; ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite embedding");
        dot += double(a[i]) * b[i];
        aa += double(a[i]) * a[i];
        bb += double(b[i]) * b[i];
    }
    require(aa > 0 && bb > 0, "zero embedding");
    return dot / sqrt(aa * bb);
}

static void check_normalized(const std::vector<float> & values) {
    for (size_t row = 0; row < values.size() / 768; ++row) {
        double squared = 0;
        for (int col = 0; col < 768; ++col) {
            double value = values[row * 768 + col];
            require(std::isfinite(value), "non-finite embedding");
            squared += value * value;
        }
        require(std::abs(sqrt(squared) - 1.0) < 0.01, "embedding is not L2 normalized");
    }
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 10) {
        fprintf(stderr, "usage: %s model.gguf coreml-directory prefixed-texts.txt [repetitions=5] [cpu|metal=metal] [workers=1] [parallel=8] [batch=512] [max-inflight=1]\n", argv[0]);
        return 1;
    }
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
        int max_inflight = argc > 9 ? std::stoi(argv[9]) : 1;
        require(max_inflight >= 1 && max_inflight <= 8, "max-inflight must be 1..8");
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
        require(llama_model_n_embd_out(model.get()) == 768, "reference output must have 768 dimensions");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        require(llama_vocab_n_tokens(vocab) == 262144 && llama_vocab_bos(vocab) == 2 && llama_vocab_eos(vocab) == 1,
                "GGUF vocabulary does not match the Core ML model");
        std::ifstream file(argv[3]);
        require(file.is_open(), "failed to open input text file");
        std::vector<std::vector<int32_t>> tokens;
        std::string line;
        while (std::getline(file, line)) {
            require(line.size() <= 1024 * 1024, "input line exceeds 1 MiB");
            int32_t count = llama_tokenize(vocab, line.c_str(), int32_t(line.size()), nullptr, 0, false, false);
            require(count <= 0 && count != std::numeric_limits<int32_t>::min(), "tokenization failed");
            require(-count <= 510, "input exceeds 512 tokens; shorten it explicitly");
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
        std::vector<float> single(tokens.size() * 768), packed(single.size()), reference(single.size());
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
            minimum_packed = std::min(minimum_packed, cosine(single.data() + i * 768, packed.data() + i * 768));
        }
        // Check errors through the public C boundary using a real loaded context.
        int32_t invalid = 262144;
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
        auto baseline = [&]() {
            llama_batch batch = llama_batch_init(batch_size, 0, 1);
            for (size_t start = 0; start < tokens.size();) {
                batch.n_tokens = 0;
                size_t end = start;
                while (end < tokens.size() && end - start < size_t(parallel) && batch.n_tokens + tokens[end].size() <= size_t(batch_size)) {
                    for (size_t pos = 0; pos < tokens[end].size(); ++pos) {
                        int i = batch.n_tokens++;
                        batch.token[i] = tokens[end][pos];
                        batch.pos[i] = int32_t(pos);
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = int32_t(end - start);
                        batch.logits[i] = true;
                    }
                    ++end;
                }
                if (llama_encode(ctx.get(), batch) != 0) {
                    llama_batch_free(batch);
                    throw std::runtime_error("reference inference failed");
                }
                for (size_t i = start; i < end; ++i) {
                    const float * out = llama_get_embeddings_seq(ctx.get(), int32_t(i - start));
                    if (!out) {
                        llama_batch_free(batch);
                        throw std::runtime_error("reference embedding missing");
                    }
                    std::copy(out, out + 768, reference.data() + i * 768);
                }
                start = end;
            }
            llama_batch_free(batch);
        };
        baseline();
        double minimum_reference = 1, mean_reference = 0, minimum_packed_reference = 1;
        for (size_t i = 0; i < tokens.size(); ++i) {
            double value = cosine(reference.data() + i * 768, single.data() + i * 768);
            minimum_reference = std::min(minimum_reference, value);
            mean_reference += value / tokens.size();
            minimum_packed_reference = std::min(minimum_packed_reference, cosine(reference.data() + i * 768, packed.data() + i * 768));
        }
        printf("texts=%zu tokens=%zu packed_vs_single_min_cos=%.8f reference_min_cos=%.8f reference_mean_cos=%.8f packed_reference_min_cos=%.8f\n",
                tokens.size(), total_tokens, minimum_packed, minimum_reference, mean_reference, minimum_packed_reference);
        require(minimum_reference >= 0.998, "Core ML output differs from GGUF reference");
        require(minimum_packed >= 0.998, "packed output differs from single output");
        require(minimum_packed_reference >= 0.998, "packed Core ML output differs from GGUF reference");
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
                                    int32_t(end - start), true, packed.data() + start * 768, (end - start) * 768,
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
                require(cosine(single.data() + i * 768, packed.data() + i * 768) >= 0.998,
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
