#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include "llama.h"

#include <algorithm>
#include <climits>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// Only the behavioral-test executable plants a stale recurrent snapshot. The
// ordinary tool and the inference libraries never contain this fault injection.
#ifdef LLAMA_SPECULATIVE_QUALITY_WRONG_SLOT
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"
#endif

using json = nlohmann::ordered_json;

namespace {

constexpr uint32_t max_continuation = 1024;
constexpr uint32_t format_version = 1;
constexpr char magic[8] = { 'L', 'L', 'Q', 'L', 'O', 'G', 'I', 'T' };
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "FP32 logits required");

struct quality_options {
    std::string tokens;
    std::string record;
    std::string reference;
    uint32_t prefix = 512;
    uint32_t width = 1;
    uint32_t rs = 0;
    uint32_t rollback = 0;
};

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static uint32_t parse_count(const std::string & text) {
    require(!text.empty(), "empty quality count");
    uint32_t value = 0;
    for (char c : text) {
        require(c >= '0' && c <= '9', "invalid quality count: " + text);
        require(value <= (uint32_t(INT_MAX) - (c - '0')) / 10, "quality count too large: " + text);
        value = value * 10 + c - '0';
    }
    return value;
}

static std::vector<char *> parse_quality(int argc, char ** argv, quality_options & options) {
    std::vector<char *> common_args = { argv[0] };
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.compare(0, 10, "--quality-") != 0) {
            common_args.push_back(argv[i]);
            continue;
        }
        require(seen.insert(key).second, "duplicate option: " + key);
        require(i + 1 < argc, "missing value for " + key);
        const std::string value = argv[++i];
        if      (key == "--quality-tokens")    { options.tokens    = value; }
        else if (key == "--quality-record")    { options.record    = value; }
        else if (key == "--quality-reference") { options.reference = value; }
        else if (key == "--quality-prefix")    { options.prefix    = parse_count(value); }
        else if (key == "--quality-width")     { options.width     = parse_count(value); }
        else if (key == "--quality-rs")        { options.rs        = parse_count(value); }
        else if (key == "--quality-rollback")  { options.rollback  = parse_count(value); }
        else { throw std::runtime_error("unknown option: " + key); }
    }
    require(!options.tokens.empty(), "--quality-tokens FILE is required");
    require(options.record.empty() != options.reference.empty(),
            "specify exactly one of --quality-record FILE / --quality-reference FILE");
    require(options.prefix > 0 && options.prefix <= uint32_t(INT_MAX) - max_continuation,
            "quality prefix must be positive and leave room for continuation tokens");
    require(options.width > 0 && options.width <= max_continuation, "quality width must be in [1, 1024]");
    require(options.rollback < options.width && options.rollback <= options.rs,
            "quality rollback must be smaller than width and no larger than rs");
    common_args.push_back(nullptr);
    return common_args;
}

static std::vector<llama_token> load_tokens(const quality_options & options) {
    std::ifstream file(options.tokens);
    require(bool(file), "cannot open token input: " + options.tokens);
    const json input = json::parse(file);
    require(input.is_array(), "token input must be a JSON array");
    require(input.size() > options.prefix && input.size() <= size_t(options.prefix) + max_continuation,
            "token input must contain the prefix and 1..1024 continuation token IDs");
    std::vector<llama_token> tokens;
    tokens.reserve(input.size());
    for (const auto & id : input) {
        require(id.is_number_integer() && id >= 0 && id <= INT_MAX, "invalid integer token ID");
        tokens.push_back(id.get<llama_token>());
    }
    return tokens;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void check_logits(const float * row, uint32_t vocab) {
    require(row != nullptr, "missing logits");
    for (uint32_t i = 0; i < vocab; ++i) {
        // std::isfinite can be optimized away under -ffast-math.
        if ((float_bits(row[i]) & 0x7f800000u) == 0x7f800000u) {
            throw std::runtime_error("non-finite logit at vocabulary index " + std::to_string(i));
        }
    }
}

static bool little_endian() {
    const uint32_t one = 1;
    return *reinterpret_cast<const unsigned char *>(&one) == 1;
}

static uint32_t swap_bytes(uint32_t value) {
    return (value >> 24) | ((value >> 8) & 0x0000ff00u) |
           ((value << 8) & 0x00ff0000u) | (value << 24);
}

// Header integers and FP32 bit patterns use little endian on disk. Memory use
// is one reference row, independent of the number of scored tokens.
class logit_file {
public:
    logit_file(const std::string & path, bool recording) : recording(recording), owned(path != "-") {
        file = owned ? std::fopen(path.c_str(), recording ? "wb" : "rb") : (recording ? stdout : stdin);
        require(file != nullptr, "cannot open logits file: " + path);
#ifdef _WIN32
        if (!owned) {
            require(_setmode(_fileno(file), _O_BINARY) != -1, "cannot set pipe to binary mode");
        }
#endif
    }

    ~logit_file() {
        if (owned) {
            std::fclose(file);
        }
    }

    void header(const json & expected) {
        std::string text = expected.dump();
        if (recording) {
            write_bytes(magic, sizeof(magic));
            write_u32(format_version);
            write_u32(uint32_t(text.size()));
            write_bytes(text.data(), text.size());
        } else {
            char actual_magic[sizeof(magic)];
            read_bytes(actual_magic, sizeof(actual_magic));
            require(std::memcmp(actual_magic, magic, sizeof(magic)) == 0, "invalid logits magic");
            require(read_u32() == format_version, "incompatible logits format version");
            const uint32_t length = read_u32();
            // Our writer emits canonical JSON; bound allocation before parsing an
            // untrusted reference (including stdin) using the expected input size.
            require(length > 0 && length <= text.size() * 2 + 1024, "invalid reference metadata length");
            text.resize(length);
            read_bytes(&text[0], length);
            require(json::parse(text) == expected, "reference metadata mismatch (schedule, context or token IDs)");
        }
    }

    void record(uint32_t phase, uint32_t position, const float * logits, uint32_t vocab) {
        write_u32(phase);
        write_u32(position);
        if (little_endian()) {
            write_bytes(logits, size_t(vocab) * sizeof(float));
        } else {
            for (uint32_t i = 0; i < vocab; ++i) {
                write_u32(float_bits(logits[i]));
            }
        }
    }

    void reference(uint32_t phase, uint32_t position, std::vector<float> & logits) {
        const uint32_t actual_phase = read_u32();
        const uint32_t actual_position = read_u32();
        require(actual_phase == phase && actual_position == position, "reference row schedule mismatch");
        read_bytes(logits.data(), logits.size() * sizeof(float));
        if (!little_endian()) {
            for (float & value : logits) {
                const uint32_t bits = swap_bytes(float_bits(value));
                std::memcpy(&value, &bits, sizeof(bits));
            }
        }
    }

    void finish() {
        if (recording) {
            require(std::fflush(file) == 0, "failed to flush logits reference");
        } else {
            require(std::fgetc(file) == EOF && !std::ferror(file), "trailing bytes or read error in reference");
        }
    }

private:
    FILE * file = nullptr;
    bool recording;
    bool owned;

    void write_bytes(const void * data, size_t size) {
        require(std::fwrite(data, 1, size, file) == size, "failed to write logits reference");
    }
    void read_bytes(void * data, size_t size) {
        require(std::fread(data, 1, size, file) == size, "truncated or unreadable logits reference");
    }
    void write_u32(uint32_t value) {
        if (!little_endian()) {
            value = swap_bytes(value);
        }
        write_bytes(&value, sizeof(value));
    }
    uint32_t read_u32() {
        uint32_t value;
        read_bytes(&value, sizeof(value));
        return little_endian() ? value : swap_bytes(value);
    }
};

struct distribution {
    uint32_t top = 0;
    double maximum;
    double log_sum;

    distribution(const float * logits, uint32_t vocab) {
        for (uint32_t i = 1; i < vocab; ++i) {
            if (logits[i] > logits[top]) {
                top = i;
            }
        }
        maximum = logits[top];
        double sum = 0;
        for (uint32_t i = 0; i < vocab; ++i) {
            sum += std::exp(double(logits[i]) - maximum);
        }
        log_sum = std::log(sum);
    }

    double log_probability(float logit) const {
        return (double(logit) - maximum) - log_sum;
    }
};

struct quality_metrics {
    uint64_t rows = 0;
    uint64_t exact = 0;
    uint64_t top_match = 0;
    double kl = 0;
    double reference_nll = 0;
    double candidate_nll = 0;

    void add(const float * reference, const float * candidate, uint32_t vocab, llama_token truth) {
        const distribution ref(reference, vocab);
        const distribution cand(candidate, vocab);
        ++rows;
        exact += std::memcmp(reference, candidate, size_t(vocab) * sizeof(float)) == 0;
        top_match += ref.top == cand.top;
        double row_kl = 0;
        for (uint32_t i = 0; i < vocab; ++i) {
            const double log_p = ref.log_probability(reference[i]);
            const double log_q = cand.log_probability(candidate[i]);
            row_kl += std::exp(log_p) * (log_p - log_q);
        }
        kl += row_kl;
        reference_nll -= ref.log_probability(reference[truth]);
        candidate_nll -= cand.log_probability(candidate[truth]);
    }

    void merge(const quality_metrics & other) {
        rows += other.rows;
        exact += other.exact;
        top_match += other.top_match;
        kl += other.kl;
        reference_nll += other.reference_nll;
        candidate_nll += other.candidate_nll;
    }

    json result() const {
        require(rows > 0, "no scored rows");
        const double agreement = double(top_match) / rows;
        const double mean_kl = kl / rows;
        const double log_ratio = (candidate_nll - reference_nll) / rows;
        const double ratio = std::exp(std::min(log_ratio, std::log(std::numeric_limits<double>::max())));
        return {
            { "rows", rows }, { "exact_match_count", exact }, { "top_match_count", top_match },
            { "top_token_agreement", agreement }, { "mean_kl", mean_kl },
            { "reference_mean_nll", reference_nll / rows }, { "candidate_mean_nll", candidate_nll / rows },
            { "log_perplexity_ratio", log_ratio },
            { "perplexity_ratio", log_ratio > std::log(std::numeric_limits<double>::max()) ? json(nullptr) : json(ratio) },
            { "passed", agreement >= 0.99 && mean_kl <= 0.002 && log_ratio <= std::log(1.01) },
        };
    }
};

static uint32_t scored_rows(const quality_options & options, uint32_t continuation) {
    uint32_t rows = continuation;
    for (uint32_t offset = 0; offset < continuation; offset += options.width) {
        const uint32_t count = std::min(options.width, continuation - offset);
        rows += std::min(options.rollback, count);
    }
    // The last decoded token has no supplied next token, including its replay.
    return rows - (options.rollback > 0 ? 1 : 0);
}

struct batch_owner {
    llama_batch batch;
    explicit batch_owner(uint32_t size) : batch(llama_batch_init(size, 0, 1)) {}
    ~batch_owner() { llama_batch_free(batch); }
};

#ifdef LLAMA_SPECULATIVE_QUALITY_WRONG_SLOT
static bool test_wrong_slot = true;

// Independent same-build rollback-vs-fresh oracle, following
// test-recurrent-state-rollback: export the pending snapshot into a fresh
// context and decode the SAME replay shape. Comparing to the earlier full-width
// verify state would conflate ordinary row-width arithmetic with state restore.
struct replay_state_oracle {
    std::unique_ptr<llama_context, decltype(&llama_free)> fresh;
    llama_memory_recurrent * memory;

    replay_state_oracle(llama_context * ctx, llama_model * model, llama_context_params cparams) :
        fresh(llama_init_from_model(model, cparams), llama_free) {
        require(fresh != nullptr, "cannot create fresh state-oracle context");
        memory = recurrent(ctx);
        common_prompt_checkpoint checkpoint;
        checkpoint.update_tgt(ctx, 0, 0);
        checkpoint.load_tgt(fresh.get(), 0, 0);
    }

    static llama_memory_recurrent * recurrent(llama_context * ctx) {
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
        require(hybrid != nullptr, "state-oracle behavioral test requires hybrid memory");
        return hybrid->get_mem_recr();
    }

    static std::vector<float> read_row(llama_memory_recurrent * mem, ggml_tensor * tensor) {
        const int32_t cell = mem->cells[0].tail;
        require(cell >= 0 && tensor->type == GGML_TYPE_F32, "state oracle requires a live FP32 cache row");
        std::vector<float> values(tensor->ne[0]);
        ggml_backend_tensor_get(tensor, values.data(), cell * tensor->nb[1], values.size() * sizeof(float));
        check_logits(values.data(), uint32_t(values.size()));
        return values;
    }

    void check(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t begin, uint32_t count) const {
        batch_owner batch(count);
        for (uint32_t i = 0; i < count; ++i) {
            common_batch_add(batch.batch, tokens[begin + i], begin + i, { 0 }, true);
        }
        require(llama_decode(fresh.get(), batch.batch) == 0, "fresh state-oracle replay failed");
        llama_synchronize(fresh.get());
        llama_synchronize(ctx);
        auto * reference_memory = recurrent(fresh.get());
        uint32_t checked = 0;
        for (bool conv : { true, false }) {
            const auto & expected = conv ? reference_memory->r_l : reference_memory->s_l;
            const auto & actual = conv ? memory->r_l : memory->s_l;
            for (size_t layer = 0; layer < expected.size(); ++layer) {
                if (expected[layer] == nullptr) {
                    continue;
                }
                const auto reference = read_row(reference_memory, expected[layer]);
                const auto candidate = read_row(memory, actual[layer]);
                require(reference.size() == candidate.size(), "state-oracle cache shape mismatch");
                double error = 0;
                double norm = 0;
                for (size_t i = 0; i < reference.size(); ++i) {
                    const double delta = double(candidate[i]) - reference[i];
                    error += delta * delta;
                    norm += double(reference[i]) * reference[i];
                }
                if (error > 1e-10 * norm) {
                    throw std::runtime_error(std::string("same-prefix replay state mismatch: ") + actual[layer]->name);
                }
                ++checked;
            }
        }
        require(checked > 0, "state oracle found no recurrent cache");
    }
};
#endif

static int run_quality(common_params & params, const quality_options & options) {
    const auto tokens = load_tokens(options);
    const uint32_t continuation = uint32_t(tokens.size()) - options.prefix;
    require(params.n_ctx > 0 && tokens.size() <= size_t(params.n_ctx), "token schedule exceeds requested context");
    require(params.n_batch > 0 && params.n_ubatch > 0 && options.width <= uint32_t(params.n_batch) &&
            options.width <= uint32_t(params.n_ubatch), "quality width exceeds batch or ubatch size");
    require(options.rs < uint32_t(params.n_ubatch), "quality rs + 1 must fit in an ubatch");
    require(params.lora_adapters.empty() && params.control_vectors.empty(),
            "quality tool supports base-model logits, not adapters or control vectors");

    ggml_backend_load_all();
    auto initialized = common_init_from_params(params, true);
    llama_model * model = initialized->model();
    require(model != nullptr, "failed to load model");
    const uint32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    require(vocab > 0, "model has no vocabulary");
    for (llama_token id : tokens) {
        if (uint32_t(id) >= vocab) {
            throw std::runtime_error("token ID outside model vocabulary: " + std::to_string(id));
        }
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq = options.rs;
    cparams.embeddings = false;
    cparams.n_outputs_max = 0;
    cparams.n_outputs_max_per_seq = 0;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cparams), llama_free);
    require(ctx != nullptr, "failed to initialize quality context");
    const bool recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    const uint32_t actual_rs = llama_n_rs_seq(ctx.get());
    require(!recurrent || actual_rs == options.rs, "runtime changed requested snapshot count");
    require(tokens.size() <= llama_n_ctx(ctx.get()), "token schedule exceeds actual context");

    const uint32_t rows = scored_rows(options, continuation);
    const json metadata = {
        { "vocab", vocab }, { "prefix", options.prefix }, { "continuation", continuation },
        { "width", options.width }, { "rs_requested", options.rs }, { "rs_actual", actual_rs },
        { "gdn_k", recurrent ? actual_rs + 1 : 0 }, { "rollback", options.rollback },
        { "rollback_tail", "min(rollback,batch_rows)" }, { "rows", rows },
        { "n_ctx", llama_n_ctx(ctx.get()) }, { "n_batch", llama_n_batch(ctx.get()) },
        { "n_ubatch", llama_n_ubatch(ctx.get()) },
        { "cache_type_k", int(cparams.type_k) }, { "cache_type_v", int(cparams.type_v) },
        { "flash_attn", int(cparams.flash_attn_type) }, { "tokens", tokens },
    };
    json schedule = metadata;
    schedule.erase("tokens");
    std::fprintf(stderr, "quality_schedule %s\n", schedule.dump().c_str());

    const bool recording = !options.record.empty();
    logit_file file(recording ? options.record : options.reference, recording);
    file.header(metadata); // Reject incompatible schedules before any decoding.
    std::vector<float> reference(recording ? 0 : vocab);
    quality_metrics metrics;
    quality_metrics replay_metrics;
    uint32_t written = 0;
    uint32_t replay_rows = 0;
    const auto score = [&](uint32_t phase, uint32_t position, const float * logits) {
        check_logits(logits, vocab);
        if (position + 1 == tokens.size()) {
            return; // No next-token label: do not accidentally score this token itself.
        }
        if (recording) {
            file.record(phase, position, logits, vocab);
        } else {
            file.reference(phase, position, reference);
            check_logits(reference.data(), vocab);
            quality_metrics row;
            row.add(reference.data(), logits, vocab, tokens[position + 1]);
            metrics.merge(row);
            if (phase == 2) {
                replay_metrics.merge(row);
            }
        }
        ++written;
        replay_rows += phase == 2;
    };

    batch_owner owned_batch(llama_n_batch(ctx.get()));
    llama_batch & batch = owned_batch.batch;
    const auto decode = [&](uint32_t begin, uint32_t count, bool all_logits) {
        common_batch_clear(batch);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t pos = begin + i;
            common_batch_add(batch, tokens[pos], pos, { 0 }, all_logits || pos + 1 == options.prefix);
        }
        if (llama_decode(ctx.get(), batch) != 0) {
            throw std::runtime_error("decode failed at position " + std::to_string(begin));
        }
    };

    // Preserve the production prefill shape (default batch 2048 / ubatch 512),
    // with rollback enabled from context construction, not only during verify.
    for (uint32_t begin = 0; begin < options.prefix;) {
        const uint32_t count = std::min(llama_n_batch(ctx.get()), options.prefix - begin);
        decode(begin, count, false);
        begin += count;
    }
    score(0, options.prefix - 1, llama_get_logits_ith(ctx.get(), -1));

    for (uint32_t begin = options.prefix; begin < tokens.size();) {
        const uint32_t count = std::min(options.width, uint32_t(tokens.size()) - begin);
        decode(begin, count, true);
        for (uint32_t i = 0; i < count; ++i) {
            score(1, begin + i, llama_get_logits_ith(ctx.get(), i));
        }
        const uint32_t rollback = std::min(options.rollback, count);
        if (rollback > 0) {
            const uint32_t replay_begin = begin + count - rollback;
            if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, replay_begin, -1)) {
                throw std::runtime_error("rollback refused at position " + std::to_string(replay_begin));
            }
#ifdef LLAMA_SPECULATIVE_QUALITY_WRONG_SLOT
            const replay_state_oracle oracle(ctx.get(), model, cparams);
            if (test_wrong_slot) {
                // Fault: consume the verify-end state instead of its rollback
                // snapshot; no inference-library code is changed.
                oracle.memory->set_rs_idx(0, 0);
            }
#endif
            decode(replay_begin, rollback, true);
            for (uint32_t i = 0; i < rollback; ++i) {
                score(2, replay_begin + i, llama_get_logits_ith(ctx.get(), i));
            }
#ifdef LLAMA_SPECULATIVE_QUALITY_WRONG_SLOT
            oracle.check(ctx.get(), tokens, replay_begin, rollback);
#endif
        }
        begin += count;
    }
    require(written == rows, "internal scored-row count mismatch");
    file.finish(); // No successful metrics until the full stream and EOF validate.
    json result = { { "mode", recording ? "record" : "compare" }, { "rows", rows },
                    { "continuation_rows", continuation }, { "replay_rows", replay_rows }, { "passed", true } };
    if (!recording) {
        result.update(metrics.result());
        if (replay_metrics.rows > 0) {
            result["replay"] = replay_metrics.result();
            result["passed"] = result["passed"].get<bool>() && result["replay"]["passed"].get<bool>();
        }
    }
    std::fprintf(stderr, "quality_metrics %s\n", result.dump().c_str());
    return result["passed"].get<bool>() ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
#ifdef LLAMA_SPECULATIVE_QUALITY_WRONG_SLOT
        if (argc > 1 && std::strcmp(argv[1], "--check-correct-slot") == 0) {
            test_wrong_slot = false;
            --argc;
            ++argv;
        }
#endif
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                std::fprintf(stderr,
                    "Teacher-forced full-logit quality gate (up to 1024 continuation tokens).\n"
                    "Use standard common model/backend/context arguments, plus:\n"
                    "  --quality-tokens FILE       JSON array: prefix + 1..1024 continuation token IDs\n"
                    "  --quality-prefix N          fixed prefill length (default 512)\n"
                    "  --quality-width N           verify batch width (default 1)\n"
                    "  --quality-rs N              recurrent snapshots (default 0; GDN K=N+1)\n"
                    "  --quality-rollback N        remove/replay trailing rows (default 0)\n"
                    "  --quality-record FILE       write lossless FP32 reference; '-' is stdout\n"
                    "  --quality-reference FILE    compare reference; '-' is stdin\n"
                    "Exactly one record/reference option is required. Diagnostics and JSON metrics use stderr.\n");
                return 0;
            }
        }
        quality_options options;
        auto common_args = parse_quality(argc, argv, options);
        common_params params;
        params.n_ctx = 16384;
        params.n_batch = 2048;
        params.n_ubatch = 512;
        params.n_parallel = 1;
        params.fit_params = false;
        params.sampling.seed = 1234;
        common_init();
        require(common_params_parse(int(common_args.size()) - 1, common_args.data(), params, LLAMA_EXAMPLE_COMMON),
                "invalid common arguments");
        return run_quality(params, options);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "quality_error %s\n", json({ { "error", error.what() }, { "passed", false } }).dump().c_str());
        return 1;
    }
}
