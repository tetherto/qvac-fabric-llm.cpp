// Model-backed fixed-input parity. This deliberately uses teacher forcing,
// rather than comparing trajectories after a backend picks a different token.
#include "audio8/internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::audio8;
using namespace tts_cpp::audio8::detail;

namespace {
constexpr double MIN_COSINE = 0.9999;
constexpr double MAX_NMSE = 2e-4;

std::string environment(const char * name, const char * fallback = "") {
    const char * value = std::getenv(name);
    return value ? value : fallback;
}

struct options {
    std::string lm = environment("AUDIO8_PARITY_LM");
    std::string codec = environment("AUDIO8_PARITY_CODEC");
    std::string backend = environment("AUDIO8_PARITY_BACKEND", "cpu");
    std::string text = "The signal is clear.";
    int frames = 3;
    int threads = 4;
    int slow_layers = 0;
    bool skip_codec = false;
    bool precise_outputs = false;
};

void usage(const char * executable) {
    std::printf("usage: %s --lm model.gguf --codec decoder.gguf [--backend cpu|hexagon|opencl|xdna]\n"
                "       [--text TEXT] [--frames N] [--threads N] [--precise-outputs]\n"
                "       [--slow-layers N] [--skip-codec]\n"
                "Models/backend also accept AUDIO8_PARITY_LM/CODEC/BACKEND. Missing models exit 77.\n",
                executable);
}

int positive_integer(const std::string & value) {
    size_t used = 0;
    const int result = std::stoi(value, &used);
    if (used != value.size() || result <= 0) throw std::runtime_error("expected positive integer: " + value);
    return result;
}

options parse(int argc, char ** argv) {
    options out;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--precise-outputs") { out.precise_outputs = true; continue; }
        if (key == "--skip-codec") { out.skip_codec = true; continue; }
        std::string value;
        const size_t equals = key.find('=');
        if (equals != std::string::npos) {
            value = key.substr(equals + 1);
            key.resize(equals);
        } else {
            if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
            value = argv[++i];
        }
        if (key == "--lm") out.lm = value;
        else if (key == "--codec") out.codec = value;
        else if (key == "--backend") out.backend = value;
        else if (key == "--text") out.text = value;
        else if (key == "--frames") out.frames = positive_integer(value);
        else if (key == "--threads") out.threads = positive_integer(value);
        else if (key == "--slow-layers") out.slow_layers = positive_integer(value);
        else throw std::runtime_error("unknown option " + key);
    }
    if (out.backend != "cpu" && out.backend != "hexagon" && out.backend != "opencl" &&
        out.backend != "xdna") {
        throw std::runtime_error("backend must be cpu, hexagon, opencl or xdna");
    }
    if (out.text.empty()) throw std::runtime_error("text must not be empty");
    return out;
}

struct lm_owner { lm_model model; ~lm_owner() { free_lm(model); } };
struct codec_owner { codec_model model; ~codec_owner() { free_codec(model); } };

void verify_backend(ggml_backend_t backend, const std::string & requested, bool reference = false) {
    const auto dev = ggml_backend_get_device(backend);
    const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    if (!backend_selection_matches(requested, reg, ggml_backend_dev_name(dev), ggml_backend_dev_type(dev))) {
        throw std::runtime_error("requested " + requested + ", resolved " + ggml_backend_name(backend));
    }
    std::printf("backend requested=%s resolved=%s registry=%s\n", requested.c_str(), ggml_backend_name(backend), reg);
    if (reference) {
        using set_ref_fn = void (*)(ggml_backend_t, bool);
        const auto set_ref = reinterpret_cast<set_ref_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cpu_set_use_ref"));
        if (!set_ref) throw std::runtime_error("CPU reference implementation unavailable");
        set_ref(backend, true);
    }
}

bool finite(const std::vector<float> & values) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

// Ranks use the host picker's first-index tie break; report margins so a
// near-tie disagreement is visible alongside the continuous error metrics.
int rank_of(const std::vector<float> & values, int index) {
    int rank = 1;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] > values[index] || (values[i] == values[index] && int(i) < index)) ++rank;
    }
    return rank;
}

double margin(const std::vector<float> & values, int best) {
    float runner_up = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < values.size(); ++i) if (int(i) != best) runner_up = std::max(runner_up, values[i]);
    return double(values[best]) - runner_up;
}

struct reporter {
    int numeric_failures = 0;
    int token_mismatches = 0;
    int chain_mismatches = 0;
    int execution_failures = 0;

    void failure(const std::string & stage, const std::string & error) {
        ++execution_failures;
        std::printf("execution stage=%s FAIL %s\n", stage.c_str(), error.c_str());
    }

    void compare(const std::string & stage, const std::vector<float> & ref, const std::vector<float> & got,
                 bool logits = false, int eos = -1) {
        const bool ref_finite = finite(ref), got_finite = finite(got);
        const bool same_size = ref.size() == got.size();
        double dot = 0, ref_energy = 0, got_energy = 0, error_energy = 0, max_abs = 0;
        double cosine = std::numeric_limits<double>::quiet_NaN();
        double nmse = cosine;
        if (ref_finite && got_finite && same_size) {
            for (size_t i = 0; i < ref.size(); ++i) {
                const double r = ref[i], g = got[i], delta = r - g;
                dot += r * g; ref_energy += r * r; got_energy += g * g;
                error_energy += delta * delta; max_abs = std::max(max_abs, std::abs(delta));
            }
            cosine = ref_energy && got_energy ? dot / std::sqrt(ref_energy * got_energy)
                     : (ref_energy == got_energy ? 1.0 : 0.0);
            cosine = std::max(-1.0, std::min(1.0, cosine));
            nmse = ref_energy ? error_energy / ref_energy
                   : (error_energy == 0 ? 0.0 : std::numeric_limits<double>::infinity());
        } else {
            max_abs = std::numeric_limits<double>::quiet_NaN();
        }
        const bool pass = ref_finite && got_finite && same_size &&
                          std::isfinite(cosine) && std::isfinite(nmse) &&
                          cosine >= MIN_COSINE && nmse <= MAX_NMSE;
        numeric_failures += !pass;
        std::printf("metric stage=%s n_ref=%zu n_test=%zu finite_ref=%d finite_test=%d cosine=%.10f "
                    "nmse=%.10g max_abs=%.10g gate=%s\n", stage.c_str(), ref.size(), got.size(),
                    ref_finite, got_finite, cosine, nmse, max_abs, pass ? "PASS" : "FAIL");
        if (!logits || !ref_finite || !got_finite || !same_size) return;
        const int ref_top = argmax_of(ref), got_top = argmax_of(got);
        token_mismatches += ref_top != got_top;
        std::printf("rank stage=%s top1_ref=%d top1_test=%d ref_winner_rank_test=%d "
                    "margin_ref=%.10g margin_test=%.10g top1_match=%d",
                    stage.c_str(), ref_top, got_top, rank_of(got, ref_top),
                    margin(ref, ref_top), margin(got, got_top), ref_top == got_top);
        if (eos >= 0 && size_t(eos) < ref.size()) {
            std::printf(" eos_index=%d eos_rank_ref=%d eos_rank_test=%d eos_top1_ref=%d eos_top1_test=%d",
                        eos, rank_of(ref, eos), rank_of(got, eos), ref_top == eos, got_top == eos);
        }
        std::puts("");
    }
};

using logit_trace = std::vector<std::vector<float>>;

std::vector<int32_t> lm_parity(const options & opts, reporter & report, int & books) {
    lm_owner ref_owner, test_owner;
    auto & ref = ref_owner.model;
    auto & test = test_owner.model;
    std::string error;
    if (!load_lm(opts.lm, 0, "cpu", ref, &error) ||
        !load_lm(opts.lm, opts.backend == "cpu" ? 0 : 99, opts.backend, test, &error)) {
        throw std::runtime_error(error);
    }
    verify_backend(ref.backend, "cpu", true);
    verify_backend(test.backend, opts.backend);
    if (opts.precise_outputs) test.precise_outputs = true;
    if (opts.slow_layers) {
        if (size_t(opts.slow_layers) > ref.blocks.size() || size_t(opts.slow_layers) > test.blocks.size()) {
            throw std::runtime_error("--slow-layers exceeds loaded model block count");
        }
        std::printf("diagnostic slow_layers=%d loaded_layers=%zu (truncated model output)\n",
                    opts.slow_layers, ref.blocks.size());
        // Keep the allocated KV cache and weight storage unchanged. Graph
        // construction visits only the selected prefix of block descriptors.
        ref.blocks.resize(opts.slow_layers);
        test.blocks.resize(opts.slow_layers);
    }
    std::printf("precision lm_reference=%d lm_candidate=%d\n", ref.precise_outputs, test.precise_outputs);
    books = ref.hp.num_codebooks;
    const Tokenizer tokenizer(ref.tokenizer);
    const auto prompt = build_frames(ref.hp, build_prompt(tokenizer, opts.text, "", false), {}, 0);
    if (prompt.width <= 0 || opts.frames > ref.hp.max_seq_len - prompt.width) {
        throw std::runtime_error("prompt plus requested frames exceeds context, or prompt is empty");
    }
    std::printf("prompt text=%s tokens=%d requested_frames=%d books=%d\n", opts.text.c_str(), prompt.width, opts.frames, books);
    std::vector<int32_t> frame_codes, column = prompt.values;
    int width = prompt.width, n_past = 0;
    for (int frame = 0; frame < opts.frames; ++frame) {
        const std::string tag = "frame" + std::to_string(frame);
        std::vector<float> ref_logits, ref_input, test_logits, test_input;
        if (!slow_step(ref, column.data(), width, n_past, opts.threads, ref_logits, ref_input, &error)) {
            report.failure(tag + ".slow.reference", error); break;
        }
        if (!slow_step(test, column.data(), width, n_past, opts.threads, test_logits, test_input, &error)) {
            report.failure(tag + ".slow.candidate", error);
        }
        report.compare(tag + (frame == 0 ? ".prefill.logits" : ".slow.logits"), ref_logits, test_logits, true, ref.hp.codebook_size);
        report.compare(tag + ".slow.fast_input", ref_input, test_input);
        if (!finite(ref_logits) || !finite(ref_input)) { report.failure(tag, "CPU reference is nonfinite"); break; }
        const int semantic_index = argmax_of(ref_logits);
        if (semantic_index == ref.hp.codebook_size) {
            std::printf("trajectory frame=%d CPU_EOS=1 stopping_teacher_forcing\n", frame); break;
        }
        const int semantic = ref.hp.semantic_begin + semantic_index;
        logit_trace ref_trace(books), test_trace(books);
        std::vector<int32_t> ref_codes, forced_codes;
        const auto ref_picker = [&](const std::vector<float> & logits, int position) {
            ref_trace.at(position) = logits;
            return argmax_of(logits);
        };
        if (!fast_step(ref, ref_input, semantic, opts.threads, ref_picker, ref_codes, &error)) {
            report.failure(tag + ".fast.reference", error); break;
        }
        if (ref_codes.size() != size_t(books)) {
            report.failure(tag + ".fast.reference", "incorrect code count"); break;
        }
        const auto forced_picker = [&](const std::vector<float> & logits, int position) {
            test_trace.at(position) = logits;
            return ref_codes.at(position);
        };
        if (!fast_step(test, ref_input, semantic, opts.threads, forced_picker, forced_codes, &error)) {
            report.failure(tag + ".fast.candidate", error);
        }
        if (forced_codes.size() != size_t(books)) {
            report.failure(tag + ".fast.candidate", "incorrect code count");
        }
        bool valid_reference = true;
        for (int book = 1; book < books; ++book) {
            report.compare(tag + ".fast.book" + std::to_string(book), ref_trace[book], test_trace[book], true);
            valid_reference = finite(ref_trace[book]) && valid_reference;
        }
        if (!valid_reference) { report.failure(tag, "CPU fast reference is nonfinite"); break; }

        // Separate check: same candidate backend, normal greedy decisions,
        // same CPU-carried input, comparing per-position against chained graphs.
        std::vector<int32_t> per_position, chained;
        const auto greedy = [&](const std::vector<float> & logits, int position) {
            if (!finite(logits) || logits.size() != size_t(test.hp.codebook_size)) {
                report.failure(tag + ".chain.per_position.book" + std::to_string(position),
                               "nonfinite logits or incorrect logit count");
                return 0;
            }
            return argmax_of(logits);
        };
        const bool step_ok = fast_step(test, ref_input, semantic, opts.threads, greedy, per_position, &error);
        if (!step_ok) report.failure(tag + ".chain.per_position", error);
        const bool chain_ok = fast_frame(test, ref_input, semantic, opts.threads, chained, &error);
        if (!chain_ok) report.failure(tag + ".chain.chained", error);
        if (step_ok && chain_ok) {
            if (per_position.size() != size_t(books) || chained.size() != size_t(books)) {
                report.failure(tag + ".chain", "incorrect code count");
            }
            for (size_t book = 0; book < size_t(books); ++book) {
                const bool same = book < per_position.size() && book < chained.size() &&
                                  per_position[book] == chained[book];
                report.chain_mismatches += !same;
                std::printf("chain frame=%d book=%zu per_position=%d chained=%d match=%d\n", frame, book,
                            book < per_position.size() ? per_position[book] : -1,
                            book < chained.size() ? chained[book] : -1, same);
            }
        }
        std::printf("teacher_codes frame=%d", frame);
        for (int32_t code : ref_codes) std::printf(" %d", code);
        std::puts("");
        frame_codes.insert(frame_codes.end(), ref_codes.begin(), ref_codes.end());
        column.resize(books + 1);
        column[0] = semantic;
        std::copy(ref_codes.begin(), ref_codes.end(), column.begin() + 1);
        n_past += width;
        width = 1;
    }
    return frame_codes;
}

void codec_parity(const options & opts, reporter & report, const std::vector<int32_t> & frames, int books) {
    if (frames.empty()) { report.failure("codec", "no CPU reference frames available"); return; }
    const int count = int(frames.size() / books);
    const auto rows = codebook_rows(frames, books, 0, count);
    codec_owner ref_owner, test_owner;
    auto & ref = ref_owner.model;
    auto & test = test_owner.model;
    std::string error;
    if (!load_codec(opts.codec, 0, "cpu", ref, &error) ||
        !load_codec(opts.codec, opts.backend == "cpu" ? 0 : 99, opts.backend, test, &error)) {
        report.failure("codec.load", error); return;
    }
    verify_backend(ref.backend, "cpu", true);
    verify_backend(test.backend, opts.backend);
    if (ref.hp.num_codebooks != books || test.hp.num_codebooks != books) {
        report.failure("codec", "LM and codec codebook counts differ"); return;
    }
    if (opts.precise_outputs) test.precise_outputs = true;
    ref.synthesis_block_frames = test.synthesis_block_frames = count;
    std::printf("precision codec_reference=%d codec_candidate=%d fixed_code_frames=%d\n",
                ref.precise_outputs, test.precise_outputs, count);
    decode_taps ref_taps, test_taps;
    std::vector<float> ref_pcm, test_pcm;
    if (!decode_codes(ref, rows.data(), count, opts.threads, {}, ref_pcm, &error, &ref_taps)) {
        report.failure("codec.reference", error); return;
    }
    if (!decode_codes(test, rows.data(), count, opts.threads, {}, test_pcm, &error, &test_taps)) {
        report.failure("codec.candidate", error);
    }
    report.compare("codec.semantic", ref_taps.semantic, test_taps.semantic);
    report.compare("codec.residual", ref_taps.residual, test_taps.residual);
    report.compare("codec.post", ref_taps.post, test_taps.post);
    report.compare("codec.latent", ref_taps.latent, test_taps.latent);
    report.compare("codec.pcm", ref_pcm, test_pcm);
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(argv[0]); return 0; }
    try {
        const auto opts = parse(argc, argv);
        if (opts.lm.empty() || !std::ifstream(opts.lm).good() ||
            (!opts.skip_codec && (opts.codec.empty() || !std::ifstream(opts.codec).good()))) {
            std::puts("SKIP: supply readable Audio8 LM and decoder GGUF files (--lm/--codec or AUDIO8_PARITY_LM/CODEC; --skip-codec needs only LM)");
            return 77;
        }
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        std::printf("audio8 parity cosine_gate=%.4f nmse_gate=%.4g threads=%d backend=%s precise_outputs_override=%d\n",
                    MIN_COSINE, MAX_NMSE, opts.threads, opts.backend.c_str(), opts.precise_outputs);
        reporter report;
        int books = 0;
        const auto codes = lm_parity(opts, report, books);
        if (!opts.skip_codec) codec_parity(opts, report, codes, books);
        else std::puts("diagnostic codec=SKIPPED (--skip-codec)");
        const bool pass = report.numeric_failures == 0 && report.token_mismatches == 0 &&
                          report.chain_mismatches == 0 && report.execution_failures == 0;
        std::printf("summary completed_frames=%zu requested_frames=%d numeric_failures=%d token_mismatches=%d chain_mismatches=%d execution_failures=%d %s\n",
                    books > 0 ? codes.size() / books : size_t(0), opts.frames,
                    report.numeric_failures, report.token_mismatches, report.chain_mismatches,
                    report.execution_failures, pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "audio8 parity: %s\n", error.what());
        return 1;
    }
}
