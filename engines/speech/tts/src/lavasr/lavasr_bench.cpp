// Native LavaSR correctness and timing consumer. File I/O and reference
// comparisons are outside synthesis timers; resampling/audio DSP are included.
#include "tts-cpp/lavasr/denoiser.h"
#include "tts-cpp/lavasr/enhancer.h"
#include "bench_validation.h"
#include "bench_timing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

namespace {
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;
using tts_cpp::lavasr::Denoiser;
using tts_cpp::lavasr::Enhancer;
using tts_cpp::lavasr::bench::Difference;
using tts_cpp::lavasr::bench::Timing;

struct Options {
    std::string denoiser, enhancer, input, out_dn, out_enh, json_out;
    std::string denoiser_coreml, enhancer_coreml, coreml_compute_units;
    std::string backend = "auto", stage = "pipeline";
    int rate = 24000, runs = 5, warmup = 1, dn_layers = 0, enh_layers = 0;
    double seconds = 6.0;
    double min_seconds = 0.0;
    bool verify = false;
};
struct Verification { std::string stage; Difference difference; bool passed; std::string reference_backend; };

void usage(const char * exe) {
    std::fprintf(stderr,
        "usage: %s --stage denoiser|enhancer|pipeline --backend cpu|opencl|hexagon|auto\n"
        "  [--denoiser DN.gguf] [--enhancer ENH.gguf] [--in input.wav]\n"
        "  [--seconds 6] [--in-rate 24000] [--runs 5] [--warmup 1] [--verify]\n"
        "  [--min-seconds 0] (minimum sum of measured processing seconds)\n"
        "  [--out-denoised dn.wav] [--out-enhanced enh.wav] [--json-out result.json]\n"
        "  Auto mode: [--denoiser-coreml DN.mlmodelc] [--enhancer-coreml ENH.mlmodelc]\n"
        "  [--coreml-compute-units CPU_ONLY|CPU_AND_GPU|CPU_AND_NE|ALL]\n"
        "  Legacy: --dn-gpu-layers N --n-gpu-layers N (0=CPU default, >0=accelerator)\n"
        "  --runs 1 --warmup 0 --min-seconds 0 measures the first call once.\n"
        "  Other modes keep the first call and warmups separate from measured calls.\n", exe);
}

int integer(const std::string & s) {
    size_t used = 0;
    int value = std::stoi(s, &used);
    if (used != s.size()) throw std::invalid_argument("invalid integer: " + s);
    return value;
}
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 == argc) throw std::invalid_argument(a + " requires a value");
            return std::string(argv[++i]);
        };
        if      (a == "--denoiser") o.denoiser = next();
        else if (a == "--enhancer") o.enhancer = next();
        else if (a == "--in") o.input = next();
        else if (a == "--out-denoised") o.out_dn = next();
        else if (a == "--out-enhanced") o.out_enh = next();
        else if (a == "--json-out") o.json_out = next();
        else if (a == "--backend") o.backend = next();
        else if (a == "--stage") o.stage = next();
        else if (a == "--denoiser-coreml") o.denoiser_coreml = next();
        else if (a == "--enhancer-coreml") o.enhancer_coreml = next();
        else if (a == "--coreml-compute-units") o.coreml_compute_units = next();
        else if (a == "--runs") o.runs = integer(next());
        else if (a == "--warmup") o.warmup = integer(next());
        else if (a == "--in-rate") o.rate = integer(next());
        else if (a == "--dn-gpu-layers") o.dn_layers = integer(next());
        else if (a == "--n-gpu-layers") o.enh_layers = integer(next());
        else if (a == "--seconds" || a == "--min-seconds") {
            const std::string s = next(); size_t used = 0;
            const double value = std::stod(s, &used);
            if (used != s.size()) throw std::invalid_argument("invalid duration: " + s);
            if (a == "--seconds") o.seconds = value;
            else o.min_seconds = value;
        }
        else if (a == "--verify") o.verify = true;
        else throw std::invalid_argument("unknown option: " + a);
    }
    if (o.stage != "denoiser" && o.stage != "enhancer" && o.stage != "pipeline")
        throw std::invalid_argument("stage must be denoiser, enhancer or pipeline");
    if (o.backend != "cpu" && o.backend != "opencl" && o.backend != "hexagon" && o.backend != "auto")
        throw std::invalid_argument("backend must be cpu, opencl, hexagon or auto");
    if (o.runs <= 0 || o.warmup < 0 || o.warmup > 100000 || o.runs > 100000 ||
        o.rate <= 0 || o.rate > 384000 || !std::isfinite(o.seconds) || o.seconds <= 0.0 ||
        !std::isfinite(o.min_seconds) || o.min_seconds < 0.0 || o.min_seconds > 3600.0)
        throw std::invalid_argument("invalid runs, warmup, sample rate or duration");
    if (o.stage != "enhancer" && o.denoiser.empty()) throw std::invalid_argument("--denoiser is required");
    if (o.stage != "denoiser" && o.enhancer.empty()) throw std::invalid_argument("--enhancer is required");
    if (o.stage == "enhancer" && !o.out_dn.empty()) throw std::invalid_argument("enhancer-only has no denoised output");
    if (o.stage == "denoiser" && !o.out_enh.empty()) throw std::invalid_argument("denoiser-only has no enhanced output");
    const bool coreml = !o.denoiser_coreml.empty() || !o.enhancer_coreml.empty();
    if (coreml && o.backend != "auto")
        throw std::invalid_argument("Core ML options require --backend auto; explicit CPU/OpenCL/Hexagon selects the native backend");
    if (coreml != !o.coreml_compute_units.empty())
        throw std::invalid_argument("Core ML requires a model path and --coreml-compute-units together");
    if (coreml && o.coreml_compute_units != "CPU_ONLY" && o.coreml_compute_units != "CPU_AND_GPU" &&
        o.coreml_compute_units != "CPU_AND_NE" && o.coreml_compute_units != "ALL")
        throw std::invalid_argument("invalid Core ML compute policy");
    if ((!o.denoiser_coreml.empty() && o.stage == "enhancer") ||
        (!o.enhancer_coreml.empty() && o.stage == "denoiser"))
        throw std::invalid_argument("Core ML options name an unused stage");
    if ((!o.denoiser_coreml.empty() && o.dn_layers != 0) ||
        (!o.enhancer_coreml.empty() && o.enh_layers > 0))
        throw std::invalid_argument("a Core ML stage cannot also request native GPU layers");
    if (o.backend != "auto") {
        o.dn_layers = o.backend == "cpu" ? -1 : 1;
        o.enh_layers = o.backend == "cpu" ? 0 : 1;
    }
    return o;
}

std::string runtime_precision(const std::string & backend) {
    if (backend == "unused") return "unused";
    return backend.rfind("coreml-", 0) == 0 ? "coreml-model-defined" : "f32";
}

void select_backend(const std::string & backend) {
    if (backend == "auto" || backend == "cpu") return;
#if defined(_WIN32)
    if (_putenv_s("TTS_CPP_GPU_BACKEND", backend.c_str()) != 0)
#else
    if (setenv("TTS_CPP_GPU_BACKEND", backend.c_str(), 1) != 0)
#endif
        throw std::runtime_error("cannot set requested backend");
}
void require_backend(const std::string & requested, const std::string & actual) {
    if (requested == "auto") return;
    const bool matches = requested == "cpu" ? actual == "CPU" :
                         requested == "hexagon" ? actual == "HTP0" : actual.rfind("OpenCL", 0) == 0;
    if (!matches) throw std::runtime_error("requested " + requested + " but actual backend is " + actual);
}
void require_pcm(const std::vector<float> & pcm, size_t expected, const char * label) {
    if (pcm.size() != expected) throw std::runtime_error(std::string(label) + ": output length mismatch");
    if (!std::all_of(pcm.begin(), pcm.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error(std::string(label) + ": non-finite PCM");
}
std::vector<float> load_audio(const Options & o, int & rate) {
    if (o.input.empty()) {
        const double length = std::round(o.seconds * rate);
        if (length < 1.0 || length > (double) std::numeric_limits<int>::max())
            throw std::invalid_argument("synthetic input is too short or too long");
        std::vector<float> pcm((size_t) length);
        constexpr double pi = 3.14159265358979323846;
        for (size_t i = 0; i < pcm.size(); ++i) {
            const double t = (double) i / rate;
            pcm[i] = (float) (0.15 * std::sin(2*pi*180*t) + 0.10 * std::sin(2*pi*440*t) + 0.05 * std::sin(2*pi*900*t));
        }
        return pcm;
    }
    unsigned int channels = 0, sample_rate = 0;
    drwav_uint64 frames = 0;
    std::unique_ptr<float, void (*)(float *)> data(
        drwav_open_file_and_read_pcm_frames_f32(o.input.c_str(), &channels, &sample_rate, &frames, nullptr),
        [](float * p) { drwav_free(p, nullptr); });
    if (!data || channels == 0 || sample_rate == 0 || sample_rate > 384000 ||
        frames == 0 || frames > (drwav_uint64) std::numeric_limits<int>::max())
        throw std::runtime_error("cannot read supported non-empty WAV: " + o.input);
    rate = (int) sample_rate;
    std::vector<float> mono((size_t) frames);
    for (size_t i = 0; i < mono.size(); ++i) {
        double sum = 0.0;
        for (unsigned int c = 0; c < channels; ++c) sum += data.get()[i * channels + c];
        mono[i] = (float) (sum / channels);
    }
    return mono;
}
void write_wav(const std::string & path, const std::vector<float> & pcm, int rate) {
    if (path.empty()) return;
    const drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_IEEE_FLOAT, 1, (drwav_uint32) rate, 32};
    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) throw std::runtime_error("cannot open WAV: " + path);
    const drwav_uint64 written = drwav_write_pcm_frames(&wav, pcm.size(), pcm.data());
    const drwav_result closed = drwav_uninit(&wav);
    if (written != pcm.size() || closed != DRWAV_SUCCESS) throw std::runtime_error("cannot finish WAV: " + path);
}
std::string quoted(const std::string & s) {
    std::string out = "\"";
    const char * hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += (char)c; }
        else if (c < 0x20) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += (char)c;
    }
    return out + '"';
}
void timing_json(std::ostream & out, const Timing & t) {
    out << "{\"denoiser_s\":" << t.denoiser << ",\"enhancer_s\":" << t.enhancer << ",\"total_s\":" << t.total << '}';
}
double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid-1] + values[mid]) * 0.5;
}
void report(const char * label, const std::vector<Timing> & calls, double Timing::* field, double audio_s) {
    double sum = 0.0;
    std::vector<double> values;
    for (const Timing & t : calls) { sum += t.*field; values.push_back(t.*field); }
    std::printf("  %s: mean=%.6f s median=%.6f s RTF=%.6f (%zu measured calls)\n",
                label, sum / values.size(), median(values), median(values) / audio_s, values.size());
}

int run(const Options & o) {
    select_backend(o.backend);
    int rate = o.rate;
    const std::vector<float> input = load_audio(o, rate);
    require_pcm(input, input.size(), "input");
    const double audio_s = (double) input.size() / rate;
    const bool do_dn = o.stage != "enhancer", do_enh = o.stage != "denoiser";
    std::unique_ptr<Denoiser> dn;
    std::unique_ptr<Enhancer> enh;
    Timing constructor;
    if (do_dn) {
        tts_cpp::lavasr::DenoiserOptions opts;
        opts.n_gpu_layers = o.dn_layers;
        if (!o.denoiser_coreml.empty()) opts.coreml = {o.denoiser_coreml, o.coreml_compute_units};
        const auto t = Clock::now();
        dn = Denoiser::load(o.denoiser, opts);
        constructor.denoiser = Seconds(Clock::now() - t).count();
        require_backend(o.backend, dn->backend_name());
    }
    if (do_enh) {
        tts_cpp::lavasr::EnhancerOptions opts;
        opts.use_gpu = o.enh_layers > 0;
        opts.verbose = true;
        if (!o.enhancer_coreml.empty()) opts.coreml = {o.enhancer_coreml, o.coreml_compute_units};
        const auto t = Clock::now();
        enh = Enhancer::load(o.enhancer, opts);
        constructor.enhancer = Seconds(Clock::now() - t).count();
        require_backend(o.backend, enh->backend_name());
    }
    constructor.total = constructor.denoiser + constructor.enhancer;
    const std::string dn_precision = runtime_precision(dn ? dn->backend_name() : "unused");
    const std::string enh_precision = runtime_precision(enh ? enh->backend_name() : "unused");
    const bool any_coreml = dn_precision == "coreml-model-defined" || enh_precision == "coreml-model-defined";
    const bool any_native = dn_precision == "f32" || enh_precision == "f32";
    const std::string precision = any_coreml ? (any_native ? "mixed" : "coreml-model-defined") : "f32";
    const int output_rate = do_enh ? enh->output_sample_rate() : rate;
    if (do_enh && output_rate != 48000) throw std::runtime_error("enhancer output rate must be 48000 Hz");
    const double output_count = std::round((double) input.size() * output_rate / rate);
    if (output_count <= 0 || output_count > std::numeric_limits<int>::max()) throw std::runtime_error("unsupported output length");
    std::vector<float> cleaned, enhanced;
    auto call = [&]() {
        Timing t;
        const auto start = Clock::now();
        if (do_dn) cleaned = dn->denoise(input, rate);
        const auto split = Clock::now();
        if (do_enh) enhanced = enh->enhance(do_dn ? cleaned : input, rate);
        const auto end = Clock::now();
        t.denoiser = do_dn ? Seconds(split - start).count() : 0.0;
        t.enhancer = do_enh ? Seconds(end - split).count() : 0.0;
        t.total = Seconds(end - start).count();
        if (do_dn) require_pcm(cleaned, input.size(), "denoiser");
        if (do_enh) require_pcm(enhanced, (size_t) output_count, "enhancer");
        return t;
    };
    const auto measurements = tts_cpp::lavasr::bench::measure(o.runs, o.warmup, o.min_seconds, call);
    const auto & first = measurements.first;
    const auto & warmups = measurements.warmups;
    const auto & calls = measurements.calls;
    const double measured_seconds = measurements.measured_seconds;
    std::printf("LavaSR %s: denoiser=%s enhancer=%s\n", o.stage.c_str(),
                dn ? dn->backend_name().c_str() : "unused", enh ? enh->backend_name().c_str() : "unused");
    std::printf("  neural precision: denoiser=%s enhancer=%s%s\n", dn_precision.c_str(), enh_precision.c_str(),
                any_coreml ? "; Core ML precision follows the compiled model; placement is framework-managed" : "; F32 graph / F32 matmul");
    std::printf("  constructor=%.6f s first=%.6f s (%s) warmup=%d (excluded)\n",
                constructor.total, first.total, measurements.first_call_measured ? "measured" : "excluded", o.warmup);
    if (do_dn) report("denoiser", calls, &Timing::denoiser, audio_s);
    if (do_enh) report("enhancer", calls, &Timing::enhancer, audio_s);
    report("total", calls, &Timing::total, audio_s);

    std::vector<Verification> verification;
    if (o.verify) {
        auto verify = [&](const char * stage, const std::vector<float> & actual, const std::vector<float> & reference) {
            Difference diff = tts_cpp::lavasr::bench::compare(actual, reference);
            // Existing denoiser pipeline/max-abs and neural NRMSE gates; enhancer
            // waveform max-abs gate from test-lavasr-enhancer-gguf. Spectral
            // parity retains its tighter existing gates in the neural tests.
            const bool denoiser = std::string(stage) == "denoiser";
            const bool pass = tts_cpp::lavasr::bench::passes(diff, denoiser ? 2e-3 : 5e-2,
                                                          denoiser ? 2e-3 : 1e100, 0.9999);
            const std::string reference_backend = denoiser ? "scalar" : std::string(stage) == "pipeline" ? "scalar+CPU" : "CPU";
            verification.push_back({stage, diff, pass, reference_backend});
            std::printf("  VERIFY %s %s length=%d finite=%d max_abs=%.6g nrmse=%.6g cosine=%.9g\n",
                        stage, pass ? "PASS" : "FAIL", diff.length_ok, diff.finite,
                        diff.max_abs, diff.nrmse, diff.cosine);
        };
        std::vector<float> cpu_cleaned;
        if (do_dn) {
            tts_cpp::lavasr::DenoiserOptions reference_options;
            reference_options.n_gpu_layers = 0;
            reference_options.coreml = {};
            auto reference = Denoiser::load(o.denoiser, reference_options);
            if (reference->backend_name() != "scalar") throw std::runtime_error("denoiser reference must use scalar CPU");
            cpu_cleaned = reference->denoise(input, rate);
            verify("denoiser", cleaned, cpu_cleaned);
        }
        if (do_enh) {
            tts_cpp::lavasr::EnhancerOptions reference_options;
            reference_options.use_gpu = false;
            reference_options.coreml = {};
            auto reference = Enhancer::load(o.enhancer, reference_options);
            if (reference->backend_name() != "CPU" || reference->backend_device() != tts_cpp::BackendDevice::CPU)
                throw std::runtime_error("enhancer reference must use native CPU");
            const auto same_input = reference->enhance(do_dn ? cleaned : input, rate);
            verify("enhancer", enhanced, same_input);
            if (do_dn) verify("pipeline", enhanced, reference->enhance(cpu_cleaned, rate));
        }
    }
    const bool passed = std::all_of(verification.begin(), verification.end(), [](const Verification & v) { return v.passed; });
    write_wav(o.out_dn, cleaned, rate);
    write_wav(o.out_enh, enhanced, output_rate);
    if (!o.json_out.empty()) {
        std::ofstream out(o.json_out, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open JSON: " + o.json_out);
        out << std::setprecision(17) << "{\"schema_version\":1,\"stage\":" << quoted(o.stage)
            << ",\"requested_backend\":" << quoted(o.backend)
            << ",\"denoiser_backend\":" << quoted(dn ? dn->backend_name() : "unused")
            << ",\"enhancer_backend\":" << quoted(enh ? enh->backend_name() : "unused")
            << ",\"graph_dtype\":" << quoted(precision) << ",\"matmul_precision\":" << quoted(precision)
            << ",\"denoiser_runtime_precision\":" << quoted(dn_precision)
            << ",\"enhancer_runtime_precision\":" << quoted(enh_precision)
            << ",\"denoiser_coreml_model\":" << quoted(o.denoiser_coreml)
            << ",\"enhancer_coreml_model\":" << quoted(o.enhancer_coreml)
            << ",\"coreml_compute_units\":" << quoted(o.coreml_compute_units) << ",\"host_audio_dsp\":true"
            << ",\"input_samples\":" << input.size() << ",\"input_rate\":" << rate
            << ",\"denoised_samples\":" << cleaned.size() << ",\"enhanced_samples\":" << enhanced.size()
            << ",\"output_rate\":" << output_rate << ",\"audio_s\":" << audio_s
            << ",\"min_seconds\":" << o.min_seconds << ",\"measured_seconds\":" << measured_seconds
            << ",\"denoiser_model\":" << quoted(o.denoiser) << ",\"enhancer_model\":" << quoted(o.enhancer)
            << ",\"first_call_measured\":" << (measurements.first_call_measured ? "true" : "false")
            << ",\"timing_scope\":" << quoted(measurements.first_call_measured
                ? "native processing including host audio DSP; excludes I/O, verification, constructor and warmups; first call included in measured calls"
                : "native processing including host audio DSP; excludes I/O, verification, constructor, first call and warmups")
            << ",\"constructor\":"; timing_json(out, constructor);
        out << ",\"first\":"; timing_json(out, first);
        out << ",\"warmups\":[";
        for (size_t i = 0; i < warmups.size(); ++i) { if (i) out << ','; timing_json(out, warmups[i]); }
        out << "],\"calls\":[";
        for (size_t i = 0; i < calls.size(); ++i) { if (i) out << ','; timing_json(out, calls[i]); }
        out << "],\"verification_requested\":" << (o.verify ? "true" : "false") << ",\"verification\":[";
        for (size_t i = 0; i < verification.size(); ++i) {
            if (i) out << ',';
            const auto & v = verification[i]; const auto & d = v.difference;
            out << "{\"stage\":" << quoted(v.stage) << ",\"passed\":" << (v.passed ? "true" : "false")
                << ",\"reference_backend\":" << quoted(v.reference_backend)
                << ",\"length_ok\":" << (d.length_ok ? "true" : "false") << ",\"finite\":" << (d.finite ? "true" : "false")
                << ",\"max_abs\":" << d.max_abs << ",\"nrmse\":" << d.nrmse << ",\"cosine\":" << d.cosine << '}';
        }
        out << "],\"passed\":" << (passed ? "true" : "false") << "}\n";
        out.close();
        if (!out) throw std::runtime_error("cannot finish JSON: " + o.json_out);
    }
    return passed ? 0 : 1;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        usage(argv[0]); return 0;
    }
    Options options;
    try { options = parse(argc, argv); }
    catch (const std::exception & e) { std::fprintf(stderr, "argument error: %s\n", e.what()); usage(argv[0]); return 2; }
    try { return run(options); }
    catch (const std::exception & e) { std::fprintf(stderr, "LavaSR failed: %s\n", e.what()); return 1; }
}
