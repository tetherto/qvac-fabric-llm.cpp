// music-cli: end-to-end ACE-Step text-to-music harness.
//
// Drives tts_cpp::acestep::Engine (text prompt -> LM codes -> FSQ detok ->
// text/cond encoders -> DiT flow-matching -> VAE) and writes a 48 kHz stereo
// WAV. This is the first fully-native music generation path (no acestep.cpp
// binaries) — the same C++ that the @qvac/audiogen-ggml addon links.
//
// Usage:
//   music-cli --models <dir>            [--out song.wav] [--dur 8] [--seed 42]
//   music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf ...
//   optional: --caption "..." --lyrics "..." --steps 8 --shift 3.0
//             --bpm 128 --key "C major" --tsig 4/4 --lang en
//             --gpu --threads N --dump-stages <existing dir>

#include "audiogen-cpp/acestep/engine.h"
#include "mini_json.h"
#include "music_cli_edit.h"
#include "wav_reader.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static const char * arg_val(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
}

static bool arg_flag(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], key)) return true;
    return false;
}

static constexpr int kRequiredAudioSampleRate = 48000;
static constexpr const char *kNormalizeOption = "--normalize";
static constexpr const char *kSourceAudioOption = "--src-audio";

static std::vector<float> wav_read(const char * path, int * frames, int * rate) {
    tts_cpp::acestep::WavReadResult result = tts_cpp::acestep::load_pcm16_wav(path);
    if (!result.error.empty()) {
        fprintf(stderr, "[music-cli] %s: %s\n", path, result.error.c_str());
        return {};
    }
    *frames = result.frames;
    *rate   = result.sample_rate;
    return std::move(result.pcm);
}

static bool load_source_audio(int argc, char ** argv,
                              tts_cpp::acestep::GenerateParams & params) {
    const char * source_path = arg_val(argc, argv, kSourceAudioOption);
    if (!source_path) return true;
    int source_frames = 0;
    int source_rate = 0;
    params.source_audio = wav_read(source_path, &source_frames, &source_rate);
    if (params.source_audio.empty()) return false;
    if (source_rate != kRequiredAudioSampleRate) {
        fprintf(stderr, "[music-cli] source WAV is %d Hz; convert it to 48 kHz before generation\n",
                source_rate);
        return false;
    }
    fprintf(stderr, "[music-cli] source audio: %s (%.2fs, 48 kHz stereo)\n", source_path,
            (float) source_frames / kRequiredAudioSampleRate);
    return true;
}

static bool should_normalize_output(
        int argc, char ** argv,
        const tts_cpp::acestep::GenerateParams & params) {
    return params.edit_plan.empty() || arg_flag(argc, argv, kNormalizeOption);
}

static void wav_write(const char * path, const std::vector<float> & pcm,
                      int frames, int rate, bool normalize) {
    float peak = 1e-9f;
    for (int i = 0; i < frames * 2; i++) peak = std::fmax(peak, std::fabs(pcm[i]));
    float  gain = normalize ? 0.9f / peak : 1.0f;
    FILE * f    = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    const int ch = 2, bits = 16;
    uint32_t  db  = (uint32_t) frames * ch * (bits / 8);
    auto      w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto      w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + db); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(ch); w32((uint32_t) rate);
    w32((uint32_t) rate * ch * (bits / 8)); w16(ch * (bits / 8)); w16(bits);
    fwrite("data", 1, 4, f); w32(db);
    for (int i = 0; i < frames * 2; i++) {
        float v = pcm[i] * gain * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        w16((uint16_t) (int16_t) lrintf(v));
    }
    fclose(f);
    fprintf(stderr, "[music-cli] wrote %s: %d frames, %.2fs @ %d Hz stereo\n", path, frames,
            (float) frames / rate, rate);
}

int main(int argc, char ** argv) {
    using namespace tts_cpp::acestep;
    using music_cli::json_field;
    using music_cli::read_text_file;

    EngineOptions o;
    o.verbose = true;
    if (arg_val(argc, argv, "--models")) o.models_dir = arg_val(argc, argv, "--models");
    if (arg_val(argc, argv, "--dit"))    o.dit_model_path = arg_val(argc, argv, "--dit");
    if (arg_val(argc, argv, "--lm"))     o.lm_model_path = arg_val(argc, argv, "--lm");
    if (arg_val(argc, argv, "--text"))   o.text_enc_model_path = arg_val(argc, argv, "--text");
    if (arg_val(argc, argv, "--vae"))    o.vae_model_path = arg_val(argc, argv, "--vae");
    // Offer every stage to a GPU backend when one is available. Engine::create
    // selects validated Vulkan/Metal or Adreno 700+ OpenCL where possible; DiT,
    // VAE and encoders use the selected GPU, while the LM and FSQ detokenizer
    // apply their independent backend allowlists and generic CPU fallback.
    if (arg_flag(argc, argv, "--gpu"))   o.n_gpu_layers = 99;
    if (arg_val(argc, argv, "--threads")) o.n_threads = atoi(arg_val(argc, argv, "--threads"));
    if (arg_val(argc, argv, "--backend")) o.backend = arg_val(argc, argv, "--backend");
    if (arg_val(argc, argv, "--lm-backend")) o.lm_backend = arg_val(argc, argv, "--lm-backend");
    // Required wherever ggml ships its backends as dlopen'd MODULE .so files
    // (GGML_BACKEND_DL, i.e. every Android/arm64 build): without it the registry is
    // empty and even the CPU backend fails to init.
    if (arg_val(argc, argv, "--backends-dir")) o.backends_dir = arg_val(argc, argv, "--backends-dir");
    // Parity aid: write one .bin per stage so a CPU/GPU divergence can be traced
    // to the stage that introduces it rather than inferred from the final WAV.
    if (arg_val(argc, argv, "--dump-stages")) o.dump_stages_dir = arg_val(argc, argv, "--dump-stages");

    if (o.models_dir.empty() && o.dit_model_path.empty()) {
        fprintf(stderr,
                "usage: music-cli --models <dir> [--out song.wav] [--dur 8] [--seed 42]\n"
                "   or: music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf\n"
                "  prompt:  [--caption \"...\"] [--lyrics \"...\"] [--bpm 128] [--key \"C major\"]\n"
                "           [--tsig 4/4] [--lang en] [--req request.json]\n"
                "           [--simple]  (Simple Mode: expand --caption into a full request;\n"
                "           the LM writes lyrics unless --lyrics \"[Instrumental]\" is given,\n"
                "           and fills any metadata left unset, duration included with --dur 0)\n"
                "           [--rewrite]  (Query Rewriting: the LM formats --caption and --lyrics\n"
                "           into a detailed request, preserving the lyric content; needs the 1.7B LM)\n"
                "  reverse: [--understand <48-kHz PCM16 WAV>]  (describe audio: metadata +\n"
                "           caption + recovered codes; --lang forces the language field)\n"
                "  audio:   [--ref-audio <48-kHz PCM16 WAV>]  (timbre reference)\n"
                "           [--src-audio <48-kHz PCM16 WAV>]  (cover source structure)\n"
                "           [--task text2music|cover-nofsq|lego]   (default text2music)\n"
                "           [--track vocals|drums|bass|guitar|...]  (lego target layer)\n"
                "           [--cover-strength F] [--cover-noise F]  (cover defaults 1.0 / 0.0)\n"
                "  editing: [--repaint-start SEC] [--repaint-end SEC|-1]\n"
                "           [--repaint-mode conservative|balanced|aggressive]\n"
                "           [--repaint-strength 0..1]\n"
                "           [--flow-source-caption TEXT] [--flow-source-lyrics TEXT]\n"
                "           [--flow-n-min F] [--flow-n-max F] [--flow-n-avg N]\n"
                "           [--edit-plan plan.json]  (ordered/repeated operations)\n"
                "           plan.json: {\"operations\":[{\"type\":\"repaint\",...},\n"
                "             {\"type\":\"flow-edit\",\"source_caption\":\"...\",\n"
                "              \"target_caption\":\"...\",\"n_min\":0,\"n_max\":1,\"n_avg\":1}]}\n"
                "  sampler: [--steps N] [--shift F] [--guidance F]  (default: auto from the\n"
                "           DiT variant, turbo 8 / 3.0 / 1.0, base and sft 50 / 1.0 / 7.0)\n"
                "           [--no-dcw]  (Haar DCW double mode is enabled by default)\n"
                "           [--no-loudness]  (skip the percentile loudness normalization)\n"
                "           [--lrc out.lrc]  (write synchronized lyric timestamps; needs lyrics)\n"
                "  output:  [--normalize]  (peak-normalize edit output before PCM quantization)\n"
                "           [--score]  (teacher-forced LM quality score of the generated codes)\n"
                "           [--temp 0.85] [--cfg 2.0] [--topp 0.9] [--topk 0 (off)]\n"
                "           [--no-phase1]  (values shown are the defaults)\n"
                "  backend: [--gpu] [--backend auto|cpu|opencl|webgpu|hexagon|xdna|<device>] [--lm-backend <same names>]\n"
                "           [--threads N] [--backends-dir <dir>]\n"
                "           [--dump-stages <existing dir>]\n");
        return 1;
    }

    GenerateParams p;
    p.caption         = arg_val(argc, argv, "--caption") ? arg_val(argc, argv, "--caption")
                                                         : "Upbeat pop rock with driving electric guitars, punchy drums and a catchy hook";
    p.lyrics          = arg_val(argc, argv, "--lyrics") ? arg_val(argc, argv, "--lyrics") : "[Instrumental]";
    p.duration        = arg_val(argc, argv, "--dur")   ? (float) atof(arg_val(argc, argv, "--dur"))   : 8.0f;
    p.inference_steps = arg_val(argc, argv, "--steps") ? atoi(arg_val(argc, argv, "--steps"))         : 0;
    p.shift           = arg_val(argc, argv, "--shift") ? (float) atof(arg_val(argc, argv, "--shift")) : 0.0f;
    p.seed            = arg_val(argc, argv, "--seed")  ? strtoll(arg_val(argc, argv, "--seed"), nullptr, 10) : 42;
    if (arg_val(argc, argv, "--bpm"))  p.bpm = atoi(arg_val(argc, argv, "--bpm"));
    if (arg_val(argc, argv, "--key"))  p.keyscale = arg_val(argc, argv, "--key");
    if (arg_val(argc, argv, "--tsig")) p.timesignature = arg_val(argc, argv, "--tsig");
    if (arg_val(argc, argv, "--lang")) p.vocal_language = arg_val(argc, argv, "--lang");
    if (arg_val(argc, argv, "--temp")) p.lm_temperature = (float) atof(arg_val(argc, argv, "--temp"));
    if (arg_val(argc, argv, "--cfg"))  p.lm_cfg_scale = (float) atof(arg_val(argc, argv, "--cfg"));
    if (arg_val(argc, argv, "--topk")) p.lm_top_k = atoi(arg_val(argc, argv, "--topk"));
    if (arg_val(argc, argv, "--topp")) p.lm_top_p = (float) atof(arg_val(argc, argv, "--topp"));
    if (arg_flag(argc, argv, "--no-phase1")) p.lm_phase1 = false;
    if (arg_flag(argc, argv, "--no-dcw")) p.dcw_enabled = false;
    if (arg_flag(argc, argv, "--no-loudness")) p.normalize_loudness = false;
    if (arg_val(argc, argv, "--lrc")) p.generate_lrc = true;
    if (arg_flag(argc, argv, "--simple")) {
        p.simple_mode = true;
        if (!arg_val(argc, argv, "--lyrics")) p.lyrics.clear();
    }
    if (arg_flag(argc, argv, "--score")) p.compute_quality_score = true;
    if (arg_flag(argc, argv, "--rewrite")) p.rewrite_query = true;

    // --req <json>: load caption/lyrics/metas and (if present) audio_codes to
    // bypass our LM — used for parity against acestep.cpp's ace-lm output.
    if (arg_val(argc, argv, "--req")) {
      std::string j = read_text_file(arg_val(argc, argv, "--req"));
      if (j.empty()) {
        fprintf(stderr, "[music-cli] cannot read --req json\n");
        return 1;
      }
        std::string v;
        if (json_field(j, "caption", v)) p.caption = v;
        if (json_field(j, "lyrics", v)) p.lyrics = v;
        if (json_field(j, "keyscale", v)) p.keyscale = v;
        if (json_field(j, "timesignature", v)) p.timesignature = v;
        if (json_field(j, "vocal_language", v)) p.vocal_language = v;
        if (json_field(j, "bpm", v)) p.bpm = atoi(v.c_str());
        if (json_field(j, "duration", v)) p.duration = (float) atof(v.c_str());
        if (json_field(j, "shift", v)) p.shift = (float) atof(v.c_str());
        if (json_field(j, "inference_steps", v)) p.inference_steps = atoi(v.c_str());
        if (json_field(j, "seed", v)) p.seed = strtoll(v.c_str(), nullptr, 10);
        if (json_field(j, "dcw_enabled", v)) p.dcw_enabled = v != "false" && v != "0";
        if (json_field(j, "dcw_scaler", v)) p.dcw_scaler = (float) atof(v.c_str());
        if (json_field(j, "dcw_high_scaler", v)) p.dcw_high_scaler = (float) atof(v.c_str());
        if (json_field(j, "task_type", v)) p.task_type = v;
        if (json_field(j, "track", v)) p.track = v;
        if (json_field(j, "guidance_scale", v)) p.guidance_scale = (float) atof(v.c_str());
        if (json_field(j, "audio_cover_strength", v)) p.audio_cover_strength = (float) atof(v.c_str());
        if (json_field(j, "cover_noise_strength", v)) p.cover_noise_strength = (float) atof(v.c_str());
        if (json_field(j, "audio_codes", v) && !v.empty()) {
            size_t start = 0;
            while (start < v.size()) {
                size_t comma = v.find(',', start);
                std::string tok = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) p.audio_codes.push_back(atoi(tok.c_str()));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            fprintf(stderr, "[music-cli] --req: %zu pre-supplied audio codes (LM bypass)\n", p.audio_codes.size());
        }
    }

    if (const char * reference_path = arg_val(argc, argv, "--ref-audio")) {
        int reference_frames = 0;
        int reference_rate   = 0;
        p.reference_audio = wav_read(reference_path, &reference_frames, &reference_rate);
        if (p.reference_audio.empty()) return 1;
        if (reference_rate != 48000) {
            fprintf(stderr, "[music-cli] reference WAV is %d Hz; convert it to 48 kHz before generation\n",
                    reference_rate);
            return 1;
        }
        fprintf(stderr, "[music-cli] reference audio: %s (%.2fs, 48 kHz stereo)\n", reference_path,
                (float) reference_frames / 48000.0f);
    }

    if (!load_source_audio(argc, argv, p)) return 1;

    if (const char * task = arg_val(argc, argv, "--task")) p.task_type = task;
    if (const char * track = arg_val(argc, argv, "--track")) p.track = track;
    if (arg_val(argc, argv, "--guidance"))
        p.guidance_scale = (float) atof(arg_val(argc, argv, "--guidance"));
    if (arg_val(argc, argv, "--cover-strength"))
        p.audio_cover_strength = (float) atof(arg_val(argc, argv, "--cover-strength"));
    if (arg_val(argc, argv, "--cover-noise"))
        p.cover_noise_strength = (float) atof(arg_val(argc, argv, "--cover-noise"));

    std::string edit_error;
    if (!music_cli::parse_edit_flags(argc, argv, p, edit_error) ||
        !music_cli::validate_edit_configuration(p, edit_error)) {
      fprintf(stderr, "[music-cli] %s\n", edit_error.c_str());
      return 1;
    }
    if (!p.edit_plan.empty()) {
      fprintf(stderr, "[music-cli] edit plan: %zu ordered operation(s)\n",
              p.edit_plan.size());
    }

    const char * out_path = arg_val(argc, argv, "--out") ? arg_val(argc, argv, "--out") : "music_out.wav";

    std::unique_ptr<Engine> eng;
    try {
        eng = Engine::create(o);
    } catch (const std::exception & e) {
        fprintf(stderr, "[music-cli] engine create failed: %s\n", e.what());
        return 1;
    }

    auto t_start = std::chrono::steady_clock::now();
    auto t_vae0  = t_start;
    auto progress = [&](const std::string & stage, int step, int total) -> bool {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t_start).count();
        if (stage == "vae" && step == 0) t_vae0 = now;
        if (stage == "vae" && step == total) {
            double vae_ms = std::chrono::duration<double, std::milli>(now - t_vae0).count();
            fprintf(stderr, "[music-cli] %-4s %d/%d  (t=%.0fms) VAE render=%.0fms\n",
                    stage.c_str(), step, total, ms, vae_ms);
        } else {
            fprintf(stderr, "[music-cli] %-4s %d/%d  (t=%.0fms)\n", stage.c_str(), step, total, ms);
        }
        return true;
    };

    if (const char * understand_path = arg_val(argc, argv, "--understand")) {
        UnderstandParams up;
        int frames = 0;
        int rate   = 0;
        up.audio = wav_read(understand_path, &frames, &rate);
        if (up.audio.empty()) return 1;
        if (rate != kRequiredAudioSampleRate) {
            fprintf(stderr, "[music-cli] understand WAV is %d Hz; convert it to 48 kHz first\n", rate);
            return 1;
        }
        if (arg_val(argc, argv, "--lang")) up.vocal_language = arg_val(argc, argv, "--lang");
        if (arg_val(argc, argv, "--seed")) up.seed = atoll(arg_val(argc, argv, "--seed"));
        if (arg_val(argc, argv, "--temp")) up.lm_temperature = (float) atof(arg_val(argc, argv, "--temp"));
        if (arg_val(argc, argv, "--topp")) up.lm_top_p = (float) atof(arg_val(argc, argv, "--topp"));
        if (arg_val(argc, argv, "--topk")) up.lm_top_k = atoi(arg_val(argc, argv, "--topk"));

        UnderstandResult u;
        try {
            u = eng->understand(up, progress);
        } catch (const std::exception & e) {
            fprintf(stderr, "[music-cli] understand failed: %s\n", e.what());
            return 1;
        }
        if (u.caption.empty() && u.audio_codes.empty()) {
            fprintf(stderr, "[music-cli] no result (cancelled?)\n");
            return 1;
        }
        printf("caption: %s\n", u.caption.c_str());
        if (u.bpm > 0) printf("bpm: %d\n", u.bpm);
        if (u.duration > 0) printf("duration: %.0f\n", u.duration);
        if (!u.keyscale.empty()) printf("keyscale: %s\n", u.keyscale.c_str());
        if (!u.timesignature.empty()) printf("timesignature: %s\n", u.timesignature.c_str());
        if (!u.vocal_language.empty()) printf("language: %s\n", u.vocal_language.c_str());
        printf("audio_codes: %zu (%.1fs @ 5Hz)\n", u.audio_codes.size(),
               (float) u.audio_codes.size() / 5.0f);
        return 0;
    }

    GenerateResult r;
    try {
        r = eng->generate(p, progress);
    } catch (const std::exception & e) {
        fprintf(stderr, "[music-cli] generate failed: %s\n", e.what());
        return 1;
    }

    if (r.pcm.empty()) { fprintf(stderr, "[music-cli] no audio (cancelled?)\n"); return 1; }

    int frames = (int) (r.pcm.size() / 2);
    fprintf(stderr, "[music-cli] generated %d codes, seed=%lld, %d frames (%.2fs)\n", r.metadata.n_codes,
            r.metadata.seed, frames, (float) frames / r.sample_rate);
    wav_write(out_path, r.pcm, frames, r.sample_rate,
              should_normalize_output(argc, argv, p));
    if (const char * lrc_path = arg_val(argc, argv, "--lrc")) {
        FILE * lrc_file = fopen(lrc_path, "wb");
        if (!lrc_file) {
            fprintf(stderr, "[music-cli] cannot write %s\n", lrc_path);
            return 1;
        }
        const size_t written = fwrite(r.metadata.lrc.data(), 1, r.metadata.lrc.size(), lrc_file);
        fclose(lrc_file);
        if (written != r.metadata.lrc.size()) {
            fprintf(stderr, "[music-cli] short write on %s\n", lrc_path);
            return 1;
        }
        fprintf(stderr, "[music-cli] wrote %s (lyrics score %.4f)\n", lrc_path, r.metadata.lyrics_score);
    }
    if (p.compute_quality_score) {
        fprintf(stderr, "[music-cli] quality score %.4f\n%s\n", r.metadata.quality_score,
                r.metadata.quality_report.c_str());
    }
    return 0;
}
