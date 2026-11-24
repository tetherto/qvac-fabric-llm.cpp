// Validation harness for per-chunk streaming mel parity.
//
// Reads Python's streaming dumps produced by
// `scripts/dump-streaming-reference.py`, runs the C++ pipeline once per
// chunk with the matching `finalize` flag and token prefix, and compares
// per-chunk mel output against Python's `chunk_{k}_mels_new.npy`.
//
// Carries `hift_cache_source` across chunks, applies
// `apply_trim_fade` only to chunk 0, concatenates the per-chunk wavs into
// `streamed_wav_cpp.npy`, and reports RMS error vs Python's
// `streamed_wav.npy` — HiFT sinegen uses a different RNG than Python's
// torch.randn so bit-exact parity is not expected, but the streamed RMS
// should be close to Python's streamed-vs-batch gap (≈3%).
//
// Usage:
//   ./build/test-streaming  models/chatterbox-s3gen.gguf  /tmp/streaming_ref/

#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "parity_test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>

static bool path_exists(const std::string & p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s S3GEN.gguf STREAMING_REF_DIR\n", argv[0]);
        return 1;
    }
    const std::string gguf = argv[1];
    const std::string ref  = argv[2];

    npy_array all_tokens = npy_load(ref + "/speech_tokens.npy");
    const int n_speech = (int)all_tokens.n_elements();
    const int32_t * tok_ptr = npy_as_i32(all_tokens);
    if (n_speech == 0) { fprintf(stderr, "error: empty speech tokens\n"); return 1; }
    fprintf(stderr, "loaded %d speech tokens\n", n_speech);

    int k = 0;
    int prev_mels_emitted = 0;
    int previous_tokens = 0;
    int failures = 0;
    double max_rel = 0;
    double total_rms = 0;
    std::vector<float> hift_cache_source;   // tail of prev chunk's post-SineGen source
    std::vector<float> streamed_wav_cpp;    // accumulated across chunks
    double wav_max_rel = 0, wav_total_rms = 0; int wav_chunks = 0;
    while (true) {
        char kbuf[16];
        std::snprintf(kbuf, sizeof(kbuf), "%02d", k + 1);
        const std::string tok_path = ref + "/chunk_" + kbuf + "_tokens.npy";
        const std::string mel_path = ref + "/chunk_" + kbuf + "_mels_new.npy";
        if (!path_exists(tok_path)) break;

        npy_array toks_chunk = npy_load(tok_path);
        const int32_t * cp = npy_as_i32(toks_chunk);
        const int n_cum = (int)toks_chunk.n_elements();
        if (n_cum <= previous_tokens || n_cum > n_speech) {
            fprintf(stderr, "error: chunk tokens must advance within the global sequence\n");
            return 1;
        }
        const bool is_last = (n_cum == n_speech);

        fprintf(stderr, "\n=== chunk %d ===  tokens_total=%d  finalize=%s  "
                        "prev_mels_emitted=%d\n",
                k + 1, n_cum, is_last ? "true" : "false", prev_mels_emitted);

        // Sanity-check the cumulative token prefix matches the global sequence.
        for (int i = 0; i < n_cum; ++i) {
            if (cp[i] != tok_ptr[i]) {
                fprintf(stderr, "error: chunk tokens diverge from global sequence at i=%d\n", i);
                return 1;
            }
        }

        std::vector<int32_t> chunk_tokens(cp, cp + n_cum);

        const std::string dump_path = ref + "/cpp_chunk_" + kbuf + "_mels_new.npy";

        s3gen_synthesize_opts opts;
        opts.s3gen_gguf_path           = gguf;
        opts.out_wav_path              = ref + "/cpp_chunk_" + kbuf + ".wav";
        opts.ref_dir                   = ref;
        opts.seed                      = 42;
        // Streaming semantics mirrored from scripts/dump-streaming-reference.py:
        //   * the Python loop appends 3 silence tokens ONCE at the top,
        //     before chunking, so per-chunk calls don't re-append.  We match
        //     that with append_lookahead_silence=false.
        //   * `finalize` only controls the tail 6-frame trim.
        //   * `skip_mel_frames` drops the frames already emitted by earlier
        //     chunks, leaving only what's new this chunk.
        opts.append_lookahead_silence  = false;
        opts.finalize                  = is_last;
        opts.skip_mel_frames           = prev_mels_emitted;
        opts.dump_mel_path             = dump_path;

        // HiFT streaming: carry source tail across chunks, fade-in only on
        // chunk 0 (matches Python dump-streaming-reference.py).
        opts.hift_cache_source         = hift_cache_source;
        opts.apply_trim_fade           = (k == 0);
        std::vector<float> source_tail_out;
        opts.hift_source_tail_out      = &source_tail_out;
        opts.source_tail_samples       = 480;   // 1 mel hop at 24 kHz = 20 ms

        // Inject Python's exact per-chunk CFM noise so the two pipelines
        // become bit-exact comparable (bypasses the torch.randn vs
        // std::mt19937 divergence).
        // Use chunk_KK_step0_x_in.npy (the ACTUAL z passed into estimator.forward
        // at step 0 — includes the `noised_mels` overlay from flow_inference).
        // The older chunk_KK_cfm_z.npy only captured the first torch.randn_like
        // call and missed the meanflow-specific speech-region overwrite, which
        // caused post-prompt divergence.
        const std::string z_path = ref + "/chunk_" + kbuf + "_step0_x_in.npy";
        if (path_exists(z_path)) {
            npy_array z = npy_load(z_path);
            const float * zp = npy_as_f32(z);
            opts.cfm_z0_override.assign(zp, zp + z.n_elements());
        } else {
            fprintf(stderr, "error: missing per-chunk CFM noise; regenerate streaming references\n");
            return 1;
        }

        int rc = s3gen_synthesize_to_wav(chunk_tokens, opts);
        if (rc != 0) {
            fprintf(stderr, "error: s3gen_synthesize_to_wav returned %d\n", rc);
            return rc;
        }

        // Compare mel against Python.
        npy_array py_mel  = npy_load(mel_path);
        npy_array cpp_mel = npy_load(dump_path);
        if (cpp_mel.shape.size() != 2 || cpp_mel.shape[1] != 80) {
            fprintf(stderr, "error: expected C++ mel shape (T, 80)\n");
            return 1;
        }
        const float * cpp = npy_as_f32(cpp_mel);
        const std::vector<float> mel(cpp, cpp + cpp_mel.n_elements());
        constexpr double kMelRelativeTolerance = 2e-3;
        if (!tts_test::check_f32("streaming mel", mel, cpp_mel.shape, py_mel,
                                 kMelRelativeTolerance)) ++failures;
        const auto stats = compare_f32(cpp, npy_as_f32(py_mel),
                                        std::min(mel.size(), py_mel.n_elements()));
        max_rel = std::max(max_rel, stats.rel_err);
        total_rms += stats.rms_err;
        prev_mels_emitted += (int) cpp_mel.shape[0];
        previous_tokens = n_cum;

        // Carry cache_source over to the next chunk.
        hift_cache_source = std::move(source_tail_out);

        // Read the per-chunk C++ wav (just written to disk) and splice into
        // the running streamed wav.  Compare to Python's chunk_KK_wav.npy:
        // C++ SineGen RNG differs from torch.randn so bit-exact parity is
        // not achievable here — we just report RMS as an informational
        // sanity signal.
        const std::string cpp_wav_path = opts.out_wav_path;
        // Minimal WAV reader (mono 16-bit PCM from write_wav()).
        std::vector<float> cpp_wav;
        {
            std::ifstream wf(cpp_wav_path, std::ios::binary);
            if (wf) {
                char hdr[44]; wf.read(hdr, 44);
                // Read 16-bit PCM samples
                std::vector<int16_t> pcm;
                int16_t s16;
                while (wf.read((char*)&s16, 2)) pcm.push_back(s16);
                cpp_wav.resize(pcm.size());
                for (size_t i = 0; i < pcm.size(); ++i) cpp_wav[i] = (float)pcm[i] / 32768.0f;
            }
        }
        if (cpp_wav.empty()) {
            fprintf(stderr, "error: missing or empty chunk WAV\n");
            return 1;
        }
        const std::string py_wav_path = ref + "/chunk_" + kbuf + "_wav.npy";
        if (path_exists(py_wav_path) && !cpp_wav.empty()) {
            npy_array py_wav_arr = npy_load(py_wav_path);
            const float * pywav = (const float *)py_wav_arr.data.data();
            size_t wn = std::min((size_t)py_wav_arr.n_elements(), cpp_wav.size());
            auto ws = compare_f32(cpp_wav.data(), pywav, wn);
            fprintf(stderr, "  wav  diff:    max_abs=%.4e  rms=%.4e  rel=%.4e  (informational, RNG mismatch expected)\n",
                    ws.max_abs_err, ws.rms_err, ws.rel_err);
            if (ws.rel_err > wav_max_rel) wav_max_rel = ws.rel_err;
            wav_total_rms += ws.rms_err;
            wav_chunks += 1;
        }
        streamed_wav_cpp.insert(streamed_wav_cpp.end(), cpp_wav.begin(), cpp_wav.end());
        k += 1;
    }

    fprintf(stderr, "\nresult: %d chunks processed, worst mel rel=%.4e, mean mel rms=%.4e\n",
            k, max_rel, total_rms / std::max(k, 1));
    if (wav_chunks > 0) {
        fprintf(stderr, "        per-chunk wav: worst rel=%.4e, mean rms=%.4e\n",
                wav_max_rel, wav_total_rms / wav_chunks);
    }

    // Save streamed_wav_cpp for offline comparison vs Python's streamed_wav.
    if (!streamed_wav_cpp.empty()) {
        npy_save_f32(ref + "/streamed_wav_cpp.npy",
                     {(int64_t)streamed_wav_cpp.size()}, streamed_wav_cpp.data());
        fprintf(stderr, "        streamed wav (%zu samples) → %s/streamed_wav_cpp.npy\n",
                streamed_wav_cpp.size(), ref.c_str());
        const std::string py_streamed = ref + "/streamed_wav.npy";
        if (path_exists(py_streamed)) {
            npy_array py_sw = npy_load(py_streamed);
            const float * py = (const float *)py_sw.data.data();
            size_t n = std::min((size_t)py_sw.n_elements(), streamed_wav_cpp.size());
            auto sw = compare_f32(streamed_wav_cpp.data(), py, n);
            fprintf(stderr, "        streamed vs Python: rms=%.4e max_abs=%.4e rel=%.4e\n",
                    sw.rms_err, sw.max_abs_err, sw.rel_err);
        }
    }
    if (k == 0 || previous_tokens != n_speech) {
        fprintf(stderr, "error: streaming fixtures do not cover the complete token sequence\n");
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
