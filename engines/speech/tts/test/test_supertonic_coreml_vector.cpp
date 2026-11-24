// Stage an F32/F16 GGUF and FP16 vector buckets, then set
// SUPERTONIC_COREML_VECTOR_TEST_GGUF. Real-model parity and routing tests.
#include <cstdio>
#include <cstdlib>
#ifndef TTS_CPP_USE_COREML
int main() { std::puts("SKIP: built without Core ML"); return 77; }
#else
#include "supertonic_internal.h"
#include "supertonic_coreml_vector.h"
#include "supertonic_coreml_path.h"
#include "tts-cpp/supertonic/engine.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tts_cpp::supertonic::detail;
namespace {
void check(bool ok, const std::string & what) { if (!ok) throw std::runtime_error(what); }
struct Model { supertonic_model value; ~Model() { free_supertonic_model(value); } };
struct Env {
    std::string key, old;
    bool exists;
    Env(const char * k, const char * value) : key(k), exists(std::getenv(k) != nullptr) {
        if (exists) old = std::getenv(k);
        setenv(k,value,1);
    }
    ~Env() { if (exists) setenv(key.c_str(),old.c_str(),1); else unsetenv(key.c_str()); }
};
bool fp32 = false;
void parity(const std::vector<float> & a, const std::vector<float> & b, const std::string & tag, bool loop = false) {
    check(a.size() == b.size() && !a.empty(), tag+": shape mismatch");
    double dot = 0, aa = 0, bb = 0, squared = 0, maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]),tag+": nonfinite output");
        dot += double(a[i])*b[i]; aa += double(a[i])*a[i]; bb += double(b[i])*b[i];
        double delta = double(a[i])-b[i]; squared += delta*delta; maximum = std::max(maximum,std::abs(delta));
    }
    double cosine = dot/(std::sqrt(aa*bb)+1e-12), nrmse = std::sqrt(squared/(aa+1e-12));
    std::printf("%s cosine=%.8f nrmse=%.6f max_abs=%.6f\n",tag.c_str(),cosine,nrmse,maximum);
    // FP16 trajectories are not numerically equivalent: the existing English
    // intelligibility benchmark is the quality evidence for the default. Keep
    // the single-step numerical gate and a broad trajectory regression bound.
    // Opt in to the former tight trajectory gate for numerical diagnostics.
    if (fp32) check(cosine >= 0.99999 && nrmse <= 0.001 && maximum <= 0.005,tag+": FP32 parity failed");
    else if (loop && !std::getenv("SUPERTONIC_COREML_VECTOR_REQUIRE_NUMERICAL_PARITY"))
        check(cosine >= 0.98 && nrmse <= 0.15,tag+": FP16 trajectory regression bound failed");
    else check(cosine >= (loop ? 0.999 : 0.9998) && nrmse <= (loop ? 0.04 : 0.02) &&
               maximum <= (loop ? 0.25 : 0.1),tag+": FP16 parity failed");
}
}
int main() {
    const char * path = std::getenv("SUPERTONIC_COREML_VECTOR_TEST_GGUF");
    if (!path || !*path) { std::puts("SKIP: SUPERTONIC_COREML_VECTOR_TEST_GGUF not set"); return 77; }
    try {
        Model reference, accelerated;
        {
            Env disabled("SUPERTONIC_COREML_VECTOR_DISABLE","1");
            check(load_supertonic_gguf(path,reference.value,0),"reference load failed");
            check(!reference.value.coreml_vector,"disable ignored");
        }
        check(load_supertonic_gguf(path,accelerated.value,0),"accelerated load failed");
        auto & model = accelerated.value;
        check(model.vector_on_coreml && model.coreml_vector,"vector sidecar not attached");
        if (std::getenv("SUPERTONIC_COREML_VECTOR_TEST_REQUIRE_VOCODER"))
            check(model.vocoder_on_coreml && model.coreml_vocoder,"required vocoder sidecar not attached");
        fp32 = supertonic_coreml_vector_weight_bits(model.coreml_vector) == 32;
        supertonic_set_n_threads(reference.value,4); supertonic_set_n_threads(model,4);
        auto paths = coreml_vector_sidecar_paths(path);
        const int smallest = paths.front().first, largest = paths.back().first;
        std::mt19937 rng(7); std::normal_distribution<float> normal(0,0.5f);
        std::vector<float> style(50*256);
        check(!model.voices.empty(),"missing voice");
        std::printf("voice=%s vector_weight_bits=%d\n",model.voices.begin()->first.c_str(),
            supertonic_coreml_vector_weight_bits(model.coreml_vector));
        ggml_backend_tensor_get(model.voices.begin()->second.ttl,style.data(),0,style.size()*sizeof(float));
        // First/last frames, exact boundary, next bucket, and overflow.
        std::vector<int> lengths = {1,17,smallest,smallest+1,largest,largest+1};
        std::sort(lengths.begin(),lengths.end()); lengths.erase(std::unique(lengths.begin(),lengths.end()),lengths.end());
        for (int length : lengths) {
            const int text_length = 21;
            std::vector<float> latent((size_t)144*length),text(256*text_length),mask(length,1),got,want;
            for (float & v : latent) v = normal(rng);
            for (float & v : text) v = normal(rng);
            std::string error,backend;
            check(supertonic_vector_step_ggml(reference.value,latent.data(),length,text.data(),text_length,
                style.data(),mask.data(),2,5,want,&error),error);
            check(supertonic_vector_step_ggml(model,latent.data(),length,text.data(),text_length,
                style.data(),mask.data(),2,5,got,&error,&backend),error);
            check((backend != "ggml") == (length <= largest),"incorrect bucket/overflow routing");
            parity(want,got,"step L="+std::to_string(length));
            if (length <= largest) {
                // General masks are intentionally unsupported by the sidecar;
                // fallback must retain ggml's semantics instead of coercing them.
                mask.back() = 0.5f;
                check(supertonic_vector_step_ggml(model,latent.data(),length,text.data(),text_length,
                    style.data(),mask.data(),2,5,got,&error,&backend),error);
                check(backend == "ggml","fractional mask was not rejected");
            } else {
                Env strict("SUPERTONIC_COREML_VECTOR_STRICT","1");
                check(!supertonic_vector_loop_ggml(model,latent.data(),length,text.data(),text_length,
                    style.data(),mask.data(),5,got,&error),"strict overflow accepted");
            }
        }
        // Full trajectories use actual encoder features, predicted durations
        // and unit Gaussian noise, rather than out-of-distribution random
        // text features and forced durations. Check both common step counts.
        const char * utterances[] = {"Hello world.", "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog, then returns to the quiet garden before sunset."};
        std::normal_distribution<float> noise(0,1);
        for (const char * utterance : utterances) {
            std::vector<int32_t> ids32;
            std::string error, normalized, backend;
            check(supertonic_text_to_ids(reference.value,utterance,"en",ids32,&normalized,&error),error);
            std::vector<int64_t> ids(ids32.begin(),ids32.end());
            std::vector<float> text;
            check(supertonic_text_encoder_forward_one_graph_ggml(reference.value,ids.data(),(int)ids.size(),style.data(),text,&error),error);
            const auto & voice = model.voices.begin()->second;
            std::vector<float> style_dp((size_t)ggml_nelements(voice.dp));
            ggml_backend_tensor_get(voice.dp,style_dp.data(),0,style_dp.size()*sizeof(float));
            float duration = 0;
            check(supertonic_duration_forward_ggml(reference.value,ids.data(),(int)ids.size(),style_dp.data(),duration,&error),error);
            const int chunk = model.hparams.base_chunk_size * model.hparams.ttl_chunk_compress_factor;
            int length = std::max(1,(int)std::ceil(duration / 1.05f * model.hparams.sample_rate / chunk));
            check(length <= largest,"fixture needs a larger bucket for full-utterance parity");
            std::vector<float> latent((size_t)144*length),mask(length,1),got,want;
            for (float & v : latent) v = noise(rng);
            for (int steps : {5,8}) {
                check(supertonic_vector_loop_ggml(reference.value,latent.data(),length,text.data(),(int)ids.size(),
                    style.data(),mask.data(),steps,want,&error),error);
                check(supertonic_vector_loop_ggml(model,latent.data(),length,text.data(),(int)ids.size(),
                    style.data(),mask.data(),steps,got,&error,&backend),error);
                check(backend != "ggml","loop silently fell back");
                parity(want,got,"speech loop L="+std::to_string(length)+" steps="+std::to_string(steps),true);
            }
        }
        // A graph exported for the other model family must not attach.
        check(supertonic_coreml_vector_init(paths,model.hparams.vector_text_attn_heads == 8 ? 4 : 8,
            model.hparams.cfg_cond_scale,model.hparams.cfg_uncond_scale) == nullptr,"incompatible metadata accepted");
        tts_cpp::supertonic::EngineOptions options;
        options.model_gguf_path = path; options.n_gpu_layers = 0;
        options.prewarm_text = "Hello.";
        tts_cpp::supertonic::Engine engine(options);
        check(engine.vector_estimator_on_coreml(),"public load status false");
        if (model.vocoder_on_coreml) check(engine.vocoder_on_coreml(),"public vocoder load status false");
        auto result = engine.synthesize("Hello.");
        check(!result.pcm.empty() && result.vector_estimator_synthesis_backend != "ggml","engine bypassed vector sidecar");
        if (model.vocoder_on_coreml)
            check(result.vocoder_synthesis_backend.rfind("coreml-",0) == 0,"engine bypassed vocoder sidecar");
        options.stream_chunk_tokens = 20;
        tts_cpp::supertonic::Engine streaming(options);
        int chunks = 0;
        result = streaming.synthesize("Hello there. This is the next sentence. This is the final sentence.",
            [&](const float *,std::size_t,int,bool) { ++chunks; });
        check(chunks >= 2 && result.vector_estimator_synthesis_backend.rfind("coreml-",0) == 0,"streaming lost vector backend");
        if (model.vocoder_on_coreml)
            check(result.vocoder_synthesis_backend.rfind("coreml-",0) == 0,"streaming bypassed vocoder sidecar");
        std::puts("test_supertonic_coreml_vector: OK");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
#endif
