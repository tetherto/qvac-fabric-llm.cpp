// Model-free ACE-Step quantization regression tests.

#include "acestep_test_support.h"
#include "quantize_gguf.h"
#include "quantize_policy.h"
#include "gguf.h"

namespace {

void test_quantize_policy() {
    using namespace tts_cpp::acestep;

    const QuantVariant * q4km = find_quant_variant("q4_k_m");
    const QuantVariant * q80  = find_quant_variant("Q8_0");
    const QuantVariant * q2k  = find_quant_variant("Q2_K");
    const QuantVariant * q3kl = find_quant_variant("Q3_K_L");
    CHECK(q4km != nullptr && q4km->base == GGML_TYPE_Q4_K);
    CHECK(q80 != nullptr && q2k != nullptr && q3kl != nullptr);
    CHECK(find_quant_variant("Q4_K") == nullptr);

    CHECK(quant_layer_index("model.layers.12.self_attn.v_proj.weight") == 12);
    CHECK(quant_layer_index("model.embed_tokens.weight") == -1);

    // The deny-list: VAE files, 1-D tensors, the text encoder's embedding, and
    // the DiT's quality-critical extras are never quantized.
    CHECK(quant_pick_type("decoder.conv1.weight", 3, "acestep-vae", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("model.norm.weight", 1, "acestep-lm", *q4km, 24) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("embed_tokens.weight", 2, "acestep-text-enc", *q4km, 28) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("silence_latent", 2, "acestep-dit", *q4km, 24) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("decoder.scale_shift_table", 2, "acestep-dit", *q4km, 24) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("null_condition_emb", 2, "acestep-dit", *q4km, 24) == GGML_TYPE_COUNT);

    // The LM embedding takes the variant's embed type, everything ordinary the base.
    CHECK(quant_pick_type("model.embed_tokens.weight", 2, "acestep-lm", *q4km, 24) == GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.embed_tokens.weight", 2, "acestep-lm", *q80, 24) == GGML_TYPE_Q8_0);
    CHECK(quant_pick_type("model.layers.4.self_attn.q_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q4_K);

    // M-variant bump: first n/9, last n/7, and every 3rd layer of v_proj/down_proj.
    CHECK(quant_pick_type("model.layers.0.self_attn.v_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.layers.3.mlp.down_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.layers.22.self_attn.v_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.layers.4.self_attn.v_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q4_K);

    // S-variant bump: only the first bump_layer_count layers.
    CHECK(quant_pick_type("model.layers.3.self_attn.v_proj.weight", 2, "acestep-lm", *q2k, 24) ==
          GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("model.layers.4.self_attn.v_proj.weight", 2, "acestep-lm", *q2k, 24) ==
          GGML_TYPE_Q2_K);

    // L-variant bump: o_proj joins the important set, at every layer.
    CHECK(quant_pick_type("model.layers.10.self_attn.o_proj.weight", 2, "acestep-lm", *q3kl, 24) ==
          GGML_TYPE_Q5_K);
    CHECK(quant_pick_type("model.layers.10.self_attn.o_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q4_K);

    CHECK(quant_should_promote_f32(1));
    CHECK(!quant_should_promote_f32(2));
}

// MM3's LM (arch "qwen3") uses llama.cpp-style tensor names, and its synth file
// (arch "mm3") bundles a DiT with a condition encoder, RVQ depth decoder, vocoder,
// and timestep Fourier basis that must stay untouched. Neither reuses the HF-style
// v_proj/down_proj/o_proj bump machinery: MM3's own attn_v.weight/ffn_down.weight/
// attn_output.weight names are shared verbatim between the LM and the (denylisted)
// depth decoder, and the DiT's fused attn_qkv/ffn_in/ffn_out layout has no bump
// target to protect, so quantization is flat aside from the embed/output-head
// overrides checked here.
void test_quantize_policy_mm3() {
    using namespace tts_cpp::acestep;

    const QuantVariant * q4km = find_quant_variant("q4_k_m");
    CHECK(q4km != nullptr);

    // LM embedding table and untied output head both take the embed type.
    CHECK(quant_pick_type("token_embd.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("output.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q6_K);

    // Everything else in the LM is flat base-type: no bump for MM3's naming.
    CHECK(quant_pick_type("blk.0.attn_v.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("blk.35.ffn_down.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("blk.0.attn_output.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q4_K);

    // The untied-output rule is scoped to "qwen3" only; it must not leak into ACE-Step's LM.
    CHECK(quant_pick_type("output.weight", 2, "acestep-lm", *q4km, 24) == GGML_TYPE_Q4_K);

    // Synth policy: condition encoder and vocoder never quantize; the RVQ depth
    // decoder follows the requested variant like the DiT, keeping its positional
    // embedding raw and its get_rows code table at Q8_0 (CUDA get_rows has no
    // k-quant path).
    CHECK(quant_pick_type("depth.proj.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("depth.blk.0.attn_v.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("depth.audio_embd.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q8_0);
    CHECK(quant_pick_type("depth.audio_embd.weight", 2, "mm3", *find_quant_variant("Q8_0"), 0) ==
          GGML_TYPE_Q8_0);
    CHECK(quant_pick_type("depth.blk.0.input_norm.weight", 1, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("depth.pos_embd.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("cond.proj.weight", 3, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("voc.conv_in.weight", 3, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("dit.time_fourier.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_COUNT);

    // The DiT is the one synth component that quantizes, flat, including the
    // attn_output.weight name it shares with the LM and the denylisted depth decoder.
    CHECK(quant_pick_type("dit.blk.0.attn_qkv.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("dit.proj_in.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("dit.blk.0.attn_output.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
}

// End-to-end guard on the offset/padding planning and the streaming writer: a
// regression there emits a byte-shifted GGUF that only fails at engine load.
// Quantizes a synthetic BF16-era file and reads the result back through the
// gguf loader (which validates offsets and alignment structurally), then
// checks every tensor's type and dequantized values against the source.
void test_quantize_gguf_roundtrip() {
    using namespace tts_cpp::acestep;

    const std::string in_path  = test_temp_dir() + "/qvac-acestep-quantize-in.gguf";
    const std::string out_path = test_temp_dir() + "/qvac-acestep-quantize-out.gguf";

    // 32-wide rows quantize under Q8_0 (block size 32); the 24-wide tensor is
    // unaligned and must be kept as stored; the BF16 1-D norm must be promoted
    // to F32. Q8_0 row size is 34 bytes, so quantized tensors end misaligned
    // and the writer's padding path is exercised between every pair.
    std::vector<float> wide(32 * 4);
    for (size_t i = 0; i < wide.size(); ++i) {
        wide[i] = ((float) i - 64.0f) / 32.0f;
    }
    std::vector<float> narrow(24 * 2);
    for (size_t i = 0; i < narrow.size(); ++i) {
        narrow[i] = ((float) i - 24.0f) / 8.0f;
    }
    std::vector<float> norm(32);
    for (size_t i = 0; i < norm.size(); ++i) {
        norm[i] = 1.0f + (float) i / 100.0f;
    }

    ggml_init_params ip{ 4 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    gguf_set_val_str(gc, "general.architecture", "acestep-lm");
    gguf_set_val_u32(gc, "acestep-lm.block_count", 1);

    ggml_tensor * embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
    ggml_set_name(embed, "model.embed_tokens.weight");
    std::memcpy(embed->data, wide.data(), wide.size() * sizeof(float));
    gguf_add_tensor(gc, embed);

    ggml_tensor * vproj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
    ggml_set_name(vproj, "model.layers.0.self_attn.v_proj.weight");
    std::memcpy(vproj->data, wide.data(), wide.size() * sizeof(float));
    gguf_add_tensor(gc, vproj);

    ggml_tensor * ln = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 32);
    ggml_set_name(ln, "model.layers.0.input_layernorm.weight");
    ggml_fp32_to_bf16_row(norm.data(), (ggml_bf16_t *) ln->data, (int64_t) norm.size());
    gguf_add_tensor(gc, ln);

    std::vector<float> norm_bf16(norm.size());
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) ln->data, norm_bf16.data(), (int64_t) norm.size());

    ggml_tensor * odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 2);
    ggml_set_name(odd, "model.layers.0.mlp.gate_proj.weight");
    std::memcpy(odd->data, narrow.data(), narrow.size() * sizeof(float));
    gguf_add_tensor(gc, odd);

    CHECK(gguf_write_to_file(gc, in_path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);

    const QuantVariant * q80 = find_quant_variant("Q8_0");
    QuantizeStats        stats;
    std::string          error;
    CHECK(quantize_gguf_file(in_path, out_path, *q80, stats, error));
    CHECK(error.empty());
    CHECK(stats.n_tensors == 4);
    CHECK(stats.n_quantized == 2);
    CHECK(stats.n_promoted == 1);

    ggml_context *   out_meta   = nullptr;
    gguf_init_params out_params = { /*no_alloc=*/false, /*ctx=*/&out_meta };
    gguf_context *   out        = gguf_init_from_file(out_path.c_str(), out_params);
    CHECK(out != nullptr);
    if (out) {
        const int64_t ft_idx = gguf_find_key(out, "general.file_type");
        CHECK(ft_idx >= 0 && gguf_get_kv_type(out, ft_idx) == GGUF_TYPE_UINT32);
        CHECK(gguf_get_val_u32(out, ft_idx) == (uint32_t) GGML_FTYPE_MOSTLY_Q8_0);

        const size_t alignment = gguf_get_alignment(out);
        for (int64_t i = 0; i < gguf_get_n_tensors(out); ++i) {
            CHECK(gguf_get_tensor_offset(out, i) % alignment == 0);
        }

        ggml_tensor * q_vproj = ggml_get_tensor(out_meta, "model.layers.0.self_attn.v_proj.weight");
        CHECK(q_vproj && q_vproj->type == GGML_TYPE_Q8_0);
        if (q_vproj) {
            std::vector<float> back(wide.size());
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q_vproj->data, back.data(),
                                                           (int64_t) back.size());
            for (size_t i = 0; i < back.size(); ++i) {
                CHECK(std::fabs(back[i] - wide[i]) < 0.02f);
            }
        }

        ggml_tensor * q_embed = ggml_get_tensor(out_meta, "model.embed_tokens.weight");
        CHECK(q_embed && q_embed->type == GGML_TYPE_Q8_0);

        ggml_tensor * q_ln = ggml_get_tensor(out_meta, "model.layers.0.input_layernorm.weight");
        CHECK(q_ln && q_ln->type == GGML_TYPE_F32);
        if (q_ln) {
            const float * vals = (const float *) q_ln->data;
            for (size_t i = 0; i < norm_bf16.size(); ++i) {
                CHECK(vals[i] == norm_bf16[i]);
            }
        }

        ggml_tensor * q_odd = ggml_get_tensor(out_meta, "model.layers.0.mlp.gate_proj.weight");
        CHECK(q_odd && q_odd->type == GGML_TYPE_F32);
        if (q_odd) {
            CHECK(std::memcmp(q_odd->data, narrow.data(), narrow.size() * sizeof(float)) == 0);
        }

        gguf_free(out);
        ggml_free(out_meta);
    }

    std::remove(in_path.c_str());
    std::remove(out_path.c_str());
}

// Same end-to-end guard for the MM3 synth layout: the protected condition
// encoder must pass through byte-identical, the depth decoder must land at
// Q8_0, and the DiT takes the variant base type. A regression here is silent
// (the file still loads) and only shows up as degraded audio.
// Each tensor gets its own source values so an expectation built from the
// wrong tensor's bytes cannot pass.
std::vector<float> quant_source_row(size_t count, float offset) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = ((float) i - offset) / 64.0f;
    }
    return values;
}

void test_quantize_gguf_roundtrip_mm3() {
    using namespace tts_cpp::acestep;

    const std::string in_path  = test_temp_dir() + "/qvac-mm3-quantize-in.gguf";
    const std::string out_path = test_temp_dir() + "/qvac-mm3-quantize-out.gguf";

    // 256-wide rows satisfy the k-quant superblock alignment, so only the
    // policy (not the alignment fallback) decides who quantizes.
    const std::vector<float> cond_row  = quant_source_row(256 * 2, 256.0f);
    const std::vector<float> depth_row = quant_source_row(256 * 2, 320.0f);
    const std::vector<float> dit_row   = quant_source_row(256 * 2, 400.0f);

    ggml_init_params ip{ 4 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    gguf_set_val_str(gc, "general.architecture", "mm3");

    ggml_tensor * cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 2);
    ggml_set_name(cond, "cond.proj.weight");
    ggml_fp32_to_fp16_row(cond_row.data(), (ggml_fp16_t *) cond->data, (int64_t) cond_row.size());
    gguf_add_tensor(gc, cond);
    std::vector<ggml_fp16_t> cond_bytes((size_t) cond_row.size());
    std::memcpy(cond_bytes.data(), cond->data, cond_row.size() * sizeof(ggml_fp16_t));

    ggml_tensor * depth = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 2);
    ggml_set_name(depth, "depth.blk.0.attn_v.weight");
    ggml_fp32_to_fp16_row(depth_row.data(), (ggml_fp16_t *) depth->data, (int64_t) depth_row.size());
    gguf_add_tensor(gc, depth);
    std::vector<ggml_fp16_t> depth_bytes((size_t) depth_row.size());
    std::memcpy(depth_bytes.data(), depth->data, depth_row.size() * sizeof(ggml_fp16_t));

    ggml_tensor * dit = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 2);
    ggml_set_name(dit, "dit.blk.0.attn_qkv.weight");
    ggml_fp32_to_fp16_row(dit_row.data(), (ggml_fp16_t *) dit->data, (int64_t) dit_row.size());
    gguf_add_tensor(gc, dit);
    std::vector<ggml_fp16_t> dit_bytes((size_t) dit_row.size());
    std::memcpy(dit_bytes.data(), dit->data, dit_row.size() * sizeof(ggml_fp16_t));

    CHECK(gguf_write_to_file(gc, in_path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);

    const QuantVariant * q4km = find_quant_variant("Q4_K_M");
    QuantizeStats        stats;
    std::string          error;
    CHECK(quantize_gguf_file(in_path, out_path, *q4km, stats, error));
    CHECK(error.empty());
    CHECK(stats.n_tensors == 3);
    CHECK(stats.n_quantized == 2);

    ggml_context *   out_meta   = nullptr;
    gguf_init_params out_params = { /*no_alloc=*/false, /*ctx=*/&out_meta };
    gguf_context *   out        = gguf_init_from_file(out_path.c_str(), out_params);
    CHECK(out != nullptr);
    if (out) {
        ggml_tensor * q_cond = ggml_get_tensor(out_meta, "cond.proj.weight");
        CHECK(q_cond && q_cond->type == GGML_TYPE_F16);
        if (q_cond) {
            CHECK(std::memcmp(q_cond->data, cond_bytes.data(),
                              cond_bytes.size() * sizeof(ggml_fp16_t)) == 0);
        }

        ggml_tensor * q_depth = ggml_get_tensor(out_meta, "depth.blk.0.attn_v.weight");
        CHECK(q_depth && q_depth->type == GGML_TYPE_Q4_K);
        if (q_depth) {
            std::vector<float> f32(depth_row.size());
            ggml_fp16_to_fp32_row(depth_bytes.data(), f32.data(), (int64_t) f32.size());
            std::vector<uint8_t> expected(ggml_row_size(GGML_TYPE_Q4_K, 256) * 2);
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f32.data(), expected.data(), 0, 2, 256, nullptr);
            CHECK(std::memcmp(q_depth->data, expected.data(), expected.size()) == 0);
        }

        ggml_tensor * q_dit = ggml_get_tensor(out_meta, "dit.blk.0.attn_qkv.weight");
        CHECK(q_dit && q_dit->type == GGML_TYPE_Q4_K);
        if (q_dit) {
            // The writer must emit exactly ggml's reference quantization of the
            // F32-converted source rows (ggml_quantize_chunk, no imatrix).
            std::vector<float> f32(dit_row.size());
            ggml_fp16_to_fp32_row(dit_bytes.data(), f32.data(), (int64_t) f32.size());
            std::vector<uint8_t> expected(ggml_row_size(GGML_TYPE_Q4_K, 256) * 2);
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f32.data(), expected.data(), 0, 2, 256, nullptr);
            CHECK(std::memcmp(q_dit->data, expected.data(), expected.size()) == 0);

            std::vector<float> back(dit_row.size());
            ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(q_dit->data, back.data(),
                                                           (int64_t) back.size());
            for (size_t i = 0; i < back.size(); ++i) {
                CHECK(std::fabs(back[i] - dit_row[i]) < 0.25f);
            }
        }

        gguf_free(out);
        ggml_free(out_meta);
    }

    std::remove(in_path.c_str());
    std::remove(out_path.c_str());
}


}  // namespace

int main() {
    test_quantize_policy();
    test_quantize_policy_mm3();
    test_quantize_gguf_roundtrip();
    test_quantize_gguf_roundtrip_mm3();
    std::fprintf(stderr, "[test-acestep-quantization] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
