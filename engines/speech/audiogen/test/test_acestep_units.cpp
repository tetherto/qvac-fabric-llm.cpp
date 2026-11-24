// Model-free ACE-Step units regression tests.

#include "acestep_test_support.h"
#include "dit_ggml.h"
#include "philox.h"
#include "tok_ggml.h"
#include "detok_ggml.h"
#include "lm_pipeline.h"
#include "metadata_fsm.h"
#include "generation_plan.h"

namespace {

void test_schedule() {
    using tts_cpp::acestep::dit_build_schedule;
    std::vector<float> sched;

    // shift == 1.0 collapses to a linear ramp: schedule[i] = 1 - i/N.
    dit_build_schedule(1.0f, 8, sched);
    CHECK(sched.size() == 8);
    CHECK(approx(sched[0], 1.0f));                 // first step starts at pure noise
    CHECK(approx(sched[4], 1.0f - 4.0f / 8.0f));   // == 0.5
    for (size_t i = 1; i < sched.size(); ++i) CHECK(sched[i] < sched[i - 1]);  // strictly decreasing

    // shift > 1 (turbo uses 3.0) still starts at 1.0 and stays monotone.
    dit_build_schedule(3.0f, 8, sched);
    CHECK(sched.size() == 8);
    CHECK(approx(sched[0], 1.0f));
    for (size_t i = 1; i < sched.size(); ++i) CHECK(sched[i] < sched[i - 1]);
    for (float v : sched) CHECK(v > 0.0f && v <= 1.0f);
}

void test_haar_dcw() {
    using tts_cpp::acestep::dit_apply_haar_dcw;

    // Low-band correction moves both samples in a pair together.
    std::vector<float> x = { 2.0f, 4.0f };
    const std::vector<float> y_low = { 1.0f, 3.0f };
    dit_apply_haar_dcw(x, y_low, /*T=*/2, /*C=*/1, /*N=*/1, 0.1f, 0.0f);
    CHECK(approx(x[0], 2.1f));
    CHECK(approx(x[1], 4.1f));

    // High-band correction changes the contrast inside the temporal pair.
    x = { 2.0f, 4.0f };
    const std::vector<float> y_high = { 1.0f, 5.0f };
    dit_apply_haar_dcw(x, y_high, 2, 1, 1, 0.0f, 0.2f);
    CHECK(approx(x[0], 2.2f));
    CHECK(approx(x[1], 3.8f));

    // Odd temporal lengths use a zero-padded partner and discard it after IDWT.
    x = { 2.0f };
    const std::vector<float> y_odd = { 1.0f };
    dit_apply_haar_dcw(x, y_odd, 1, 1, 1, 0.1f, 0.2f);
    CHECK(approx(x[0], 2.15f));

    // Disabled coefficients are an exact no-op.
    x = { -3.0f, 7.0f };
    const std::vector<float> before = x;
    dit_apply_haar_dcw(x, y_low, 2, 1, 1, 0.0f, 0.0f);
    CHECK(x == before);
}

void test_philox() {
    using tts_cpp::acestep::philox_randn;
    using tts_cpp::acestep::philox_randn_from;

    // Golden vector for seed 42 (bf16-rounded, the mode the engine uses for
    // the DiT initial noise). These values were validated corr=1.0 against
    // torch.randn() on CUDA via acestep.cpp's --dump; they lock the port.
    const float golden[8] = { 0.194335938f,  2.156250000f,  -0.171875000f, 0.847656250f,
                              -1.921875000f, 0.652343750f,  -0.648437500f, -0.816406250f };
    float out[8];
    philox_randn(42, out, 8, /*bf16_round=*/true);
    for (int i = 0; i < 8; ++i) CHECK(approx(out[i], golden[i], 1e-6f));

    // Determinism: same seed -> identical stream.
    float a[16], b[16];
    philox_randn(1234, a, 16, true);
    philox_randn(1234, b, 16, true);
    for (int i = 0; i < 16; ++i) CHECK(a[i] == b[i]);

    // Repeated draws from one generator continue at the next subsequence,
    // matching one torch.Generator whose state advances across randn calls.
    float first[7], second[9];
    philox_randn_from(1234, 0, first, 7, true);
    philox_randn_from(1234, 7, second, 9, true);
    for (int i = 0; i < 7; ++i)
      CHECK(first[i] == a[i]);
    for (int i = 0; i < 9; ++i)
      CHECK(second[i] == a[i + 7]);

    // Different seeds -> different stream (extremely unlikely to collide).
    float c[16];
    philox_randn(4321, c, 16, true);
    bool differs = false;
    for (int i = 0; i < 16; ++i) differs |= (a[i] != c[i]);
    CHECK(differs);

    // Statistical sanity over a large draw: mean ~ 0, std ~ 1.
    const int          N = 8192;
    std::vector<float> big(N);
    philox_randn(7, big.data(), N, false);
    double mean = 0.0;
    for (float v : big) mean += v;
    mean /= N;
    double var = 0.0;
    for (float v : big) var += (v - mean) * (v - mean);
    var /= N;
    CHECK(std::fabs(mean) < 0.05);
    CHECK(std::fabs(std::sqrt(var) - 1.0) < 0.05);
}

void test_fsq() {
    using tts_cpp::acestep::fsq_decode_index;
    float out[6];

    // index 0 -> all digits 0 -> every dim at the low end (-1).
    fsq_decode_index(0, out);
    for (int d = 0; d < 6; ++d) CHECK(approx(out[d], -1.0f));

    // digit 1 in dim0 (L=8, half=3.5): 1/3.5 - 1.
    fsq_decode_index(1, out);
    CHECK(approx(out[0], 1.0f / 3.5f - 1.0f));
    for (int d = 1; d < 6; ++d) CHECK(approx(out[d], -1.0f));

    // Max index across all dims -> every dim at the high end (+1).
    int max_index = 8 * 8 * 8 * 5 * 5 * 5 - 1;
    fsq_decode_index(max_index, out);
    for (int d = 0; d < 6; ++d) CHECK(approx(out[d], 1.0f));

    // Output is always in [-1, 1] (small FP slack at the boundaries).
    const float slack = 1e-4f;
    for (int idx = 0; idx < 500; idx += 37) {
        fsq_decode_index(idx, out);
        for (int d = 0; d < 6; ++d) CHECK(out[d] >= -1.0f - slack && out[d] <= 1.0f + slack);
    }
}

// FSQ encode: the tokenizer-side inverse of fsq_decode_index.
void test_fsq_encode() {
    using tts_cpp::acestep::fsq_decode_index;
    using tts_cpp::acestep::fsq_encode_index;

    // Saturated projector outputs land on the level extremes.
    const float low[6]  = { -100.0f, -100.0f, -100.0f, -100.0f, -100.0f, -100.0f };
    const float high[6] = { 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f };
    CHECK(fsq_encode_index(low) == 0);
    CHECK(fsq_encode_index(high) == 8 * 8 * 8 * 5 * 5 * 5 - 1);

    // Zero raw values quantize to the middle level of every dim:
    // codes {4,4,4,2,2,2} with strides {1,8,64,512,2560,12800}.
    const float mid[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    CHECK(fsq_encode_index(mid) == 4 + 4 * 8 + 4 * 64 + 2 * 512 + 2 * 2560 + 2 * 12800);

    // Roundtrip through the decoder: interior indices survive
    // decode -> atanh -> encode exactly (the extremes would need atanh(1)).
    for (int idx = 0; idx < 64000; idx += 1237) {
        float decoded[6];
        fsq_decode_index(idx, decoded);
        bool interior = true;
        for (float v : decoded) {
            if (v <= -0.999f || v >= 0.999f) interior = false;
        }
        if (!interior) continue;
        float raw[6];
        for (int d = 0; d < 6; ++d) raw[d] = std::atanh(decoded[d]);
        CHECK(fsq_encode_index(raw) == idx);
    }
}

void test_sampler() {
    using tts_cpp::acestep::sample_top_k_p;

    const int         V = 32;
    std::vector<float> logits(V, 0.0f);
    logits[17] = 10.0f;  // clearly dominant token

    // top_k = 1 keeps only the argmax, so the pick is deterministic regardless
    // of the RNG draw.
    for (uint32_t seed = 0; seed < 8; ++seed) {
        std::vector<float> l = logits;
        std::mt19937       rng(seed);
        CHECK(sample_top_k_p(l.data(), V, 1.0f, 1.0f, /*top_k=*/1, rng) == 17);
    }

    // Same RNG seed -> identical token sequence (reproducible generation).
    std::vector<float> l1 = logits, l2 = logits;
    std::mt19937       r1(99), r2(99);
    for (int step = 0; step < 5; ++step) {
        std::vector<float> a = l1, b = l2;
        int ta = sample_top_k_p(a.data(), V, 0.85f, 0.9f, 0, r1);
        int tb = sample_top_k_p(b.data(), V, 0.85f, 0.9f, 0, r2);
        CHECK(ta == tb);
        CHECK(ta >= 0 && ta < V);
    }
}

// Compact-slice sampling must match the historical full-vocab path exactly:
// entries below the EOS cut were only ever masked to -1e9, so sampling the
// [eos, V) slice via index_base is an identity, RNG stream included.
void test_sampler_compact_equivalence() {
    using tts_cpp::acestep::sample_top_k_p;

    const int V = 1000, eos = 100, base = 124;  // mirrors TOKEN_IM_END/AUDIO_CODE_BASE layout
    std::mt19937                          gen(1234);
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);

    for (int trial = 0; trial < 20; ++trial) {
        std::vector<float> raw(V);
        for (float & v : raw) v = ud(gen);

        std::vector<float> full(raw);
        for (int v = 0; v < base; ++v)
            if (v != eos) full[v] = -1e9f;
        std::vector<float> slice(raw.begin() + eos, raw.end());
        for (int v = 1; v < base - eos; ++v) slice[v] = -1e9f;

        std::mt19937 r_full(trial), r_slice(trial);
        const float  temp  = (trial % 5 == 0) ? 0.0f : 0.85f;  // argmax path every 5th trial
        const float  top_p = (trial % 7 == 0) ? 0.0f : 0.9f;   // also cover the disabled-top_p path
        int tf = sample_top_k_p(full.data(), V, temp, top_p, 0, r_full);
        int ts = sample_top_k_p(slice.data(), V - eos, temp, top_p, 0, r_slice, eos);
        CHECK(tf == ts);
        CHECK(r_full == r_slice);

        // Prefix form (phase-1 FSM shape): tail [base2, V) masked -> sampling
        // only [0, base2) is the same identity.
        const int          base2 = 900;
        std::vector<float> full2(raw);
        for (int v = base2; v < V; ++v) full2[v] = -1e9f;
        std::vector<float> pre(raw.begin(), raw.begin() + base2);
        std::mt19937       r_f2(trial + 100), r_p2(trial + 100);
        int tf2 = sample_top_k_p(full2.data(), V, temp, top_p, 0, r_f2);
        int tp2 = sample_top_k_p(pre.data(), base2, temp, top_p, 0, r_p2);
        CHECK(tf2 == tp2);
        CHECK(r_f2 == r_p2);
    }
}

// r==0 edge of the compact form. dist(0, sum) yields exactly 0.0f only when
// the raw 32-bit mt19937 output is 0 (sum >= 1.0, so no underflow path); an
// all-zero engine state produces that draw deterministically. The historical
// full-vector walk then lands on absolute index 0, and the index_base form
// must land on the same spot - NOT on index_base + 0 (= EOS in phase 2).
std::mt19937 make_zero_draw_rng() {
    std::stringstream ss;
    for (int i = 0; i < 624; ++i) ss << 0 << ' ';
    ss << 0;
    std::mt19937 rng;
    ss >> rng;
    return rng;
}

void test_sampler_r0_edge() {
    using tts_cpp::acestep::sample_top_k_p;

    {  // Portability guard: stream format / canonical mapping are stdlib-specific.
        std::mt19937                          probe = make_zero_draw_rng();
        std::uniform_real_distribution<float> d01(0.0f, 1.0f);
        if (d01(probe) != 0.0f) {
            std::fprintf(stderr, "[test-acestep-units] r==0 rng state not reproducible here, subtest skipped\n");
            return;
        }
    }

    const int                             V = 1000, eos = 100, base = 124;
    std::mt19937                          gen(4321);
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
    std::vector<float>                    raw(V);
    for (float & v : raw) v = ud(gen);

    for (float top_p : { 0.9f, 0.0f }) {  // top_p and disabled-top_p branches
        std::vector<float> full(raw);
        for (int v = 0; v < base; ++v)
            if (v != eos) full[v] = -1e9f;
        std::vector<float> slice(raw.begin() + eos, raw.end());
        for (int v = 1; v < base - eos; ++v) slice[v] = -1e9f;

        std::mt19937 r_full = make_zero_draw_rng(), r_slice = make_zero_draw_rng();
        int tf = sample_top_k_p(full.data(), V, 0.85f, top_p, 0, r_full);
        int ts = sample_top_k_p(slice.data(), V - eos, 0.85f, top_p, 0, r_slice, eos);
        CHECK(tf == 0);
        CHECK(ts == 0);
        CHECK(r_full == r_slice);
    }
}

// lm_consume_forced must be indistinguishable from masking all but one token
// and running the full sampler: same pick, same RNG consumption.
void test_sampler_forced_fast_path() {
    using tts_cpp::acestep::lm_consume_forced;
    using tts_cpp::acestep::sample_top_k_p;

    const int V = 500;
    for (int trial = 0; trial < 12; ++trial) {
        const int   live = 37 + trial * 13;
        const float temp = (trial % 4 == 0) ? 0.0f : 0.85f;

        std::vector<float> l(V, -1e9f);
        l[live] = 1.5f;
        std::mt19937 r_full(trial), r_fast(trial);
        int tf = sample_top_k_p(l.data(), V, temp, 0.9f, 0, r_full);
        int tc = lm_consume_forced(live, temp, r_fast);
        CHECK(tf == tc);
        CHECK(r_full == r_fast);
    }
}

// MetadataFSM::forced_token must agree with apply_mask: it returns exactly the
// single surviving token when one exists, -1 when the LM still has a choice.
void test_fsm_forced_token() {
    using tts_cpp::acestep::MetadataFSM;

    const int V         = 64;
    auto      survivors = [&](MetadataFSM & f) {
        std::vector<float> l(V, 1.0f);
        f.apply_mask(l.data());
        std::vector<int> alive;
        for (int v = 0; v < V; ++v)
            if (l[v] > -1e8f) alive.push_back(v);
        return alive;
    };
    auto agree = [&](MetadataFSM & f) {
        MetadataFSM probe = f;  // forced_token may seed inject_queue; probe a copy
        int         t     = probe.forced_token();
        auto        alive = survivors(f);
        if (alive.size() == 1) CHECK(t == alive[0]);
        else CHECK(t == -1);
    };

    MetadataFSM fsm;
    fsm.enabled    = true;
    fsm.vocab_size = V;

    fsm.state    = MetadataFSM::BPM_NAME;  // name tokens force one token at a time
    fsm.bpm_name = { 5, 7 };
    fsm.name_pos = 0;
    agree(fsm);
    fsm.name_pos = 1;
    agree(fsm);

    fsm.inject_queue = { 9 };  // injected value: forced
    agree(fsm);
    fsm.inject_queue.clear();

    fsm.state         = MetadataFSM::THINK_END;  // </think> is forced
    fsm.think_end_tok = 20;                      // keep it inside the synthetic vocab
    agree(fsm);

    fsm.state       = MetadataFSM::BPM_VALUE;  // value tree: forced only when 1 child
    fsm.name_pos    = (int) fsm.bpm_name.size();
    fsm.newline_tok = 3;
    fsm.bpm_tree.add({ 11, 12 });
    fsm.value_acc.clear();
    agree(fsm);  // single child {11}
    fsm.bpm_tree.add({ 13, 12 });
    agree(fsm);  // two children {11,13} -> free choice
    fsm.value_acc = { 11 };
    agree(fsm);  // single child {12}
    fsm.value_acc = { 40 };
    agree(fsm);  // unknown prefix -> forced newline
}

// Verbatim pre-fusion sampler, kept as an oracle: the fused sample_top_k_p must
// match it token-for-token and draw-for-draw.
struct RefTokenProb {
    int   id;
    float prob;
};

int reference_sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        return (int) (std::max_element(logits, logits + V) - logits);
    }

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) logits[i] *= inv_temp;

    if (top_k > 0 && top_k < V) {
        std::vector<float> tmp(logits, logits + V);
        std::nth_element(tmp.begin(), tmp.begin() + (top_k - 1), tmp.end(), std::greater<float>());
        float threshold = tmp[top_k - 1];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) logits[i] = -INFINITY;
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float max_logit = -INFINITY;
        for (int i = 0; i < V; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++) sum_exp += expf(logits[i] - max_logit);
        float inv_sum = 1.0f / sum_exp;

        float                     cutoff = max_logit - 16.0f;
        std::vector<RefTokenProb> sorted;
        for (int i = 0; i < V; i++) {
            if (logits[i] >= cutoff) {
                sorted.push_back({ i, expf(logits[i] - max_logit) * inv_sum });
            } else {
                logits[i] = -INFINITY;
            }
        }

        int K = (int) sorted.size();
        if (K > 0) {
            std::sort(sorted.begin(), sorted.end(),
                      [](const RefTokenProb & a, const RefTokenProb & b) { return a.prob > b.prob; });
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) logits[sorted[i].id] = -INFINITY;
                cum += sorted[i].prob;
            }
        }
    }

    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float                                 r   = dist(rng);
    float                                 acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) return i;
    }
    return 0;
}

void test_sampler_fusion_oracle() {
    using tts_cpp::acestep::sample_top_k_p;

    const int                             V = 4096;
    std::mt19937                          gen(77);
    std::uniform_real_distribution<float> ud(-6.0f, 6.0f);

    for (int trial = 0; trial < 30; ++trial) {
        std::vector<float> raw(V);
        for (float & v : raw) v = ud(gen);
        // Adversarial shapes: -1e9 masses, -inf islands, exact ties.
        if (trial % 3 == 0)
            for (int i = 0; i < V; i += 2) raw[i] = -1e9f;
        if (trial % 4 == 0)
            for (int i = 0; i < V / 8; ++i) raw[i] = raw[V - 1 - i];
        if (trial % 5 == 0)
            for (int i = 100; i < 200; ++i) raw[i] = -INFINITY;
        const float temp  = (trial % 6 == 0) ? 0.0f : 0.85f;
        const float top_p = (trial % 7 == 0) ? 0.0f : 0.9f;  // also cover the disabled-top_p path
        const int   top_k = (trial % 8 == 0) ? 40 : 0;

        std::vector<float> a = raw, b = raw;
        std::mt19937       r1(trial), r2(trial);
        int ta = reference_sample_top_k_p(a.data(), V, temp, top_p, top_k, r1);
        int tb = sample_top_k_p(b.data(), V, temp, top_p, top_k, r2);
        CHECK(ta == tb);
        CHECK(r1 == r2);
    }
}

void test_guidance_and_dcw_policy() {
    using tts_cpp::acestep::resolve_dcw_enabled;
    using tts_cpp::acestep::resolve_guidance_scale;

    CHECK(approx(resolve_guidance_scale(0.0f, true), 1.0f));
    CHECK(approx(resolve_guidance_scale(7.0f, true), 1.0f));
    CHECK(approx(resolve_guidance_scale(0.0f, false), 7.0f));
    CHECK(approx(resolve_guidance_scale(3.5f, false), 3.5f));

    CHECK(resolve_dcw_enabled(true, true));
    CHECK(!resolve_dcw_enabled(true, false));
    CHECK(!resolve_dcw_enabled(false, true));
    CHECK(!resolve_dcw_enabled(false, false));
}

// APG guide: golden values hand-derived from the reference apg_forward
// (momentum -0.75, norm_threshold 2.5, projection per channel over T).
void test_apg_guide() {
    using tts_cpp::acestep::dit_apg_guide;

    std::vector<double> momentum(2, 0.0);
    std::vector<float> cond = { 3.0f, 4.0f };
    std::vector<float> uncond = { 2.0f, 2.0f };
    std::vector<float> velocity = cond;
    dit_apg_guide(velocity, uncond, momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], 1.08f, 1e-4f));
    CHECK(approx(velocity[1], 5.44f, 1e-4f));

    velocity = cond;
    dit_apg_guide(velocity, uncond, momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], 2.52f, 1e-4f));
    CHECK(approx(velocity[1], 4.36f, 1e-4f));

    std::vector<double> fresh_momentum(2, 0.0);
    std::vector<float> parallel_cond = { 10.0f, 0.0f };
    std::vector<float> parallel_uncond = { 6.0f, 0.0f };
    velocity = parallel_cond;
    dit_apg_guide(velocity, parallel_uncond, fresh_momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], 10.0f, 1e-4f));
    CHECK(approx(velocity[1], 0.0f, 1e-4f));

    std::vector<double> zero_momentum(2, 0.0);
    std::vector<float> equal = { 1.5f, -0.5f };
    velocity = equal;
    dit_apg_guide(velocity, equal, zero_momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], equal[0]));
    CHECK(approx(velocity[1], equal[1]));
}


}  // namespace

int main() {
    test_schedule();
    test_haar_dcw();
    test_philox();
    test_fsq();
    test_fsq_encode();
    test_sampler();
    test_sampler_compact_equivalence();
    test_sampler_r0_edge();
    test_sampler_forced_fast_path();
    test_fsm_forced_token();
    test_sampler_fusion_oracle();
    test_guidance_and_dcw_policy();
    test_apg_guide();
    std::fprintf(stderr, "[test-acestep-units] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
