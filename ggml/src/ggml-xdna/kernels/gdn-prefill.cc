#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

// Qwen3.5 GDN prefill step. Semantics match GGML_OP_GATED_DELTA_NET (K=1,
// scalar gate): each state row r in [j0, j0+ROWS):
//   row *= eg; delta = (v[r] - row.k) * beta; row += delta * k;
//   attn[r] = scale * row.q
//
// Rows are independent, so a 64-row strip is a complete slice. Packed object
// is [state strip | attn strip]; the strip is updated in place for all CS
// tokens of this head.
//
// The strip is held transposed, [DH][ROWS]. Both dot products then accumulate
// across the DH loop with r on the vector lanes, so a token costs no lane
// reductions at all - only ROWS-wide vector MACs. Row-major would need two
// aie::reduce_add per row per token, and those do not overlap.

#ifndef DH
#define DH 128
#endif
#ifndef ROWS
#define ROWS 64
#endif
#ifndef VEC
#define VEC 16
#endif

extern "C" void ggml_xdna_gdn_copy_strip(bfloat16 * dst, bfloat16 * src) {
    for (int i = 0; i < ROWS * DH; i += VEC) {
        ::aie::store_v(dst + i, ::aie::load_v<VEC>(src + i));
    }
}

extern "C" void ggml_xdna_gdn_token_bf16(
    bfloat16 * packed,
    bfloat16 * tok,
    int32_t t,
    int32_t j0) {
    const bfloat16 * q = tok;
    const bfloat16 * k = tok + DH;
    const bfloat16 * v = tok + 2 * DH + (int) j0;
    const float      eg    = (float) tok[3 * DH];
    const float      beta  = (float) tok[3 * DH + 1];
    const float      scale = 0.08838834764831845f;
    bfloat16 * state = packed;
    bfloat16 * attn  = packed + ROWS * DH + (int) t * ROWS;

    // the default floor mode biases every bf16 state store, and the error
    // compounds linearly over the recurrence
    ::aie::set_rounding(::aie::rounding_mode::conv_even);

    // ROWS must be one accumulator value, not an array of NB: indexing an accum
    // array makes the compiler spill it to stack and reload it every MAC
    ::aie::accum<accfloat, ROWS> sk = ::aie::zeros<accfloat, ROWS>();
#pragma clang loop unroll_count(8)
    for (int i = 0; i < DH; i++) {
        sk = ::aie::mac(sk, ::aie::load_v<ROWS>(state + i * ROWS),
                        ::aie::broadcast<bfloat16, ROWS>(k[i]));
    }

    ::aie::accum<accfloat, ROWS> va;
    // j0 = ROWS leaves v short of the ROWS-wide load alignment
    va.from_vector(::aie::load_unaligned_v<ROWS>(v));
    auto df = ::aie::sub(va.to_vector<float>(),
                         ::aie::mul(sk.to_vector<float>(), eg).to_vector<float>());
    ::aie::accum<accfloat, ROWS> da;
    da.from_vector(::aie::mul(df, beta).to_vector<float>());
    const auto dl = da.to_vector<bfloat16>();

    ::aie::accum<accfloat, ROWS> at = ::aie::zeros<accfloat, ROWS>();
    const auto egv = ::aie::broadcast<bfloat16, ROWS>((bfloat16) eg);
#pragma clang loop unroll_count(8)
    for (int i = 0; i < DH; i++) {
        bfloat16 * p = state + i * ROWS;
        auto acc = ::aie::mul(::aie::load_v<ROWS>(p), egv);
        acc = ::aie::mac(acc, dl, ::aie::broadcast<bfloat16, ROWS>(k[i]));
        const auto nv = acc.to_vector<bfloat16>();
        ::aie::store_v(p, nv);
        at = ::aie::mac(at, nv, ::aie::broadcast<bfloat16, ROWS>(q[i]));
    }

    ::aie::accum<accfloat, ROWS> sa;
    sa.from_vector(::aie::mul(at.to_vector<float>(), scale).to_vector<float>());
    ::aie::store_v(attn, sa.to_vector<bfloat16>());
}
