const bool QJL_FULL_PROJ = (Flags & 16) != 0;

// Output accumulator type. The b9518 generic FA shaders define O_TYPE/O_TYPEV4
// in the LLAMA_UPSTREAM_FA_MIXED_TYPES branch of flash_attn_base.glsl; the TBQ
// per-type compile-time path (this include) is fp16-only, so mirror FLOAT_TYPE.
#ifndef O_TYPE
#define O_TYPE   FLOAT_TYPE
#endif
#ifndef O_TYPEV4
#define O_TYPEV4 FLOAT_TYPEV4
#endif

// ============================================================================
// Backward compatibility: map DATA_A_* to both DATA_K_* and DATA_V_*
// ============================================================================
#if defined(DATA_A_F32) && !defined(DATA_K_F32)
#define DATA_K_F32
#define DATA_V_F32
#endif

#if defined(A_TYPE_PACKED32)
layout (binding = 1) readonly buffer K_PACKED32 {A_TYPE_PACKED32 k_data_packed32[];} k_packed32;
layout (binding = 2) readonly buffer V_PACKED32 {A_TYPE_PACKED32 v_data_packed32[];} v_packed32;
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#if defined(DATA_A_Q4_0) && !defined(DATA_K_Q4_0)
#define DATA_K_Q4_0
#define DATA_V_Q4_0
#endif

#if defined(DATA_A_Q8_0) && !defined(DATA_K_Q8_0)
#define DATA_K_Q8_0
#define DATA_V_Q8_0
#endif

#if defined(DATA_A_TBQ3_0) && !defined(DATA_K_TBQ3_0)
#define DATA_K_TBQ3_0
#define DATA_V_TBQ3_0
#endif

#if defined(DATA_A_TBQ4_0) && !defined(DATA_K_TBQ4_0)
#define DATA_K_TBQ4_0
#define DATA_V_TBQ4_0
#endif

#if defined(DATA_A_PQ3_0) && !defined(DATA_K_PQ3_0)
#define DATA_K_PQ3_0
#define DATA_V_PQ3_0
#endif

#if defined(DATA_A_PQ4_0) && !defined(DATA_K_PQ4_0)
#define DATA_K_PQ4_0
#define DATA_V_PQ4_0
#endif

// _64 variants (head_dim=64): same dequant logic, different block struct size
#if defined(DATA_A_TBQ3_0_64) && !defined(DATA_K_TBQ3_0_64)
#define DATA_K_TBQ3_0_64
#define DATA_V_TBQ3_0_64
#endif
#if defined(DATA_A_TBQ4_0_64) && !defined(DATA_K_TBQ4_0_64)
#define DATA_K_TBQ4_0_64
#define DATA_V_TBQ4_0_64
#endif
#if defined(DATA_A_PQ3_0_64) && !defined(DATA_K_PQ3_0_64)
#define DATA_K_PQ3_0_64
#define DATA_V_PQ3_0_64
#endif
#if defined(DATA_A_PQ4_0_64) && !defined(DATA_K_PQ4_0_64)
#define DATA_K_PQ4_0_64
#define DATA_V_PQ4_0_64
#endif

// ============================================================================
// For mixed-type mode, ensure QUANT_K is defined for QJL correction.
// In same-type mode DATA_A_* sets QUANT_K via types.glsl; in mixed-type mode
// we derive it from the K type.
// ============================================================================
#if !defined(QUANT_K)
#if defined(DATA_K_TBQ3_0_64) || defined(DATA_K_PQ3_0_64)
#define QUANT_K QUANT_K_TBQ3_0_64
#elif defined(DATA_K_TBQ4_0_64) || defined(DATA_K_PQ4_0_64)
#define QUANT_K QUANT_K_TBQ4_0_64
#elif defined(DATA_K_TBQ3_0) || defined(DATA_K_PQ3_0)
#define QUANT_K QUANT_K_TBQ3_0
#elif defined(DATA_K_TBQ4_0) || defined(DATA_K_PQ4_0)
#define QUANT_K QUANT_K_TBQ4_0
#elif defined(DATA_K_Q8_0)
#define QUANT_K QUANT_K_Q8_0
#elif defined(DATA_K_Q4_0)
#define QUANT_K QUANT_K_Q4_0
#endif
#endif

// ============================================================================
// Include tq_utils.comp if needed by any K or V type
// ============================================================================
#if defined(DATA_K_TBQ3_0) || defined(DATA_K_TBQ4_0) || defined(DATA_K_PQ3_0) || defined(DATA_K_PQ4_0) || \
    defined(DATA_V_TBQ3_0) || defined(DATA_V_TBQ4_0) || defined(DATA_V_PQ3_0) || defined(DATA_V_PQ4_0) || \
    defined(DATA_K_TBQ3_0_64) || defined(DATA_K_TBQ4_0_64) || defined(DATA_K_PQ3_0_64) || defined(DATA_K_PQ4_0_64) || \
    defined(DATA_V_TBQ3_0_64) || defined(DATA_V_TBQ4_0_64) || defined(DATA_V_PQ3_0_64) || defined(DATA_V_PQ4_0_64)
#include "../tq_utils.glsl"
#endif

// ============================================================================
// K buffer declarations (binding 1)
// ============================================================================
#if defined(DATA_K_F16)
layout (binding = 1) readonly buffer KV4_K {f16vec4 k_data_f16v4[];};
#define K_BLOCK_SIZE 1
#define K_BLOCK_BYTE_SIZE 2
#elif defined(DATA_K_F32)
layout (binding = 1) readonly buffer K_PACKED {vec4 k_data_packed[];} k_packed;
#define K_BLOCK_SIZE 4
#define K_BLOCK_BYTE_SIZE 16
#elif defined(DATA_K_Q4_0)
layout (binding = 1) readonly buffer K_PACKED16 {block_q4_0_packed16 k_data_packed16[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_Q4_0
#define K_BLOCK_BYTE_SIZE 18
#elif defined(DATA_K_Q8_0)
layout (binding = 1) readonly buffer K_PACKED16 {block_q8_0_packed16 k_data_packed16[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_Q8_0
#define K_BLOCK_BYTE_SIZE 34
#elif defined(DATA_K_TBQ3_0_64)
layout (binding = 1) readonly buffer K_TQ3 {block_tbq3_0_64 k_data_tbq3[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_TBQ3_0_64
#define K_BLOCK_BYTE_SIZE 36
#define HAS_QJL_CORRECTION
#elif defined(DATA_K_TBQ3_0)
layout (binding = 1) readonly buffer K_TQ3 {block_tbq3_0 k_data_tbq3[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_TBQ3_0
#define K_BLOCK_BYTE_SIZE 68
#define HAS_QJL_CORRECTION
#elif defined(DATA_K_TBQ4_0_64)
layout (binding = 1) readonly buffer K_TQ4 {block_tbq4_0_64 k_data_tbq4[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_TBQ4_0_64
#define K_BLOCK_BYTE_SIZE 44
#define HAS_QJL_CORRECTION
#elif defined(DATA_K_TBQ4_0)
layout (binding = 1) readonly buffer K_TQ4 {block_tbq4_0 k_data_tbq4[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_TBQ4_0
#define K_BLOCK_BYTE_SIZE 84
#define HAS_QJL_CORRECTION
#elif defined(DATA_K_PQ3_0_64)
layout (binding = 1) readonly buffer K_PQ3 {block_pq3_0_64 k_data_pq3[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_PQ3_0_64
#define K_BLOCK_BYTE_SIZE 26
#elif defined(DATA_K_PQ3_0)
layout (binding = 1) readonly buffer K_PQ3 {block_pq3_0 k_data_pq3[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_PQ3_0
#define K_BLOCK_BYTE_SIZE 50
#elif defined(DATA_K_PQ4_0_64)
layout (binding = 1) readonly buffer K_PQ4 {block_pq4_0_64 k_data_pq4[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_PQ4_0_64
#define K_BLOCK_BYTE_SIZE 34
#elif defined(DATA_K_PQ4_0)
layout (binding = 1) readonly buffer K_PQ4 {block_pq4_0 k_data_pq4[];} k_packed;
#define K_BLOCK_SIZE QUANT_K_PQ4_0
#define K_BLOCK_BYTE_SIZE 66
#endif

// Fallback for the upstream same-type f16 variant, which is emitted by
// vulkan-shaders-gen.cpp without any DATA_K_* macro. Without these defaults
// `#if K_BLOCK_SIZE == 1` style guards in the FA shaders silently evaluate
// to 0 == 1 (false), eliminating whole code blocks from the SPIR-V.
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 1
#define K_BLOCK_BYTE_SIZE 2
#endif

// Fallback for upstream legacy types (Q4_1, Q5_0, Q5_1, IQ4_NL) that go
// through the same-type code path with DATA_A_* set but no DATA_K_*.
// The per-type K_PACKED16 declarations above only cover Q4_0/Q8_0/TBQ/PQ;
// without this fallback, the dequantize4() functions below see an
// undeclared `k_packed`.
#if !defined(DATA_K_F16) && !defined(DATA_K_F32) && \
    !defined(DATA_K_Q4_0) && !defined(DATA_K_Q8_0) && \
    !defined(DATA_K_TBQ3_0_64) && !defined(DATA_K_TBQ3_0) && \
    !defined(DATA_K_TBQ4_0_64) && !defined(DATA_K_TBQ4_0) && \
    !defined(DATA_K_PQ3_0_64) && !defined(DATA_K_PQ3_0) && \
    !defined(DATA_K_PQ4_0_64) && !defined(DATA_K_PQ4_0) && \
    (defined(DATA_A_Q4_1) || defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1) || defined(DATA_A_IQ4_NL))
layout (binding = 1) readonly buffer K_PACKED16 {A_TYPE_PACKED16 k_data_packed16[];} k_packed;
#endif

#if defined(DATA_K_TBQ3_0) || defined(DATA_K_PQ3_0) || defined(DATA_K_TBQ4_0) || defined(DATA_K_PQ4_0) || \
    defined(DATA_K_TBQ3_0_64) || defined(DATA_K_PQ3_0_64) || defined(DATA_K_TBQ4_0_64) || defined(DATA_K_PQ4_0_64)
#define HAS_CENTROID_K
#endif
#if defined(DATA_K_TBQ3_0) || defined(DATA_K_PQ3_0) || defined(DATA_K_TBQ3_0_64) || defined(DATA_K_PQ3_0_64)
#define K_NUM_CENTROIDS 8
#elif defined(DATA_K_TBQ4_0) || defined(DATA_K_PQ4_0) || defined(DATA_K_TBQ4_0_64) || defined(DATA_K_PQ4_0_64)
#define K_NUM_CENTROIDS 16
#endif

// ============================================================================
// V buffer declarations (binding 2)
// ============================================================================
#if defined(DATA_V_F16)
layout (binding = 2) readonly buffer VV4_V {f16vec4 v_data_f16v4[];};
#define V_BLOCK_SIZE 1
#define V_BLOCK_BYTE_SIZE 2
#elif defined(DATA_V_F32)
layout (binding = 2) readonly buffer V_PACKED {vec4 v_data_packed[];} v_packed;
#define V_BLOCK_SIZE 4
#define V_BLOCK_BYTE_SIZE 16
#elif defined(DATA_V_Q4_0)
layout (binding = 2) readonly buffer V_PACKED16 {block_q4_0_packed16 v_data_packed16[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_Q4_0
#define V_BLOCK_BYTE_SIZE 18
#elif defined(DATA_V_Q8_0)
layout (binding = 2) readonly buffer V_PACKED16 {block_q8_0_packed16 v_data_packed16[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_Q8_0
#define V_BLOCK_BYTE_SIZE 34
#elif defined(DATA_V_TBQ3_0_64)
layout (binding = 2) readonly buffer V_TQ3 {block_tbq3_0_64 v_data_tbq3[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_TBQ3_0_64
#define V_BLOCK_BYTE_SIZE 36
#elif defined(DATA_V_TBQ3_0)
layout (binding = 2) readonly buffer V_TQ3 {block_tbq3_0 v_data_tbq3[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_TBQ3_0
#define V_BLOCK_BYTE_SIZE 68
#elif defined(DATA_V_TBQ4_0_64)
layout (binding = 2) readonly buffer V_TQ4 {block_tbq4_0_64 v_data_tbq4[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_TBQ4_0_64
#define V_BLOCK_BYTE_SIZE 44
#elif defined(DATA_V_TBQ4_0)
layout (binding = 2) readonly buffer V_TQ4 {block_tbq4_0 v_data_tbq4[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_TBQ4_0
#define V_BLOCK_BYTE_SIZE 84
#elif defined(DATA_V_PQ3_0_64)
layout (binding = 2) readonly buffer V_PQ3 {block_pq3_0_64 v_data_pq3[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_PQ3_0_64
#define V_BLOCK_BYTE_SIZE 26
#elif defined(DATA_V_PQ3_0)
layout (binding = 2) readonly buffer V_PQ3 {block_pq3_0 v_data_pq3[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_PQ3_0
#define V_BLOCK_BYTE_SIZE 50
#elif defined(DATA_V_PQ4_0_64)
layout (binding = 2) readonly buffer V_PQ4 {block_pq4_0_64 v_data_pq4[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_PQ4_0_64
#define V_BLOCK_BYTE_SIZE 34
#elif defined(DATA_V_PQ4_0)
layout (binding = 2) readonly buffer V_PQ4 {block_pq4_0 v_data_pq4[];} v_packed;
#define V_BLOCK_SIZE QUANT_K_PQ4_0
#define V_BLOCK_BYTE_SIZE 66
#endif

// Same fallback as K_BLOCK_SIZE above; needed by the upstream same-type f16
// variant where no DATA_V_* macro is set.
#ifndef V_BLOCK_SIZE
#define V_BLOCK_SIZE 1
#define V_BLOCK_BYTE_SIZE 2
#endif

// Same fallback as K_PACKED16 above — for legacy types (Q4_1, Q5_0, Q5_1,
// IQ4_NL) that go through the same-type path with DATA_A_* but no DATA_V_*.
#if !defined(DATA_V_F16) && !defined(DATA_V_F32) && \
    !defined(DATA_V_Q4_0) && !defined(DATA_V_Q8_0) && \
    !defined(DATA_V_TBQ3_0_64) && !defined(DATA_V_TBQ3_0) && \
    !defined(DATA_V_TBQ4_0_64) && !defined(DATA_V_TBQ4_0) && \
    !defined(DATA_V_PQ3_0_64) && !defined(DATA_V_PQ3_0) && \
    !defined(DATA_V_PQ4_0_64) && !defined(DATA_V_PQ4_0) && \
    (defined(DATA_A_Q4_1) || defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1) || defined(DATA_A_IQ4_NL))
layout (binding = 2) readonly buffer V_PACKED16 {A_TYPE_PACKED16 v_data_packed16[];} v_packed;
#endif

// ============================================================================
// Backward compatibility: define BLOCK_SIZE/BLOCK_BYTE_SIZE when K and V match
// ============================================================================
#if defined(DATA_A_F32)
#undef BLOCK_SIZE
#define BLOCK_SIZE K_BLOCK_SIZE
#define BLOCK_BYTE_SIZE K_BLOCK_BYTE_SIZE
#elif defined(DATA_A_Q4_0) || defined(DATA_A_Q8_0) || defined(DATA_A_TBQ3_0) || defined(DATA_A_TBQ4_0) || defined(DATA_A_PQ3_0) || defined(DATA_A_PQ4_0) || \
      defined(DATA_A_TBQ3_0_64) || defined(DATA_A_TBQ4_0_64) || defined(DATA_A_PQ3_0_64) || defined(DATA_A_PQ4_0_64)
#define BLOCK_BYTE_SIZE K_BLOCK_BYTE_SIZE
#endif

#if defined(LLAMA_UPSTREAM_FA_MIXED_TYPES)
    #define FA_BLOCK_BYTES_K FaBlockBytesK
#elif defined(K_BLOCK_SIZE)  && (K_BLOCK_SIZE > 1)
    #define FA_BLOCK_BYTES_K K_BLOCK_BYTE_SIZE;
#elif defined(BLOCK_SIZE) && (BLOCK_SIZE > 1)
    #define FA_BLOCK_BYTES_K BLOCK_BYTE_SIZE;
#else
    #define FA_BLOCK_BYTES_K 2
#endif

#if defined(LLAMA_UPSTREAM_FA_MIXED_TYPES)
    #define FA_BLOCK_BYTES_V FaBlockBytesV
#elif defined(V_BLOCK_SIZE) && V_BLOCK_SIZE > 1
    #define FA_BLOCK_BYTES_V V_BLOCK_BYTE_SIZE;
#elif defined (BLOCK_SIZE) && BLOCK_SIZE > 1
    #define FA_BLOCK_BYTES_V BLOCK_BYTE_SIZE;
#else
    #define FA_BLOCK_BYTES_V 2
#endif

#undef USE_DECODE_K
#if (defined(K_BLOCK_SIZE) && (K_BLOCK_SIZE > 1)) || \
    (defined(BLOCK_SIZE) && (BLOCK_SIZE > 1))
#define USE_DECODE_K true
#else
#define USE_DECODE_K false
#endif

#undef USE_DECODE_V
#if (defined(V_BLOCK_SIZE) && (V_BLOCK_SIZE > 1)) || \
    (defined(BLOCK_SIZE) && (BLOCK_SIZE > 1))
#define USE_DECODE_V true
#else
#define USE_DECODE_V false
#endif

#undef BLOCK_SIZE_K
#if defined(K_BLOCK_SIZE) && (K_BLOCK_SIZE > 1)
    #define BLOCK_SIZE_K K_BLOCK_SIZE
#elif defined(BLOCK_SIZE) && (BLOCK_SIZE > 1)
    #define BLOCK_SIZE_K BLOCK_SIZE
#else
    // Plain f16 variant: no dequantize4_a exists, and the call sits in dead
    // `if (USE_DECODE_K)` branches that GLSL still type-checks.
    #define BLOCK_SIZE_K 1
#endif

#undef BLOCK_SIZE_V
#if defined(V_BLOCK_SIZE) && (V_BLOCK_SIZE > 1)
    #define BLOCK_SIZE_V V_BLOCK_SIZE
#elif defined(BLOCK_SIZE) && (BLOCK_SIZE > 1)
    #define BLOCK_SIZE_V BLOCK_SIZE
#else
    // Plain f16 variant: no dequantize4_a exists, and the call sits in dead
    // `if (USE_DECODE_K)` branches that GLSL still type-checks.
    #define BLOCK_SIZE_V 1
#endif

#if defined(DATA_A_F32)
FLOAT_TYPEV4 dequantize4_a(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // iqs is currently always zero in the flash attention shaders
    if (binding_idx == BINDING_IDX_K) {
        return FLOAT_TYPEV4(k_packed.k_data_packed[a_offset + ib]);
    } else {
        return FLOAT_TYPEV4(v_packed.v_data_packed[a_offset + ib]);
    }
}
#endif
// ============================================================================
// dequantize4_k — K dequantization (binding 1)
// ============================================================================
#if defined(DATA_K_F16)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    return vec4(k_data_f16v4[a_offset + ib]);
}
#elif defined(DATA_K_F32)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    return k_packed.k_data_packed[a_offset + ib];
}
#elif defined(DATA_K_Q4_0)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
    uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
    uint shift = (iqs & 0x10) >> 2;
    vui_lo >>= shift;
    vui_hi >>= shift;
    return float(k_packed.k_data_packed16[a_offset + ib].d) * (vec4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - 8.0f);
}
#elif defined(DATA_K_Q8_0)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2])).xy;
    const i8vec2 v1 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;
    return float(k_packed.k_data_packed16[a_offset + ib].d) * vec4(v0.x, v0.y, v1.x, v1.y);
}
#elif defined(DATA_K_TBQ3_0) || defined(DATA_K_TBQ3_0_64)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    float d = float(k_packed.k_data_tbq3[a_offset + ib].d);
    uint bit_pos = iqs * 3u;
    uint byte_off = bit_pos >> 3u;
    uint bits16 = uint(k_packed.k_data_tbq3[a_offset + ib].qs[byte_off])
                | (uint(k_packed.k_data_tbq3[a_offset + ib].qs[byte_off + 1u]) << 8u);
    uint shift = bit_pos & 7u;
    return d * vec4(TBQ3_CB[(bits16 >> shift) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 3u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 6u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 9u)) & 7u]);
}

float qjl_correction_k(uint k_idx, uint k_off, float proj_q_sum, vec4 proj_q_v4[QUANT_K / 4]) {
    float d_r = float(k_packed.k_data_tbq3[k_off + k_idx].d_r);
    if (d_r < 1e-15) return 0.0;
    float pos_sum = 0.0;
    [[unroll]] for (uint w = 0u; w < QUANT_K / 32u; w++) {
        uint base = w * 4u;
        uint bits = uint(k_packed.k_data_tbq3[k_off + k_idx].qjl[base])
                  | (uint(k_packed.k_data_tbq3[k_off + k_idx].qjl[base + 1u]) << 8u)
                  | (uint(k_packed.k_data_tbq3[k_off + k_idx].qjl[base + 2u]) << 16u)
                  | (uint(k_packed.k_data_tbq3[k_off + k_idx].qjl[base + 3u]) << 24u);
        uint v0 = w * 8u;
        [[unroll]] for (uint q = 0u; q < 8u; q++) {
            vec4 pq = proj_q_v4[v0 + q];
            vec4 mask = vec4(float(bits & 1u), float((bits >> 1u) & 1u),
                             float((bits >> 2u) & 1u), float((bits >> 3u) & 1u));
            pos_sum += dot(mask, pq);
            bits >>= 4u;
        }
    }
    return d_r * sqrt(1.5707963) / float(QUANT_K) * (2.0 * pos_sum - proj_q_sum);
}
#elif defined(DATA_K_TBQ4_0) || defined(DATA_K_TBQ4_0_64)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    float d = float(k_packed.k_data_tbq4[a_offset + ib].d);
    uint vui0 = uint(k_packed.k_data_tbq4[a_offset + ib].qs[iqs / 2]);
    uint vui1 = uint(k_packed.k_data_tbq4[a_offset + ib].qs[iqs / 2 + 1u]);
    return d * vec4(TBQ4_CB[vui0 & 0xFu], TBQ4_CB[vui0 >> 4u],
                    TBQ4_CB[vui1 & 0xFu], TBQ4_CB[vui1 >> 4u]);
}

float qjl_correction_k(uint k_idx, uint k_off, float proj_q_sum, vec4 proj_q_v4[QUANT_K / 4]) {
    float d_r = float(k_packed.k_data_tbq4[k_off + k_idx].d_r);
    if (d_r < 1e-15) return 0.0;
    float pos_sum = 0.0;
    [[unroll]] for (uint w = 0u; w < QUANT_K / 32u; w++) {
        uint base = w * 4u;
        uint bits = uint(k_packed.k_data_tbq4[k_off + k_idx].qjl[base])
                  | (uint(k_packed.k_data_tbq4[k_off + k_idx].qjl[base + 1u]) << 8u)
                  | (uint(k_packed.k_data_tbq4[k_off + k_idx].qjl[base + 2u]) << 16u)
                  | (uint(k_packed.k_data_tbq4[k_off + k_idx].qjl[base + 3u]) << 24u);
        uint v0 = w * 8u;
        [[unroll]] for (uint q = 0u; q < 8u; q++) {
            vec4 pq = proj_q_v4[v0 + q];
            vec4 mask = vec4(float(bits & 1u), float((bits >> 1u) & 1u),
                             float((bits >> 2u) & 1u), float((bits >> 3u) & 1u));
            pos_sum += dot(mask, pq);
            bits >>= 4u;
        }
    }
    return d_r * sqrt(1.5707963) / float(QUANT_K) * (2.0 * pos_sum - proj_q_sum);
}
#elif defined(DATA_K_PQ3_0) || defined(DATA_K_PQ3_0_64)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    float d = float(k_packed.k_data_pq3[a_offset + ib].d);
    uint bit_pos = iqs * 3u;
    uint byte_off = bit_pos >> 3u;
    uint bits16 = uint(k_packed.k_data_pq3[a_offset + ib].qs[byte_off])
                | (uint(k_packed.k_data_pq3[a_offset + ib].qs[byte_off + 1u]) << 8u);
    uint shift = bit_pos & 7u;
    return d * vec4(TBQ3_CB[(bits16 >> shift) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 3u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 6u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 9u)) & 7u]);
}
#elif defined(DATA_K_PQ4_0) || defined(DATA_K_PQ4_0_64)
vec4 dequantize4_k(uint ib, uint iqs, uint a_offset) {
    float d = float(k_packed.k_data_pq4[a_offset + ib].d);
    uint vui0 = uint(k_packed.k_data_pq4[a_offset + ib].qs[iqs / 2]);
    uint vui1 = uint(k_packed.k_data_pq4[a_offset + ib].qs[iqs / 2 + 1u]);
    return d * vec4(TBQ4_CB[vui0 & 0xFu], TBQ4_CB[vui0 >> 4u],
                    TBQ4_CB[vui1 & 0xFu], TBQ4_CB[vui1 >> 4u]);
}
#endif

#if defined(DATA_A_Q4_0)
#undef BLOCK_BYTE_SIZE
#define BLOCK_BYTE_SIZE 18
#elif defined(DATA_A_Q4_1)
#undef BLOCK_BYTE_SIZE
#define BLOCK_BYTE_SIZE 20
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
FLOAT_TYPEV4 dequantize4_a(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q4_1
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * nibbles + FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));
#endif
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q4_1
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * nibbles + FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));
#endif
    }
}
#endif

#if defined(DATA_A_Q5_0)
#undef BLOCK_BYTE_SIZE
#define BLOCK_BYTE_SIZE 22
#elif defined(DATA_A_Q5_1)
#undef BLOCK_BYTE_SIZE
#define BLOCK_BYTE_SIZE 24
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
FLOAT_TYPEV4 dequantize4_a(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

#ifdef DATA_A_Q5_1
        uint qh = k_packed.k_data_packed16[a_offset + ib].qh;
#else
        uint qh = uint(k_packed.k_data_packed16[a_offset + ib].qh[0]) | (uint(k_packed.k_data_packed16[a_offset + ib].qh[1]) << 16);
#endif
        FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs) & 1, (qh >> (iqs + 1)) & 1, (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1) * FLOAT_TYPE(16.0f);

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q5_1
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles + hb) + FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));
#endif
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

#ifdef DATA_A_Q5_1
        uint qh = v_packed.v_data_packed16[a_offset + ib].qh;
#else
        uint qh = uint(v_packed.v_data_packed16[a_offset + ib].qh[0]) | (uint(v_packed.v_data_packed16[a_offset + ib].qh[1]) << 16);
#endif
        FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs) & 1, (qh >> (iqs + 1)) & 1, (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1) * FLOAT_TYPE(16.0f);

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q5_1
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles + hb) + FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));
#endif
    }
}
#endif


#ifdef HAS_CENTROID_K
#if defined(DATA_K_TBQ3_0) || defined(DATA_K_TBQ3_0_64)
float k_get_scale(uint ib, uint a_offset) {
    return float(k_packed.k_data_tbq3[a_offset + ib].d);
}
uvec4 k_get_indices4(uint ib, uint iqs, uint a_offset) {
    uint bit_pos = iqs * 3u;
    uint byte_off = bit_pos >> 3u;
    uint bits16 = uint(k_packed.k_data_tbq3[a_offset + ib].qs[byte_off])
                | (uint(k_packed.k_data_tbq3[a_offset + ib].qs[byte_off + 1u]) << 8u);
    uint s = bit_pos & 7u;
    return uvec4((bits16 >> s) & 7u, (bits16 >> (s + 3u)) & 7u,
                 (bits16 >> (s + 6u)) & 7u, (bits16 >> (s + 9u)) & 7u);
}
#elif defined(DATA_K_PQ3_0) || defined(DATA_K_PQ3_0_64)
float k_get_scale(uint ib, uint a_offset) {
    return float(k_packed.k_data_pq3[a_offset + ib].d);
}
uvec4 k_get_indices4(uint ib, uint iqs, uint a_offset) {
    uint bit_pos = iqs * 3u;
    uint byte_off = bit_pos >> 3u;
    uint bits16 = uint(k_packed.k_data_pq3[a_offset + ib].qs[byte_off])
                | (uint(k_packed.k_data_pq3[a_offset + ib].qs[byte_off + 1u]) << 8u);
    uint s = bit_pos & 7u;
    return uvec4((bits16 >> s) & 7u, (bits16 >> (s + 3u)) & 7u,
                 (bits16 >> (s + 6u)) & 7u, (bits16 >> (s + 9u)) & 7u);
}
#elif defined(DATA_K_TBQ4_0) || defined(DATA_K_TBQ4_0_64)
float k_get_scale(uint ib, uint a_offset) {
    return float(k_packed.k_data_tbq4[a_offset + ib].d);
}
uvec4 k_get_indices4(uint ib, uint iqs, uint a_offset) {
    uint vui0 = uint(k_packed.k_data_tbq4[a_offset + ib].qs[iqs / 2]);
    uint vui1 = uint(k_packed.k_data_tbq4[a_offset + ib].qs[iqs / 2 + 1u]);
    return uvec4(vui0 & 0xFu, vui0 >> 4u, vui1 & 0xFu, vui1 >> 4u);
}
#elif defined(DATA_K_PQ4_0) || defined(DATA_K_PQ4_0_64)
float k_get_scale(uint ib, uint a_offset) {
    return float(k_packed.k_data_pq4[a_offset + ib].d);
}
uvec4 k_get_indices4(uint ib, uint iqs, uint a_offset) {
    uint vui0 = uint(k_packed.k_data_pq4[a_offset + ib].qs[iqs / 2]);
    uint vui1 = uint(k_packed.k_data_pq4[a_offset + ib].qs[iqs / 2 + 1u]);
    return uvec4(vui0 & 0xFu, vui0 >> 4u, vui1 & 0xFu, vui1 >> 4u);
}
#endif
#endif

#if defined(DATA_A_IQ4_NL)
#define BLOCK_BYTE_SIZE 18


FLOAT_TYPEV4 dequantize4_a(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(
            kvalues_iq4nl[vui_lo & 0xF],
            kvalues_iq4nl[(vui_lo >> 8) & 0xF],
            kvalues_iq4nl[vui_hi & 0xF],
            kvalues_iq4nl[(vui_hi >> 8) & 0xF]);
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(
            kvalues_iq4nl[vui_lo & 0xF],
            kvalues_iq4nl[(vui_lo >> 8) & 0xF],
            kvalues_iq4nl[vui_hi & 0xF],
            kvalues_iq4nl[(vui_hi >> 8) & 0xF]);
    }
}
#endif
#if defined(DATA_A_Q8_0)
FLOAT_TYPEV4 dequantize4_a(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        const i8vec2 v0 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    } else {
        const i8vec2 v0 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    }
}
#endif

// ============================================================================
// dequantize4_v — V dequantization (binding 2)
// ============================================================================
#if defined(DATA_V_F16)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    return vec4(v_data_f16v4[a_offset + ib]);
}
#elif defined(DATA_V_F32)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    return v_packed.v_data_packed[a_offset + ib];
}
#elif defined(DATA_V_Q4_0)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
    uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
    uint shift = (iqs & 0x10) >> 2;
    vui_lo >>= shift;
    vui_hi >>= shift;
    return float(v_packed.v_data_packed16[a_offset + ib].d) * (vec4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - 8.0f);
}
#elif defined(DATA_V_Q8_0)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2])).xy;
    const i8vec2 v1 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;
    return float(v_packed.v_data_packed16[a_offset + ib].d) * vec4(v0.x, v0.y, v1.x, v1.y);
}
#elif defined(DATA_V_TBQ3_0) || defined(DATA_V_TBQ3_0_64)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    float d = float(v_packed.v_data_tbq3[a_offset + ib].d);
    uint bit_pos = iqs * 3u;
    uint byte_off = bit_pos >> 3u;
    uint bits16 = uint(v_packed.v_data_tbq3[a_offset + ib].qs[byte_off])
                | (uint(v_packed.v_data_tbq3[a_offset + ib].qs[byte_off + 1u]) << 8u);
    uint shift = bit_pos & 7u;
    return d * vec4(TBQ3_CB[(bits16 >> shift) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 3u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 6u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 9u)) & 7u]);
}
#elif defined(DATA_V_TBQ4_0) || defined(DATA_V_TBQ4_0_64)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    float d = float(v_packed.v_data_tbq4[a_offset + ib].d);
    uint vui0 = uint(v_packed.v_data_tbq4[a_offset + ib].qs[iqs / 2]);
    uint vui1 = uint(v_packed.v_data_tbq4[a_offset + ib].qs[iqs / 2 + 1u]);
    return d * vec4(TBQ4_CB[vui0 & 0xFu], TBQ4_CB[vui0 >> 4u],
                    TBQ4_CB[vui1 & 0xFu], TBQ4_CB[vui1 >> 4u]);
}
#elif defined(DATA_V_PQ3_0) || defined(DATA_V_PQ3_0_64)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    float d = float(v_packed.v_data_pq3[a_offset + ib].d);
    uint bit_pos = iqs * 3u;
    uint byte_off = bit_pos >> 3u;
    uint bits16 = uint(v_packed.v_data_pq3[a_offset + ib].qs[byte_off])
                | (uint(v_packed.v_data_pq3[a_offset + ib].qs[byte_off + 1u]) << 8u);
    uint shift = bit_pos & 7u;
    return d * vec4(TBQ3_CB[(bits16 >> shift) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 3u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 6u)) & 7u],
                    TBQ3_CB[(bits16 >> (shift + 9u)) & 7u]);
}
#elif defined(DATA_V_PQ4_0) || defined(DATA_V_PQ4_0_64)
vec4 dequantize4_v(uint ib, uint iqs, uint a_offset) {
    float d = float(v_packed.v_data_pq4[a_offset + ib].d);
    uint vui0 = uint(v_packed.v_data_pq4[a_offset + ib].qs[iqs / 2]);
    uint vui1 = uint(v_packed.v_data_pq4[a_offset + ib].qs[iqs / 2 + 1u]);
    return d * vec4(TBQ4_CB[vui0 & 0xFu], TBQ4_CB[vui0 >> 4u],
                    TBQ4_CB[vui1 & 0xFu], TBQ4_CB[vui1 >> 4u]);
}
#endif

// dequantize4_a only exists for the legacy same-type quants handled above.
// Other quant types (e.g. Q1_0, IQ1_M) are FA cm2-only: they dequantize via
// DEQUANTFUNC, but still include this file and type-check dequantize4.
#if defined(DATA_A_F32) || defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1) || \
    defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1) || defined(DATA_A_IQ4_NL) || defined(DATA_A_Q8_0)
#define HAS_DEQUANTIZE4_A
#endif

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
#if defined(K_BLOCK_SIZE) && (K_BLOCK_SIZE > 1)
        return FLOAT_TYPEV4(dequantize4_k(ib, iqs, a_offset));
#elif defined(HAS_DEQUANTIZE4_A) && defined(BLOCK_SIZE) && (BLOCK_SIZE > 1)
        return dequantize4_a(ib, iqs, a_offset, binding_idx);
#else
        // No dequantize4_a exists (plain f16, or a cm2-only quant type), and
        // the call sits in dead `if (USE_DECODE_K)` branches that GLSL still
        // type-checks.
        return FLOAT_TYPEV4(0);
#endif
    } else {
#if defined(V_BLOCK_SIZE) && (V_BLOCK_SIZE > 1)
        return FLOAT_TYPEV4(dequantize4_v(ib, iqs, a_offset));
#elif defined(HAS_DEQUANTIZE4_A) && defined(BLOCK_SIZE) && (BLOCK_SIZE > 1)
        return dequantize4_a(ib, iqs, a_offset, binding_idx);
#else
        // No dequantize4_a exists (plain f16, or a cm2-only quant type), and
        // the call sits in dead `if (USE_DECODE_K)` branches that GLSL still
        // type-checks.
        return FLOAT_TYPEV4(0);
#endif
    }
}

