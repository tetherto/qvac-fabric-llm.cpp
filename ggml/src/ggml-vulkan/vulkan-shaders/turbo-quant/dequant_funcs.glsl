#if defined(DATA_A_ANY_TBQ3_OR_PQ3_0)
#include "../tq_utils.glsl"

float tbq3_dequantize1(uint ib, uint iqs, uint a_offset) {
    const uint bit_pos = iqs * 3u;
    uint raw = uint(data_a[a_offset + ib].qs[bit_pos / 8u]);
    if ((bit_pos % 8u) + 3u > 8u)
        raw |= uint(data_a[a_offset + ib].qs[bit_pos / 8u + 1u]) << 8u;
    return tbq3_dequant_raw(raw, bit_pos % 8u);
}

vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(tbq3_dequantize1(ib, iqs, a_offset), tbq3_dequantize1(ib, iqs + 1u, a_offset));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(
        tbq3_dequantize1(ib, iqs + 0u, a_offset),
        tbq3_dequantize1(ib, iqs + 1u, a_offset),
        tbq3_dequantize1(ib, iqs + 2u, a_offset),
        tbq3_dequantize1(ib, iqs + 3u, a_offset)
    );
}
#endif

#if defined(DATA_A_ANY_TBQ4_OR_PQ4_0)
#include "../tq_utils.glsl"

// iqs is the element index (consistent with other QUANT_R=1 types). TBQ4/PQ4 packs
// 2 elements per byte, so byte = iqs/2 and nibble = iqs&1.
float tbq4_dequantize1(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs >> 1u]);
    return tbq4_dequant_raw(vui, iqs & 1u);
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(tbq4_dequant_raw(vui, 0u), tbq4_dequant_raw(vui, 1u));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(
        tbq4_dequantize1(ib, iqs + 0u, a_offset),
        tbq4_dequantize1(ib, iqs + 1u, a_offset),
        tbq4_dequantize1(ib, iqs + 2u, a_offset),
        tbq4_dequantize1(ib, iqs + 3u, a_offset)
    );
}
#endif
