#if defined(DATA_A_TQ2_0) || \
    defined(DATA_A_TBQ3_0) || defined(DATA_A_PQ3_0) || \
    defined(DATA_A_TBQ4_0) || defined(DATA_A_PQ4_0) || \
    defined(DATA_A_TBQ3_0_64) || defined(DATA_A_PQ3_0_64) || \
    defined(DATA_A_TBQ4_0_64) || defined(DATA_A_PQ4_0_64)
#include "tq_utils.glsl"
#endif

// k_pair is the K coordinate measured in FLOAT_TYPEV2 elements.
uint a_shmem_index(uint m, uint k_pair) {
    if (APPLY_SLM_A_RESHAPE) {
        const uint tile_width = TK / 2;
        return (k_pair / tile_width) * BM * tile_width
             + m * tile_width
             + k_pair % tile_width;
    }
    return m * SHMEM_STRIDE + k_pair;
}

uint a_shmem_stride() {
    return APPLY_SLM_A_RESHAPE ? TK / 2 : SHMEM_STRIDE;
}

void store_a(uint m, uint k_pair, FLOAT_TYPEV2 value) {
    buf_a[a_shmem_index(m, k_pair)] = value;
}

// bounds_check is a literal at every call site: the caller only passes true for
// the ragged last tile in M, so interior tiles fold the checks away entirely.
void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uint idx_m, const bool bounds_check, const uint block, const uint end_k) {
#if defined(DATA_A_F32) || defined(DATA_A_F16)
#if LOAD_VEC_A == 8
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint k_pair = row * LOAD_VEC_A / 2;
                // ALIGNED only guarantees that K is a whole number of tiles; it says
                // nothing about M. Nothing pads A out to a whole BM tile, so the last
                // tile must still be bounded or this vector load runs past the tensor.
                if (!bounds_check || idx_m < p.M) {
                    FLOAT_TYPEV8 aa = FLOAT_TYPEV8(data_a[idx]);
                    store_a(col, k_pair,     aa[0].xy);
                    store_a(col, k_pair + 1, aa[0].zw);
                    store_a(col, k_pair + 2, aa[1].xy);
                    store_a(col, k_pair + 3, aa[1].zw);
                } else {
                    store_a(col, k_pair,     FLOAT_TYPEV2(0.0f));
                    store_a(col, k_pair + 1, FLOAT_TYPEV2(0.0f));
                    store_a(col, k_pair + 2, FLOAT_TYPEV2(0.0f));
                    store_a(col, k_pair + 3, FLOAT_TYPEV2(0.0f));
                }
                return;
            }
#elif LOAD_VEC_A == 4
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint k_pair = row * LOAD_VEC_A / 2;
                // ALIGNED only guarantees that K is a whole number of tiles; it says
                // nothing about M. Nothing pads A out to a whole BM tile, so the last
                // tile must still be bounded or this vector load runs past the tensor.
                if (!bounds_check || idx_m < p.M) {
                    FLOAT_TYPEV4 aa = FLOAT_TYPEV4(data_a[idx]);
                    store_a(col, k_pair,     aa.xy);
                    store_a(col, k_pair + 1, aa.zw);
                } else {
                    store_a(col, k_pair,     FLOAT_TYPEV2(0.0f));
                    store_a(col, k_pair + 1, FLOAT_TYPEV2(0.0f));
                }
                return;
            }
#endif
            const uint idx = pos_a + col * p.stride_a + row * 2;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(data_a_scalar[idx],
                                               data_a_scalar[idx + 1]));
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(data_a_scalar[idx], 0.0f));
            } else {
                store_a(col, row, FLOAT_TYPEV2(0.0f));
            }
#elif defined(DATA_A_BF16)
#if LOAD_VEC_A == 4
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint k_pair = row * LOAD_VEC_A / 2;
                // ALIGNED only guarantees that K is a whole number of tiles; it says
                // nothing about M. Nothing pads A out to a whole BM tile, so the last
                // tile must still be bounded or this vector load runs past the tensor.
                if (!bounds_check || idx_m < p.M) {
                    FLOAT_TYPEV4 aa = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_a[idx]));
                    store_a(col, k_pair,     aa.xy);
                    store_a(col, k_pair + 1, aa.zw);
                } else {
                    store_a(col, k_pair,     FLOAT_TYPEV2(0.0f));
                    store_a(col, k_pair + 1, FLOAT_TYPEV2(0.0f));
                }
                return;
            }
#endif
            const uint idx = pos_a + col * p.stride_a + row * 2;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(TO_FLOAT_TYPE(data_a_scalar[idx]),
                                               TO_FLOAT_TYPE(data_a_scalar[idx + 1])));
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(TO_FLOAT_TYPE(data_a_scalar[idx]), 0.0f));
            } else {
                store_a(col, row, FLOAT_TYPEV2(0.0f));
            }
#elif defined(DATA_A_Q4_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const float d = float(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[2*iqs]) | (uint(data_a_packed16[ib].qs[2*iqs + 1]) << 16);

#if defined(ADRENO)
            const vec4 v0 = (vec4(
                float((vui >> 0)  & 0xF),
                float((vui >> 8)  & 0xF),
                float((vui >> 16) & 0xF),
                float((vui >> 24) & 0xF)
            ) - 8.0) * d;

            const vec4 v1 = (vec4(
                float((vui >> 4)  & 0xF),
                float((vui >> 12) & 0xF),
                float((vui >> 20) & 0xF),
                float((vui >> 28) & 0xF)
            ) - 8.0) * d;
#else
            const vec4 v0 = (vec4(unpack8(vui & 0x0F0F0F0F)) - 8.0f) * d;
            const vec4 v1 = (vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) - 8.0f) * d;
#endif

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     FLOAT_TYPEV2(v0.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v0.zw));
            store_a(col, k_pair + 8, FLOAT_TYPEV2(v1.xy));
            store_a(col, k_pair + 9, FLOAT_TYPEV2(v1.zw));
#elif defined(DATA_A_Q4_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const vec2 dm = vec2(data_a_packed32[ib].dm);
            const uint vui = data_a_packed32[ib].qs[iqs];
#if defined(ADRENO)
            const vec4 v0 = vec4(
                float((vui >> 0)  & 0xF),
                float((vui >> 8)  & 0xF),
                float((vui >> 16) & 0xF),
                float((vui >> 24) & 0xF)
            ) * dm.x + dm.y;
            const vec4 v1 = vec4(
                float((vui >> 4)  & 0xF),
                float((vui >> 12) & 0xF),
                float((vui >> 20) & 0xF),
                float((vui >> 28) & 0xF)
            ) * dm.x + dm.y;
#else
            const vec4 v0 = vec4(unpack8(vui & 0x0F0F0F0F)) * dm.x + dm.y;
            const vec4 v1 = vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) * dm.x + dm.y;
#endif

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     FLOAT_TYPEV2(v0.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v0.zw));
            store_a(col, k_pair + 8, FLOAT_TYPEV2(v1.xy));
            store_a(col, k_pair + 9, FLOAT_TYPEV2(v1.zw));
#elif defined(DATA_A_Q5_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const uint uint_qh = uint(data_a_packed16[ib].qh[1]) << 16 | uint(data_a_packed16[ib].qh[0]);
            const ivec2 qh0 = ivec2(((uint_qh >> 2*iqs) << 4) & 0x10, (uint_qh >> (2*iqs + 12)) & 0x10);
            const ivec2 qh1 = ivec2(((uint_qh >> (2*iqs + 1)) << 4) & 0x10, (uint_qh >> (2*iqs + 13)) & 0x10);

            const uint vui = uint(data_a_packed16[ib].qs[iqs]);
            const vec4 v = (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f) * d;
            store_a(col, row,     FLOAT_TYPEV2(v.xz));
            store_a(col, row + 8, FLOAT_TYPEV2(v.yw));
#elif defined(DATA_A_Q5_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const vec2 dm = vec2(data_a_packed32[ib].dm);
            const uint uint_qh = data_a_packed32[ib].qh;
            const uvec2 qh0 = uvec2(((uint_qh >> 4*iqs) << 4) & 0x10, (uint_qh >> (4*iqs + 12)) & 0x10);
            const uvec2 qh1 = uvec2(((uint_qh >> (4*iqs + 1)) << 4) & 0x10, (uint_qh >> (4*iqs + 13)) & 0x10);
            const uvec2 qh2 = uvec2(((uint_qh >> (4*iqs + 2)) << 4) & 0x10, (uint_qh >> (4*iqs + 14)) & 0x10);
            const uvec2 qh3 = uvec2(((uint_qh >> (4*iqs + 3)) << 4) & 0x10, (uint_qh >> (4*iqs + 15)) & 0x10);

            const uint vui = data_a_packed32[ib].qs[iqs];
            const vec4 v0 = vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, ((vui >> 12) & 0xF) | qh1.y) * dm.x + dm.y;
            const vec4 v1 = vec4(((vui >> 16) & 0xF) | qh2.x, ((vui >> 20) & 0xF) | qh2.y, ((vui >> 24) & 0xF) | qh3.x, ((vui >> 28) & 0xF) | qh3.y) * dm.x + dm.y;

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     FLOAT_TYPEV2(v0.xz));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v1.xz));
            store_a(col, k_pair + 8, FLOAT_TYPEV2(v0.yw));
            store_a(col, k_pair + 9, FLOAT_TYPEV2(v1.yw));
#elif defined(DATA_A_Q8_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

#if defined(ADRENO)
            const float d = float(data_a[ib].d);
            const vec4 v = vec4(
                int(data_a[ib].qs[4*iqs]),
                int(data_a[ib].qs[4*iqs + 1]),
                int(data_a[ib].qs[4*iqs + 2]),
                int(data_a[ib].qs[4*iqs + 3])
            ) * d;
#else
            const float d = float(data_a_packed16[ib].d);
            const i8vec2 v0 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs])).xy; // vec4 used due to #12147
            const i8vec2 v1 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs + 1])).xy;
            const vec4 v = vec4(v0.x, v0.y, v1.x, v1.y) * d;
#endif

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_Q1_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xfu;

            const float d = float(data_a[ib].d);
            const uint bits = uint(data_a[ib].qs[iqs]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2((bits & 0x01u) != 0u ? d : -d, (bits & 0x02u) != 0u ? d : -d));
            store_a(col, k_pair + 1, FLOAT_TYPEV2((bits & 0x04u) != 0u ? d : -d, (bits & 0x08u) != 0u ? d : -d));
            store_a(col, k_pair + 2, FLOAT_TYPEV2((bits & 0x10u) != 0u ? d : -d, (bits & 0x20u) != 0u ? d : -d));
            store_a(col, k_pair + 3, FLOAT_TYPEV2((bits & 0x40u) != 0u ? d : -d, (bits & 0x80u) != 0u ? d : -d));
#elif defined(DATA_A_Q2_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xfu;

            const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib].d);
            const uint bits = uint(data_a[ib].qs[iqs]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     d * (FLOAT_TYPEV2(bits & 3u, (bits >> 2u) & 3u) - FLOAT_TYPEV2(1.0f)));
            store_a(col, k_pair + 1, d * (FLOAT_TYPEV2((bits >> 4u) & 3u, bits >> 6u) - FLOAT_TYPEV2(1.0f)));
#elif defined(DATA_A_TQ1_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 128;
            const uint iqs = idx % 128;

            const float d = float(data_a[ib].d);

            const uint pow3[6] = uint[6](1u, 3u, 9u, 27u, 81u, 243u);

            const uint e0 = 2u * iqs;
            const uint e1 = e0 + 1u;

            float v0;
            if (e0 < 160u) {
                const uint n0 = e0 / 32u;
                const uint m0 = e0 % 32u;
                const uint q0 = uint(data_a[ib].qs[m0]);
                const uint xi0 = (((q0 * pow3[n0]) & 255u) * 3u) >> 8;
                v0 = float(int(xi0) - 1);
            } else if (e0 < 240u) {
                const uint ee0 = e0 - 160u;
                const uint n0 = ee0 / 16u;
                const uint m0 = ee0 % 16u;
                const uint q0 = uint(data_a[ib].qs[32u + m0]);
                const uint xi0 = (((q0 * pow3[n0]) & 255u) * 3u) >> 8;
                v0 = float(int(xi0) - 1);
            } else {
                const uint ee0 = e0 - 240u;
                const uint n0 = ee0 / (QUANT_K / 64u);
                const uint j0 = ee0 % (QUANT_K / 64u);
                const uint q0 = uint(data_a[ib].qh[j0]);
                const uint xi0 = (((q0 * pow3[n0]) & 255u) * 3u) >> 8;
                v0 = float(int(xi0) - 1);
            }

            float v1;
            if (e1 < 160u) {
                const uint n1 = e1 / 32u;
                const uint m1 = e1 % 32u;
                const uint q1 = uint(data_a[ib].qs[m1]);
                const uint xi1 = (((q1 * pow3[n1]) & 255u) * 3u) >> 8;
                v1 = float(int(xi1) - 1);
            } else if (e1 < 240u) {
                const uint ee1 = e1 - 160u;
                const uint n1 = ee1 / 16u;
                const uint m1 = ee1 % 16u;
                const uint q1 = uint(data_a[ib].qs[32u + m1]);
                const uint xi1 = (((q1 * pow3[n1]) & 255u) * 3u) >> 8;
                v1 = float(int(xi1) - 1);
            } else {
                const uint ee1 = e1 - 240u;
                const uint n1 = ee1 / (QUANT_K / 64u);
                const uint j1 = ee1 % (QUANT_K / 64u);
                const uint q1 = uint(data_a[ib].qh[j1]);
                const uint xi1 = (((q1 * pow3[n1]) & 255u) * 3u) >> 8;
                v1 = float(int(xi1) - 1);
            }

            const vec2 v = d * vec2(v0, v1);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair, FLOAT_TYPEV2(v.xy));
#elif defined(DATA_A_TBQ3_0) || defined(DATA_A_PQ3_0) || defined(DATA_A_TBQ3_0_64) || defined(DATA_A_PQ3_0_64) || \
      defined(DATA_A_TBQ4_0) || defined(DATA_A_PQ4_0) || defined(DATA_A_TBQ4_0_64) || defined(DATA_A_PQ4_0_64)
            // LOAD_VEC_A is 2 for TBQ/PQ 3/4-bit variants (see vulkan-shaders-gen.cpp).
            // One idx step covers a pair of consecutive elements e0 = 2*iqs, e1 = e0 + 1.
            const uint idx     = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib  = idx / (QUANT_K / 2u);
            const uint iqs = idx % (QUANT_K / 2u);

            const float d = float(data_a[ib].d);

#if defined(DATA_A_TBQ3_0) || defined(DATA_A_PQ3_0) || defined(DATA_A_TBQ3_0_64) || defined(DATA_A_PQ3_0_64)
            const uint bit_pos0 = (2u * iqs) * 3u;
            const uint bit_pos1 = bit_pos0 + 3u;

            uint raw0 = uint(data_a[ib].qs[bit_pos0 / 8u]);
            if ((bit_pos0 % 8u) + 3u > 8u) {
                raw0 |= uint(data_a[ib].qs[bit_pos0 / 8u + 1u]) << 8u;
            }
            uint raw1 = uint(data_a[ib].qs[bit_pos1 / 8u]);
            if ((bit_pos1 % 8u) + 3u > 8u) {
                raw1 |= uint(data_a[ib].qs[bit_pos1 / 8u + 1u]) << 8u;
            }

            const float v0 = tbq3_dequant_raw(raw0, bit_pos0 % 8u) * d;
            const float v1 = tbq3_dequant_raw(raw1, bit_pos1 % 8u) * d;

#elif defined(DATA_A_TBQ4_0) || defined(DATA_A_PQ4_0) || defined(DATA_A_TBQ4_0_64) || defined(DATA_A_PQ4_0_64)
            const uint vui = uint(data_a[ib].qs[iqs]);

            const float v0 = tbq4_dequant_raw(vui, 0u) * d;
            const float v1 = tbq4_dequant_raw(vui, 1u) * d;
#endif
            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair, FLOAT_TYPEV2(v0, v1));
#elif defined(DATA_A_Q2_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;                          // 4 values per idx
            const uint iqs = (idx % 64) * 2;                   // 0,2,4..126

            const uint qsi = (iqs / 64) * 16 + (iqs % 16);     // 0..15
            const uint scalesi = iqs / 8;                      // 0..15
            const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

            const vec4 qs = vec4(unpack8((data_a_packed32[ib].qs[qsi / 2] >> qsshift) & 0x03030303));
            const uint scales = data_a[ib].scales[scalesi];
            const vec2 dm = vec2(data_a[ib].dm);

            const vec4 v = dm.x * float(scales & 0xF) * qs - dm.y * float(scales >> 4);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_TQ2_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 128;                         // 2 values per idx
            const uint iqs = (idx % 128) * 2;                  // elem 0,2,4..254

            const uint qsi   = (iqs / 128) * 32 + (iqs % 32);  // byte pair start
            const uint shift = 2 * ((iqs % 128) / 32);         // 0,2,4,6

            const uvec2 qs = uvec2(data_a[ib].qs[qsi], data_a[ib].qs[qsi + 1]);
            const float d = float(data_a[ib].d);

            const vec2 v = d * (vec2((qs >> shift) & 3) - 1.0);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair, FLOAT_TYPEV2(v.xy));
#elif defined(DATA_A_Q3_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 128;                   // 2 values per idx
            const uint iqs = idx % 128;                  // 0..127

            const uint n = iqs / 64;                     // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
            const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
            const uint j = (iqs % 64) / 4;               // 0..3
            const uint is = iqs / 8;                     // 0..15
            const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
            const uint qsshift = halfsplit * 2;          // 0,2,4,6

            const int8_t us = int8_t(((data_a[ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                                  | (((data_a[ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
            const float dl = float(data_a[ib].d) * float(us - 32);

            const vec2 qs = vec2(unpack8((uint(data_a_packed16[ib].qs[qsi / 2]) >> qsshift) & 0x0303).xy);
            const vec2 hm = vec2(unpack8(((uint(data_a_packed16[ib].hmask[hmi / 2]) >> (4 * n + halfsplit)) & 0x0101 ^ 0x0101) << 2).xy);

            store_a(col, row * LOAD_VEC_A / 2, FLOAT_TYPEV2(dl * (qs.x - hm.x),
                                                              dl * (qs.y - hm.y)));
#elif defined(DATA_A_Q4_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;                  // 4 values per idx
            const uint iqs = (idx % 64) * 2;           // 0,2,4..126

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

            const vec2 loadd = vec2(data_a[ib].dm);

            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
                                       data_a_packed32[ib].scales[1],
                                       data_a_packed32[ib].scales[2]);
            const uint scalesoffs = (is & 3) * 8;

            const uint scidx0 = (is < 4) ? 0 : 2;
            const uint scidxshift0 = scalesoffs;
            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
            const uint mbidx0 = (is < 4) ? 1 : 2;
            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            const vec4 q = vec4(unpack8((data_a_packed32[ib].qs[qsi / 4] >> (b * 4)) & 0x0F0F0F0F));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m)));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(fma(d, q.z, m), fma(d, q.w, m)));
#elif defined(DATA_A_Q5_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;                  // 4 values per idx
            const uint iqs = (idx % 64) * 2;           // 0,2,4..126

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
            const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

            const vec2 loadd = vec2(data_a[ib].dm);

            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
                                       data_a_packed32[ib].scales[1],
                                       data_a_packed32[ib].scales[2]);
            const uint scalesoffs = (is & 3) * 8;

            const uint scidx0 = (is < 4) ? 0 : 2;
            const uint scidxshift0 = scalesoffs;
            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
            const uint mbidx0 = (is < 4) ? 1 : 2;
            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            const uint qs = (data_a_packed32[ib].qs[qsi / 4] >> (b * 4)) & 0x0F0F0F0F;
            const uint qh = ((data_a_packed32[ib].qh[qhi / 4] >> (iqs / 16)) & 0x01010101) << 4;
            const vec4 q = vec4(unpack8(qs | qh));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m)));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(fma(d, q.z, m), fma(d, q.w, m)));
#elif defined(DATA_A_Q6_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 128;                  // 2 values per idx
            const uint iqs = idx % 128;                 // 0..127

            const uint n = iqs / 64;                    // 0,1
            const uint b = ((iqs % 64) / 32) * 4;       // 0,4
            const uint is_b = (iqs % 16) / 8;           // 0,1
            const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
            const uint is = 8 * n + qhshift + is_b;     // 0..15
            const uint qsi = n * 32 + (iqs % 32);       // 0..63
            const uint qhi = n * 16 + (iqs % 16);       // 0..31

            const float dscale = float(data_a[ib].d) * float(data_a[ib].scales[is]);

            const uint ql = (uint(data_a_packed16[ib].ql[qsi]) >> b) & 0x0F0F;
            const uint qh = (uint(data_a_packed16[ib].qh[qhi]) >> qhshift) & 0x0303;
            const vec2 q = (vec2(unpack8(ql | (qh << 4)).xy) - 32) * dscale;

            store_a(col, row * LOAD_VEC_A / 2, FLOAT_TYPEV2(q.x, q.y));
#elif defined(DATA_A_IQ1_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;                  // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 32;

            const float d = float(data_a[ib].d);
            const uint qh = data_a[ib].qh[ib32];
            const uint qs = data_a[ib].qs[ib8];
            const float dl = d * (2 * bitfieldExtract(qh, 12, 3) + 1);
            const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            [[unroll]] for (int k = 0; k < 4; ++k) {
                store_a(col, k_pair + k, FLOAT_TYPEV2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                       dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta)));
            }
#elif defined(DATA_A_IQ1_M)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32;
            const uint ib16 = ib8 / 2;

            const uint16_t[4] scales = data_a[ib].scales;
            const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
            const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
            const uint sc = scales[ib8 / 8];
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib16] >> (4 * (ib8 & 1));
            const float dl = d * (2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1);
            const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            [[unroll]] for (int k = 0; k < 4; ++k) {
                store_a(col, k_pair + k, FLOAT_TYPEV2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                       dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta)));
            }
#elif defined(DATA_A_IQ2_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;                 // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 4;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[8 * ib32 + ib8];
            const uint signs = pack32(u8vec4(
                data_a[ib].qs[8*ib32 + 4],
                data_a[ib].qs[8*ib32 + 5],
                data_a[ib].qs[8*ib32 + 6],
                data_a[ib].qs[8*ib32 + 7]
            ));
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + (signs >> 28)));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xxs_grid[qs];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                        (sign &   2) != 0 ? -grid0.y : grid0.y));
            store_a(col, k_pair + 1, db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                        (sign &   8) != 0 ? -grid0.w : grid0.w));
            store_a(col, k_pair + 2, db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                        (sign &  32) != 0 ? -grid1.y : grid1.y));
            store_a(col, k_pair + 3, db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                        (sign & 128) != 0 ? -grid1.w : grid1.w));
#elif defined(DATA_A_IQ2_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;            // 8 values per idx
            const uint ib32 = (idx % 32) / 4;    // 0..7
            const uint ib8 = idx % 4;            // 0..3

            const float d = float(data_a[ib].d);
            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uint qs = data_a[ib].qs[4 * ib32 + ib8];
            const uint sign7 = qs >> 9;
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xs_grid[qs & 511];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                        (sign &   2) != 0 ? -grid0.y : grid0.y));
            store_a(col, k_pair + 1, db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                        (sign &   8) != 0 ? -grid0.w : grid0.w));
            store_a(col, k_pair + 2, db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                        (sign &  32) != 0 ? -grid1.y : grid1.y));
            store_a(col, k_pair + 3, db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                        (sign & 128) != 0 ? -grid1.w : grid1.w));
#elif defined(DATA_A_IQ2_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32; // 0..31
            const uint ib32 = ib8 / 4; // 0..7

            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib32];
            const uint qhshift = 2 * (ib8 % 4);
            const uint sign = data_a[ib].qs[QUANT_K / 8 + ib8];

            const float d = float(data_a[ib].d);
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uvec2 grid = iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                        (sign &   2) != 0 ? -grid0.y : grid0.y));
            store_a(col, k_pair + 1, db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                        (sign &   8) != 0 ? -grid0.w : grid0.w));
            store_a(col, k_pair + 2, db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                        (sign &  32) != 0 ? -grid1.y : grid1.y));
            store_a(col, k_pair + 3, db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                        (sign & 128) != 0 ? -grid1.w : grid1.w));
#elif defined(DATA_A_IQ3_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint is = QUANT_K / 4 + 4 * (iqs / 8); // 8 values

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint signs = pack32(u16vec2(
                data_a_packed16[ib].qs[is/2],
                data_a_packed16[ib].qs[is/2+1]
            ));
            const float db = d * 0.5 * (0.5 + (signs >> 28));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * (int(iqs / 2) % 4), 7);
            const uint sign = (sign7 | (bitCount(sign7) << 7)) >> (4 * (idx % 2));
            const uint grid = iq3xxs_grid[qs];
            const vec4 v = db * vec4(unpack8(grid));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2((sign &   1) != 0 ? -v.x : v.x,
                                                   (sign &   2) != 0 ? -v.y : v.y));
            store_a(col, k_pair + 1, FLOAT_TYPEV2((sign &   4) != 0 ? -v.z : v.z,
                                                   (sign &   8) != 0 ? -v.w : v.w));
#elif defined(DATA_A_IQ3_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint iqh = iqs / 8;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint qh = data_a[ib].qh[iqh];
            const int8_t sign = int8_t(data_a[ib].signs[iqs / 2] >> (4 * (idx % 2)));
            const uint scale = data_a[ib].scales[iqs / 16];
            const i8vec2 sign01 = i8vec2(1 - (2 & i8vec2(sign << 1, sign)));
            const float db = d * (1 + 2 * ((scale >> (4 * (iqh & 1))) & 0xf));
            const uint32_t grid = iq3s_grid[qs | ((qh << (8 - (iqs % 8))) & 256)];
            const vec4 v = db * vec4(unpack8(grid));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2((sign &   1) != 0 ? -v.x : v.x,
                                                   (sign &   2) != 0 ? -v.y : v.y));
            store_a(col, k_pair + 1, FLOAT_TYPEV2((sign &   4) != 0 ? -v.z : v.z,
                                                   (sign &   8) != 0 ? -v.w : v.w));
#elif defined(DATA_A_IQ4_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;            // 4 values per idx
            const uint ib32 = (idx % 64) / 8;    // 0..7
            const uint iq = 4 * ib32 + (idx % 4);

            const uint sl = (data_a[ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
            const uint sh = ((data_a[ib].scales_h) >> (2 * ib32)) & 3;
            const uint qshift = idx & 4;
            u8vec4 qs = unpack8((uint(data_a_packed32[ib].qs[iq]) >> qshift) & 0x0F0F0F0F);

            const float d = float(data_a[ib].d);
            const vec4 v = d * float(int(sl | (sh << 4)) - 32) * vec4(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y], kvalues_iq4nl[qs.z], kvalues_iq4nl[qs.w]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_IQ4_NL)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const FLOAT_TYPE d = FLOAT_TYPE(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[iqs]);

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     d * FLOAT_TYPEV2(kvalues_iq4nl[vui & 0xF],
                                                       kvalues_iq4nl[bitfieldExtract(vui, 8, 4)]));
            store_a(col, k_pair + 8, d * FLOAT_TYPEV2(kvalues_iq4nl[bitfieldExtract(vui, 4, 4)],
                                                       kvalues_iq4nl[vui >> 12]));
#elif defined(DATA_A_MXFP4)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 2;

            const uint vui = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

#ifdef USE_OCP_FP4
            const float d = e8m0_to_fp32(data_a[ib].e);
            const u8vec2 packed = u8vec2(vui, vui2);
            store_a(col, row,     FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 0u)) * FLOAT_TYPE(d));
            store_a(col, row + 8, FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 4u)) * FLOAT_TYPE(d));
#else
            const float d = e8m0_to_fp32(data_a[ib].e) * 0.5;
            store_a(col, row,     FLOAT_TYPEV2(kvalues_mxfp4[vui  & 0xF] * d,
                                              kvalues_mxfp4[vui2 & 0xF] * d));
            store_a(col, row + 8, FLOAT_TYPEV2(kvalues_mxfp4[vui  >>  4] * d,
                                              kvalues_mxfp4[vui2 >>  4] * d));
#endif
#elif defined(DATA_A_NVFP4)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint ib = idx / 16u;
            const uint sub = (idx & 0xC) >> 2;
            const uint iqs = (idx & 0xF) * 2;
            const uint vui = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

            // lo and hi nibbles are 8 elements apart, which doesn't quite line up with
            // how the thread mapping and buf_idx calculation works for other types.
            const uint eff_row = (row & 3) + (row & ~3) * 2;
#ifdef USE_OCP_FP4
            const FLOAT_TYPE d = FLOAT_TYPE(ue4m3_from_bits(data_a[ib].d[sub]));
            const u8vec2 packed = u8vec2(vui, vui2);
            store_a(col, eff_row,     FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 0u)) * d);
            store_a(col, eff_row + 4, FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 4u)) * d);
#else
            const float d = ue4m3_to_fp32(data_a[ib].d[sub]) * 0.5;
            store_a(col, eff_row,     FLOAT_TYPEV2(kvalues_mxfp4[vui  & 0xF] * d,
                                                    kvalues_mxfp4[vui2 & 0xF] * d));
            store_a(col, eff_row + 4, FLOAT_TYPEV2(kvalues_mxfp4[vui  >>  4] * d,
                                                    kvalues_mxfp4[vui2 >>  4] * d));
#endif
#endif
}

#if !defined(MUL_MAT_ID)
// bounds_check is a literal at every call site: the caller only passes true for
// the ragged last tile in N, so interior tiles fold the checks away entirely.
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint idx_n, const bool bounds_check, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            if (ALIGNED != 0) {
                // Not supported for b_type bf16 because bf16mat2x4 does not exist
                const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                // ALIGNED only guarantees that K is a whole number of tiles; it says
                // nothing about N. B is padded out to a whole BN tile only when it is
                // staged through prealloc_y, so when src1 is read in place the last
                // tile must still be bounded or this vector load runs past the tensor.
                if (!bounds_check || idx_n < p.N) {
                    FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
                    buf_b[buf_idx + 0] = bb[0].xy;
                    buf_b[buf_idx + 1] = bb[0].zw;
                    buf_b[buf_idx + 2] = bb[1].xy;
                    buf_b[buf_idx + 3] = bb[1].zw;
                } else {
                    buf_b[buf_idx + 0] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 1] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 2] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 3] = FLOAT_TYPEV2(0.0f);
                }
                return;
            }
#elif LOAD_VEC_B == 4
            if (ALIGNED != 0) {
                const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                // ALIGNED only guarantees that K is a whole number of tiles; it says
                // nothing about N. B is padded out to a whole BN tile only when it is
                // staged through prealloc_y, so when src1 is read in place the last
                // tile must still be bounded or this vector load runs past the tensor.
                if (!bounds_check || idx_n < p.N) {
#if defined(DATA_B_BF16)
                    FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
                    FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
                    buf_b[buf_idx + 0] = bb.xy;
                    buf_b[buf_idx + 1] = bb.zw;
                } else {
                    buf_b[buf_idx + 0] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 1] = FLOAT_TYPEV2(0.0f);
                }
                return;
            }
#endif
            const uint idx = pos_b + col * p.stride_b + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_n < p.N && block + row * 2 + 1 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                              TO_FLOAT_TYPE(data_b_scalar[idx + 1]));
            } else if (idx_n < p.N && block + row * 2 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f);
            } else {
                buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
            }
}
#else
// bounds_check is a literal at every call site: the caller only passes true for
// the tile straddling _ne1, so interior tiles fold the checks away entirely.
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint ic, const uint _ne1, const bool bounds_check, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            if (ALIGNED != 0) {
                // Not supported for b_type bf16 because bf16mat2x4 does not exist
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                // ALIGNED only guarantees that K is a whole number of tiles; it says
                // nothing about the expert row count. Past _ne1 the row_ids entry is
                // unset, so this must be bounded or it becomes a wild offset into B.
                if (!bounds_check || ic * BN + col < _ne1) {
                    const u16vec2 row_idx = row_ids[col];
                    const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
                    FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
                    buf_b[buf_idx + 0] = bb[0].xy;
                    buf_b[buf_idx + 1] = bb[0].zw;
                    buf_b[buf_idx + 2] = bb[1].xy;
                    buf_b[buf_idx + 3] = bb[1].zw;
                } else {
                    buf_b[buf_idx + 0] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 1] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 2] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 3] = FLOAT_TYPEV2(0.0f);
                }
                return;
            }
#elif LOAD_VEC_B == 4
            if (ALIGNED != 0) {
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                // ALIGNED only guarantees that K is a whole number of tiles; it says
                // nothing about the expert row count. Past _ne1 the row_ids entry is
                // unset, so this must be bounded or it becomes a wild offset into B.
                if (!bounds_check || ic * BN + col < _ne1) {
                    const u16vec2 row_idx = row_ids[col];
                    const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
#if defined(DATA_B_BF16)
                    FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
                    FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
                    buf_b[buf_idx + 0] = bb.xy;
                    buf_b[buf_idx + 1] = bb.zw;
                } else {
                    buf_b[buf_idx + 0] = FLOAT_TYPEV2(0.0f);
                    buf_b[buf_idx + 1] = FLOAT_TYPEV2(0.0f);
                }
                return;
            }
#endif
            const uint row_i = ic * BN + col;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (row_i < _ne1 && block + row * 2 + 1 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                              TO_FLOAT_TYPE(data_b_scalar[idx + 1]));
            } else if (row_i < _ne1 && block + row * 2 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f);
            } else {
                buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
            }
}
#endif
