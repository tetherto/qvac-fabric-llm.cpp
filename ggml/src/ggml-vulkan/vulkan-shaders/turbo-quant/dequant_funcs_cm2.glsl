#if defined(DATA_A_TBQ3_0) || defined(DATA_K_TBQ3_0) || defined(DATA_V_TBQ3_0) || \
    defined(DATA_A_TBQ4_0) || defined(DATA_K_TBQ4_0) || defined(DATA_V_TBQ4_0) || \
    defined(DATA_A_PQ3_0) || defined(DATA_K_PQ3_0) || defined(DATA_V_PQ3_0) || \
    defined(DATA_A_PQ4_0) || defined(DATA_K_PQ4_0) || defined(DATA_V_PQ4_0) || \
    defined(DATA_A_TBQ3_0_64) || defined(DATA_K_TBQ3_0_64) || defined(DATA_V_TBQ3_0_64) || \
    defined(DATA_A_TBQ4_0_64) || defined(DATA_K_TBQ4_0_64) || defined(DATA_V_TBQ4_0_64) || \
    defined(DATA_A_PQ3_0_64) || defined(DATA_K_PQ3_0_64) || defined(DATA_V_PQ3_0_64) || \
    defined(DATA_A_PQ4_0_64) || defined(DATA_K_PQ4_0_64) || defined(DATA_V_PQ4_0_64)
#include "../tq_utils.glsl"

// cm2 decode wrappers: read raw bytes from buffer-reference block, delegate to shared helpers.
#define DEQUANT_CM2_3BIT(NAME, BLOCK_TYPE) \
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBuf##NAME { BLOCK_TYPE block; }; \
float16_t dequantFunc##NAME(const in decodeBuf##NAME bl, const in uint blockCoords[2], const in uint coordInBlock[2]) { \
    const uint bit_pos = coordInBlock[1] * 3u;                 \
    const uint byte_off = bit_pos >> 3u;                       \
    uint bits16 = uint(bl.block.qs[byte_off])                  \
                | (uint(bl.block.qs[byte_off + 1u]) << 8u);    \
    return bl.block.d * float16_t(TBQ3_CB[(bits16 >> (bit_pos & 7u)) & 7u]); \
}

#define DEQUANT_CM2_4BIT(NAME, BLOCK_TYPE) \
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBuf##NAME { BLOCK_TYPE block; }; \
float16_t dequantFunc##NAME(const in decodeBuf##NAME bl, const in uint blockCoords[2], const in uint coordInBlock[2]) { \
    const uint idx = coordInBlock[1];                          \
    const uint raw = uint(bl.block.qs[idx >> 1u]);             \
    return bl.block.d * float16_t(TBQ4_CB[(idx & 1u) != 0u ? (raw >> 4u) : (raw & 0xFu)]); \
}

DEQUANT_CM2_3BIT(TBQ3_0, block_tbq3_0)
DEQUANT_CM2_3BIT(PQ3_0,  block_pq3_0)
DEQUANT_CM2_4BIT(TBQ4_0, block_tbq4_0)
DEQUANT_CM2_4BIT(PQ4_0,  block_pq4_0)

DEQUANT_CM2_3BIT(TBQ3_0_64, block_tbq3_0_64)
DEQUANT_CM2_3BIT(PQ3_0_64,  block_pq3_0_64)
DEQUANT_CM2_4BIT(TBQ4_0_64, block_tbq4_0_64)
DEQUANT_CM2_4BIT(PQ4_0_64,  block_pq4_0_64)

#undef DEQUANT_CM2_3BIT
#undef DEQUANT_CM2_4BIT

#endif
