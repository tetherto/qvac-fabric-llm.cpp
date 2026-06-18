#ifndef TQ_UTILS_COMP
#define TQ_UTILS_COMP

#if defined(DATA_A_TQ2_0)
#if defined(TQ2_CM2)
int tq2_dequantize(const in decodeBufTQ2_0 bl, uint iqs) {
#else
int tq2_dequantize(uint ib, uint iqs) {
#endif
    const uint upper = iqs / 128;

    const uint byte = (upper * 32) + (iqs % 32);
    const uint shift = ((iqs % 128) / 32) * 2;

    #if defined(TQ2_CM2)
    const int c = (int(bl.block.qs[byte]) >> shift) & 3;
    #else
    const int c = (int(data_a[ib].qs[byte]) >> shift) & 3;
    #endif
    return c - 1;
}
#endif

// TurboQuant (TBQ3_0, TBQ4_0) shared constants for inverse Hadamard transform.
// Signs generated from seed 42 via xoshiro256**, packed as bitmasks.
#if defined(DATA_A_TBQ3_0) || defined(DATA_A_TBQ4_0) || defined(DATA_A_PQ3_0) || defined(DATA_A_PQ4_0) || \
    defined(DATA_K_TBQ3_0) || defined(DATA_K_TBQ4_0) || defined(DATA_K_PQ3_0) || defined(DATA_K_PQ4_0) || \
    defined(DATA_V_TBQ3_0) || defined(DATA_V_TBQ4_0) || defined(DATA_V_PQ3_0) || defined(DATA_V_PQ4_0) || \
    defined(DATA_A_TBQ3_0_64) || defined(DATA_A_TBQ4_0_64) || defined(DATA_A_PQ3_0_64) || defined(DATA_A_PQ4_0_64) || \
    defined(DATA_K_TBQ3_0_64) || defined(DATA_K_TBQ4_0_64) || defined(DATA_K_PQ3_0_64) || defined(DATA_K_PQ4_0_64) || \
    defined(DATA_V_TBQ3_0_64) || defined(DATA_V_TBQ4_0_64) || defined(DATA_V_PQ3_0_64) || defined(DATA_V_PQ4_0_64)

// Pick sign / codebook constants based on the block size.
// QK_TQ_64 blocks (d=64) use different seeds and a wider Lloyd-Max codebook
// (sigma = 1/sqrt(d) is larger at d=64 than at d=128).
#if defined(DATA_A_TBQ3_0_64) || defined(DATA_A_TBQ4_0_64) || defined(DATA_A_PQ3_0_64) || defined(DATA_A_PQ4_0_64) || \
    defined(DATA_K_TBQ3_0_64) || defined(DATA_K_TBQ4_0_64) || defined(DATA_K_PQ3_0_64) || defined(DATA_K_PQ4_0_64) || \
    defined(DATA_V_TBQ3_0_64) || defined(DATA_V_TBQ4_0_64) || defined(DATA_V_PQ3_0_64) || defined(DATA_V_PQ4_0_64)
#define TQ_D64 1
#endif

#if defined(TQ_D64)
// Stage 1 sign diagonal D (seed 43, d=64), packed as bitmasks
const uint TQ_SIGN_BITS[2] = uint[2](
    0x57661b0eu, 0xcdf0a1a5u
);

// QJL Stage 2 sign diagonal (seed 139, d=64)
const uint QJL_SIGN_BITS[2] = uint[2](
    0xbcd2ccddu, 0xe0279308u
);

// d=64 Lloyd-Max codebooks (wider spread since sigma = 1/sqrt(d) is larger).
// Must match TQ3_CODEBOOK_64 / TQ4_CODEBOOK_64 in ggml-quants.c.
const float TBQ3_CB[8] = float[8](
    -0.26391393084454512, -0.16616785892516461,
    -0.09383226321833739, -0.03046917893115905,
     0.03046917893115905,  0.09383226321833739,
     0.16616785892516461,  0.26391393084454512
);

const float TBQ4_CB[16] = float[16](
    -0.33074821159014389, -0.25285715281341298,
    -0.19879720552558833, -0.15486925951295250,
    -0.11643764752566743, -0.08127367507061777,
    -0.04806567112944460, -0.01591077077846402,
     0.01591077077846402,  0.04806567112944460,
     0.08127367507061777,  0.11643764752566743,
     0.15486925951295250,  0.19879720552558833,
     0.25285715281341298,  0.33074821159014389
);
#else
// Stage 1 sign diagonal D (seed 42, d=128), packed as bitmasks
const uint TQ_SIGN_BITS[4] = uint[4](
    0x40f54e8cu, 0x6587b7b0u, 0xc31220eau, 0x32f6449bu
);

// QJL Stage 2 sign diagonal (seed 137, d=128), independent from Stage 1
const uint QJL_SIGN_BITS[4] = uint[4](
    0x4a492032u, 0x1adafe4bu, 0xac005e9bu, 0x0808dc78u
);

// d=128 Lloyd-Max codebooks.
// Must match TQ3_CODEBOOK_128 / TQ4_CODEBOOK_128 in ggml-quants.c.
const float TBQ3_CB[8] = float[8](
    -0.18839718597003241, -0.11813976699668613,
    -0.06658560804735174, -0.02160431064212660,
     0.02160431064212660,  0.06658560804735174,
     0.11813976699668613,  0.18839718597003241
);

const float TBQ4_CB[16] = float[16](
    -0.23762692286887249, -0.18079342531272283,
    -0.14176134070424901, -0.11024676790280842,
    -0.08279230816984559, -0.05774433563409530,
    -0.03413390187425037, -0.01129645493594766,
     0.01129645493594766,  0.03413390187425037,
     0.05774433563409530,  0.08279230816984559,
     0.11024676790280842,  0.14176134070424901,
     0.18079342531272283,  0.23762692286887249
);
#endif

float tq_get_sign(uint tid) {
    return ((TQ_SIGN_BITS[tid / 32] >> (tid % 32)) & 1u) != 0u ? 1.0 : -1.0;
}

float qjl_get_sign(uint tid) {
    return ((QJL_SIGN_BITS[tid / 32] >> (tid % 32)) & 1u) != 0u ? 1.0 : -1.0;
}

// Shared codebook lookup helpers.
// Callers read raw bytes from their own data source, then call these.

// 3-bit: caller passes the 1-2 raw bytes spanning the 3-bit field and the bit offset.
float tbq3_dequant_raw(uint raw_bytes, uint bit_off) {
    return TBQ3_CB[(raw_bytes >> bit_off) & 0x7u];
}

// 4-bit: caller passes the raw byte containing the nibble pair.
float tbq4_dequant_raw(uint raw_byte, uint idx) {
    return TBQ4_CB[(idx & 1u) != 0u ? (raw_byte >> 4u) : (raw_byte & 0xFu)];
}

#endif

#endif
