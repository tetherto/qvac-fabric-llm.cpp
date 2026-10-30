/*!
 * tubro-quant/types.glsl - do not use directly, included by ../types.glsl
 */

#define DEF_PQ_BLOCK_TYPE(name, qs_size)                       \
    struct block_##name {                                      \
        uint8_t   qs[qs_size]; /* bit-packed 3-bit indices */  \
        float16_t d;           /* L2 norm */                   \
    }

#define DEF_TBQ_BLOCK_TYPE(name, qs_size, qjl_size)              \
    struct block_##name {                                        \
        uint8_t   qs[qs_size];   /* bit-packed 3-bit indices */  \
        float16_t d;             /* L2 norm */                   \
        uint8_t   qjl[qjl_size]; /* QJL Stage 2 sign bits */     \
        float16_t d_r;           /* residual L2 norm */          \
    }

// TBQ3_0 (TurboQuant 3-bit, block=128)
#define QUANT_K_TBQ3_0 128
#define QUANT_R_TBQ3_0 1

DEF_TBQ_BLOCK_TYPE(tbq3_0, (QUANT_K_TBQ3_0 * 3 + 7) / 8, QUANT_K_TBQ3_0 / 8);

#if defined(DATA_A_TBQ3_0)
#define QUANT_K QUANT_K_TBQ3_0
#define QUANT_R QUANT_R_TBQ3_0
#define A_TYPE block_tbq3_0
#endif

// TBQ4_0 (TurboQuant 4-bit + QJL Stage 2, block=128)
#define QUANT_K_TBQ4_0 128
#define QUANT_R_TBQ4_0 1

DEF_TBQ_BLOCK_TYPE(tbq4_0, QUANT_K_TBQ4_0 / 2, QUANT_K_TBQ4_0 / 8);

#if defined(DATA_A_TBQ4_0)
#define QUANT_K QUANT_K_TBQ4_0
#define QUANT_R QUANT_R_TBQ4_0
#define A_TYPE block_tbq4_0
#endif

// PQ3_0 (PolarQuant 3-bit, Stage 1 only, block=128)
#define QUANT_K_PQ3_0 128
#define QUANT_R_PQ3_0 1

DEF_PQ_BLOCK_TYPE(pq3_0, (QUANT_K_PQ3_0 * 3 + 7) / 8);

#if defined(DATA_A_PQ3_0)
#define QUANT_K QUANT_K_PQ3_0
#define QUANT_R QUANT_R_PQ3_0
#define A_TYPE block_pq3_0
#endif

// PQ4_0 (PolarQuant 4-bit, Stage 1 only, block=128)
#define QUANT_K_PQ4_0 128
#define QUANT_R_PQ4_0 1

DEF_PQ_BLOCK_TYPE(pq4_0, QUANT_K_PQ4_0 / 2);

#if defined(DATA_A_PQ4_0)
#define QUANT_K QUANT_K_PQ4_0
#define QUANT_R QUANT_R_PQ4_0
#define A_TYPE block_pq4_0
#endif

// --- block=64 variants (head_dim=64 models) ---

#define QUANT_K_TBQ3_0_64 64
#define QUANT_R_TBQ3_0_64 1

DEF_TBQ_BLOCK_TYPE(tbq3_0_64, (QUANT_K_TBQ3_0_64 * 3 + 7) / 8, QUANT_K_TBQ3_0_64 / 8);

#if defined(DATA_A_TBQ3_0_64)
#define QUANT_K QUANT_K_TBQ3_0_64
#define QUANT_R QUANT_R_TBQ3_0_64
#define A_TYPE block_tbq3_0_64
#endif

#define QUANT_K_TBQ4_0_64 64
#define QUANT_R_TBQ4_0_64 1

DEF_TBQ_BLOCK_TYPE(tbq4_0_64, QUANT_K_TBQ4_0_64 / 2, QUANT_K_TBQ4_0_64 / 8);

#if defined(DATA_A_TBQ4_0_64)
#define QUANT_K QUANT_K_TBQ4_0_64
#define QUANT_R QUANT_R_TBQ4_0_64
#define A_TYPE block_tbq4_0_64
#endif

#define QUANT_K_PQ3_0_64 64
#define QUANT_R_PQ3_0_64 1

DEF_PQ_BLOCK_TYPE(pq3_0_64, (QUANT_K_PQ3_0_64 * 3 + 7) / 8);

#if defined(DATA_A_PQ3_0_64)
#define QUANT_K QUANT_K_PQ3_0_64
#define QUANT_R QUANT_R_PQ3_0_64
#define A_TYPE block_pq3_0_64
#endif

#define QUANT_K_PQ4_0_64 64
#define QUANT_R_PQ4_0_64 1

DEF_PQ_BLOCK_TYPE(pq4_0_64, QUANT_K_PQ4_0_64 / 2);

#if defined(DATA_A_PQ4_0_64)
#define QUANT_K QUANT_K_PQ4_0_64
#define QUANT_R QUANT_R_PQ4_0_64
#define A_TYPE block_pq4_0_64
#endif

#undef DEF_TBQ_BLOCK_TYPE
#undef DEF_PQ_BLOCK_TYPE

#if defined(DATA_A_PQ3_0) || defined(DATA_A_PQ3_0_64)
#define DATA_A_ANY_PQ3_0
#endif
#if defined(DATA_A_PQ4_0) || defined(DATA_A_PQ4_0_64)
#define DATA_A_ANY_PQ4_0
#endif

#if defined(DATA_A_TBQ3_0) || defined(DATA_A_TBQ3_0_64)
#define DATA_A_ANY_TBQ3_0
#endif
#if defined(DATA_A_TBQ4_0) || defined(DATA_A_TBQ4_0_64)
#define DATA_A_ANY_TBQ4_0
#endif

#if defined(DATA_A_ANY_PQ3_0) || defined(DATA_A_ANY_TBQ3_0)
#define DATA_A_ANY_TBQ3_OR_PQ3_0
#elif defined(DATA_A_ANY_PQ4_0) || defined(DATA_A_ANY_TBQ4_0)
#define DATA_A_ANY_TBQ4_OR_PQ4_0
#endif

#if defined(DATA_A_ANY_PQ3_0) || defined(DATA_A_ANY_PQ4_0)
#define DATA_A_ANY_PQ3_OR_4_0
#endif

#if defined(DATA_A_ANY_TBQ3_0) || defined(DATA_A_ANY_TBQ4_0)
#define DATA_A_ANY_TBQ3_OR_4_0
#endif
