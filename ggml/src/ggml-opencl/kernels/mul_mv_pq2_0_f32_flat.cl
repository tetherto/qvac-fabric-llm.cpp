#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK_PQ2_0         128
#define PQ2_0_WORDS      8           // quant words per block, 16 codes each
#define PQ2_0_FIELD_MASK 0x03030303u // the 2-bit field at the same position of every byte
#define QK_ACT           32          // src1 elements sharing one int8 scale
#define ACT_GROUP        16          // src1 elements matching one quant word

// Element e of a quant word sits at bits 2e, so (w >> 2k) & PQ2_0_FIELD_MASK holds elements k, 4+k, 8+k, 12+k in its
// bytes. The int8 src1 words of a group of ACT_GROUP elements are stored in that order, so one dp4a covers a field.
inline uint4 pq2_0_pack_act_group(int4 q0, int4 q1, int4 q2, int4 q3) {
    return (uint4)(
        as_uint(convert_char4((int4)(q0.x, q1.x, q2.x, q3.x))),
        as_uint(convert_char4((int4)(q0.y, q1.y, q2.y, q3.y))),
        as_uint(convert_char4((int4)(q0.z, q1.z, q2.z, q3.z))),
        as_uint(convert_char4((int4)(q0.w, q1.w, q2.w, q3.w))));
}

inline int pq2_0_sum_act_group(int4 q0, int4 q1, int4 q2, int4 q3) {
    const int4 s = q0 + q1 + q2 + q3;
    return s.x + s.y + s.z + s.w;
}

// Quantize contiguous f32 src1 to int8 per QK_ACT elements like the CPU q8_0 reference, one work item per QK_ACT
// elements. Each ACT_GROUP elements keep (scale, scale * sum of their int8 values) for the -1 offset of the codes.
kernel void kernel_quantize_pq2_0_act(
    global const char * src1,
    ulong               offset1,
    global uint4      * act_q,
    global float2     * act_ds,
    int                 n_blocks
) {
    const int blk = get_global_id(0);
    if (blk >= n_blocks) {
        return;
    }

    global const float * x = (global const float *) (src1 + offset1) + (ulong)blk*QK_ACT;

    const float4 x0 = vload4(0, x);
    const float4 x1 = vload4(1, x);
    const float4 x2 = vload4(2, x);
    const float4 x3 = vload4(3, x);
    const float4 x4 = vload4(4, x);
    const float4 x5 = vload4(5, x);
    const float4 x6 = vload4(6, x);
    const float4 x7 = vload4(7, x);

    const float4 m = fmax(fmax(fmax(fabs(x0), fabs(x1)), fmax(fabs(x2), fabs(x3))),
                          fmax(fmax(fabs(x4), fabs(x5)), fmax(fabs(x6), fabs(x7))));
    const float amax = fmax(fmax(m.x, m.y), fmax(m.z, m.w));

    const float d  = amax/127.0f;
    const float id = d != 0.0f ? 1.0f/d : 0.0f;
    const float dh = (float)(half)d; // the CPU reference keeps the scale as fp16

    const int4 q0 = convert_int4_rte(x0*id);
    const int4 q1 = convert_int4_rte(x1*id);
    const int4 q2 = convert_int4_rte(x2*id);
    const int4 q3 = convert_int4_rte(x3*id);
    const int4 q4 = convert_int4_rte(x4*id);
    const int4 q5 = convert_int4_rte(x5*id);
    const int4 q6 = convert_int4_rte(x6*id);
    const int4 q7 = convert_int4_rte(x7*id);

    act_q[2*blk + 0] = pq2_0_pack_act_group(q0, q1, q2, q3);
    act_q[2*blk + 1] = pq2_0_pack_act_group(q4, q5, q6, q7);

    act_ds[2*blk + 0] = (float2)(dh, dh*pq2_0_sum_act_group(q0, q1, q2, q3));
    act_ds[2*blk + 1] = (float2)(dh, dh*pq2_0_sum_act_group(q4, q5, q6, q7));
}

// Expand X once per src1 column (N_COLS <= 2) and per row of a lane (N_ROWS == 4). Named variables instead of private
// arrays: E031.41 placed the private arrays in local memory and gave wrong, run-to-run varying results.
#if N_COLS > 2 || N_ROWS != 4
#error "kernel_mul_mv_pq2_0_f32_flat expands at most 2 columns and exactly 4 rows per lane"
#endif
#if N_COLS > 1
#define FOR_COLS(X) X(0) X(1)
#else
#define FOR_COLS(X) X(0)
#endif
#define FOR_ROWS(X)         X(0) X(1) X(2) X(3)
#define FOR_ROWS_OF(X, c)   X(c, 0) X(c, 1) X(c, 2) X(c, 3)

// a row past ne01 reads the last row and its result is dropped
#define ROW_PTRS(k)                                                                          \
    const int           row##k = row_base + (k)*N_LANES;                                     \
    global const uint * q##k   = src0_q + mat_row0 + min(row##k, ne01 - 1);                  \
    global const half * d##k   = src0_d + mat_row0 + min(row##k, ne01 - 1);
#define ROW_SCALE(k) const float dw##k = d##k[ib*n_rows];
#define ROW_WORD(k)                                                                          \
    const uint w##k    = q##k[g*n_rows];                                                     \
    const uint f##k##0 =  w##k       & PQ2_0_FIELD_MASK;                                     \
    const uint f##k##1 = (w##k >> 2) & PQ2_0_FIELD_MASK;                                     \
    const uint f##k##2 = (w##k >> 4) & PQ2_0_FIELD_MASK;                                     \
    const uint f##k##3 = (w##k >> 6) & PQ2_0_FIELD_MASK;

// accumulate with plain a*b + c, never fma(): this GPU has no native FMA, so fma() is emulated in software
#define CR_SUM_INIT(c, k) float sum##c##_##k = 0.0f;
#define CR_BLK_INIT(c, k) float blk##c##_##k = 0.0f;
#define CR_BLK_ACC(c, k)  sum##c##_##k += dw##k*blk##c##_##k;

// sum_e (q_e - 1)*dw*dx*xq_e = dw*(dx*sum_e q_e*xq_e - dx*sum_e xq_e)
#define CR_WORD(c, k) {                                                                      \
    int isum = dot_acc_sat_4x8packed_ss_int(f##k##0, xq.x, 0);                               \
    isum     = dot_acc_sat_4x8packed_ss_int(f##k##1, xq.y, isum);                            \
    isum     = dot_acc_sat_4x8packed_ss_int(f##k##2, xq.z, isum);                            \
    isum     = dot_acc_sat_4x8packed_ss_int(f##k##3, xq.w, isum);                            \
    blk##c##_##k += ds.x*(float)isum - ds.y;                                                 \
}

#define COL_SUM_INIT(c) FOR_ROWS_OF(CR_SUM_INIT, c)
#define COL_BLK_INIT(c) FOR_ROWS_OF(CR_BLK_INIT, c)
#define COL_BLK_ACC(c)  FOR_ROWS_OF(CR_BLK_ACC, c)
#define COL_WORD(c) {                                                                        \
    const uint4  xq = aq [(c)*n_groups + g];                                                 \
    const float2 ds = ads[(c)*n_groups + g];                                                 \
    FOR_ROWS_OF(CR_WORD, c)                                                                  \
}

#define CR_RED_STORE(c, k) red[((ks - 1)*N_COLS + (c))*N_LANES + lid] = sum##c##_##k;
#define CR_RED_OUT(c, k) {                                                                   \
    float tot = sum##c##_##k;                                                                \
    for (int s = 0; s < N_KSPLIT - 1; ++s) {                                                 \
        tot += red[(s*N_COLS + (c))*N_LANES + lid];                                          \
    }                                                                                        \
    dst_f32[(first_col + (c))*ne0 + row##k] = tot;                                           \
}
#define CR_RED_STORE_0(c) CR_RED_STORE(c, 0)
#define CR_RED_STORE_1(c) CR_RED_STORE(c, 1)
#define CR_RED_STORE_2(c) CR_RED_STORE(c, 2)
#define CR_RED_STORE_3(c) CR_RED_STORE(c, 3)
#define CR_RED_OUT_0(c)   CR_RED_OUT(c, 0)
#define CR_RED_OUT_1(c)   CR_RED_OUT(c, 1)
#define CR_RED_OUT_2(c)   CR_RED_OUT(c, 2)
#define CR_RED_OUT_3(c)   CR_RED_OUT(c, 3)

// The N_KSPLIT subgroups of a work group hold partial sums of the same rows; subgroup 0 adds and stores them.
#define ROW_REDUCE(k)                                                                        \
    if (ks > 0) { FOR_COLS(CR_RED_STORE_##k) }                                               \
    barrier(CLK_LOCAL_MEM_FENCE);                                                            \
    if (ks == 0 && row##k < ne01) { FOR_COLS(CR_RED_OUT_##k) }                               \
    barrier(CLK_LOCAL_MEM_FENCE);

// Weights are word-major (word w of row r at w*n_rows + r, the scale of block b at b*n_rows + r), so the N_LANES
// lanes of a subgroup read consecutive rows while every lane reads the same int8 src1 words. Each lane computes
// N_ROWS rows N_LANES apart, so one src1 load serves N_ROWS rows, and N_KSPLIT subgroups split the blocks of K.
// N_COLS, N_LANES, N_ROWS and N_KSPLIT are set by the host.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_pq2_0_f32_flat(
    global const uint   * src0_q,
    global const half   * src0_d,
    global const uint4  * act_q,
    global const float2 * act_ds,
    global char         * dst,
    ulong                 offsetd,
    int                   ne00,
    int                   ne01,
    int                   n_rows,
    int                   mat_row0,
    int                   act_col0,
    int                   ne0,
    int                   col0
) {
    const int lid = get_local_id(0);
    const int ks  = get_local_id(1);

    const int row_base = get_group_id(0)*N_LANES*N_ROWS + lid;
    FOR_ROWS(ROW_PTRS)

    const int nb        = ne00/QK_PQ2_0;
    const int n_groups  = ne00/ACT_GROUP;
    const int first_col = col0 + get_group_id(2)*N_COLS;

    // src1 is contiguous, so its columns over all batches are consecutive in the quantized copy
    global const uint4  * aq  = act_q  + (act_col0 + first_col)*n_groups;
    global const float2 * ads = act_ds + (act_col0 + first_col)*n_groups;

    FOR_COLS(COL_SUM_INIT)

    for (int ib = ks; ib < nb; ib += N_KSPLIT) {
        FOR_ROWS(ROW_SCALE)
        FOR_COLS(COL_BLK_INIT)

        #pragma unroll
        for (int j = 0; j < PQ2_0_WORDS; ++j) {
            const int g = ib*PQ2_0_WORDS + j;
            FOR_ROWS(ROW_WORD)
            FOR_COLS(COL_WORD)
        }

        FOR_COLS(COL_BLK_ACC)
    }

    local float red[(N_KSPLIT - 1)*N_COLS*N_LANES];
    global float * dst_f32 = (global float *) (dst + offsetd);
    FOR_ROWS(ROW_REDUCE)
}
