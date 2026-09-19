#ifndef LLAMA_UPSTREAM_FA_MIXED_TYPES

#ifdef HAS_QJL_CORRECTION
shared float Qf_qjl_proj[Br][QUANT_K];

void qjl_project_q_block(uint32_t q_offset, uint32_t qjl_block_start, uint32_t tid) {
    // Pre-compute QJL projection of one K block of Q:
    // proj_q = FHT(D_qjl * q[qjl_block_start:qjl_block_start+QUANT_K]).
    //
    // Read Q straight from the input SSBO (data_qv4). The alternative is to
    // read the already-loaded Qf shared array and divide by p.scale to undo
    // the p.scale that was multiplied into Qf for the main Q·K dot. For the
    // scalar FA path Qf is f32 and p.scale is usually a power of two
    // (1/sqrt(head_dim)), so the round-trip is mathematically exact; for
    // the coopmat1 path Qf is f16, which makes the round-trip lossy for
    // large-magnitude activations. Reading the raw f32 Q directly avoids
    // both issues and matches what the non-FA mul_mm_tbq_qjl_correction pass
    // does with src1.
    [[unroll]] for (uint32_t r = 0; r < Br; ++r) {
        if (i * Br + r < N) {
            for (uint32_t idx = tid; idx < QUANT_K; idx += gl_WorkGroupSize.x) {
                vec4  qv    = vec4(data_qv4[q_offset / 4 + (i * Br + r) * q_stride / 4 + (qjl_block_start + idx) / 4]);
                float q_val = qv[idx % 4];
                Qf_qjl_proj[r][idx] = q_val * qjl_get_sign(idx);
            }
        }
        barrier();

        if (i * Br + r < N) {
            for (uint h = 1u; h < QUANT_K; h <<= 1u) {
                for (uint idx = tid; idx < QUANT_K / 2u; idx += gl_WorkGroupSize.x) {
                    uint  grp             = idx / h;
                    uint  pos             = idx % h;
                    uint  j               = grp * (h << 1u) + pos;
                    float a               = Qf_qjl_proj[r][j];
                    float b               = Qf_qjl_proj[r][j + h];
                    Qf_qjl_proj[r][j]     = a + b;
                    Qf_qjl_proj[r][j + h] = a - b;
                }
                barrier();
            }
        }
        barrier();
    }
}
#endif
#endif
