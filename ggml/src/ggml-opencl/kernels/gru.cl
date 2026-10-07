// Fused batched GRU (PyTorch: gate order r,z,n, reset on hh new-gate, h0=0): one workgroup per
// batch b, H threads, L steps looped in local mem. whh [H,3H], gi [3H,B,L], bhh [3H] -> dst [H,B,L].
#define GRU_MAX_H 128

kernel void kernel_gru_f32(
    global const char * whh, ulong off_whh,
    global const char * gi,  ulong off_gi,
    global const char * bhh, ulong off_bhh,
    global       char * dst, ulong off_dst,
    int H, int B, int L, int reverse
) {
    const int j = get_local_id(0);     // hidden unit  (0..H-1)
    const int b = get_group_id(1);     // batch element

    global const float * whh_d = (global const float *)(whh + off_whh);
    global const float * gi_d  = (global const float *)(gi  + off_gi);
    global const float * bhh_d = (global const float *)(bhh + off_bhh);
    global       float * dst_d = (global       float *)(dst + off_dst);

    const int H3 = 3 * H;

    __local float h_l[GRU_MAX_H];
    __local float gh_l[3 * GRU_MAX_H];

    h_l[j] = 0.0f;                      // zero initial state
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 0; s < L; ++s) {
        const int t = reverse ? (L - 1 - s) : s;
        global const float * git = gi_d + (long) H3 * b + (long) H3 * B * t;

        // pass 1: gh[g] = sum_k whh[k,g]*h[k] + bhh[g] for this thread's 3 gate rows.
        for (int gg = 0; gg < 3; ++gg) {
            const int g = gg * H + j;
            global const float * wcol = whh_d + (long) H * g;
            float acc = bhh_d[g];
            for (int k = 0; k < H; ++k) acc += wcol[k] * h_l[k];
            gh_l[g] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // pass 2: gates + state update (h_l[j] read then written; no cross-thread h read here).
        const float r  = 1.0f / (1.0f + exp(-(git[j]         + gh_l[j])));
        const float z  = 1.0f / (1.0f + exp(-(git[H + j]     + gh_l[H + j])));
        const float nc = tanh(git[2 * H + j] + r * gh_l[2 * H + j]);
        const float hn = nc + z * (h_l[j] - nc);
        dst_d[(long) H * b + (long) H * B * t + j] = hn;
        h_l[j] = hn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
