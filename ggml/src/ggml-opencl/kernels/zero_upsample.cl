// Zero-insertion upsample of ne0 by s: thread f writes dst[f*s]=src[f] then s-1 zeros.
// src [F,R] contiguous -> dst [Fu,R], Fu=(F-1)*s+1, R=ne1*ne2*ne3.
kernel void kernel_zero_upsample_f32(
    global const char * src, ulong off_src,
    global       char * dst, ulong off_dst,
    int F, int R, int Fu, int s
) {
    const int f = get_global_id(0);
    const int r = get_global_id(1);
    if (f >= F || r >= R) return;

    global const float * sp = (global const float *)(src + off_src) + (long) r * F;
    global       float * dp = (global       float *)(dst + off_dst) + (long) r * Fu;

    const int obase = f * s;
    dp[obase] = sp[f];
    for (int p = 1; p < s; ++p) {
        if (obase + p < Fu) dp[obase + p] = 0.0f;
    }
}
