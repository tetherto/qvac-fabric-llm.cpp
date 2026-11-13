// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC] for a 1-D transpose conv
// (ACE-Step Oobleck VAE), T_out = (T_in-1)*s0 + K - 2*p0.
// Written as a gather so each thread owns one output element and writes are
// disjoint -- no atomics. Accumulation walks t_in ascending, matching the CPU
// kernel's summation order.
kernel void kernel_col2im_1d_f32(
    global const char * src, ulong off_src,
    global       char * dst, ulong off_dst,
    int K_OC, int T_in, int K, int T_out, int OC, int s0, int p0
) {
    const int t_out = get_global_id(0);
    const int oc    = get_global_id(1);
    if (t_out >= T_out || oc >= OC) return;

    global const float * col = (global const float *)(src + off_src);
    global       float * dp  = (global       float *)(dst + off_dst);

    const int t_abs = t_out + p0;  // position in the uncropped signal

    // Every (t_in, k) with t_in*s0 + k == t_abs, 0 <= k < K. Integer division
    // truncates toward zero here exactly as it does on the CPU, and any negative
    // result is clamped to 0, so the two bounds agree.
    int t_in_min = (t_abs - K + 1 + s0 - 1) / s0;
    if (t_in_min < 0) t_in_min = 0;
    int t_in_max = t_abs / s0;
    if (t_in_max >= T_in) t_in_max = T_in - 1;

    float sum = 0.0f;
    for (int t_in = t_in_min; t_in <= t_in_max; ++t_in) {
        const int k = t_abs - t_in * s0;
        if (k >= 0 && k < K) {
            sum += col[(long) (oc * K + k) + (long) t_in * K_OC];
        }
    }
    dp[(long) t_out + (long) oc * T_out] = sum;
}
