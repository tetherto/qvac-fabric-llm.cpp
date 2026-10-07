// Snake activation y = x + sin^2(a*x) * inv_b, per-channel a / inv_b (ACE-Step Oobleck VAE).
// x / dst are [T, C] contiguous; a and inv_b hold one F32 per channel.
// One thread per element; `sin` (not `native_sin`) to keep the CPU's sinf accuracy.
kernel void kernel_snake_f32(
    global const char * x,     ulong off_x,
    global const char * a,     ulong off_a,
    global const char * inv_b, ulong off_b,
    global       char * dst,   ulong off_d,
    int T, int C
) {
    const int t = get_global_id(0);
    const int c = get_global_id(1);
    if (t >= T || c >= C) return;

    global const float * xp = (global const float *)(x   + off_x) + (long) c * T;
    global       float * dp = (global       float *)(dst + off_d) + (long) c * T;

    const float ac = ((global const float *)(a     + off_a))[c];
    const float bc = ((global const float *)(inv_b + off_b))[c];

    const float xi = xp[t];
    const float si = sin(ac * xi);
    dp[t] = xi + si * si * bc;
}
