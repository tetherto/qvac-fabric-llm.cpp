// Channel shuffle (G groups) along ne2 (PyTorch): output plane c' copies input plane
// (c'%G)*(C/G) + c'/G. One workgroup per output plane; coalesced FT copy.
kernel void kernel_channel_shuffle_f32(
    global const char * src, ulong off_src,
    global       char * dst, ulong off_dst,
    int FT, int C, int G
) {
    const int ft = get_global_id(0);
    if (ft >= FT) return;

    const int p      = get_group_id(2);          // output plane = c' + C*b
    const int cprime = p % C;
    const int b      = p / C;
    const int in_c   = (cprime % G) * (C / G) + cprime / G;
    const int in_p   = in_c + C * b;

    global const float * sp = (global const float *)(src + off_src) + (long) in_p * FT;
    global       float * dp = (global       float *)(dst + off_dst) + (long) p    * FT;
    dp[ft] = sp[ft];
}
