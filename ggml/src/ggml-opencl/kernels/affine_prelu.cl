// Fused affine + per-channel PReLU (LavaSR denoiser), scalar op order for bit-exactness:
// out = x*aw[f,c] + ab[f,c] + max(x,0) + slope[c]*min(x,0). x/out [F,T,C,Bc], aw,ab [F,C], slope [C].
kernel void kernel_affine_prelu_f32(
    global const char * x,  ulong ox,
    global const char * aw, ulong oaw,
    global const char * ab, ulong oab,
    global const char * sl, ulong osl,
    global       char * dst, ulong od,
    int F, int T, int C
) {
    const int f = get_global_id(0);
    if (f >= F) return;
    const int t = get_group_id(1);
    const int p = get_group_id(2);        // c + C*b
    const int c = p % C;

    global const float * xg  = (global const float *)(x  + ox);
    global const float * awg = (global const float *)(aw + oaw);
    global const float * abg = (global const float *)(ab + oab);
    global const float * slg = (global const float *)(sl + osl);
    global       float * dg  = (global       float *)(dst + od);

    const long idx = (long) f + (long) F * t + (long) F * T * p;
    const float xv = xg[idx];
    const float r  = fmax(xv, 0.0f);
    const float n  = xv - r;
    const float pr = r + slg[c] * n;
    const float a  = xv * awg[f + F * c] + abg[f + F * c];
    dg[idx] = a + pr;
}
