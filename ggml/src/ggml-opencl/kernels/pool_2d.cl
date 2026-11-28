#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// pool_2d
//------------------------------------------------------------------------------
// Contiguous NCHW F32 src/dst. op: 0 = MAX, 1 = AVG.
kernel void kernel_pool_2d_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1,
        int IW,
        int IH,
        int OW,
        int OH,
        int np
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int idx = get_global_id(0);
    if (idx >= np) {
        return;
    }

    const int nc     = idx / (OW * OH);
    const int cur_oh = (idx / OW) % OH;
    const int cur_ow = idx % OW;

    global float * i_ptr = src0 + nc * IW * IH;
    global float * o_ptr = dst  + nc * OW * OH;

    const int start_h = cur_oh * s1 - p1;
    const int bh      = max(0,  start_h);
    const int eh      = min(IH, start_h + k1);

    const int start_w = cur_ow * s0 - p0;
    const int bw      = max(0,  start_w);
    const int ew      = min(IW, start_w + k0);

    if (op == 0) {
        // MAX
        float res = -INFINITY;
        for (int i = bh; i < eh; ++i) {
            for (int j = bw; j < ew; ++j) {
                res = fmax(res, i_ptr[i * IW + j]);
            }
        }
        o_ptr[cur_oh * OW + cur_ow] = res;
    } else {
        // AVG
        float res = 0.0f;
        for (int i = bh; i < eh; ++i) {
            for (int j = bw; j < ew; ++j) {
                res += i_ptr[i * IW + j];
            }
        }
        const float scale = 1.0f / (float)(k0 * k1);
        o_ptr[cur_oh * OW + cur_ow] = res * scale;
    }
}
