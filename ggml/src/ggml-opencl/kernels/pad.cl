// Fast path for the common shape: nothing padded on dim0, both tensors contiguous and
// ne0 a multiple of 4.  Every output row is then either a whole source row or all zeros,
// so this copies float4 per work-item from one flat 256-wide launch instead of the
// generic kernel's one-float-per-item, 64-wide, one-workgroup-per-row geometry.
kernel void kernel_pad_f32_4(
        global void * src0,
        ulong offset0,
        global void * dst,
        ulong offsetd,
        int ne0v, int ne01, int ne02,
        int ne1, int ne2, int ne3,
        int lp1, int rp1,
        int lp2, int rp2,
        int lp3, int rp3,
        int total_v
) {
    int idx = get_global_id(0);
    if (idx >= total_v) {
        return;
    }

    global float4 * src0_v = (global float4 *)((global char *)src0 + offset0);
    global float4 * dst_v  = (global float4 *)((global char *)dst  + offsetd);

    // One division in the common ne2 == ne3 == 1 case; i0 comes back by multiply-subtract.
    int r  = idx / ne0v;
    int i0 = idx - r*ne0v;

    int i1 = r;
    int i2 = 0;
    int i3 = 0;
    if (ne2 > 1 || ne3 > 1) {
        i1 = r % ne1;
        int r2 = r / ne1;
        i2 = r2 % ne2;
        i3 = r2 / ne2;
    }

    bool in_src_bounds = (i1 >= lp1 && i1 < ne1 - rp1) &&
                         (i2 >= lp2 && i2 < ne2 - rp2) &&
                         (i3 >= lp3 && i3 < ne3 - rp3);

    dst_v[idx] = in_src_bounds
        ? src0_v[(((i3 - lp3)*ne02 + (i2 - lp2))*ne01 + (i1 - lp1))*ne0v + i0]
        : (float4)(0.0f);
}

kernel void kernel_pad(
        global void * src0,
        ulong offset0,
        global void * dst,
        ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne0, int ne1, int ne2, int ne3,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int lp0, int rp0,
        int lp1, int rp1,
        int lp2, int rp2,
        int lp3, int rp3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst  + offsetd);

    int i0 = get_global_id(0);
    int i1 = get_group_id(1);
    int i2 = get_group_id(2) % ne2;
    int i3 = get_group_id(2) / ne2;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    // 32-bit index arithmetic: Adreno emulates 64-bit multiplies in software, and both
    // sums are truncated to uint anyway, so the low 32 bits are unchanged.
    uint src0_idx = (uint)(i3 - lp3)*(uint)nb03 + (uint)(i2 - lp2)*(uint)nb02 +
                    (uint)(i1 - lp1)*(uint)nb01 + (uint)(i0 - lp0)*(uint)nb00;
    uint dst_idx  = (uint)i3*(uint)nb3  + (uint)i2*(uint)nb2  +
                    (uint)i1*(uint)nb1  + (uint)i0*(uint)nb0;

    global float * src0_ptr = (global float *)((global char *)src0 + src0_idx);
    global float * dst_ptr  = (global float *)((global char *)dst  + dst_idx);

    bool in_src_bounds = (i0 >= lp0 && i0 < ne0 - rp0) &&
                         (i1 >= lp1 && i1 < ne1 - rp1) &&
                         (i2 >= lp2 && i2 < ne2 - rp2) &&
                         (i3 >= lp3 && i3 < ne3 - rp3);

    *dst_ptr = in_src_bounds ? *src0_ptr : 0.0f;
}
