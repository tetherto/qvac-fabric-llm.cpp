#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_im2col_f16(
        global float * src1,
        ulong offset1,
        global half  * dst,
        ulong offsetd,
        ulong batch_offset,
        ulong delta_offset,
        long IW,
        long IH,
        long IC,
        long OW,
        long OH,
        long KW,
        long KH,
        long pelements,
        long CHW,
        int  s0,
        int  s1,
        int  p0,
        int  p1,
        int  d0,
        int  d1
) {
    // One workgroup per output position (in, ioh, iow); its threads walk that position's
    // CHW columns, which are CONTIGUOUS in dst, so the stores coalesce.
    //
    // The previous mapping put ix on the fastest-varying work-item index while dst is
    // addressed as (... + ix)*CHW + col, so adjacent lanes wrote CHW elements apart --
    // up to 1792 bytes in the ACE-Step VAE. Every 2-byte store then owned a cache line,
    // and im2col measured 81% of the entire VAE decode on an Adreno 740.
    // Mirrors the ggml-metal fix in 1bc4628b; im2col is a pure gather, so bit-exact.
    const long iow = get_group_id(0);
    const long ioh = get_group_id(1);
    const long in  = get_group_id(2);

    src1 = (global float*)((global char*)src1 + offset1);
    dst  = (global half *)((global char*)dst  + offsetd);

    // Column decomposition in 32-bit: CHW/KW/KH are small, and Adreno emulates 64-bit
    // integer division in software -- doing two of them per element made im2col ~280x
    // less efficient per element than a plain elementwise kernel on the same device.
    // Only the final addresses stay 64-bit. KH == 1 (every 1-D conv) skips a division.
    const int  KWi  = (int) KW;
    const int  KHWi = (int) (KW * KH);
    const int  CHWi = (int) CHW;
    const int  lsz  = (int) get_local_size(0);

    const long offset_row = ((in * OH + ioh) * OW + iow) * CHW;
    const long offset_bat = in * batch_offset;

    for (int col = (int) get_local_id(0); col < CHWi; col += lsz) {
        const int iic = col / KHWi;
        const int rem = col - iic * KHWi;
        int ikh, ikw;
        if (KH == 1) { ikh = 0; ikw = rem; }
        else         { ikh = rem / KWi; ikw = rem - ikh * KWi; }

        const long iiw = iow * s0 + (long) ikw * d0 - p0;
        const long iih = ioh * s1 + (long) ikh * d1 - p1;

        dst[offset_row + col] = (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW)
            ? (half) 0
            : (half) src1[offset_bat + (long) iic * delta_offset + iih * IW + iiw];
    }
}
