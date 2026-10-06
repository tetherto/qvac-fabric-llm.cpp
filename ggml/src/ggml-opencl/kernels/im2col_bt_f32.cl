#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Builds the transposed im2col matrix a GEMM actually consumes, straight from the signal:
//
//   bt[chw][ix] = src[ic*delta + ix*s0 + kx*d0 - p0],   chw = ic*KW + kx
//
// For a fixed chw that is contiguous in ix on both sides, so lanes walking ix read and
// write coalesced. Materialising im2col as [ix][chw] and transposing it afterwards costs
// two passes over a matrix KW times larger than the signal, both of them strided.
kernel void kernel_im2col_bt_f32(
        global const float * src,
        ulong  offset_src,
        global float * bt,
        ulong  offset_bt,
        int    IW,
        int    IC,
        int    OW,
        int    KW,
        int    padded_OW,
        ulong  delta_offset,
        int    s0,
        int    p0,
        int    d0
) {
    src = (global const float *)((global const char *)src + offset_src);
    bt  = (global float *)((global char *)bt + offset_bt);

    const int ix  = get_global_id(0);
    const int chw = get_global_id(1);

    if (ix >= padded_OW || chw >= IC * KW) {
        return;
    }

    // Columns past OW exist only to pad the GEMM's N up to a multiple of 8.
    if (ix >= OW) {
        bt[(long) chw * padded_OW + ix] = 0.0f;
        return;
    }

    const int ic  = chw / KW;
    const int kx  = chw - ic * KW;
    const int iiw = ix * s0 + kx * d0 - p0;

    bt[(long) chw * padded_OW + ix] =
        (iiw < 0 || iiw >= IW) ? 0.0f : src[(long) ic * delta_offset + iiw];
}
