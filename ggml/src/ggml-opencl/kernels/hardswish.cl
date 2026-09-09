#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// hardswish
//------------------------------------------------------------------------------
kernel void kernel_hardswish_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x * fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f));
}
