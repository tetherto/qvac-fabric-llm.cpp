#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// conv_2d_dw (depthwise) - F32 weights + F32 input + F32 output
//------------------------------------------------------------------------------
// Ported from the ggml Metal kernel_conv_2d_dw_f32_f32, adapted for Adreno.
//
// IMPORTANT (Adreno correctness): the Metal port read every element in the inner
// loop via byte-pointer type punning, i.e. `*(global const float *)(charptr +
// byte_offset)`. On Qualcomm Adreno the OpenCL compiler miscompiles such
// per-iteration char->float type-punned loads, producing wrong results (DocTR
// detection yielded 0 regions). The working sibling kernel (conv2d.cl)
// avoids this by casting the base pointer to `global float*`
// ONCE and then indexing it as a typed array with *element* strides. We do the
// same here.
//
// All strides below are ELEMENT strides (nb / sizeof(float)), supplied by the
// host dispatch. Because the kernel only ever uses nb[] strides (never assumes a
// particular contiguity), it is layout-agnostic and matches the CPU reference for
// BOTH the WHCN (standard contiguous) and CWHN (channel-contiguous) layouts:
//   - WHCN input : nb10=1, nb11=IW, nb12=IW*IH, nb13=IW*IH*OC
//                  weight nb00=1, nb01=KW, nb03=KW*KH
//                  output nb0=1, nb1=OW, nb2=OW*OH, nb3=OW*OH*OC
//   - CWHN input : nb10=OC, nb11=IW*OC, nb12=1, nb13=IW*IH*OC
//                  weight nb00=OC, nb01=KW*OC, nb03=1
//                  output nb0=OC, nb1=OW*OC, nb2=1, nb3=OW*OH*OC
// In both cases the element-offset arithmetic reproduces the exact indices used
// by ggml_compute_forward_conv_2d_dw_whcn / _cwhn.
kernel void kernel_conv_2d_dw_f32_f32(
        global char * weights,
        ulong off_weights,
        global char * src,
        ulong off_src,
        global char * dst,
        ulong off_dst,
        int IW, int IH,
        int OW, int OH,
        int OC,
        int N,
        int KW, int KH,
        int s0, int s1,
        int p0, int p1,
        int d0, int d1,
        ulong nb00, ulong nb01, ulong nb03,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        ulong nb0,  ulong nb1,  ulong nb2,  ulong nb3
) {
    // Cast bases to typed float pointers ONCE (off_* are byte offsets and are
    // always a multiple of sizeof(float) for F32 tensors). All nb* below are
    // element strides, so the per-iteration loads are plain typed-array indexes
    // rather than char->float type punning.
    global const float * weights_base = (global const float *)((global char*)weights + off_weights);
    global const float * src_base     = (global const float *)((global char*)src     + off_src);
    global       float * dst_base     = (global       float *)((global char*)dst     + off_dst);

    const ulong total_threads = (ulong)get_global_size(0);
    const ulong total_outputs = (ulong)N * OC * OH * OW;

    for (ulong index = get_global_id(0); index < total_outputs; index += total_threads) {
        ulong tmp = index;

        const int ow = (int)(tmp % OW); tmp /= OW;
        const int oh = (int)(tmp % OH); tmp /= OH;
        const int oc = (int)(tmp % OC); tmp /= OC;
        const int n  = (int)tmp;

        // Dead-simple form, matching the CPU reference
        // ggml_compute_forward_conv_2d_dw_whcn EXACTLY: per-tap forward map
        // src = dst*stride + knl*dilation - pad, with a plain in-bounds check.
        // (Replaced the Metal ky_start/ky_end clamping, which gave wrong results
        // on Adreno.) Layout-agnostic via element strides nb*.
        float acc = 0.0f;
        const ulong src_b = (ulong)n * nb13 + (ulong)oc * nb12;
        const ulong w_b   = (ulong)oc * nb03;

        for (int ky = 0; ky < KH; ++ky) {
            const int iy = oh*s1 + ky*d1 - p1;
            if (iy < 0 || iy >= IH) {
                continue;
            }
            for (int kx = 0; kx < KW; ++kx) {
                const int ix = ow*s0 + kx*d0 - p0;
                if (ix < 0 || ix >= IW) {
                    continue;
                }
                const ulong src_idx = src_b + (ulong)iy * nb11 + (ulong)ix * nb10;
                const ulong w_idx   = w_b   + (ulong)ky * nb01 + (ulong)kx * nb00;
                acc += src_base[src_idx] * weights_base[w_idx];
            }
        }

        const ulong dst_idx =
            (ulong)n  * nb3 +
            (ulong)oc * nb2 +
            (ulong)oh * nb1 +
            (ulong)ow * nb0;

        dst_base[dst_idx] = acc;
    }
}

// NOTE: DocTR promotes depthwise weights to F32 (so the direct
// GGML_OP_CONV_2D_DW kernel runs on every backend), so only the F32-weight
// variant above is used / shipped. An F16-weight variant was dropped as
// unused — re-add it (plus an F16 supports_op case + host dispatch) if a
// future consumer needs F16 depthwise weights on OpenCL.
