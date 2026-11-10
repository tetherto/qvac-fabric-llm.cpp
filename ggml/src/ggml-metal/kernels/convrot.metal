#include "common.h"

template <typename T>
inline void convrot_rotate_activation(
        constant ggml_metal_kargs_mul_mat_convrot & args, device const char * activations,
        device float * rotated, threadgroup float * values, uint3 group, ushort tid) {
    #pragma clang fp reassociate(off) contract(off)
    const uint i2 = group.z % args.ne02, i3 = group.z / args.ne02;
    for (uint i = tid; i < 256; i += 64) {
        values[i] = float(*((device const T *)(activations +
            (group.x*256 + i)*args.nb00 + group.y*args.nb01 + i2*args.nb02 + i3*args.nb03)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride *= 4) {
        const uint j = tid/stride*(4*stride) + tid%stride;
        const float a = values[j], b = values[j + stride], c = values[j + 2*stride], d = values[j + 3*stride];
        values[j]            = ( a + b + c - d)*0.5f;
        values[j + stride]   = ( a + b - c + d)*0.5f;
        values[j + 2*stride] = ( a - b + c + d)*0.5f;
        values[j + 3*stride] = (-a + b + c + d)*0.5f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const ulong offset = (ulong(group.z)*args.ne01 + group.y)*args.k + group.x*256;
    for (uint i = tid; i < 256; i += 64) {
        rotated[offset + i] = values[i];
    }
}

// H256 is symmetric: the activations are rotated once, then a tiled I8 matmul
// scales the FP32 result per output row.
inline void convrot_i8mm(
        constant ggml_metal_kargs_mul_mat_convrot & args, device const char * weights,
        device const char * scales, device char * dst, device const float * rotated,
        threadgroup float * shared, uint3 group, ushort tid, ushort sg) {
    threadgroup float * wa = shared;
    threadgroup float * xa = shared + 1024;
    const uint row0 = group.x*32, col0 = group.y*64;
    const uint i2 = group.z % args.ne02, i3 = group.z / args.ne02;
    simdgroup_float8x8 sums[4];
    for (uint c = 0; c < 4; ++c) {
        sums[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < uint(args.k); k0 += 32) {
        for (uint i = tid; i < 1024; i += 256) {
            const uint row = row0 + i/32;
            wa[i] = row < uint(args.out_features) ? float(*((device const int8_t *)(weights +
                row*args.nb11 + (k0 + i%32)*args.nb10))) : 0.0f;
        }
        for (uint i = tid; i < 2048; i += 256) {
            const uint col = col0 + i/32;
            xa[i] = col < uint(args.ne01) ? rotated[(ulong(group.z)*args.ne01 + col)*args.k + k0 + i%32] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 ma;
            simdgroup_load(ma, wa + (sg/2)*8*32 + k, 32);
            for (uint c = 0; c < 4; ++c) {
                simdgroup_float8x8 mb;
                simdgroup_load(mb, xa + ((sg%2)*32 + c*8)*32 + k, 32, 0, true);
                simdgroup_multiply_accumulate(sums[c], ma, mb, sums[c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint c = 0; c < 4; ++c) {
        simdgroup_store(sums[c], shared + (sg/2)*8*64 + (sg%2)*32 + c*8, 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 2048; i += 256) {
        const uint row = row0 + i/64, col = col0 + i%64;
        if (row < uint(args.out_features) && col < uint(args.ne01)) {
            *((device float *)(dst + row*args.nb0 + col*args.nb1 + i2*args.nb2 + i3*args.nb3)) =
                shared[i]*(*((device const float *)(scales + row*args.nb20)));
        }
    }
}

// One threadgroup per group_size-wide block, staged in threadgroup memory, so dst may alias src0.
kernel void kernel_convrot_f32(
        constant ggml_metal_kargs_convrot & args,
        device const char * src0,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        uint3  ntg3 [[threads_per_threadgroup]]) {
    #pragma clang fp reassociate(off) contract(off)
    threadgroup float values[1024];

    const uint ntg = ntg3.x;
    const uint g   = (uint) args.group_size;
    const uint k0  = tgpig.x*g;
    const uint i1  = tgpig.y;
    const uint i2  = tgpig.z % args.ne02;
    const uint i3  = tgpig.z / args.ne02;

    device const char * src_row = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device       char * dst_row = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    for (uint i = tid; i < g; i += ntg) {
        values[i] = *(device const float *) (src_row + (k0 + i)*args.nb00);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < g; stride *= 4) {
        for (uint b = tid; b < g/4; b += ntg) {
            const uint j = (b/stride)*(4*stride) + b%stride;
            const float a = values[j], bb = values[j + stride], c = values[j + 2*stride], d = values[j + 3*stride];
            values[j]            = ( a + bb + c - d)*0.5f;
            values[j + stride]   = ( a + bb - c + d)*0.5f;
            values[j + 2*stride] = ( a - bb + c + d)*0.5f;
            values[j + 3*stride] = (-a + bb + c + d)*0.5f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < g; i += ntg) {
        *(device float *) (dst_row + (k0 + i)*args.nb0) = values[i];
    }
}

#define CONVROT_NATIVE_KERNELS(T, suffix) \
[[host_name("kernel_mul_mat_convrot_rotate_" #suffix)]] \
kernel void kernel_mul_mat_convrot_rotate_##suffix( \
        constant ggml_metal_kargs_mul_mat_convrot & args, device const char * activations, \
        device const char * weights, device const char * scales, device char * dst, device float * rotated, \
        uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) { \
    threadgroup float values[256]; \
    convrot_rotate_activation<T>(args, activations, rotated, values, group, tid); \
} \
[[host_name("kernel_mul_mat_convrot_i8mm_" #suffix)]] \
kernel void kernel_mul_mat_convrot_i8mm_##suffix( \
        constant ggml_metal_kargs_mul_mat_convrot & args, device const char * activations, \
        device const char * weights, device const char * scales, device char * dst, device const float * rotated, \
        uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], \
        ushort sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float shared[3072]; \
    convrot_i8mm(args, weights, scales, dst, rotated, shared, group, tid, sg); \
}
CONVROT_NATIVE_KERNELS(float, f32)
CONVROT_NATIVE_KERNELS(half, f16)
#undef CONVROT_NATIVE_KERNELS

// F16 compatibility: each threadgroup reconstructs one 16-row weight tile once
// and reuses it across 16 activation columns, in 32 KiB of threadgroup memory.
template <typename T>
inline void kernel_mul_mat_convrot_mm_impl(
        constant ggml_metal_kargs_mul_mat_convrot & args,
        device const char * activations, device const char * weights,
        device const char * scales, device char * dst,
        threadgroup float * scratch, uint3 group, ushort tid, ushort sg) {
    #pragma clang fp reassociate(off) contract(off)
    threadgroup float * w = scratch;
    threadgroup float * a = scratch + 4096;
    const uint row0 = group.x * 16;
    const uint col0 = group.y * 16;
    const uint i2 = group.z % args.ne02;
    const uint i3 = group.z / args.ne02;
    simdgroup_float8x8 accum = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint k0 = 0; k0 < uint(args.k); k0 += 256) {
        for (uint i = tid; i < 4096; i += 128) {
            const uint row = row0 + i/256;
            const uint col = col0 + i/256;
            const uint k = k0 + i%256;
            w[i] = row < uint(args.out_features)
                ? float(*((device const int8_t *)(weights + row*args.nb11 + k*args.nb10))) *
                    *((device const float *)(scales + row*args.nb20)) : 0.0f;
            const float x = col < uint(args.ne01) ? float(*((device const T *)(activations +
                k*args.nb00 + col*args.nb01 + i2*args.nb02 + i3*args.nb03))) : 0.0f;
            a[i] = args.f16_compat ? float(half(x)) : x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < 256; stride *= 4) {
            for (uint i = tid; i < 1024; i += 128) {
                const uint j = i/64*256 + (i%64)/stride*(4*stride) + i%stride;
                const float v0 = w[j], v1 = w[j + stride], v2 = w[j + 2*stride], v3 = w[j + 3*stride];
                w[j]            = ( v0 + v1 + v2 - v3)*0.5f;
                w[j + stride]   = ( v0 + v1 - v2 + v3)*0.5f;
                w[j + 2*stride] = ( v0 - v1 + v2 + v3)*0.5f;
                w[j + 3*stride] = (-v0 + v1 + v2 + v3)*0.5f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (args.f16_compat) {
            for (uint i = tid; i < 4096; i += 128) {
                w[i] = float(half(w[i]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint k = 0; k < 256; k += 8) {
            simdgroup_float8x8 ma, mb;
            simdgroup_load(ma, w + (sg/2)*8*256 + k, 256);
            simdgroup_load(mb, a + (sg%2)*8*256 + k, 256, 0, true);
            simdgroup_multiply_accumulate(accum, ma, mb, accum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accum, scratch + (sg/2)*8*16 + (sg%2)*8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 256; i += 128) {
        const uint row = row0 + i/16, col = col0 + i%16;
        if (row < uint(args.out_features) && col < uint(args.ne01)) {
            float value = scratch[i];
            if (args.f16_compat) {
                value = float(half(value));
            }
            *((device float *)(dst + row*args.nb0 + col*args.nb1 + i2*args.nb2 + i3*args.nb3)) = value;
        }
    }
}

[[host_name("kernel_mul_mat_convrot_mm_f32")]]
kernel void kernel_mul_mat_convrot_mm_f32(
        constant ggml_metal_kargs_mul_mat_convrot & args,
        device const char * activations, device const char * weights, device const char * scales, device char * dst,
        uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8192];
    kernel_mul_mat_convrot_mm_impl<float>(args, activations, weights, scales, dst, scratch, group, tid, sg);
}

[[host_name("kernel_mul_mat_convrot_mm_f16")]]
kernel void kernel_mul_mat_convrot_mm_f16(
        constant ggml_metal_kargs_mul_mat_convrot & args,
        device const char * activations, device const char * weights, device const char * scales, device char * dst,
        uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8192];
    kernel_mul_mat_convrot_mm_impl<half>(args, activations, weights, scales, dst, scratch, group, tid, sg);
}
