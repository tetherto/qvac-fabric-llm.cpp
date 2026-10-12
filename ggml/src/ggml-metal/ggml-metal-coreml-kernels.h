#pragma once

static const char * ggml_metal_coreml_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void pack_q8(device const uchar * src [[buffer(0)]],
                    device atomic_uint * bad [[buffer(1)]],
                    texture2d<half, access::write> dst [[texture(0)]],
                    uint2 p [[thread_position_in_grid]]) {
    if (p.x >= dst.get_width() || p.y >= dst.get_height()) { return; }
    const uint block = (p.y * dst.get_width() + p.x) / 32;
    const ushort bits = ushort(src[34*block]) | (ushort(src[34*block + 1]) << 8);
    const float value = float(as_type<half>(bits)) * float(as_type<char>(src[34*block + 2 + p.x%32]));
    if (!isfinite(value) || abs(value) > 65504.0f) { atomic_store_explicit(bad, 1u, memory_order_relaxed); }
    dst.write(half4(half(value)), p);
}

kernel void pack_f32(device const float * src [[buffer(0)]],
                     device atomic_uint * bad [[buffer(1)]],
                     texture2d<half, access::write> dst [[texture(0)]],
                     uint2 p [[thread_position_in_grid]]) {
    if (p.x >= dst.get_width() || p.y >= dst.get_height()) { return; }
    const float value = src[p.y * dst.get_width() + p.x];
    if (!isfinite(value) || abs(value) > 65504.0f) { atomic_store_explicit(bad, 1u, memory_order_relaxed); }
    dst.write(half4(half(value)), p);
}

kernel void pack_f16(device const half * src [[buffer(0)]],
                     device atomic_uint * bad [[buffer(1)]],
                     texture2d<half, access::write> dst [[texture(0)]],
                     uint2 p [[thread_position_in_grid]]) {
    if (p.x >= dst.get_width() || p.y >= dst.get_height()) { return; }
    const half value = src[p.y * dst.get_width() + p.x];
    if (!isfinite(value)) { atomic_store_explicit(bad, 1u, memory_order_relaxed); }
    dst.write(half4(value), p);
}

kernel void unpack_f32(device float * dst [[buffer(0)]],
                       texture2d<half, access::read> src [[texture(0)]],
                       uint2 p [[thread_position_in_grid]]) {
    if (p.x >= src.get_width() || p.y >= src.get_height()) { return; }
    dst[p.y * src.get_width() + p.x] = float(src.read(p).x);
}
)metal";
