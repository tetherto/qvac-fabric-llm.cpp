// Integer IEEE conversion fixes the rounding mode independently of the device's
// floating-point controls. ConvRot compatibility requires round-to-nearest-even.
uint convrot_f16_rne(float value) {
    const uint bits = floatBitsToUint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u : 0u);
    }
    if (magnitude >= 0x477ff000u) return sign | 0x7c00u;
    if (magnitude <= 0x33000000u) return sign;
    if (magnitude < 0x38800000u) {
        const uint shift = 126u - (magnitude >> 23);
        const uint mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        uint rounded = mantissa >> shift;
        const uint remainder = mantissa & ((1u << shift) - 1u);
        const uint midpoint = 1u << (shift - 1u);
        rounded += uint(remainder > midpoint || (remainder == midpoint && (rounded & 1u) != 0));
        return sign | rounded;
    }
    const uint rounded = magnitude + 0xfffu + ((magnitude >> 13) & 1u);
    return sign | ((rounded - 0x38000000u) >> 13);
}

uint convrot_pack_f16_rne(vec2 value) {
    return convrot_f16_rne(value.x) | (convrot_f16_rne(value.y) << 16);
}
