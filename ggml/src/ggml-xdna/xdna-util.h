#pragma once

// Small helpers shared by more than one translation unit. Anything used by a
// single one stays local to it.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// f32 -> bf16, round to nearest even. The device side is bf16 throughout.
static inline uint16_t xdna_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (uint16_t) ((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

// Value of an integer environment variable, or `def` when unset.
static inline int xdna_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// True when the variable is unset or set to a non-zero number.
static inline bool xdna_env_on(const char * name) {
    const char * v = getenv(name);
    return v == nullptr || atoi(v) != 0;
}
