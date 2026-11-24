#pragma once

// SHA-256 (FIPS 180-4). Portable, no platform crypto: the hint and cache keys
// hashed with it must be identical on Linux and Apple.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace detail {

class Sha256 {
public:
    Sha256();
    void update(const void * data, size_t len);
    void update(std::string_view s) { update(s.data(), s.size()); }
    // Field encoders of the hashed keys (little-endian, length-prefixed strings).
    void u32le(uint32_t v);
    void u64le(uint64_t v);
    void i64le(int64_t v) { u64le(static_cast<uint64_t>(v)); }
    void str(std::string_view s);  // u32le(len) || bytes
    std::array<uint8_t, 32> finish();

private:
    void block(const uint8_t * p);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t buf_len_ = 0;
    uint64_t total_ = 0;
};

std::array<uint8_t, 32> sha256(const void * data, size_t len);
std::string to_hex(const uint8_t * data, size_t len);
inline std::string to_hex(const std::array<uint8_t, 32> & d) { return to_hex(d.data(), d.size()); }

}  // namespace detail
}  // namespace QVAC_COREML_NS
