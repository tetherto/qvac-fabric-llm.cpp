#pragma once

// MILBlob v2 writer: the weights/weight.bin of an .mlpackage, in the layout
// coremltools writes.
//
//   [0, 64)    header:   u32le count | u32le version (2) | zeros
//   per blob:  64-byte metadata at a 64-aligned offset:
//              u32le 0xDEADBEEF | u32le dtype | u64le data bytes | u64le data offset (= metadata + 64) | zeros
//              then the data, then zero padding up to the next 64-byte boundary
//
// The recipe states every metadata offset and the final file size; the writer
// asserts each of them against its running offset, so a template whose
// weight.bin layout drifted fails deterministically instead of writing a file
// Core ML would misread.

#include "qvac_coreml/slot.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace milblob {

constexpr uint64_t ALIGN = 64;
constexpr uint64_t HEADER_BYTES = 64;
constexpr uint64_t METADATA_BYTES = 64;
constexpr uint32_t VERSION = 2;
constexpr uint32_t SENTINEL = 0xDEADBEEFu;

// MILBlob BlobDataType values.
enum MilDtype : uint32_t { FLOAT16 = 1, FLOAT32 = 2 };

inline uint64_t align_up(uint64_t v) { return (v + ALIGN - 1) / ALIGN * ALIGN; }
// The metadata offset of the blob after one at `meta_offset` with `data_bytes`.
inline uint64_t next_offset(uint64_t meta_offset, uint64_t data_bytes) {
    return align_up(meta_offset + METADATA_BYTES + data_bytes);
}
// coremltools may or may not pad after the last blob; both totals are valid.
inline bool valid_total(uint64_t last_data_end, uint64_t total) {
    return total == last_data_end || total == align_up(last_data_end);
}

class Writer {
public:
    // Bytes go to `out` in file order; its errors pass through unchanged.
    using Out = std::function<Error(const uint8_t * data, size_t n)>;

    Writer(uint32_t count, uint64_t total_size, Out out);

    Error begin();  // the header
    // Errors (deterministic): a blob beyond `count`, an offset other than the
    // running one, a data offset that would not be 64-aligned.
    Error blob(uint64_t meta_offset, uint32_t dtype, const uint8_t * data, uint64_t bytes);
    // Errors (deterministic): fewer blobs than `count`, a final size other
    // than `total_size`.
    Error finish();

    uint64_t written() const { return pos_; }
    uint32_t blobs_written() const { return n_; }

private:
    Error pad_to(uint64_t target);
    Error emit(const uint8_t * data, size_t n);

    uint32_t count_;
    uint64_t total_;
    Out out_;
    uint64_t pos_ = 0;      // bytes written
    uint64_t data_end_ = 0; // end of the last blob's data (before padding)
    uint32_t n_ = 0;
    bool begun_ = false;
};

struct Entry {
    uint64_t offset = 0;  // metadata offset
    uint32_t dtype = 0;
    uint64_t bytes = 0;
    uint64_t data_offset = 0;
};

// Parses and validates a weight.bin (verify, tests). nullopt with `err` set
// when the header, a sentinel, an offset or a size is malformed.
std::optional<std::vector<Entry>> read_index(const uint8_t * data, size_t size, std::string * err = nullptr);

}  // namespace milblob
}  // namespace QVAC_COREML_NS
