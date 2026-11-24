#include "milblob.h"

#include <algorithm>
#include <cstring>

namespace QVAC_COREML_NS {
namespace milblob {

namespace {

void put_u32(uint8_t * p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
void put_u64(uint8_t * p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
uint32_t get_u32(const uint8_t * p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
uint64_t get_u64(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

const uint8_t kZeros[ALIGN] = {};

}  // namespace

Writer::Writer(uint32_t count, uint64_t total_size, Out out) : count_(count), total_(total_size), out_(std::move(out)) {}

Error Writer::emit(const uint8_t * data, size_t n) {
    if (n == 0) return {};
    Error e = out_(data, n);
    if (e.ok()) pos_ += n;
    return e;
}

Error Writer::pad_to(uint64_t target) {
    while (pos_ < target) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(ALIGN, target - pos_));
        Error e = emit(kZeros, n);
        if (!e.ok()) return e;
    }
    return {};
}

Error Writer::begin() {
    if (begun_) return Error::deterministic("milblob: header written twice");
    begun_ = true;
    uint8_t h[HEADER_BYTES] = {};
    put_u32(h, count_);
    put_u32(h + 4, VERSION);
    data_end_ = HEADER_BYTES;
    return emit(h, sizeof(h));
}

Error Writer::blob(uint64_t meta_offset, uint32_t dtype, const uint8_t * data, uint64_t bytes) {
    if (!begun_) return Error::deterministic("milblob: blob before header");
    if (n_ >= count_) {
        return Error::deterministic("milblob: blob " + std::to_string(n_ + 1) + " exceeds the declared count " +
                                   std::to_string(count_));
    }
    const uint64_t expect = align_up(data_end_);
    if (meta_offset != expect) {
        return Error::deterministic("milblob: blob " + std::to_string(n_) + " at offset " +
                                    std::to_string(meta_offset) + ", the writer is at " + std::to_string(expect));
    }
    Error e = pad_to(expect);
    if (!e.ok()) return e;
    uint8_t m[METADATA_BYTES] = {};
    put_u32(m, SENTINEL);
    put_u32(m + 4, dtype);
    put_u64(m + 8, bytes);
    put_u64(m + 16, meta_offset + METADATA_BYTES);
    e = emit(m, sizeof(m));
    if (!e.ok()) return e;
    e = emit(data, static_cast<size_t>(bytes));
    if (!e.ok()) return e;
    data_end_ = pos_;
    ++n_;
    return {};
}

Error Writer::finish() {
    if (n_ != count_) {
        return Error::deterministic("milblob: " + std::to_string(n_) + " blobs written, the template declares " +
                                    std::to_string(count_));
    }
    if (!valid_total(data_end_, total_)) {
        return Error::deterministic("milblob: weight.bin would be " + std::to_string(align_up(data_end_)) +
                                    " bytes, the template declares " + std::to_string(total_));
    }
    return pad_to(total_);
}

std::optional<std::vector<Entry>> read_index(const uint8_t * data, size_t size, std::string * err) {
    auto fail = [&](const std::string & m) -> std::optional<std::vector<Entry>> {
        if (err) *err = m;
        return std::nullopt;
    };
    if (size < HEADER_BYTES) return fail("weight.bin shorter than its header");
    const uint32_t count = get_u32(data);
    if (get_u32(data + 4) != VERSION) return fail("weight.bin is not MILBlob version 2");
    std::vector<Entry> out;
    out.reserve(count);
    uint64_t off = HEADER_BYTES;
    uint64_t end = HEADER_BYTES;
    for (uint32_t i = 0; i < count; ++i) {
        off = align_up(end);
        if (off + METADATA_BYTES > size) return fail("blob " + std::to_string(i) + ": metadata beyond the file");
        const uint8_t * m = data + off;
        if (get_u32(m) != SENTINEL) return fail("blob " + std::to_string(i) + ": bad sentinel");
        Entry e;
        e.offset = off;
        e.dtype = get_u32(m + 4);
        e.bytes = get_u64(m + 8);
        e.data_offset = get_u64(m + 16);
        if (e.data_offset != off + METADATA_BYTES) return fail("blob " + std::to_string(i) + ": bad data offset");
        if (e.data_offset + e.bytes > size) return fail("blob " + std::to_string(i) + ": data beyond the file");
        end = e.data_offset + e.bytes;
        out.push_back(e);
    }
    if (!valid_total(end, size)) return fail("weight.bin has trailing bytes");
    return out;
}

}  // namespace milblob
}  // namespace QVAC_COREML_NS
