// test-coreml-milblob: the MILBlob v2 writer's header, alignment,
// offsets, size and count assertions. Portable: Linux and macOS.

#include "generator_test_util.h"

using namespace gtest;

namespace {

struct Mem {
    std::vector<uint8_t> bytes;
    milblob::Writer::Out out() {
        return [this](const uint8_t * p, size_t n) -> Error {
            bytes.insert(bytes.end(), p, p + n);
            return {};
        };
    }
};

uint32_t u32_at(const std::vector<uint8_t> & b, size_t o) {
    uint32_t v;
    std::memcpy(&v, b.data() + o, 4);
    return v;
}
uint64_t u64_at(const std::vector<uint8_t> & b, size_t o) {
    uint64_t v;
    std::memcpy(&v, b.data() + o, 8);
    return v;
}
bool zeros(const std::vector<uint8_t> & b, size_t from, size_t to) {
    for (size_t i = from; i < to; ++i) {
        if (b[i] != 0) return false;
    }
    return true;
}

void case_header() {
    Mem m;
    milblob::Writer w(0, 64, m.out());
    CHECK(w.begin().ok());
    CHECK(w.finish().ok());
    CHECK(m.bytes.size() == 64);
    CHECK(u32_at(m.bytes, 0) == 0);
    CHECK(u32_at(m.bytes, 4) == 2);  // version
    CHECK(zeros(m.bytes, 8, 64));

    Mem m2;
    milblob::Writer w2(3, 0, m2.out());
    CHECK(w2.begin().ok());
    CHECK(u32_at(m2.bytes, 0) == 3);
    CHECK(!w2.begin().ok());  // header twice
}

void case_layout() {
    // 3 fp16 (6 bytes, unaligned end) then 32 fp32 (128 bytes).
    const uint8_t a[6] = {1, 2, 3, 4, 5, 6};
    std::vector<uint8_t> b(128);
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>(i + 7);
    const uint64_t off0 = 64;
    const uint64_t off1 = milblob::next_offset(off0, 6);
    CHECK(off1 == 192);  // align64(64 + 64 + 6) = 192
    const uint64_t end = off1 + 64 + 128;
    CHECK(end == 384);
    Mem m;
    milblob::Writer w(2, end, m.out());
    CHECK(w.begin().ok());
    CHECK(w.blob(off0, milblob::FLOAT16, a, 6).ok());
    CHECK(w.blob(off1, milblob::FLOAT32, b.data(), b.size()).ok());
    CHECK(w.finish().ok());
    CHECK(m.bytes.size() == end);
    // Metadata of blob 0.
    CHECK(u32_at(m.bytes, 64) == 0xDEADBEEFu);
    CHECK(u32_at(m.bytes, 68) == milblob::FLOAT16);
    CHECK(u64_at(m.bytes, 72) == 6);
    CHECK(u64_at(m.bytes, 80) == 128);  // data offset = metadata + 64
    CHECK(zeros(m.bytes, 88, 128));
    CHECK(std::memcmp(m.bytes.data() + 128, a, 6) == 0);
    CHECK(zeros(m.bytes, 134, 192));  // padding to the next 64-byte boundary
    CHECK(u32_at(m.bytes, 192) == 0xDEADBEEFu);
    CHECK(u32_at(m.bytes, 196) == milblob::FLOAT32);
    CHECK(u64_at(m.bytes, 200) == 128);
    CHECK(u64_at(m.bytes, 208) == 256);
    CHECK(std::memcmp(m.bytes.data() + 256, b.data(), 128) == 0);

    std::string err;
    auto idx = milblob::read_index(m.bytes.data(), m.bytes.size(), &err);
    CHECK_MSG(idx.has_value(), err);
    if (idx) {
        CHECK(idx->size() == 2);
        CHECK((*idx)[0].offset == 64 && (*idx)[0].bytes == 6 && (*idx)[0].data_offset == 128);
        CHECK((*idx)[1].offset == 192 && (*idx)[1].dtype == milblob::FLOAT32);
    }
}

void case_final_padding() {
    // A last blob ending unaligned: both the unpadded and the padded total
    // are valid (coremltools' choice); anything else is not.
    const uint8_t a[2] = {9, 9};
    for (uint64_t total : {uint64_t{64 + 64 + 2}, uint64_t{192}}) {
        Mem m;
        milblob::Writer w(1, total, m.out());
        CHECK(w.begin().ok());
        CHECK(w.blob(64, milblob::FLOAT16, a, 2).ok());
        CHECK(w.finish().ok());
        CHECK(m.bytes.size() == total);
        CHECK(milblob::read_index(m.bytes.data(), m.bytes.size()).has_value());
    }
    Mem m;
    milblob::Writer w(1, 256, m.out());
    CHECK(w.begin().ok());
    CHECK(w.blob(64, milblob::FLOAT16, a, 2).ok());
    Error e = w.finish();
    CHECK(!e.ok() && e.cls == ErrorClass::deterministic);
}

void case_assertions() {
    const uint8_t a[4] = {};
    {  // wrong offset
        Mem m;
        milblob::Writer w(2, 0, m.out());
        CHECK(w.begin().ok());
        Error e = w.blob(128, milblob::FLOAT16, a, 4);
        CHECK(!e.ok() && e.cls == ErrorClass::deterministic);
        CHECK(e.message.find("running") != std::string::npos || e.message.find("writer is at") != std::string::npos);
    }
    {  // second blob at an offset that skips the alignment rule
        Mem m;
        milblob::Writer w(2, 0, m.out());
        CHECK(w.begin().ok());
        CHECK(w.blob(64, milblob::FLOAT16, a, 4).ok());
        CHECK(!w.blob(132, milblob::FLOAT16, a, 4).ok());
    }
    {  // more blobs than declared
        Mem m;
        milblob::Writer w(1, 0, m.out());
        CHECK(w.begin().ok());
        CHECK(w.blob(64, milblob::FLOAT16, a, 4).ok());
        Error e = w.blob(192, milblob::FLOAT16, a, 4);
        CHECK(!e.ok() && e.cls == ErrorClass::deterministic);
    }
    {  // fewer blobs than declared
        Mem m;
        milblob::Writer w(2, 192, m.out());
        CHECK(w.begin().ok());
        CHECK(w.blob(64, milblob::FLOAT16, a, 4).ok());
        Error e = w.finish();
        CHECK(!e.ok() && e.cls == ErrorClass::deterministic);
    }
    {  // blob before the header
        Mem m;
        milblob::Writer w(1, 0, m.out());
        CHECK(!w.blob(64, milblob::FLOAT16, a, 4).ok());
    }
    {  // an I/O error passes through with its class
        milblob::Writer w(1, 0, [](const uint8_t *, size_t) { return Error::transient("disk full", "NSPOSIXErrorDomain", 28); });
        Error e = w.begin();
        CHECK(!e.ok() && e.cls == ErrorClass::transient && e.code == 28);
    }
}

void case_read_index_rejects() {
    const uint8_t a[4] = {1, 2, 3, 4};
    Mem m;
    milblob::Writer w(1, 192, m.out());
    CHECK(w.begin().ok());
    CHECK(w.blob(64, milblob::FLOAT16, a, 4).ok());
    CHECK(w.finish().ok());
    std::vector<uint8_t> bad = m.bytes;
    bad[64] ^= 0xff;  // sentinel
    CHECK(!milblob::read_index(bad.data(), bad.size()).has_value());
    bad = m.bytes;
    bad[4] = 1;  // version
    CHECK(!milblob::read_index(bad.data(), bad.size()).has_value());
    bad = m.bytes;
    bad.resize(bad.size() + 64, 0);  // trailing bytes
    CHECK(!milblob::read_index(bad.data(), bad.size()).has_value());
    bad = m.bytes;
    bad[80] = 0;  // data offset
    CHECK(!milblob::read_index(bad.data(), bad.size()).has_value());
    bad = m.bytes;
    bad.resize(100);  // truncated
    CHECK(!milblob::read_index(bad.data(), bad.size()).has_value());
}

}  // namespace

int main(int argc, char ** argv) {
    return run_cases(argc, argv,
                     {
                         {"header", case_header},
                         {"layout", case_layout},
                         {"final-padding", case_final_padding},
                         {"assertions", case_assertions},
                         {"read-index-rejects", case_read_index_rejects},
                     },
                     "test-coreml-milblob");
}
