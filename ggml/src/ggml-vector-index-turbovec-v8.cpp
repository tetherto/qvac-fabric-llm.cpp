// ggml-vector-index-turbovec-v8.cpp - upstream turbovec v7/v8 container for
// TurboVec 1.x indexes.
//
// The layout follows upstream turbovec 1.1.2 `io_v7.rs` byte for byte, so a
// snapshot written here loads in upstream turbovec and the reverse:
//
//   [superblock]  magic "TV8\0", revision, bit width, kind (1 = id-mapped),
//                 unit layout, dim, nonce, ops capacity, codebook, TQ+
//                 calibration, CRC
//   [header A]    commit slot: generation, row count, the partial tail
//                 block's rows, pending redo ops, delta descriptor, CRC
//   [header B]    the alternate commit slot
//   [units]       one per whole 32-row block: codes, 32 scales, 32 ids
//
// Header slots have a fixed size that reserves room for 1024 redo ops, so a
// snapshot carries mostly-zero header padding. The writer leaves that
// padding as a file hole where the filesystem supports it.

#include "ggml-vector-index-impl.h"

namespace {

constexpr uint8_t kV7Magic[4] = { 'T', 'V', '7', '\0' };
constexpr uint8_t kV8Magic[4] = { 'T', 'V', '8', '\0' };
constexpr uint8_t kV7Revision = 2;
constexpr uint8_t kV8Revision = 1;
constexpr uint8_t kLayoutSeq = 0;
constexpr uint8_t kLayoutPlanes = 1;
constexpr uint8_t kKindPositional = 0;
constexpr uint8_t kKindIdMap = 1;
constexpr size_t kBlock = 32;
// Header slots reserve this many redo ops; it sizes every offset in the
// file and is stamped into the superblock (upstream MAX_OPS).
constexpr size_t kMaxOps = 1024;
// A per-vector scale above this could drive a score to infinity on its own
// (upstream io.rs MAX_VECTOR_SCALE).
constexpr float kMaxVectorScale = 1e22f;

uint32_t crc32c_raw(uint32_t crc, const uint8_t * data, size_t size) {
    return crc32c_update(crc, data, size);
}

uint32_t crc32_one(const uint8_t * data, size_t size) {
    return ~crc32c_raw(0xffffffffu, data, size);
}

// Upstream `crc32`: inputs of 4096 bytes or more checksum as three
// length-derived thirds, and the stored value is the CRC of the digests.
class Crc32Digest {
public:
    explicit Crc32Digest(size_t total) : total(total) {
        if (total >= 4096) {
            split1 = total / 3;
            split2 = split1 * 2;
        }
    }

    void push(const uint8_t * data, size_t size) {
        if (total < 4096) {
            parts[0] = crc32c_raw(parts[0], data, size);
            at += size;
            return;
        }
        while (size != 0) {
            const int part = at < split1 ? 0 : (at < split2 ? 1 : 2);
            const size_t end = part == 0 ? split1 : (part == 1 ? split2 : SIZE_MAX);
            const size_t take = std::min(size, end - at);
            parts[part] = crc32c_raw(parts[part], data, take);
            at += take;
            data += take;
            size -= take;
        }
    }

    uint32_t finish() const {
        if (total < 4096) {
            return ~parts[0];
        }
        uint8_t digest[12];
        for (int i = 0; i < 3; ++i) {
            put_u32_le(digest + 4 * i, ~parts[i]);
        }
        return crc32_one(digest, sizeof(digest));
    }

private:
    size_t total = 0;
    size_t at = 0;
    size_t split1 = 0;
    size_t split2 = 0;
    uint32_t parts[3] = { 0xffffffffu, 0xffffffffu, 0xffffffffu };
};

uint32_t crc32_upstream(const uint8_t * data, size_t size) {
    Crc32Digest digest(size);
    digest.push(data, size);
    return digest.finish();
}

// Offsets of a v7/v8 file all derive from this geometry (upstream `Geo`).
struct Geo {
    uint8_t kind = kKindIdMap;
    size_t dim = 0;
    int bits = 0;
    size_t n_calib = 0;
    int version = 8;
    uint8_t layout = kLayoutSeq;

    size_t sb_fixed() const { return version == 8 ? 24 : 23; }
    size_t n_levels() const { return size_t{ 1 } << bits; }
    // Bytes per row in the sequential-blocked layout: one byte per group of
    // 8 / bits codes (3-bit codes take 4-bit fields).
    size_t row_bytes() const { return dim / static_cast<size_t>(8 / bits); }
    size_t id_bytes(size_t rows) const { return kind == kKindIdMap ? rows * 8 : 0; }
    size_t sb_len() const { return sb_fixed() + (n_levels() - 1) * 4 + n_levels() * 4 + 4 + n_calib * 8 + 4; }
    size_t op_size() const { return 1 + row_bytes() + 4 + id_bytes(1); }
    size_t hdr_len() const {
        return 16 + 31 * (row_bytes() + 4 + id_bytes(1)) + 4 + kMaxOps * (5 + op_size()) + 4 + kMaxOps * 4 + 12 + 4;
    }
    size_t hdr_at(size_t slot) const { return sb_len() + slot * hdr_len(); }
    size_t unit_len() const { return kBlock * row_bytes() + kBlock * 4 + id_bytes(kBlock); }
    size_t unit_at(size_t block) const { return hdr_at(2) + block * unit_len(); }
};

void push_u32(std::vector<uint8_t> & out, uint32_t v) {
    uint8_t b[4];
    put_u32_le(b, v);
    out.insert(out.end(), b, b + 4);
}

void push_u64(std::vector<uint8_t> & out, uint64_t v) {
    uint8_t b[8];
    put_u64_le(b, v);
    out.insert(out.end(), b, b + 8);
}

void push_f32(std::vector<uint8_t> & out, float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    push_u32(out, bits);
}

float get_f32_le(const uint8_t * src) {
    const uint32_t bits = get_u32_le(src);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Upstream io.rs calibration bounds: a shift or scale outside them turns
// every score into Inf/NaN.
float max_tqplus_shift(size_t dim) {
    const float cap = FLT_MAX / (static_cast<float>(std::max<size_t>(dim, 1)) * kTurboVecMaxInputMagnitude);
    return cap / 10.0f;
}

float min_tqplus_scale(size_t dim) {
    const float need = static_cast<float>(std::max<size_t>(dim, 1)) * kTurboVecMaxInputMagnitude / FLT_MAX;
    return need * 10.0f;
}

bool valid_vector_scale(float v) {
    return std::isfinite(v) && v >= 0.0f && v <= kMaxVectorScale;
}

// Planes geometry (upstream pack.rs): sign-plane bytes per row and the low
// planes' bytes per row.
size_t planes_nsg(const Geo & geo) {
    return geo.dim / 8;
}

bool planes_geom_ok(const Geo & geo) {
    const size_t nbg = geo.row_bytes();
    return (geo.bits == 2 && nbg % 8 == 0) || (geo.bits == 4 && nbg % 16 == 0);
}

}  // namespace

bool turbovec_v8_magic(const uint8_t * bytes, size_t size) {
    return size >= 4 && (std::memcmp(bytes, kV8Magic, 4) == 0 || std::memcmp(bytes, kV7Magic, 4) == 0);
}

void turbovec_v8_build_image(const ggml_vec_index & idx, TurboVecV8Image & image) {
    Geo geo;
    geo.kind = kKindIdMap;
    geo.dim = static_cast<size_t>(idx.dim);
    geo.bits = idx.turbovec_bits;
    geo.n_calib = idx.turbovec_tqplus_shift.size();

    const size_t packed_bytes = turbovec_row_bytes(geo.bits, geo.dim);
    const size_t row_bytes = geo.row_bytes();
    std::vector<size_t> live;
    live.reserve(active_count(idx));
    for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
        if (slot_is_active(idx, slot)) {
            live.push_back(slot);
        }
    }
    const size_t n = live.size();

    std::vector<float> boundaries;
    std::vector<float> centroids;
    turbovec_codebook_values(geo.bits, idx.dim, false, boundaries, centroids);

    std::vector<uint8_t> & sb = image.superblock;
    sb.clear();
    sb.insert(sb.end(), kV8Magic, kV8Magic + 4);
    sb.push_back(kV8Revision);
    sb.push_back(static_cast<uint8_t>(geo.bits));
    sb.push_back(geo.kind);
    sb.push_back(kLayoutSeq);
    push_u32(sb, static_cast<uint32_t>(geo.dim));
    // An unclaimed snapshot: no index is syncing to it.
    push_u64(sb, 0);
    push_u32(sb, static_cast<uint32_t>(kMaxOps));
    for (float v : boundaries) {
        push_f32(sb, v);
    }
    for (float v : centroids) {
        push_f32(sb, v);
    }
    push_u32(sb, static_cast<uint32_t>(geo.n_calib));
    for (float v : idx.turbovec_tqplus_shift) {
        push_f32(sb, v);
    }
    for (float v : idx.turbovec_tqplus_scale) {
        push_f32(sb, v);
    }
    push_u32(sb, crc32_upstream(sb.data(), sb.size()));

    std::vector<uint8_t> seq(row_bytes);
    auto seq_row = [&](size_t slot) {
        turbovec_packed_row_to_seq(
            idx.turbovec_data.data() + slot * packed_bytes, geo.bits, idx.dim, seq.data());
        return seq.data();
    };

    // Header slot 0: generation 0, every row, the tail rows, no pending ops
    // and an empty delta whose digest covers just the generation.
    std::vector<uint8_t> & h = image.header0;
    h.clear();
    push_u64(h, 0);
    push_u64(h, static_cast<uint64_t>(n));
    const size_t first_tail = n - n % kBlock;
    for (size_t r = first_tail; r < n; ++r) {
        const size_t slot = live[r];
        const uint8_t * codes = seq_row(slot);
        h.insert(h.end(), codes, codes + row_bytes);
        push_f32(h, idx.turbovec_scale[slot]);
        push_u64(h, idx.slot_to_id[slot]);
    }
    push_u32(h, 0);
    push_u32(h, 0);
    push_u32(h, 0);
    push_u32(h, 0);
    uint8_t gen_bytes[8] = {};
    push_u32(h, crc32_upstream(gen_bytes, sizeof(gen_bytes)));
    push_u32(h, crc32_upstream(h.data(), h.size()));
    image.header_len = geo.hdr_len();

    // Whole 32-row blocks: sequential-blocked codes (group-major across the
    // block's lanes), then 32 scales, then 32 ids.
    const size_t n_blocks = n / kBlock;
    std::vector<uint8_t> & units = image.units;
    units.assign(n_blocks * geo.unit_len(), 0);
    for (size_t block = 0; block < n_blocks; ++block) {
        uint8_t * unit = units.data() + block * geo.unit_len();
        for (size_t lane = 0; lane < kBlock; ++lane) {
            const size_t slot = live[block * kBlock + lane];
            const uint8_t * codes = seq_row(slot);
            for (size_t g = 0; g < row_bytes; ++g) {
                unit[g * kBlock + lane] = codes[g];
            }
            uint32_t scale_bits = 0;
            std::memcpy(&scale_bits, &idx.turbovec_scale[slot], sizeof(scale_bits));
            put_u32_le(unit + kBlock * row_bytes + lane * 4, scale_bits);
            put_u64_le(unit + kBlock * row_bytes + kBlock * 4 + lane * 8, idx.slot_to_id[slot]);
        }
    }
}

size_t turbovec_v8_image_size(const TurboVecV8Image & image) {
    return image.superblock.size() + 2 * image.header_len + image.units.size();
}

void turbovec_v8_flatten_image(const TurboVecV8Image & image, std::vector<uint8_t> & out) {
    out.assign(turbovec_v8_image_size(image), 0);
    std::memcpy(out.data(), image.superblock.data(), image.superblock.size());
    std::memcpy(out.data() + image.superblock.size(), image.header0.data(), image.header0.size());
    if (!image.units.empty()) {
        std::memcpy(
            out.data() + image.superblock.size() + 2 * image.header_len,
            image.units.data(),
            image.units.size());
    }
}

namespace {

struct ParsedHeader {
    uint64_t gen = 0;
    size_t n = 0;
    size_t tail_at = 0;
    // (slot, payload offset) per pending redo op.
    std::vector<std::pair<size_t, size_t>> ops;
    std::vector<size_t> delta_mat;
    size_t delta_app_from = 0;
    size_t delta_app_to = 0;
    uint32_t delta_crc = 0;
};

bool parse_header(const std::vector<uint8_t> & raw, const Geo & geo, size_t slot, ParsedHeader & out) {
    const size_t at = geo.hdr_at(slot);
    const size_t hdr_len = geo.hdr_len();
    if (at > raw.size() || raw.size() - at < hdr_len) {
        return false;
    }
    const uint8_t * bytes = raw.data() + at;
    const size_t tail_row = geo.row_bytes() + 4 + geo.id_bytes(1);
    const size_t op_size = geo.op_size();
    auto need = [&](size_t p, size_t len) { return p <= hdr_len && hdr_len - p >= len; };

    out.gen = get_u64_le(bytes);
    if (out.gen % 2 != slot) {
        return false;
    }
    const uint64_t n64 = get_u64_le(bytes + 8);
    if (n64 > std::numeric_limits<size_t>::max()) {
        return false;
    }
    out.n = static_cast<size_t>(n64);
    const size_t n_blocks = out.n / kBlock;
    if (n_blocks > (std::numeric_limits<size_t>::max() - geo.unit_at(0)) / geo.unit_len() ||
        geo.unit_at(0) + n_blocks * geo.unit_len() > raw.size()) {
        return false;
    }
    size_t p = 16 + (out.n % kBlock) * tail_row;
    if (!need(p, 4)) {
        return false;
    }
    const size_t n_units = get_u32_le(bytes + p);
    if (n_units > kMaxOps) {
        return false;
    }
    p += 4;
    out.ops.clear();
    for (size_t u = 0; u < n_units; ++u) {
        if (!need(p, 5)) {
            return false;
        }
        const size_t block = get_u32_le(bytes + p);
        // Ops only target committed blocks.
        if (block >= n_blocks) {
            return false;
        }
        const size_t n_ops = bytes[p + 4];
        p += 5;
        for (size_t k = 0; k < n_ops; ++k) {
            if (!need(p, op_size)) {
                return false;
            }
            const size_t lane = bytes[p];
            if (lane >= kBlock) {
                return false;
            }
            out.ops.emplace_back(block * kBlock + lane, at + p + 1);
            p += op_size;
        }
    }
    if (!need(p, 4)) {
        return false;
    }
    const size_t n_mat = get_u32_le(bytes + p);
    if (n_mat > kMaxOps) {
        return false;
    }
    p += 4;
    out.delta_mat.clear();
    for (size_t k = 0; k < n_mat; ++k) {
        if (!need(p, 4)) {
            return false;
        }
        out.delta_mat.push_back(get_u32_le(bytes + p));
        p += 4;
    }
    if (!need(p, 16)) {
        return false;
    }
    out.delta_app_from = get_u32_le(bytes + p);
    out.delta_app_to = get_u32_le(bytes + p + 4);
    out.delta_crc = get_u32_le(bytes + p + 8);
    p += 12;
    if (crc32_upstream(bytes, p) != get_u32_le(bytes + p)) {
        return false;
    }
    out.tail_at = at + 16;
    return true;
}

// A commit is adopted only if every unit its sync wrote is present with the
// bytes it recorded (upstream `delta_verified`).
bool delta_verified(const std::vector<uint8_t> & raw, const Geo & geo, const ParsedHeader & h) {
    const size_t unit_len = geo.unit_len();
    const size_t n_app = h.delta_app_to > h.delta_app_from ? h.delta_app_to - h.delta_app_from : 0;
    const size_t count = h.delta_mat.size() + n_app;
    Crc32Digest digest(8 + count * (4 + unit_len));
    uint8_t gen_bytes[8];
    put_u64_le(gen_bytes, h.gen);
    digest.push(gen_bytes, sizeof(gen_bytes));
    auto feed = [&](size_t block) {
        const size_t at = geo.unit_at(block);
        if (at > raw.size() || raw.size() - at < unit_len) {
            return false;
        }
        uint8_t index_bytes[4];
        put_u32_le(index_bytes, static_cast<uint32_t>(block));
        digest.push(index_bytes, sizeof(index_bytes));
        digest.push(raw.data() + at, unit_len);
        return true;
    };
    for (size_t block : h.delta_mat) {
        if (!feed(block)) {
            return false;
        }
    }
    for (size_t block = h.delta_app_from; block < h.delta_app_to; ++block) {
        if (!feed(block)) {
            return false;
        }
    }
    return digest.finish() == h.delta_crc;
}

}  // namespace

ggml_vec_index_t * turbovec_v8_load_image(std::vector<uint8_t> raw, int & error) {
    error = GGML_VEC_INDEX_E_IO;
    if (raw.size() < 12 || !turbovec_v8_magic(raw.data(), raw.size())) {
        error = GGML_VEC_INDEX_E_BAD_MAGIC;
        return nullptr;
    }
    Geo geo;
    const bool v8 = std::memcmp(raw.data(), kV8Magic, 4) == 0;
    geo.version = v8 ? 8 : 7;
    if (raw[4] != (v8 ? kV8Revision : kV7Revision)) {
        error = GGML_VEC_INDEX_E_BAD_VERSION;
        return nullptr;
    }
    geo.bits = raw[5];
    if (geo.bits < 2 || geo.bits > 4) {
        return nullptr;
    }
    geo.kind = raw[6];
    // A positional (TurboQuantIndex) file has no ids to map.
    if (geo.kind != kKindIdMap) {
        error = geo.kind == kKindPositional ? GGML_VEC_INDEX_E_INVALID_ARG : GGML_VEC_INDEX_E_IO;
        return nullptr;
    }
    geo.layout = v8 ? raw[7] : kLayoutSeq;
    if (geo.layout != kLayoutSeq && geo.layout != kLayoutPlanes) {
        return nullptr;
    }
    if (geo.layout == kLayoutPlanes && geo.bits == 3) {
        return nullptr;
    }
    if (raw.size() < geo.sb_fixed()) {
        return nullptr;
    }
    geo.dim = get_u32_le(raw.data() + geo.sb_fixed() - 16);
    // dim 0 is upstream's lazy sentinel; this API has no dim-less index.
    if (geo.dim == 0 || geo.dim % 8 != 0 || geo.dim > static_cast<size_t>(kTurboVecMaxDim)) {
        return nullptr;
    }
    // A complete image reserves kMaxOps redo slots, each at least one row
    // wide: reject a truncated file before the O(dim) codebook solve.
    if (raw.size() < kMaxOps * (geo.dim * static_cast<size_t>(geo.bits) / 8)) {
        return nullptr;
    }
    if (get_u32_le(raw.data() + geo.sb_fixed() - 4) != kMaxOps) {
        error = GGML_VEC_INDEX_E_BAD_VERSION;
        return nullptr;
    }

    // The embedded codebook must be the canonical one: a drifted codebook
    // silently mis-scores (upstream #320).
    std::vector<float> boundaries;
    std::vector<float> centroids;
    turbovec_codebook_values(geo.bits, static_cast<int>(geo.dim), false, boundaries, centroids);
    size_t off = geo.sb_fixed();
    const size_t n_levels = geo.n_levels();
    if (raw.size() < off + (2 * n_levels - 1) * 4 + 4) {
        return nullptr;
    }
    for (const std::vector<float> * values : { &boundaries, &centroids }) {
        for (float expected : *values) {
            if (get_f32_le(raw.data() + off) != expected) {
                return nullptr;
            }
            off += 4;
        }
    }
    geo.n_calib = get_u32_le(raw.data() + off);
    off += 4;
    if (geo.n_calib != 0 && geo.n_calib != geo.dim) {
        return nullptr;
    }
    if (raw.size() < off + geo.n_calib * 8 + 4) {
        return nullptr;
    }
    std::vector<float> shift(geo.n_calib);
    std::vector<float> scale(geo.n_calib);
    for (size_t k = 0; k < geo.n_calib; ++k) {
        shift[k] = get_f32_le(raw.data() + off + k * 4);
    }
    off += geo.n_calib * 4;
    for (size_t k = 0; k < geo.n_calib; ++k) {
        scale[k] = get_f32_le(raw.data() + off + k * 4);
    }
    off += geo.n_calib * 4;
    const float shift_cap = max_tqplus_shift(geo.n_calib);
    const float scale_floor = min_tqplus_scale(geo.n_calib);
    for (size_t k = 0; k < geo.n_calib; ++k) {
        if (!std::isfinite(shift[k]) || std::fabs(shift[k]) > shift_cap ||
            !std::isfinite(scale[k]) || scale[k] < scale_floor) {
            return nullptr;
        }
    }
    if (crc32_upstream(raw.data(), off) != get_u32_le(raw.data() + off)) {
        return nullptr;
    }
    if (geo.layout == kLayoutPlanes && !planes_geom_ok(geo)) {
        return nullptr;
    }

    // Adopt the newest header whose delta verifies.
    ParsedHeader headers[2];
    bool valid[2] = {
        parse_header(raw, geo, 0, headers[0]),
        parse_header(raw, geo, 1, headers[1]),
    };
    int order[2] = { 0, 1 };
    if (valid[0] && valid[1] && headers[1].gen > headers[0].gen) {
        std::swap(order[0], order[1]);
    } else if (!valid[0] && valid[1]) {
        std::swap(order[0], order[1]);
    }
    const ParsedHeader * chosen = nullptr;
    for (int i = 0; i < 2; ++i) {
        const int slot = order[i];
        if (valid[slot] && delta_verified(raw, geo, headers[slot])) {
            chosen = &headers[slot];
            break;
        }
    }
    if (chosen == nullptr) {
        return nullptr;
    }

    const size_t n = chosen->n;
    if (n > kMaxIndexLen) {
        return nullptr;
    }
    const int dim = static_cast<int>(geo.dim);
    std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
        ggml_vec_index_create_turbovec_for_load(dim, geo.bits, false),
        ggml_vec_index_free);
    if (idx == nullptr) {
        error = GGML_VEC_INDEX_E_OOM;
        return nullptr;
    }
    const size_t row_bytes = geo.row_bytes();
    const size_t packed_bytes = turbovec_row_bytes(geo.bits, geo.dim);
    const size_t nsg = planes_nsg(geo);
    const size_t low_row = static_cast<size_t>(geo.bits - 1) * nsg;
    idx->turbovec_data.assign(n * packed_bytes, 0);
    idx->turbovec_scale.assign(n, 0.0f);
    idx->slot_to_id.assign(n, 0);

    // A sequential row (one byte per code group) or a planes row (low
    // planes, then the sign plane) into the packed bit-plane row.
    auto put_seq_row = [&](size_t slot, const uint8_t * codes) {
        turbovec_seq_row_to_packed(codes, geo.bits, dim, idx->turbovec_data.data() + slot * packed_bytes);
    };
    auto put_row_codes = [&](size_t slot, const uint8_t * codes) {
        // Tail rows and redo ops always carry sequential row codes.
        put_seq_row(slot, codes);
    };

    const size_t n_blocks = n / kBlock;
    std::vector<uint8_t> codes(row_bytes);
    for (size_t block = 0; block < n_blocks; ++block) {
        const uint8_t * unit = raw.data() + geo.unit_at(block);
        for (size_t lane = 0; lane < kBlock; ++lane) {
            const size_t slot = block * kBlock + lane;
            uint8_t * packed = idx->turbovec_data.data() + slot * packed_bytes;
            if (geo.layout == kLayoutPlanes) {
                // Canonical planes unit: sign bytes group-major across the
                // block's lanes, then each lane's low planes.
                std::memcpy(packed, unit + nsg * kBlock + lane * low_row, low_row);
                for (size_t g = 0; g < nsg; ++g) {
                    packed[low_row + g] = unit[g * kBlock + lane];
                }
            } else {
                for (size_t g = 0; g < row_bytes; ++g) {
                    codes[g] = unit[g * kBlock + lane];
                }
                put_seq_row(slot, codes.data());
            }
            const float v = get_f32_le(unit + kBlock * row_bytes + lane * 4);
            if (!valid_vector_scale(v)) {
                return nullptr;
            }
            idx->turbovec_scale[slot] = v;
            idx->slot_to_id[slot] = get_u64_le(unit + kBlock * row_bytes + kBlock * 4 + lane * 8);
        }
    }
    // Pending redo ops override their slots.
    for (const auto & op : chosen->ops) {
        const size_t slot = op.first;
        const uint8_t * payload = raw.data() + op.second;
        put_row_codes(slot, payload);
        const float v = get_f32_le(payload + row_bytes);
        if (!valid_vector_scale(v)) {
            return nullptr;
        }
        idx->turbovec_scale[slot] = v;
        idx->slot_to_id[slot] = get_u64_le(payload + row_bytes + 4);
    }
    const size_t tail_row = row_bytes + 4 + 8;
    for (size_t k = 0; k < n % kBlock; ++k) {
        const size_t slot = n_blocks * kBlock + k;
        const uint8_t * row = raw.data() + chosen->tail_at + k * tail_row;
        put_row_codes(slot, row);
        const float v = get_f32_le(row + row_bytes);
        if (!valid_vector_scale(v)) {
            return nullptr;
        }
        idx->turbovec_scale[slot] = v;
        idx->slot_to_id[slot] = get_u64_le(row + row_bytes + 4);
    }

    // Duplicate or reserved ids mean a corrupt file.
    idx->id_to_slot.reserve(n);
    for (size_t slot = 0; slot < n; ++slot) {
        const uint64_t id = idx->slot_to_id[slot];
        if (!is_valid_id(id) || !idx->id_to_slot.emplace(id, slot).second) {
            return nullptr;
        }
    }
    idx->slot_active.assign(n, 1);
    idx->n_active = n;
    idx->turbovec_tqplus_shift.swap(shift);
    idx->turbovec_tqplus_scale.swap(scale);
    repack_turbovec_codes(
        idx->turbovec_data.data(),
        n,
        geo.bits,
        dim,
        idx->turbovec_blocked_data,
        idx->turbovec_blocked_n_blocks);
    rebuild_state_hash(*idx);
    error = GGML_VEC_INDEX_OK;
    return idx.release();
}
