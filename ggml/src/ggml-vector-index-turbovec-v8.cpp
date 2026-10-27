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

#include <random>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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

namespace {

Geo index_geo(const ggml_vec_index & idx) {
    Geo geo;
    geo.kind = kKindIdMap;
    geo.dim = static_cast<size_t>(idx.dim);
    geo.bits = idx.turbovec_bits;
    geo.n_calib = idx.turbovec_tqplus_shift.size();
    // Upstream stores units as bit planes when the index would search in
    // stages at its current size (lib.rs `unit_layout`), sequential rows
    // otherwise, whatever its cache holds.
    geo.layout = turbovec_planes_wanted(idx.turbovec_bits, idx.dim, active_count(idx)) ? kLayoutPlanes : kLayoutSeq;
    return geo;
}

// Serializes an index as a file whose row r is slot r (upstream
// `SyncSource`). A TurboVec 1.x index swap-removes, so its slots are dense
// and already in upstream's row order.
class Source {
public:
    explicit Source(const ggml_vec_index & idx) :
        idx(idx), geo(index_geo(idx)),
        packed_bytes(turbovec_row_bytes(idx.turbovec_bits, static_cast<size_t>(idx.dim))),
        seq(geo.row_bytes()) {}

    const Geo & geometry() const { return geo; }
    size_t n() const { return idx.slot_to_id.size(); }

    // One row's sequential codes, scale and id: a tail row or a redo op.
    void append_row(size_t r, std::vector<uint8_t> & out) const {
        const size_t slot = r;
        const uint8_t * codes = seq_row(slot);
        out.insert(out.end(), codes, codes + geo.row_bytes());
        push_f32(out, idx.turbovec_scale[slot]);
        push_u64(out, idx.slot_to_id[slot]);
    }

    std::vector<uint8_t> superblock(uint64_t nonce) const {
        std::vector<float> boundaries;
        std::vector<float> centroids;
        turbovec_codebook_values(geo.bits, idx.dim, false, boundaries, centroids);
        std::vector<uint8_t> sb;
        sb.reserve(geo.sb_len());
        sb.insert(sb.end(), kV8Magic, kV8Magic + 4);
        sb.push_back(kV8Revision);
        sb.push_back(static_cast<uint8_t>(geo.bits));
        sb.push_back(geo.kind);
        sb.push_back(geo.layout);
        push_u32(sb, static_cast<uint32_t>(geo.dim));
        push_u64(sb, nonce);
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
        return sb;
    }

    // One commit header's used prefix (upstream `header_slot`): the tail
    // rows, the pending redo ops grouped per unit, the delta descriptor.
    std::vector<uint8_t> header(
            uint64_t gen,
            const std::vector<std::pair<size_t, std::vector<size_t>>> & groups,
            const std::vector<size_t> & materialized,
            size_t append_from,
            size_t append_to,
            uint32_t delta_crc) const {
        const size_t count = n();
        std::vector<uint8_t> h;
        push_u64(h, gen);
        push_u64(h, static_cast<uint64_t>(count));
        for (size_t r = count - count % kBlock; r < count; ++r) {
            append_row(r, h);
        }
        push_u32(h, static_cast<uint32_t>(groups.size()));
        for (const auto & group : groups) {
            push_u32(h, static_cast<uint32_t>(group.first));
            h.push_back(static_cast<uint8_t>(group.second.size()));
            for (size_t r : group.second) {
                h.push_back(static_cast<uint8_t>(r % kBlock));
                append_row(r, h);
            }
        }
        push_u32(h, static_cast<uint32_t>(materialized.size()));
        for (size_t block : materialized) {
            push_u32(h, static_cast<uint32_t>(block));
        }
        push_u32(h, static_cast<uint32_t>(append_from));
        push_u32(h, static_cast<uint32_t>(append_to));
        push_u32(h, delta_crc);
        push_u32(h, crc32_upstream(h.data(), h.size()));
        return h;
    }

    // One whole 32-row block: sequential-blocked codes (group-major across
    // the lanes), 32 scales, 32 ids. Lanes past the last row are zero.
    void unit(size_t block, uint8_t * out) const {
        const size_t row_bytes = geo.row_bytes();
        const size_t nsg = geo.dim / 8;
        const size_t low_row = static_cast<size_t>(geo.bits - 1) * nsg;
        std::memset(out, 0, geo.unit_len());
        for (size_t lane = 0; lane < kBlock; ++lane) {
            const size_t r = block * kBlock + lane;
            if (r >= n()) {
                break;
            }
            const size_t slot = r;
            if (geo.layout == kLayoutPlanes) {
                // Canonical planes unit: the sign bytes group-major across
                // the lanes, then each lane's low planes.
                const uint8_t * packed = packed_row(slot);
                for (size_t g = 0; g < nsg; ++g) {
                    out[g * kBlock + lane] = packed[low_row + g];
                }
                std::memcpy(out + nsg * kBlock + lane * low_row, packed, low_row);
            } else {
                const uint8_t * codes = seq_row(slot);
                for (size_t g = 0; g < row_bytes; ++g) {
                    out[g * kBlock + lane] = codes[g];
                }
            }
            uint32_t scale_bits = 0;
            std::memcpy(&scale_bits, &idx.turbovec_scale[slot], sizeof(scale_bits));
            put_u32_le(out + kBlock * row_bytes + lane * 4, scale_bits);
            put_u64_le(out + kBlock * row_bytes + kBlock * 4 + lane * 8, idx.slot_to_id[slot]);
        }
    }

private:
    const uint8_t * seq_row(size_t slot) const {
        turbovec_row_seq(idx, slot, seq.data());
        return seq.data();
    }

    const uint8_t * packed_row(size_t slot) const {
        if (!turbovec_rows_in_blocked(idx)) {
            return idx.turbovec_data.data() + slot * packed_bytes;
        }
        packed.resize(packed_bytes);
        turbovec_row_packed(idx, slot, packed.data());
        return packed.data();
    }

    const ggml_vec_index & idx;
    Geo geo;
    size_t packed_bytes;
    mutable std::vector<uint8_t> seq;
    mutable std::vector<uint8_t> packed;
};

// The Source is shared with the image, which outlives its builder.
void build_image(const ggml_vec_index & idx, uint64_t nonce, TurboVecV8Image & image) {
    auto owned = std::make_shared<Source>(idx);
    const Source & src = *owned;
    const Geo & geo = src.geometry();
    image.superblock = src.superblock(nonce);
    // Generation 0 with no ops and an empty delta, whose digest covers just
    // the generation.
    const uint8_t gen_bytes[8] = {};
    image.header0 = src.header(0, {}, {}, 0, 0, crc32_upstream(gen_bytes, sizeof(gen_bytes)));
    image.header_len = geo.hdr_len();
    image.n_units = src.n() / kBlock;
    image.unit_len = geo.unit_len();
    image.unit = [owned](size_t block, uint8_t * out) { owned->unit(block, out); };
}

}  // namespace

// The rows in upstream order: the order an upstream index with the same
// add/remove history holds them in. Falls back to slot order if the
// tracked order was lost.
void turbovec_v8_build_image(const ggml_vec_index & idx, TurboVecV8Image & image) {
    build_image(idx, 0, image);
}

size_t turbovec_v8_image_size(const TurboVecV8Image & image) {
    return image.superblock.size() + 2 * image.header_len + image.n_units * image.unit_len;
}

size_t turbovec_v8_serialized_size(const ggml_vec_index & idx) {
    return index_geo(idx).unit_at(active_count(idx) / kBlock);
}

void turbovec_v8_fill_image(const TurboVecV8Image & image, uint8_t * out) {
    std::memcpy(out, image.superblock.data(), image.superblock.size());
    std::memcpy(out + image.superblock.size(), image.header0.data(), image.header0.size());
    uint8_t * units = out + image.superblock.size() + 2 * image.header_len;
    for (size_t block = 0; block < image.n_units; ++block) {
        image.unit(block, units + block * image.unit_len);
    }
}

namespace {

struct ParsedHeader {
    uint64_t gen = 0;
    size_t n = 0;
    // File offset of the tail rows.
    size_t tail_at = 0;
    // (file row, payload file offset) per pending redo op.
    std::vector<std::pair<size_t, size_t>> ops;
    std::vector<size_t> delta_mat;
    size_t delta_app_from = 0;
    size_t delta_app_to = 0;
    uint32_t delta_crc = 0;
    // Bytes of the slot the header occupies.
    size_t used = 0;
    // The header's own CRC, which tells two commits of one generation apart.
    uint32_t crc = 0;
};

// Parses the header in `bytes` (slot `slot`, at file offset `at`, at most
// one slot long) of a file `file_len` bytes long (upstream `parse_header_at`).
bool parse_header(
        const uint8_t * bytes,
        size_t size,
        size_t at,
        const Geo & geo,
        size_t slot,
        size_t file_len,
        ParsedHeader & out) {
    const size_t tail_row = geo.row_bytes() + 4 + geo.id_bytes(1);
    const size_t op_size = geo.op_size();
    auto need = [&](size_t p, size_t len) { return p <= size && size - p >= len; };
    if (!need(0, 16)) {
        return false;
    }
    out.gen = get_u64_le(bytes);
    if (out.gen % 2 != slot) {
        return false;
    }
    const uint64_t n64 = get_u64_le(bytes + 8);
    if (n64 > std::numeric_limits<size_t>::max() / 2) {
        return false;
    }
    out.n = static_cast<size_t>(n64);
    const size_t n_blocks = out.n / kBlock;
    if (n_blocks > (std::numeric_limits<size_t>::max() - geo.unit_at(0)) / geo.unit_len() ||
        geo.unit_at(0) + n_blocks * geo.unit_len() > file_len) {
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
    out.used = p + 4;
    out.crc = get_u32_le(bytes + p);
    return true;
}

bool parse_header_in(const std::vector<uint8_t> & raw, const Geo & geo, size_t slot, ParsedHeader & out) {
    const size_t at = geo.hdr_at(slot);
    if (at > raw.size() || raw.size() - at < geo.hdr_len()) {
        return false;
    }
    return parse_header(raw.data() + at, geo.hdr_len(), at, geo, slot, raw.size(), out);
}

// A commit is adopted only if every unit its sync wrote is present with the
// bytes it recorded (upstream `delta_verified`). `feed` pushes one unit's
// bytes into the digest, or returns false when the unit is missing.
template <typename Feed>
bool delta_verified(const Geo & geo, const ParsedHeader & h, Feed feed) {
    const size_t unit_len = geo.unit_len();
    const size_t n_app = h.delta_app_to > h.delta_app_from ? h.delta_app_to - h.delta_app_from : 0;
    const size_t count = h.delta_mat.size() + n_app;
    Crc32Digest digest(8 + count * (4 + unit_len));
    uint8_t gen_bytes[8];
    put_u64_le(gen_bytes, h.gen);
    digest.push(gen_bytes, sizeof(gen_bytes));
    auto one = [&](size_t block) {
        uint8_t index_bytes[4];
        put_u32_le(index_bytes, static_cast<uint32_t>(block));
        digest.push(index_bytes, sizeof(index_bytes));
        return feed(block, digest);
    };
    for (size_t block : h.delta_mat) {
        if (!one(block)) {
            return false;
        }
    }
    for (size_t block = h.delta_app_from; block < h.delta_app_to; ++block) {
        if (!one(block)) {
            return false;
        }
    }
    return digest.finish() == h.delta_crc;
}

bool delta_verified_in(const std::vector<uint8_t> & raw, const Geo & geo, const ParsedHeader & h) {
    return delta_verified(geo, h, [&](size_t block, Crc32Digest & digest) {
        const size_t at = geo.unit_at(block);
        if (at > raw.size() || raw.size() - at < geo.unit_len()) {
            return false;
        }
        digest.push(raw.data() + at, geo.unit_len());
        return true;
    });
}

}  // namespace

uint64_t turbovec_v8_declared_size(std::istream & f, uint64_t file_size) {
    // The image size follows from the geometry and the row count, so a
    // padded or sparse file never costs more memory than its image
    // (upstream io_v7 `declared_len`, #487). Only a header whose CRC holds
    // may declare the row count. A file whose geometry is invalid yields
    // just the fixed superblock prefix, enough for the loader to name its
    // error, never the whole file.
    const uint64_t probe = std::min<uint64_t>(file_size, 24);
    auto read_at = [&](uint64_t offset, uint8_t * out, size_t size) {
        if (offset > file_size || file_size - offset < size) {
            return false;
        }
        f.clear();
        f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        f.read(reinterpret_cast<char *>(out), static_cast<std::streamsize>(size));
        return static_cast<bool>(f) && f.gcount() == static_cast<std::streamsize>(size);
    };
    uint8_t head[24];
    if (!read_at(0, head, sizeof(head)) || !turbovec_v8_magic(head, sizeof(head))) {
        return probe;
    }
    Geo geo;
    const bool v8 = std::memcmp(head, kV8Magic, 4) == 0;
    geo.version = v8 ? 8 : 7;
    geo.bits = head[5];
    geo.kind = head[6];
    geo.layout = v8 ? head[7] : kLayoutSeq;
    if (geo.bits < 2 || geo.bits > 4 || geo.kind > 1 || geo.layout > kLayoutPlanes) {
        return probe;
    }
    geo.dim = get_u32_le(head + geo.sb_fixed() - 16);
    if (geo.dim == 0 || geo.dim % 8 != 0 || geo.dim > static_cast<size_t>(kTurboVecMaxDim)) {
        return probe;
    }
    uint8_t n_calib_bytes[4];
    if (!read_at(geo.sb_fixed() + (2 * geo.n_levels() - 1) * 4, n_calib_bytes, 4)) {
        return probe;
    }
    geo.n_calib = get_u32_le(n_calib_bytes);
    if (geo.n_calib != 0 && geo.n_calib != geo.dim) {
        return probe;
    }
    // The superblock and both header slots: bounded by the geometry.
    const uint64_t fixed = std::min<uint64_t>(file_size, geo.unit_at(0));
    if (file_size > std::numeric_limits<size_t>::max()) {
        return fixed;
    }
    const size_t file_len = static_cast<size_t>(file_size);
    uint64_t n = 0;
    std::vector<uint8_t> buf;
    for (size_t slot = 0; slot < 2; ++slot) {
        const size_t at = geo.hdr_at(slot);
        if (at >= file_len) {
            continue;
        }
        buf.resize(std::min(geo.hdr_len(), file_len - at));
        ParsedHeader h;
        if (read_at(at, buf.data(), buf.size()) &&
            parse_header(buf.data(), buf.size(), at, geo, slot, file_len, h)) {
            n = std::max<uint64_t>(n, h.n);
        }
    }
    // parse_header already bounds the row count's units by the file.
    return std::min<uint64_t>(file_size, geo.unit_at(0) + n / kBlock * geo.unit_len());
}

ggml_vec_index_t * turbovec_v8_load_image(std::vector<uint8_t> raw, int & error, TurboVecV8LoadInfo * info) {
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
        parse_header_in(raw, geo, 0, headers[0]),
        parse_header_in(raw, geo, 1, headers[1]),
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
        if (valid[slot] && delta_verified_in(raw, geo, headers[slot])) {
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
    // Rows decode straight into the blocked cache. An index the planes
    // layout will promote decodes into packed rows instead, which its staged
    // search reads, and builds the blocked cache from them.
    const bool keep_packed = turbovec_planes_wanted(geo.bits, dim, n);
    idx->turbovec_packed_rows = keep_packed;
    if (keep_packed) {
        idx->turbovec_data.assign(n * packed_bytes, 0);
    } else {
        idx->turbovec_blocked_n_blocks = (n + kBlock - 1) / kBlock;
        idx->turbovec_blocked_data.assign(idx->turbovec_blocked_n_blocks * row_bytes * kBlock, 0);
    }
    idx->turbovec_scale.assign(n, 0.0f);
    idx->slot_to_id.assign(n, 0);

    // A sequential row (one byte per code group) or a planes row (low
    // planes, then the sign plane) into the index's code storage.
    auto put_seq_row = [&](size_t slot, const uint8_t * codes) {
        if (keep_packed) {
            turbovec_seq_row_to_packed(codes, geo.bits, dim, idx->turbovec_data.data() + slot * packed_bytes);
        } else {
            turbovec_blocked_set_row_seq(*idx, slot, codes);
        }
    };
    std::vector<uint8_t> packed_row(packed_bytes);
    std::vector<uint8_t> seq_row(row_bytes);
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
            if (geo.layout == kLayoutPlanes) {
                // Canonical planes unit: sign bytes group-major across the
                // block's lanes, then each lane's low planes.
                uint8_t * packed = keep_packed ? idx->turbovec_data.data() + slot * packed_bytes : packed_row.data();
                std::memcpy(packed, unit + nsg * kBlock + lane * low_row, low_row);
                for (size_t g = 0; g < nsg; ++g) {
                    packed[low_row + g] = unit[g * kBlock + lane];
                }
                if (!keep_packed) {
                    turbovec_packed_row_to_seq(packed, geo.bits, dim, seq_row.data());
                    turbovec_blocked_set_row_seq(*idx, slot, seq_row.data());
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
    // A stored identity pair declares no calibration (upstream
    // `normalize_calibration`): the index is uncalibrated and writes none.
    const bool identity =
        std::all_of(shift.begin(), shift.end(), [](float v) { return v == 0.0f; }) &&
        std::all_of(scale.begin(), scale.end(), [](float v) { return v == 1.0f; });
    if (!identity) {
        idx->turbovec_tqplus_shift.swap(shift);
        idx->turbovec_tqplus_scale.swap(scale);
    }
    if (keep_packed) {
        repack_turbovec_codes(
            idx->turbovec_data.data(),
            n,
            geo.bits,
            dim,
            idx->turbovec_blocked_data,
            idx->turbovec_blocked_n_blocks,
            idx->turbovec_blocked_vm);
    }
    turbovec_planes_refresh(*idx, 0);
    rebuild_state_hash(*idx);
    turbovec_settle_rows(*idx);
    if (info != nullptr) {
        info->gen = chosen->gen;
        info->nonce = get_u64_le(raw.data() + geo.sb_fixed() - 12);
        info->pending.clear();
        for (const auto & op : chosen->ops) {
            info->pending.push_back(op.first);
        }
    }
    error = GGML_VEC_INDEX_OK;
    return idx.release();
}

// ---------------------------------------------------------------------------
// Incremental sync (upstream turbovec `sync`)
// ---------------------------------------------------------------------------

void turbovec_sync_unbind(ggml_vec_index & idx) noexcept {
    TurboVecSyncBinding & b = idx.turbovec_sync;
    b.bound = false;
    b.path.clear();
    b.pending.clear();
    b.fresh.clear();
    b.rewrite = false;
    b.attempt_nonce = 0;
    b.attempt_gen = 0;
}

namespace {

// Callers build `path` and `pending` first, so a commit that already landed
// is never followed by a failed allocation that leaves the binding stale.
void bind_rows(
        ggml_vec_index & idx,
        std::string path,
        uint64_t gen,
        uint64_t nonce,
        std::unordered_set<size_t> pending) noexcept {
    TurboVecSyncBinding & b = idx.turbovec_sync;
    b.pending = std::move(pending);
    b.fresh.clear();
    b.path = std::move(path);
    b.rewrite = false;
    b.attempt_nonce = 0;
    b.attempt_gen = 0;
    b.gen = gen;
    b.n_synced = idx.slot_to_id.size();
    b.nonce = nonce;
    b.calib_gen = idx.turbovec_calib_gen;
    b.bound = true;
}

uint64_t new_file_nonce() {
    static std::mutex mutex;
    static std::mt19937_64 rng = [] {
        std::random_device device;
        const uint64_t seed =
            (static_cast<uint64_t>(device()) << 32) ^ device() ^
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        return std::mt19937_64(seed);
    }();
    // The clock and process id are mixed into every nonce (upstream
    // io::file_nonce), so processes forked from one parent still differ.
    auto mix = [](uint64_t x) {
        x += UINT64_C(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    };
#ifdef _WIN32
    const uint64_t pid = static_cast<uint64_t>(_getpid());
#else
    const uint64_t pid = static_cast<uint64_t>(::getpid());
#endif
    const uint64_t clock = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t nonce = 0;
    // Zero marks an unclaimed snapshot.
    while (nonce == 0) {
        nonce = rng() ^ mix(clock ^ mix(pid));
    }
    return nonce;
}

enum class CursorState { intact, foreign, replaced };

// Is the file at `path` still the one this binding last committed (upstream
// `cursor_state`)? `stale_used` reports a newer header left in the slot the
// next commit targets, which must be wiped before data moves; `file_n_calib`
// the calibration length the file's superblock declares.
int cursor_state(
        const char * path,
        const TurboVecSyncBinding & b,
        uint8_t b_layout,
        CursorState & state,
        size_t & stale_used,
        size_t & file_n_calib) {
    stale_used = 0;
    file_n_calib = 0;
    std::vector<uint8_t> head;
    uint64_t file_size = 0;
    int rc = turbovec_read_file_range(path, 0, 24, head, file_size);
    if (rc == GGML_VEC_INDEX_E_NOT_FOUND) {
        state = CursorState::replaced;
        return GGML_VEC_INDEX_OK;
    }
    if (rc != GGML_VEC_INDEX_OK) {
        return rc;
    }
    if (head.size() < 24 || !turbovec_v8_magic(head.data(), head.size())) {
        state = CursorState::replaced;
        return GGML_VEC_INDEX_OK;
    }
    const bool v8 = std::memcmp(head.data(), kV8Magic, 4) == 0;
    const size_t sb_fixed = v8 ? 24 : 23;
    if (head[4] != (v8 ? kV8Revision : kV7Revision)) {
        state = CursorState::replaced;
        return GGML_VEC_INDEX_OK;
    }
    // Generations restart at 0 on every full write, so the nonce decides
    // whose file this is. An unclaimed snapshot dropped over the path is
    // nobody's sync destination: rewrite it.
    const uint64_t nonce = get_u64_le(head.data() + sb_fixed - 12);
    if (nonce != b.nonce) {
        // Our own full write, which failed after its rename, owns the file.
        const bool ours = b.attempt_nonce != 0 && nonce == b.attempt_nonce;
        state = nonce == 0 || ours ? CursorState::replaced : CursorState::foreign;
        return GGML_VEC_INDEX_OK;
    }
    // Our own file in a form this index no longer writes (a v7 file, or a
    // layout it has since grown into) is rewritten whole, but only once its
    // commits show that no other writer has advanced it.
    const bool reform = !v8 || head[7] != b_layout;
    if (file_size > std::numeric_limits<size_t>::max()) {
        return GGML_VEC_INDEX_E_IO;
    }
    const size_t file_len = static_cast<size_t>(file_size);
    // Parse the commits with the file's own geometry, not the index's: a
    // calibrate since the last sync changes the index's superblock length,
    // and reading the file through it would miss another writer's newer
    // commits and treat the file as ours to rewrite.
    Geo geo;
    geo.version = v8 ? 8 : 7;
    geo.kind = head[6];
    geo.bits = head[5];
    geo.layout = v8 ? head[7] : kLayoutSeq;
    geo.dim = get_u32_le(head.data() + sb_fixed - 16);
    if (geo.kind != kKindIdMap || geo.bits < 2 || geo.bits > 4 || geo.dim == 0 || geo.dim % 8 != 0 ||
        geo.dim > static_cast<size_t>(kTurboVecMaxDim) || geo.layout > kLayoutPlanes) {
        state = CursorState::replaced;
        return GGML_VEC_INDEX_OK;
    }
    {
        std::vector<uint8_t> n_calib_bytes;
        uint64_t ignored = 0;
        if (turbovec_read_file_range(path, sb_fixed + (2 * geo.n_levels() - 1) * 4, 4, n_calib_bytes, ignored) !=
                GGML_VEC_INDEX_OK ||
            n_calib_bytes.size() != 4) {
            state = CursorState::replaced;
            return GGML_VEC_INDEX_OK;
        }
        geo.n_calib = get_u32_le(n_calib_bytes.data());
        if (geo.n_calib != 0 && geo.n_calib != geo.dim) {
            state = CursorState::replaced;
            return GGML_VEC_INDEX_OK;
        }
        file_n_calib = geo.n_calib;
    }

    // Read each slot's used prefix: a commit without redo ops fits in the
    // probe length; one carrying ops widens to the whole slot once.
    const size_t probe_len =
        16 + 31 * (geo.row_bytes() + 4 + geo.id_bytes(1)) + 4 + 4 + kMaxOps * 4 + 12 + 4;
    auto read_slot = [&](size_t slot, ParsedHeader & out) -> bool {
        const size_t at = geo.hdr_at(slot);
        if (at >= file_len) {
            return false;
        }
        size_t want = std::min(probe_len, file_len - at);
        for (;;) {
            std::vector<uint8_t> buf;
            uint64_t ignored = 0;
            if (turbovec_read_file_range(path, at, want, buf, ignored) != GGML_VEC_INDEX_OK ||
                buf.size() != want) {
                return false;
            }
            if (parse_header(buf.data(), buf.size(), at, geo, slot, file_len, out)) {
                return true;
            }
            const size_t full = std::min(geo.hdr_len(), file_len - at);
            if (want >= full) {
                return false;
            }
            want = full;
        }
    };
    std::vector<ParsedHeader> candidates;
    for (size_t slot = 0; slot < 2; ++slot) {
        ParsedHeader h;
        if (read_slot(slot, h)) {
            candidates.push_back(std::move(h));
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const ParsedHeader & a, const ParsedHeader & c) {
        return a.gen > c.gen;
    });
    // The binding's own commit, still newest, was verified when the binding
    // was made; it needs no delta re-read.
    const CursorState ours_now = reform ? CursorState::replaced : CursorState::intact;
    if (!candidates.empty() && candidates.front().gen == b.gen) {
        state = ours_now;
        return GGML_VEC_INDEX_OK;
    }
    for (const ParsedHeader & h : candidates) {
        if (h.gen > b.gen) {
            stale_used = std::max(stale_used, h.used);
        }
    }
    for (const ParsedHeader & h : candidates) {
        const bool verified = delta_verified(geo, h, [&](size_t block, Crc32Digest & digest) {
            std::vector<uint8_t> unit;
            uint64_t ignored = 0;
            if (turbovec_read_file_range(path, geo.unit_at(block), geo.unit_len(), unit, ignored) !=
                    GGML_VEC_INDEX_OK ||
                unit.size() != geo.unit_len()) {
                return false;
            }
            digest.push(unit.data(), unit.size());
            return true;
        });
        if (verified) {
            // A newer commit is another writer's unless it is the one our
            // last, failed sync wrote, which nobody has built on.
            const bool ours = b.attempt_gen != 0 && h.gen == b.attempt_gen && h.crc == b.attempt_crc;
            state = h.gen == b.gen ? ours_now : ours ? CursorState::replaced : CursorState::foreign;
            return GGML_VEC_INDEX_OK;
        }
    }
    // Our own file with no adoptable commit: only a full rewrite is safe.
    state = CursorState::replaced;
    return GGML_VEC_INDEX_OK;
}

// Plans one incremental sync from the bound commit to the live state
// (upstream `plan_incremental`). Returns false when the carried redo ops
// exceed the header's capacity, where a full rewrite is required.
bool plan_incremental(
        const Source & src,
        const TurboVecSyncBinding & b,
        size_t stale_used,
        std::vector<TurboVecWriteBatch> & batches,
        uint64_t & new_gen,
        std::vector<size_t> & carried) {
    const Geo & geo = src.geometry();
    const size_t old_blocks = static_cast<size_t>(b.n_synced) / kBlock;
    const size_t new_blocks = src.n() / kBlock;
    const size_t live_rows = std::min(old_blocks, new_blocks) * kBlock;
    const uint64_t gen = b.gen + 1;

    // Ops only matter for units committed under both states.
    std::unordered_set<size_t> fresh_units;
    for (size_t r : b.fresh) {
        if (r < live_rows) {
            fresh_units.insert(r / kBlock);
        }
    }
    std::vector<size_t> materialize;
    for (size_t r : b.pending) {
        if (r < live_rows && fresh_units.count(r / kBlock) == 0) {
            materialize.push_back(r / kBlock);
        }
    }
    std::sort(materialize.begin(), materialize.end());
    materialize.erase(std::unique(materialize.begin(), materialize.end()), materialize.end());

    // Every changed row, old or new, of a unit with fresh changes rides the
    // new header; those units are never touched on disk.
    carried.clear();
    for (const std::unordered_set<size_t> * set : { &b.pending, &b.fresh }) {
        for (size_t r : *set) {
            if (r < live_rows && fresh_units.count(r / kBlock) != 0) {
                carried.push_back(r);
            }
        }
    }
    std::sort(carried.begin(), carried.end());
    carried.erase(std::unique(carried.begin(), carried.end()), carried.end());
    if (carried.size() > kMaxOps) {
        return false;
    }
    std::vector<std::pair<size_t, std::vector<size_t>>> groups;
    for (size_t r : carried) {
        if (groups.empty() || groups.back().first != r / kBlock) {
            groups.emplace_back(r / kBlock, std::vector<size_t>{});
        }
        groups.back().second.push_back(r);
    }

    // One batch, one fsync: the header names every unit this sync writes
    // and their digest, so a commit that lands before its data is detected
    // at load instead of being prevented with a barrier.
    const size_t appended = new_blocks > old_blocks ? new_blocks - old_blocks : 0;
    const size_t n_written = materialize.size() + appended;
    Crc32Digest digest(8 + n_written * (4 + geo.unit_len()));
    uint8_t gen_bytes[8];
    put_u64_le(gen_bytes, gen);
    digest.push(gen_bytes, sizeof(gen_bytes));
    TurboVecWriteBatch batch;
    auto write_unit = [&](size_t block) {
        std::vector<uint8_t> bytes(geo.unit_len());
        src.unit(block, bytes.data());
        uint8_t index_bytes[4];
        put_u32_le(index_bytes, static_cast<uint32_t>(block));
        digest.push(index_bytes, sizeof(index_bytes));
        digest.push(bytes.data(), bytes.size());
        batch.emplace_back(geo.unit_at(block), std::move(bytes));
    };
    for (size_t block : materialize) {
        write_unit(block);
    }
    for (size_t block = old_blocks; block < new_blocks; ++block) {
        write_unit(block);
    }
    const size_t slot = static_cast<size_t>(gen % 2);
    batch.emplace_back(
        geo.hdr_at(slot),
        src.header(gen, groups, materialize, old_blocks, new_blocks, digest.finish()));

    batches.clear();
    if (stale_used != 0) {
        // A header from an abandoned commit sits in the target slot. Wipe
        // all of it behind its own barrier, with a generation whose parity
        // the slot rejects, so no torn prefix of the new header can
        // reassemble it.
        std::vector<uint8_t> wipe(std::min(std::max<size_t>(stale_used, 8), geo.hdr_len()), 0);
        put_u64_le(wipe.data(), (gen % 2) ^ 1);
        batches.push_back(TurboVecWriteBatch{ { geo.hdr_at(slot), std::move(wipe) } });
    }
    batches.push_back(std::move(batch));
    new_gen = gen;
    return true;
}

}  // namespace

void turbovec_sync_bind_loaded(ggml_vec_index & idx, const char * path, const TurboVecV8LoadInfo & info) {
    // An unclaimed snapshot loads unbound, so the first sync claims it.
    if (info.nonce == 0 || path == nullptr) {
        turbovec_sync_unbind(idx);
        return;
    }
    bind_rows(idx, path, info.gen, info.nonce, std::unordered_set<size_t>(info.pending.begin(), info.pending.end()));
}

void turbovec_sync_note_remove(ggml_vec_index & idx, size_t slot) noexcept {
    TurboVecSyncBinding & b = idx.turbovec_sync;
    if (!b.bound) {
        return;
    }
    try {
        // Both rows diverge from the commit (the hole takes the last row's
        // bytes, the last row dies but may be refilled), and only rows
        // inside the commit's whole blocks need redo ops (upstream
        // swap_remove + mark_dirty).
        const size_t last = idx.slot_to_id.size() - 1;
        const size_t watermark = static_cast<size_t>(b.n_synced) / kBlock * kBlock;
        for (size_t row : { slot, last }) {
            if (row < watermark) {
                b.fresh.insert(row);
            }
        }
    } catch (...) {
        // Keep the binding: a rewrite still checks the file's generation.
        b.rewrite = true;
    }
}

int turbovec_v8_sync(ggml_vec_index & idx, const char * path) {
    TurboVecSyncBinding & b = idx.turbovec_sync;
    const bool bound = b.bound && filesystem_paths_equal(b.path.c_str(), path);
    CursorState state = CursorState::replaced;
    size_t stale_used = 0;
    if (bound) {
        size_t file_n_calib = 0;
        const int rc = cursor_state(path, b, index_geo(idx).layout, state, stale_used, file_n_calib);
        if (rc != GGML_VEC_INDEX_OK) {
            return rc;
        }
        // Incremental writes use the index's superblock geometry. A file
        // that declares another calibration length (a stored identity pair,
        // which loading drops) would take them at the wrong offsets:
        // rewrite it instead.
        if (state == CursorState::intact && (file_n_calib != idx.turbovec_tqplus_shift.size() || b.rewrite)) {
            state = CursorState::replaced;
        }
    }
    // Another writer advanced or replaced the file: refuse rather than
    // overwrite their commits. Loading the file adopts its state.
    if (state == CursorState::foreign) {
        return GGML_VEC_INDEX_E_IO;
    }

    if (state == CursorState::intact && b.calib_gen == idx.turbovec_calib_gen) {
        const Source src(idx);
        std::vector<TurboVecWriteBatch> batches;
        uint64_t new_gen = 0;
        std::vector<size_t> carried;
        if (plan_incremental(src, b, stale_used, batches, new_gen, carried)) {
            std::unordered_set<size_t> pending(carried.begin(), carried.end());
            // The commit header is the last write; it ends with its CRC.
            const std::vector<uint8_t> & header = batches.back().back().second;
            b.attempt_gen = new_gen;
            b.attempt_crc = get_u32_le(header.data() + header.size() - 4);
            const int rc = turbovec_v8_apply_batches(path, batches);
            if (rc != GGML_VEC_INDEX_OK) {
                // Bytes may have landed. The binding stays, so the next
                // sync still refuses a file another writer advanced, and
                // rewrites if this commit landed.
                return rc;
            }
            b.gen = new_gen;
            b.n_synced = src.n();
            b.pending = std::move(pending);
            b.fresh.clear();
            b.attempt_gen = 0;
            return GGML_VEC_INDEX_OK;
        }
    }

    // A full write claims the file with a fresh nonce.
    const uint64_t nonce = new_file_nonce();
    std::string bound_path(path);
    std::unordered_set<size_t> no_pending;
    TurboVecV8Image image;
    build_image(idx, nonce, image);
    if (bound) {
        b.attempt_nonce = nonce;
    }
    const int rc = turbovec_v8_write_file(path, image);
    if (rc != GGML_VEC_INDEX_OK && rc != GGML_VEC_INDEX_E_NOT_DURABLE) {
        // A full write commits by rename and every failure before it leaves
        // the file untouched, so the binding still describes it.
        return rc;
    }
    bind_rows(idx, std::move(bound_path), 0, nonce, std::move(no_pending));
    return rc;
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
int turbovec_sync_plan_for_test(
        ggml_vec_index_t * idx,
        const char * path,
        std::vector<TurboVecWriteBatch> & batches) {
    std::unique_lock<std::shared_mutex> lock(idx->mutex);
    TurboVecSyncBinding & b = idx->turbovec_sync;
    if (!b.bound || !filesystem_paths_equal(b.path.c_str(), path) || b.calib_gen != idx->turbovec_calib_gen) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    CursorState state = CursorState::replaced;
    size_t stale_used = 0;
    size_t file_n_calib = 0;
    const int rc = cursor_state(path, b, index_geo(*idx).layout, state, stale_used, file_n_calib);
    if (rc != GGML_VEC_INDEX_OK || state != CursorState::intact ||
        file_n_calib != idx->turbovec_tqplus_shift.size()) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    uint64_t new_gen = 0;
    std::vector<size_t> carried;
    return plan_incremental(Source(*idx), b, stale_used, batches, new_gen, carried) ?
        GGML_VEC_INDEX_OK : GGML_VEC_INDEX_E_INVALID_ARG;
}
#endif
