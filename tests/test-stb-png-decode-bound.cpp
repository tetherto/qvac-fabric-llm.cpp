// QVAC-26299: the PNG zlib decode in vendor/stb/stb_image.h must be bounded by the size IHDR
// declares. Upstream stb grows the output buffer by doubling up to UINT_MAX/2 whatever the
// image size, so a 1x1 PNG carrying a few MB of deflated zeros allocated about 2 GiB before any
// size check ran. The fix decodes into a buffer of exactly the filtered scanline size and
// rejects a stream that inflates past it.
//
// stb's allocator hooks count live and peak bytes, so the bound is asserted on every platform
// without measuring RSS. The PNGs are built here (stored deflate blocks, hand-packed fixed
// Huffman for the bomb), so the test needs no zlib.
//
// With a file argument it only decodes that file and prints the result and the tracked peak;
// that mode is for a standalone RSS measurement of a real reproducer.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static size_t g_live = 0;
static size_t g_peak = 0;

// A 16-byte header keeps the user pointer aligned for any type and remembers the size, since
// STBI_FREE does not get one.
static const size_t HDR = 16;

static void * track_malloc(size_t sz) {
    unsigned char * p = (unsigned char *) std::malloc(sz + HDR);
    if (!p) {
        return nullptr;
    }
    std::memcpy(p, &sz, sizeof(sz));
    g_live += sz;
    if (g_live > g_peak) {
        g_peak = g_live;
    }
    return p + HDR;
}

static void track_free(void * ptr) {
    if (!ptr) {
        return;
    }
    unsigned char * p = (unsigned char *) ptr - HDR;
    size_t          sz;
    std::memcpy(&sz, p, sizeof(sz));
    g_live -= sz;
    std::free(p);
}

static void * track_realloc(void * ptr, size_t newsz) {
    if (!ptr) {
        return track_malloc(newsz);
    }
    unsigned char * p = (unsigned char *) ptr - HDR;
    size_t          oldsz;
    std::memcpy(&oldsz, p, sizeof(oldsz));
    unsigned char * q = (unsigned char *) std::realloc(p, newsz + HDR);
    if (!q) {
        return nullptr;
    }
    std::memcpy(q, &newsz, sizeof(newsz));
    g_live = g_live - oldsz + newsz;
    if (g_live > g_peak) {
        g_peak = g_live;
    }
    return q + HDR;
}

// stb requires STBI_REALLOC alongside STBI_REALLOC_SIZED; it calls the sized form.
#define STBI_MALLOC(sz)                     track_malloc(sz)
#define STBI_REALLOC(p, newsz)              track_realloc(p, newsz)
#define STBI_REALLOC_SIZED(p, oldsz, newsz) track_realloc(p, newsz)
#define STBI_FREE(p)                        track_free(p)
// Same configuration as tools/mtmd/mtmd-helper.cpp, so this compiles the decoder mtmd ships.
#define STB_IMAGE_IMPLEMENTATION
// stb casts away const throughout; that is upstream code, not something this test checks. gcc 14
// also reports tc16 in stbi__parse_png_file as maybe-uninitialized once it inlines
// stbi__compute_transparency16 into it, a false positive clang does not raise.
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    if !defined(__clang__)
#        pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#    endif
#endif
#include "stb/stb_image.h"
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

static int g_failures = 0;

typedef std::vector<uint8_t> bytes;

// ---------------------------------------------------------------- PNG writer

static uint32_t crc32_of(const uint8_t * data, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static bool     init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; i++) {
        crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    }
    return ~crc;
}

static uint32_t adler32_of(const bytes & data) {
    uint32_t a = 1, b = 0;
    for (uint8_t v : data) {
        a = (a + v) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

static void put_be32(bytes & out, uint32_t v) {
    out.push_back((uint8_t) (v >> 24));
    out.push_back((uint8_t) (v >> 16));
    out.push_back((uint8_t) (v >> 8));
    out.push_back((uint8_t) v);
}

static void put_chunk(bytes & out, const char * type, const bytes & data) {
    put_be32(out, (uint32_t) data.size());
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    put_be32(out, crc32_of(out.data() + start, out.size() - start));
}

// zlib stream of stored blocks: header 0x78 0x01, blocks of up to 65535 bytes, adler32.
static bytes zlib_stored(const bytes & raw) {
    bytes  z   = { 0x78, 0x01 };
    size_t off = 0;
    do {
        const size_t n     = raw.size() - off < 65535 ? raw.size() - off : 65535;
        const bool   final = off + n == raw.size();
        z.push_back(final ? 1 : 0);
        z.push_back((uint8_t) n);
        z.push_back((uint8_t) (n >> 8));
        z.push_back((uint8_t) ~n);
        z.push_back((uint8_t) (~n >> 8));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    } while (off < raw.size());
    put_be32(z, adler32_of(raw));
    return z;
}

struct png_spec {
    int   w, h;
    int   color;  // PNG color type: 0 gray, 2 RGB, 3 palette, 4 gray+alpha, 6 RGBA
    int   depth;
    bool  interlaced;
    bytes plte;          // RGB triplets, palette only
    bytes trns;          // alpha per palette entry, palette only
    bool  cgbi = false;  // Apple CgBI: raw deflate with no zlib header or adler32
};

static int channels_of(int color) {
    return color == 2 ? 3 : color == 4 ? 2 : color == 6 ? 4 : 1;
}

static bytes png_wrap(const png_spec & s, const bytes & zstream) {
    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    bytes                out(sig, sig + 8);
    if (s.cgbi) {
        // stb only checks that the chunk is there; its payload is ignored.
        put_chunk(out, "CgBI", bytes(4, 0));
    }
    bytes ihdr;
    put_be32(ihdr, (uint32_t) s.w);
    put_be32(ihdr, (uint32_t) s.h);
    ihdr.push_back((uint8_t) s.depth);
    ihdr.push_back((uint8_t) s.color);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(s.interlaced ? 1 : 0);
    put_chunk(out, "IHDR", ihdr);
    if (!s.plte.empty()) {
        put_chunk(out, "PLTE", s.plte);
    }
    if (!s.trns.empty()) {
        put_chunk(out, "tRNS", s.trns);
    }
    if (s.cgbi) {
        put_chunk(out, "IDAT", bytes(zstream.begin() + 2, zstream.end() - 4));
    } else {
        put_chunk(out, "IDAT", zstream);
    }
    put_chunk(out, "IEND", bytes());
    return out;
}

// Deterministic sample value for pixel (x, y), channel c, within the bit depth.
static uint32_t sample_at(const png_spec & s, int x, int y, int c) {
    const uint32_t maxv = (s.depth == 16) ? 0xffffu : (1u << s.depth) - 1;
    uint32_t       v    = (uint32_t) (x * 37 + y * 101 + c * 59 + 13) * 2654435761u;
    v ^= v >> 13;
    if (s.color == 3) {
        return v % (uint32_t) (s.plte.size() / 3);
    }
    return v & maxv;
}

// One filtered scanline (filter type 0) holding the given pixel x coordinates of row y.
static void emit_row(const png_spec & s, int y, const std::vector<int> & xs, bytes & raw) {
    const int n = channels_of(s.color);
    raw.push_back(0);
    uint32_t acc = 0;
    int      nb  = 0;
    for (int x : xs) {
        for (int c = 0; c < n; c++) {
            const uint32_t v = sample_at(s, x, y, c);
            if (s.depth == 16) {
                raw.push_back((uint8_t) (v >> 8));
                raw.push_back((uint8_t) v);
            } else if (s.depth == 8) {
                raw.push_back((uint8_t) v);
            } else {
                acc = (acc << s.depth) | v;
                nb += s.depth;
                if (nb == 8) {
                    raw.push_back((uint8_t) acc);
                    acc = 0;
                    nb  = 0;
                }
            }
        }
    }
    if (nb > 0) {
        raw.push_back((uint8_t) (acc << (8 - nb)));
    }
}

static bytes raw_scanlines(const png_spec & s) {
    bytes raw;
    if (!s.interlaced) {
        std::vector<int> xs;
        for (int x = 0; x < s.w; x++) {
            xs.push_back(x);
        }
        for (int y = 0; y < s.h; y++) {
            emit_row(s, y, xs, raw);
        }
        return raw;
    }
    static const int xorig[] = { 0, 4, 0, 2, 0, 1, 0 };
    static const int yorig[] = { 0, 0, 4, 0, 2, 0, 1 };
    static const int xspc[]  = { 8, 8, 4, 4, 2, 2, 1 };
    static const int yspc[]  = { 8, 8, 8, 4, 4, 2, 2 };
    for (int p = 0; p < 7; p++) {
        std::vector<int> xs;
        for (int x = xorig[p]; x < s.w; x += xspc[p]) {
            xs.push_back(x);
        }
        // An empty pass has no scanlines at all, not even filter bytes.
        if (xs.empty()) {
            continue;
        }
        for (int y = yorig[p]; y < s.h; y += yspc[p]) {
            emit_row(s, y, xs, raw);
        }
    }
    return raw;
}

// What stbi_load_16_from_memory / stbi_load_from_memory return with req_comp equal to the
// native channel count (4 for palette, which stb expands through PLTE and tRNS).
static std::vector<uint32_t> expected_pixels(const png_spec & s) {
    std::vector<uint32_t> out;
    const int             n = channels_of(s.color);
    for (int y = 0; y < s.h; y++) {
        for (int x = 0; x < s.w; x++) {
            if (s.color == 3) {
                const uint32_t i = sample_at(s, x, y, 0);
                out.push_back(s.plte[i * 3]);
                out.push_back(s.plte[i * 3 + 1]);
                out.push_back(s.plte[i * 3 + 2]);
                out.push_back(i < s.trns.size() ? s.trns[i] : 255);
                continue;
            }
            for (int c = 0; c < n; c++) {
                uint32_t v = sample_at(s, x, y, c);
                if (s.depth < 8) {
                    // stb scales low-depth gray to the full 8-bit range.
                    static const uint32_t scale[] = { 0, 0xff, 0x55, 0, 0x11 };
                    v *= scale[s.depth];
                }
                out.push_back(v);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------- bomb writer

struct bit_writer {
    bytes    out;
    uint32_t acc = 0;
    int      nb  = 0;

    // deflate packs bits LSB-first within each byte
    void bits(uint32_t v, int n) {
        for (int i = 0; i < n; i++) {
            acc |= ((v >> i) & 1u) << nb;
            if (++nb == 8) {
                out.push_back((uint8_t) acc);
                acc = 0;
                nb  = 0;
            }
        }
    }

    // Huffman codes are defined MSB-first, so they go in reversed.
    void code(uint32_t c, int n) {
        for (int i = n - 1; i >= 0; i--) {
            bits((c >> i) & 1u, 1);
        }
    }

    void flush() {
        if (nb > 0) {
            out.push_back((uint8_t) acc);
            acc = 0;
            nb  = 0;
        }
    }
};

// zlib stream inflating to 1 + 258 * n_copies zero bytes: one fixed-Huffman block holding a
// literal 0 and then n_copies matches of length 258 at distance 1, about 13 bits per 258 bytes.
static bytes zlib_zero_bomb(size_t n_copies, size_t * inflated) {
    bit_writer bw;
    bw.out = { 0x78, 0x01 };
    bw.bits(1, 1);         // BFINAL
    bw.bits(1, 2);         // BTYPE 01, fixed Huffman
    bw.code(0x30, 8);      // literal 0
    for (size_t i = 0; i < n_copies; i++) {
        bw.code(0xC5, 8);  // length symbol 285 = 258
        bw.code(0x00, 5);  // distance code 0 = 1
    }
    bw.code(0x00, 7);      // end of block, symbol 256
    bw.flush();
    const size_t n = 1 + 258 * n_copies;
    // adler32 of n zero bytes: s1 stays 1 and s2 gains 1 per byte.
    put_be32(bw.out, (uint32_t) ((n % 65521) << 16) | 1u);
    *inflated = n;
    return bw.out;
}

// ---------------------------------------------------------------- checks

static const char * reason() {
    const char * r = stbi_failure_reason();
    return r ? r : "(none)";
}

static void expect_roundtrip(const char * what, const png_spec & s) {
    const bytes                 png = png_wrap(s, zlib_stored(raw_scanlines(s)));
    const std::vector<uint32_t> exp = expected_pixels(s);
    const int                   req = s.color == 3 ? 4 : channels_of(s.color);
    int                         w = 0, h = 0, comp = 0;
    bool                        ok = true;
    if (s.depth == 16) {
        stbi_us * px = stbi_load_16_from_memory(png.data(), (int) png.size(), &w, &h, &comp, req);
        ok           = px && w == s.w && h == s.h;
        for (size_t i = 0; ok && i < exp.size(); i++) {
            ok = px[i] == exp[i];
        }
        if (!px) {
            std::printf("FAIL %s %dx%d: decode failed: %s\n", what, s.w, s.h, reason());
        }
        stbi_image_free(px);
    } else {
        stbi_uc * px = stbi_load_from_memory(png.data(), (int) png.size(), &w, &h, &comp, req);
        ok           = px && w == s.w && h == s.h;
        for (size_t i = 0; ok && i < exp.size(); i++) {
            ok = px[i] == exp[i];
        }
        if (!px) {
            std::printf("FAIL %s %dx%d: decode failed: %s\n", what, s.w, s.h, reason());
        }
        stbi_image_free(px);
    }
    if (!ok) {
        std::printf("FAIL %s %dx%d%s: pixels or size differ\n", what, s.w, s.h, s.interlaced ? " adam7" : "");
        g_failures++;
        return;
    }
    std::printf("ok   %s %dx%d%s\n", what, s.w, s.h, s.interlaced ? " adam7" : "");
}

static void expect_reject(const char * what, const bytes & png, const char * want_reason) {
    int       w = 0, h = 0, comp = 0;
    stbi_uc * px = stbi_load_from_memory(png.data(), (int) png.size(), &w, &h, &comp, 3);
    if (px) {
        std::printf("FAIL %s: decoded as %dx%d, expected rejection with \"%s\"\n", what, w, h, want_reason);
        stbi_image_free(px);
        g_failures++;
        return;
    }
    if (std::strstr(reason(), want_reason) == nullptr) {
        std::printf("FAIL %s: reason \"%s\", expected \"%s\"\n", what, reason(), want_reason);
        g_failures++;
        return;
    }
    std::printf("ok   %s: rejected, \"%s\"\n", what, reason());
}

static png_spec spec(int w, int h, int color, int depth, bool interlaced = false) {
    png_spec s;
    s.w          = w;
    s.h          = h;
    s.color      = color;
    s.depth      = depth;
    s.interlaced = interlaced;
    return s;
}

static png_spec palette_spec(int w, int h, int depth, bool interlaced = false) {
    png_spec  s = spec(w, h, 3, depth, interlaced);
    const int n = 1 << (depth < 4 ? depth : 4);
    for (int i = 0; i < n; i++) {
        s.plte.push_back((uint8_t) (i * 16 + 3));
        s.plte.push_back((uint8_t) (255 - i * 9));
        s.plte.push_back((uint8_t) (i * 41));
    }
    // fewer tRNS entries than palette entries: the rest stay opaque
    for (int i = 0; i < n - 1; i++) {
        s.trns.push_back((uint8_t) (i * 23));
    }
    return s;
}

static png_spec cgbi_spec(int w, int h) {
    png_spec s = spec(w, h, 6, 8);
    s.cgbi     = true;
    return s;
}

static int decode_file(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        std::printf("cannot open %s\n", path);
        return 1;
    }
    bytes   data;
    uint8_t buf[1 << 16];
    size_t  n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        data.insert(data.end(), buf, buf + n);
    }
    std::fclose(f);
    g_live = g_peak = 0;
    int       w = 0, h = 0, comp = 0;
    stbi_uc * px = stbi_load_from_memory(data.data(), (int) data.size(), &w, &h, &comp, 0);
    if (px) {
        std::printf("%s: %dx%d, decoded, tracked peak %zu bytes\n", path, w, h, g_peak);
    } else {
        stbi_info_from_memory(data.data(), (int) data.size(), &w, &h, &comp);
        std::printf("%s: %dx%d, failed \"%s\", tracked peak %zu bytes\n", path, w, h, reason(), g_peak);
    }
    stbi_image_free(px);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        return decode_file(argv[1]);
    }

    // The bomb bit packing must be right, or the bomb case would pass for the wrong reason. A
    // small bomb decoded into an image of exactly its inflated size proves it against stb.
    {
        size_t      inflated = 0;
        const bytes z        = zlib_zero_bomb(8, &inflated);
        png_spec    s        = spec((int) inflated - 1, 1, 0, 8);
        const bytes png      = png_wrap(s, z);
        int         w = 0, h = 0, comp = 0;
        stbi_uc *   px = stbi_load_from_memory(png.data(), (int) png.size(), &w, &h, &comp, 1);
        bool        ok = px && w == s.w && h == 1;
        for (int i = 0; ok && i < w; i++) {
            ok = px[i] == 0;
        }
        stbi_image_free(px);
        if (!ok) {
            std::printf("FAIL bomb encoder self-check: %zu-byte inflate into %dx1 gray: %s\n", inflated, s.w,
                        px ? "wrong pixels" : reason());
            g_failures++;
        } else {
            std::printf("ok   bomb encoder self-check: %zu-byte stream inflates to %zu bytes\n", z.size(), inflated);
        }
    }

    // a. 1x1 RGB whose stream inflates to about 160 MB. Before QVAC-26299 stb doubled its
    // buffer through the whole stream and this peaked at 256 MiB of tracked allocations; now
    // the peak is the compressed stream itself, about 1 MB, so the limit is 4 MiB.
    {
        size_t      inflated = 0;
        const bytes png      = png_wrap(spec(1, 1, 2, 8), zlib_zero_bomb(640000, &inflated));
        g_live = g_peak = 0;
        int          w = 0, h = 0, comp = 0;
        stbi_uc *    px       = stbi_load_from_memory(png.data(), (int) png.size(), &w, &h, &comp, 3);
        const size_t peak     = g_peak;
        const bool   rejected = px == nullptr && std::strstr(reason(), "output buffer limit") != nullptr;
        stbi_image_free(px);
        const size_t limit = 4u << 20;
        if (!rejected || peak >= limit) {
            std::printf(
                "FAIL bomb 1x1 rgb, %zu-byte png inflating to %zu bytes: %s, reason \"%s\", peak %zu bytes (limit "
                "%zu)\n",
                png.size(), inflated, px ? "decoded" : "rejected", px ? "(none)" : reason(), peak, limit);
            g_failures++;
        } else {
            std::printf("ok   bomb 1x1 rgb, %zu-byte png inflating to %zu bytes: rejected, \"%s\", peak %zu bytes\n",
                        png.size(), inflated, reason(), peak);
        }
    }

    // b. non-interlaced round trips
    {
        const int sizes[][2] = {
            { 1, 1 },
            { 7, 5 }
        };
        for (const auto & sz : sizes) {
            expect_roundtrip("gray8", spec(sz[0], sz[1], 0, 8));
            expect_roundtrip("gray1", spec(sz[0], sz[1], 0, 1));
            expect_roundtrip("gray2", spec(sz[0], sz[1], 0, 2));
            expect_roundtrip("graya8", spec(sz[0], sz[1], 4, 8));
            expect_roundtrip("graya16", spec(sz[0], sz[1], 4, 16));
            expect_roundtrip("rgb8", spec(sz[0], sz[1], 2, 8));
            expect_roundtrip("rgba8", spec(sz[0], sz[1], 6, 8));
            expect_roundtrip("palette8+trns", palette_spec(sz[0], sz[1], 8));
            expect_roundtrip("palette4+trns", palette_spec(sz[0], sz[1], 4));
            expect_roundtrip("rgb16", spec(sz[0], sz[1], 2, 16));
        }
        // more than one stored block
        expect_roundtrip("rgb8", spec(160, 200, 2, 8));
        // CgBI goes through the same bounded decode without the zlib header. stb leaves BGR
        // and premultiplied alpha alone unless stbi_convert_iphone_png_to_rgb is set.
        expect_roundtrip("cgbi rgba8", cgbi_spec(7, 5));
    }

    // c. Adam7 round trips; these sizes leave some passes empty
    {
        const int sizes[][2] = {
            { 1,  1  },
            { 1,  9  },
            { 9,  1  },
            { 5,  3  },
            { 9,  9  },
            { 16, 16 },
            { 17, 3  }
        };
        for (const auto & sz : sizes) {
            expect_roundtrip("rgb8", spec(sz[0], sz[1], 2, 8, true));
        }
        // sub-byte rows per pass
        expect_roundtrip("gray1", spec(17, 3, 0, 1, true));
        expect_roundtrip("gray2", spec(17, 3, 0, 2, true));
        expect_roundtrip("palette4+trns", palette_spec(9, 9, 4, true));
        expect_roundtrip("graya8", spec(5, 3, 4, 8, true));
        expect_roundtrip("rgb16", spec(5, 3, 2, 16, true));
    }

    // d. one byte past the declared size is rejected, one byte short still fails as before.
    // Round trips only catch a declared size that is too small; the Adam7 sub-byte cases with
    // empty passes catch one that is too large.
    {
        struct {
            const char * what;
            png_spec     s;
        } cases[] = {
            { "rgb8 7x5", spec(7, 5, 2, 8) },
            { "adam7 gray1 17x3", spec(17, 3, 0, 1, true) },
            { "adam7 palette4+trns 9x9", palette_spec(9, 9, 4, true) },
            { "cgbi rgba8 7x5", cgbi_spec(7, 5) },
        };

        for (const auto & c : cases) {
            const bytes raw    = raw_scanlines(c.s);
            bytes       longer = raw;
            longer.push_back(0);
            const std::string what_long = std::string(c.what) + " stream one byte long";
            expect_reject(what_long.c_str(), png_wrap(c.s, zlib_stored(longer)), "output buffer limit");
            bytes shorter = raw;
            shorter.pop_back();
            const std::string what_short = std::string(c.what) + " stream one byte short";
            expect_reject(what_short.c_str(), png_wrap(c.s, zlib_stored(shorter)), "not enough pixels");
        }
    }

    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
