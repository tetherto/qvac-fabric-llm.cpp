#include "testing.h"

#include "mtmd-audio.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// mtmd_audio
//

MAKE_TEST(test_audio_preprocessor_conformer) {
    clip_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // 0.4 s of tones, 0.2 s of silence, 0.4 s of a quiet tone
    const double pi = 3.14159265358979323846;
    const int sr = hparams.audio_sample_rate;
    std::vector<float> samples(sr, 0.0f);
    for (int i = 0; i < sr; i++) {
        const double ts = (double) i / sr;
        if (i < 0.4 * sr) {
            samples[i] = (float) (0.3 * std::sin(2 * pi * 300 * ts) + 0.2 * std::sin(2 * pi * 1200 * ts) + 0.1 * std::sin(2 * pi * 3500 * ts));
        } else if (i >= 0.6 * sr) {
            samples[i] = (float) (1e-3 * std::sin(2 * pi * 800 * ts));
        }
    }

    mtmd_audio_preprocessor_conformer preproc(hparams);
    preproc.initialize();
    std::vector<mtmd_audio_mel> mels;
    if (!t.assert_true("preprocess", preproc.preprocess(samples.data(), samples.size(), mels))) {
        return;
    }
    const mtmd_audio_mel & mel = mels[0];
    t.assert_equal("n_len", (int64_t) 101, mel.n_len);

    // reference values from NeMo AudioToMelSpectrogramPreprocessor (liquid-audio 1.3.0)
    // { mel bin, frame, value }
    const std::vector<std::tuple<int, int, float>> cases = {
        {   8,  0,  3.6104f }, // tones
        {  24, 50, -0.6371f }, // silence
        {  24, 60, -0.0476f }, // silence -> quiet tone
        { 100, 20, -0.0913f }, // tones, high bin
    };
    for (const auto & [bin, frame, expected] : cases) {
        const float actual = mel.data[bin * mel.n_len + frame];
        t.assert_true("mel[" + std::to_string(bin) + "][" + std::to_string(frame) + "] = " + std::to_string(actual) + ", expected " + std::to_string(expected), std::fabs(actual - expected) < 1e-3f);
    }
}

//
// mtmd_helper decode limits
//

// defined in mtmd-helper.cpp and declared in mtmd-helper-common.h, which cannot be included next to mtmd-internal.h (both define LOG_*)
mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf_impl(const mtmd_context * ctx, int audio_sample_rate,
                                                                 const unsigned char * buf, size_t len, bool placeholder,
                                                                 mtmd_helper_init_opt opt);

static void put_le16(std::vector<unsigned char> & out, uint32_t v) {
    out.push_back(v & 0xff);
    out.push_back((v >> 8) & 0xff);
}

static void put_le32(std::vector<unsigned char> & out, uint32_t v) {
    put_le16(out, v & 0xffff);
    put_le16(out, v >> 16);
}

static void put_tag(std::vector<unsigned char> & out, const char * tag) {
    out.insert(out.end(), tag, tag + 4);
}

struct decoded_media {
    mtmd_helper_bitmap_status status;
    uint32_t nx = 0; // samples for audio
    uint32_t ny = 0;
};

static decoded_media decode_media(const std::vector<unsigned char> & buf, int audio_sample_rate,
                                  uint64_t max_audio_frames, uint64_t max_image_pixels) {
    mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    opt.max_audio_frames = max_audio_frames;
    opt.max_image_pixels = max_image_pixels;
    // no model: the image and audio paths only need the sample rate from the context
    auto res = mtmd_helper_bitmap_init_from_buf_impl(nullptr, audio_sample_rate, buf.data(), buf.size(), false, opt);
    mtmd::bitmap_ptr bitmap(res.bitmap);
    decoded_media out = { res.status, 0, 0 };
    if (bitmap) {
        out.nx = mtmd_bitmap_get_nx(bitmap.get());
        out.ny = mtmd_bitmap_get_ny(bitmap.get());
    }
    return out;
}

static const char * status_name(mtmd_helper_bitmap_status status) {
    switch (status) {
        case MTMD_HELPER_BITMAP_OK:              return "ok";
        case MTMD_HELPER_BITMAP_FAILED:          return "failed";
        case MTMD_HELPER_BITMAP_AUDIO_TOO_LONG:  return "audio too long";
        case MTMD_HELPER_BITMAP_IMAGE_TOO_LARGE: return "image too large";
    }
    return "unknown";
}

// mono 16-bit PCM; claimed_data_bytes is what the data chunk header says it holds
static std::vector<unsigned char> make_wav(uint32_t sample_rate, uint32_t n_samples, uint32_t claimed_data_bytes) {
    std::vector<unsigned char> wav;
    put_tag(wav, "RIFF");
    put_le32(wav, 36 + claimed_data_bytes);
    put_tag(wav, "WAVE");
    put_tag(wav, "fmt ");
    put_le32(wav, 16);
    put_le16(wav, 1); // PCM
    put_le16(wav, 1); // mono
    put_le32(wav, sample_rate);
    put_le32(wav, sample_rate * 2);
    put_le16(wav, 2);  // block align
    put_le16(wav, 16); // bits per sample
    put_tag(wav, "data");
    put_le32(wav, claimed_data_bytes);
    for (uint32_t i = 0; i < n_samples; i++) {
        put_le16(wav, (i * 1000) & 0xffff);
    }
    return wav;
}

// MPEG-1 Layer III, 44.1 kHz mono, bitrate index 0 (free format): the frame size is not in the header
static std::vector<unsigned char> make_mp3_free_format(int n_frames) {
    const size_t frame_bytes = 418;
    std::vector<unsigned char> mp3;
    for (int i = 0; i < n_frames; i++) {
        const size_t start = mp3.size();
        mp3.push_back(0xff);
        mp3.push_back(0xfb); // MPEG-1, Layer III, no CRC
        mp3.push_back(0x00); // bitrate index 0, 44.1 kHz, no padding
        mp3.push_back(0xc0); // mono
        mp3.resize(start + frame_bytes, 0); // empty side info and main data: silence
    }
    return mp3;
}

// 24-bit BMP; only the header is written when with_pixels is false
static std::vector<unsigned char> make_bmp(int32_t width, int32_t height, bool with_pixels) {
    const uint32_t row_bytes   = ((uint32_t) width * 3 + 3) & ~3u;
    const uint32_t pixel_bytes = row_bytes * (uint32_t) std::abs(height);
    std::vector<unsigned char> bmp;
    bmp.push_back('B');
    bmp.push_back('M');
    put_le32(bmp, 54 + pixel_bytes);
    put_le32(bmp, 0);
    put_le32(bmp, 54); // pixel data offset
    put_le32(bmp, 40); // BITMAPINFOHEADER
    put_le32(bmp, (uint32_t) width);
    put_le32(bmp, (uint32_t) height);
    put_le16(bmp, 1);  // planes
    put_le16(bmp, 24); // bits per pixel
    put_le32(bmp, 0);  // BI_RGB
    put_le32(bmp, pixel_bytes);
    put_le32(bmp, 2835);
    put_le32(bmp, 2835);
    put_le32(bmp, 0);
    put_le32(bmp, 0);
    if (with_pixels) {
        bmp.resize(bmp.size() + pixel_bytes, 0x80);
    }
    return bmp;
}

MAKE_TEST(test_helper_audio_limit_wav_data_chunk) {
    // the header claims ~1 GB of samples, the file holds 1600: the decoder measures what it holds
    const auto wav = make_wav(16000, 1600, 0x7ffffff0);

    auto res = decode_media(wav, 16000, 1600, 0);
    t.assert_equal("status at the limit", std::string("ok"), std::string(status_name(res.status)));
    t.assert_equal("samples", (uint32_t) 1600, res.nx);

    res = decode_media(wav, 16000, 1599, 0);
    t.assert_equal("status over the limit", std::string("audio too long"), std::string(status_name(res.status)));
}

MAKE_TEST(test_helper_audio_limit_wav_sample_rate) {
    // a 2 KB file declared at 1 Hz resamples to 16 million samples (64 MB of floats), and the same at 64 KB would be 4 GB
    auto res = decode_media(make_wav(1, 1000, 2000), 16000, 600 * 16000, 0);
    t.assert_equal("status", std::string("audio too long"), std::string(status_name(res.status)));

    const auto wav = make_wav(1, 10, 20);
    res = decode_media(wav, 16000, 0, 0);
    if (!t.assert_equal("status without a limit", std::string("ok"), std::string(status_name(res.status)))) {
        return;
    }
    const uint32_t n_samples = res.nx;
    t.assert_true("resampled to ~160000 samples, got " + std::to_string(n_samples), n_samples > 150000);

    res = decode_media(wav, 16000, n_samples, 0);
    t.assert_equal("status at the limit", std::string("ok"), std::string(status_name(res.status)));

    res = decode_media(wav, 16000, n_samples - 1, 0);
    t.assert_equal("status over the limit", std::string("audio too long"), std::string(status_name(res.status)));
}

MAKE_TEST(test_helper_audio_limit_mp3_free_format) {
    const auto mp3 = make_mp3_free_format(200);

    // no limit: the decoder counts the frames by decoding them
    auto res = decode_media(mp3, 16000, 0, 0);
    if (!t.assert_equal("status without a limit", std::string("ok"), std::string(status_name(res.status)))) {
        return;
    }
    const uint32_t n_samples = res.nx;
    t.assert_true("decodes several seconds, got " + std::to_string(n_samples), n_samples > 5 * 16000);

    res = decode_media(mp3, 16000, n_samples, 0);
    t.assert_equal("status at the limit", std::string("ok"), std::string(status_name(res.status)));
    t.assert_equal("samples at the limit", n_samples, res.nx);

    res = decode_media(mp3, 16000, n_samples - 1, 0);
    t.assert_equal("status over the limit", std::string("audio too long"), std::string(status_name(res.status)));
}

MAKE_TEST(test_helper_audio_limit_counts_at_model_rate) {
    // 1600 samples at 16 kHz are 2400 at 24 kHz, and the limit is in the model's samples
    const auto wav = make_wav(16000, 1600, 1600 * 2);

    auto res = decode_media(wav, 24000, 2400, 0);
    t.assert_equal("status at the limit", std::string("ok"), std::string(status_name(res.status)));
    t.assert_equal("samples", (uint32_t) 2400, res.nx);

    res = decode_media(wav, 24000, 2399, 0);
    t.assert_equal("status over the limit", std::string("audio too long"), std::string(status_name(res.status)));
}

MAKE_TEST(test_helper_image_limit) {
    const auto small_bmp = make_bmp(4, 2, true);

    auto res = decode_media(small_bmp, -1, 0, 0);
    t.assert_equal("status without a limit", std::string("ok"), std::string(status_name(res.status)));

    res = decode_media(small_bmp, -1, 0, 8);
    t.assert_equal("status at the limit", std::string("ok"), std::string(status_name(res.status)));
    t.assert_equal("width", (uint32_t) 4, res.nx);
    t.assert_equal("height", (uint32_t) 2, res.ny);

    res = decode_media(small_bmp, -1, 0, 7);
    t.assert_equal("status over the limit", std::string("image too large"), std::string(status_name(res.status)));

    // a top-down BMP stores a negative height
    res = decode_media(make_bmp(4, -2, true), -1, 0, 8);
    t.assert_equal("top-down status at the limit", std::string("ok"), std::string(status_name(res.status)));

    // only a header, which claims 20000x20000: rejected from the header, so nothing is allocated
    res = decode_media(make_bmp(20000, 20000, false), -1, 0, 4096 * 4096);
    t.assert_equal("oversized header", std::string("image too large"), std::string(status_name(res.status)));
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
