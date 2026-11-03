#include "testing.h"

#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

MAKE_TEST(test_image_preprocessor_longest_edge) {
    clip_hparams hparams;
    hparams.image_longest_edge = 1540;
    hparams.patch_size = 14;
    hparams.n_merge = 2;

    const std::vector<std::pair<clip_image_size, clip_image_size>> cases = {
        { {  463,  463 }, {  476,  476 } },
        { {  463,  300 }, {  476,  308 } },
        { {  448,  280 }, {  448,  280 } },
        { { 2000, 1000 }, { 1540,  784 } },
    };

    for (const auto & [input, expected] : cases) {
        const clip_image_size actual = mtmd_image_preprocessor_longest_edge::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_dyn_size) {
    // { patch_size, image size, expected size }
    // a side that is an even number plus .5 merged patches rounds down, like Python round() in smart_resize
    const std::vector<std::tuple<int, clip_image_size, clip_image_size>> cases = {
        { 16, {  768, 400 }, {  768, 384 } },
        { 16, {  800, 336 }, {  800, 320 } },
        { 16, {  400, 400 }, {  384, 384 } },
        { 16, {  640, 240 }, {  640, 256 } },
        { 14, {  966, 560 }, {  952, 560 } },
    };

    for (const auto & [patch_size, input, expected] : cases) {
        clip_hparams hparams;
        hparams.patch_size = patch_size;
        hparams.n_merge = 2;
        hparams.image_round_half_even = true;
        hparams.set_limit_image_tokens(8, 4096);

        const clip_image_size actual = mtmd_image_preprocessor_dyn_size::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

static std::string image_size_to_str(const clip_image_size & size) {
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

static std::string resize_algo_to_str(resize_algo algo) {
    std::string str;
    switch (algo & ~RESIZE_ALGO_FLAG_TORCH) {
        case RESIZE_ALGO_BILINEAR: str = "bilinear"; break;
        case RESIZE_ALGO_BICUBIC:  str = "bicubic";  break;
        case RESIZE_ALGO_LANCZOS:  str = "lanczos";  break;
        default:                   str = "unknown";  break;
    }
    if (algo & RESIZE_ALGO_FLAG_TORCH) {
        str += "_torch";
    }
    return str;
}

// pseudo-random pixels: s = s * 1103515245 + 12345 from s = 1, one byte (s >> 16) & 255 per channel
static clip_image_u8 make_test_image(const clip_image_size & size) {
    clip_image_u8 img;
    img.set_size(size, false);
    uint32_t s = 1;
    auto next = [&s]() {
        s = s * 1103515245u + 12345u;
        return static_cast<uint8_t>(s >> 16);
    };
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            img.set_pixel(x, y, { next(), next(), next() });
        }
    }
    return img;
}

// FNV-1a of the pixels
static uint32_t hash_pixels(const clip_image_u8 & img, uint32_t hash = 2166136261u) {
    for (uint8_t v : img.get_ro_buf()) {
        hash = (hash ^ v) * 16777619u;
    }
    return hash;
}

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;

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

    // { image size, expected size from HF smart_resize }
    const std::vector<std::pair<clip_image_size, clip_image_size>> resize_cases = {
        { {  400, 600 }, { 384, 608 } }, // 400 / 32 = 12.5 rounds half to even
        { {  480, 336 }, { 480, 320 } },
        { {   64, 169 }, { 160, 416 } }, // 169 * beta / 32 is exactly 13 in double
        { { 1024, 768 }, { 576, 416 } },
    };

    for (const auto & [size, expected] : resize_cases) {
        t.assert_equal("smart_resize for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::smart_resize(hparams, size)));
    }

    // { image size, expected grid from HF }
    const std::vector<std::pair<clip_image_size, clip_image_size>> grid_cases = {
        { {  720, 1600 }, { 2, 4 } }, // 0.45 is an exact tie between 2/4 and 2/5 in double, not in float
        { {  420, 1440 }, { 1, 3 } },
        { {  255, 2160 }, { 1, 8 } },
        { { 1024,  768 }, { 3, 2 } },
    };

    for (const auto & [size, expected] : grid_cases) {
        t.assert_equal("grid for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::get_grid_layout(hparams, size.height, size.width)));
    }

    // do_image_splitting=false
    hparams.preproc_min_tiles = 1;
    hparams.preproc_max_tiles = 1;
    t.assert_equal("tiling for 1024x977 with max_tiles=1", std::string("single"),
            std::string(mtmd_image_preprocessor_lfm2::should_tile(hparams, { 1024, 977 }) ? "tiled" : "single"));
}

MAKE_TEST(test_image_preprocessor_lfm2_torch_resize) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;
    // Lfm2VlImageProcessor stretches the image to the tile grid
    hparams.image_pad_rf = PAD_NONE;

    // { resize algo, image size, expected hash of the tiles and the thumbnail from Lfm2VlImageProcessor (torchvision 0.29, native uint8 resize: arm64, or x86 with AVX2/AVX512) }
    const std::vector<std::tuple<resize_algo, clip_image_size, uint32_t>> cases = {
        { RESIZE_ALGO_BILINEAR, {  620,  410 }, 0xf5e9c941 }, // 608x416
        { RESIZE_ALGO_BILINEAR, { 2000, 1500 }, 0x3331f56c }, // 3x2 tiles, 576x416 thumbnail
        { RESIZE_ALGO_BICUBIC,  {  620,  410 }, 0x2b34eac9 },
        { RESIZE_ALGO_BICUBIC,  { 2000, 1500 }, 0xb035d942 },
    };

    for (const auto & [base_algo, size, expected] : cases) {
        const resize_algo algo = static_cast<resize_algo>(base_algo | RESIZE_ALGO_FLAG_TORCH);
        hparams.image_resize_algo_rf = algo;
        hparams.image_resize_algo_ov = algo;
        const mtmd_image_preprocessor_lfm2 preproc(hparams);
        const auto sliced = preproc.slice_image(make_test_image(size), preproc.get_slice_instructions(size));

        uint32_t hash = 2166136261u;
        for (const auto & slice : sliced.slices) {
            hash = hash_pixels(slice, hash);
        }
        hash = hash_pixels(sliced.overview, hash);
        t.assert_equal("pixels of " + std::to_string(size.width) + "x" + std::to_string(size.height) + " with resize algo " + resize_algo_to_str(algo), expected, hash);
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

MAKE_TEST(test_sam_relpos_quantization) {
    struct temp_files {
        std::string input;
        std::string output;
        ~temp_files() {
            std::remove(input.c_str());
            std::remove(output.c_str());
        }
    };
    const std::string prefix = "test-mtmd-sam-relpos-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    temp_files files{prefix + "-input.gguf", prefix + "-output.gguf"};
    ggml_context_ptr ctx(ggml_init({4*1024*1024, nullptr, false}));
    gguf_context_ptr source(gguf_init_empty());
    gguf_set_val_str(source.get(), "general.architecture", "clip");
    gguf_set_val_u32(source.get(), "general.file_type", LLAMA_FTYPE_ALL_F32);

    const char * names[] = {
        "v.sam.blk.0.attn.pos_h.weight", "v.sam.blk.0.attn.pos_w.weight",
        "v.sam.blk.2.attn.pos_h.weight", "v.sam.blk.2.attn.pos_w.weight",
        "v.blk.0.attn_q.weight",
    };
    std::vector<ggml_tensor *> tensors;
    for (int i = 0; i < 5; ++i) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, i == 4 ? 256 : 64, i == 4 ? 256 : (i < 2 ? 27 : 127));
        ggml_set_name(tensor, names[i]);
        auto * data = static_cast<float *>(tensor->data);
        for (int64_t j = 0; j < ggml_nelements(tensor); ++j) {
            data[j] = std::sin(0.13f*j) + 0.1f*std::cos(0.07f*j);
        }
        gguf_add_tensor(source.get(), tensor);
        tensors.push_back(tensor);
    }
    if (!t.assert_true("write SAM fixture", gguf_write_to_file(source.get(), files.input.c_str(), false))) {
        return;
    }
    const llama_ftype types[] = {
        LLAMA_FTYPE_MOSTLY_F16, LLAMA_FTYPE_MOSTLY_BF16, LLAMA_FTYPE_MOSTLY_Q8_0,
        LLAMA_FTYPE_MOSTLY_Q3_K_S, LLAMA_FTYPE_MOSTLY_Q3_K_M,
        LLAMA_FTYPE_MOSTLY_Q4_0, LLAMA_FTYPE_MOSTLY_Q4_K_M,
    };
    for (llama_ftype type : types) {
        llama_model_quantize_params params = llama_model_quantize_default_params();
        params.ftype = type;
        params.nthread = 1;
        if (!t.assert_equal("quantize SAM fixture", 0u, llama_model_quantize(files.input.c_str(), files.output.c_str(), &params))) {
            continue;
        }
        ggml_context * output_ctx = nullptr;
        gguf_context_ptr output(gguf_init_from_file(files.output.c_str(), {false, &output_ctx}));
        ggml_context_ptr data_ctx(output_ctx);
        if (!t.assert_true("read quantized SAM fixture", output != nullptr && data_ctx != nullptr)) {
            continue;
        }
        ggml_tensor * matrix = ggml_get_tensor(data_ctx.get(), names[4]);
        if (t.assert_true("ordinary matrix exists", matrix != nullptr)) {
            t.assert_true("ordinary matrix changes type", matrix->type != GGML_TYPE_F32);
        }
        for (int i = 0; i < 4; ++i) {
            ggml_tensor * tensor = ggml_get_tensor(data_ctx.get(), names[i]);
            if (!t.assert_true("SAM relative position exists", tensor != nullptr) ||
                !t.assert_equal("SAM relative position remains F32", GGML_TYPE_F32, tensor->type)) {
                continue;
            }
            t.assert_true("SAM relative position values unchanged", ggml_nbytes(tensor) == ggml_nbytes(tensors[i]) &&
                std::memcmp(tensor->data, tensors[i]->data, ggml_nbytes(tensor)) == 0);
            // DeepSeek OCR resizes these tables before gathering attention positions.
            ggml_tensor * cur = ggml_cont(ctx.get(), ggml_permute(ctx.get(), tensor, 1, 0, 2, 3));
            cur = ggml_reshape_3d(ctx.get(), cur, tensor->ne[1], 1, tensor->ne[0]);
            cur = ggml_interpolate(ctx.get(), cur, tensor->ne[1] + 3, 1, tensor->ne[0], 1, GGML_SCALE_MODE_BILINEAR);
            t.assert_equal("SAM interpolation graph type", GGML_TYPE_F32, cur->type);
        }
    }
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
