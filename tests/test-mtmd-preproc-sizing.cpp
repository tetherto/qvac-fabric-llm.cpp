// QVAC-23075: golden tests for the idefics3-family sizing rule (mtmd_calc_idefics3_sizing in
// tools/mtmd/mtmd-image.cpp, a pure function exposed via mtmd-image.h). Two bugs in this rule
// reached review: a float32 scale that bought a whole extra slice row on 4:3 inputs, and a
// missing check on the metadata it divides by. Both were invisible to the end-to-end tests,
// which only look at the answer text.
//
// The expected values come from the reference processor's own rule, transcribed to Python and
// evaluated in doubles, which is what the model was trained under:
//
//     P, M = 512, 2048
//     def ref(h, w, no_upscale):
//         long_, short = (w, h) if w >= h else (h, w)
//         if no_upscale and short < P:
//             den = short * P
//             tl = min(M, -(-(long_ * P) // den) * P)
//             ts = max(-(-(short * P) // den) * P, P)
//             return (ts, tl) if w >= h else (tl, ts)
//         tl = min(M, math.ceil(long_ / P) * P) if no_upscale else M
//         ts = max(math.ceil(short * (tl / long_) / P) * P, P)
//         return (ts, tl) if w >= h else (tl, ts)
//
// CITE: DynamicResize._get_new_hw in
// https://huggingface.co/qvac/VisionPsy-Nano-460M-Flash/blob/main/custom_transforms.py
//
// Four of these are also confirmed on hardware: the slice counts below plus one overview match
// the image-encode counts a local Metal run logs for the same inputs, 13 for 960x720 base,
// 5 for 1024x768 Flash, 3 for 640x480 Flash, and 17 for the 640x488 tools/mtmd/test-1.jpeg.

#include "mtmd-image.h"

#include <cstdio>

static int g_failures = 0;

// VisionPsy's published hparams, and the same values idefics3 uses when its mmproj carries
// clip.vision.preproc_image_size.
static const int P = 512;
static const int M = 2048;

static void expect_sizing(const char * what, int w, int h, bool no_upscale,
                          int exp_rw, int exp_rh, int exp_gx, int exp_gy) {
    const mtmd_idefics3_sizing got = mtmd_calc_idefics3_sizing(clip_image_size{w, h}, P, M, no_upscale);
    const int exp_slices = exp_gx * exp_gy;
    if (got.refined_size.width != exp_rw || got.refined_size.height != exp_rh ||
        got.grid_size.width != exp_gx || got.grid_size.height != exp_gy ||
        got.n_slices != exp_slices) {
        std::printf("FAIL %s (%dx%d, %s): refined %dx%d grid %dx%d slices %d, expected refined %dx%d grid %dx%d slices %d\n",
                    what, w, h, no_upscale ? "flash" : "base",
                    got.refined_size.width, got.refined_size.height,
                    got.grid_size.width, got.grid_size.height, got.n_slices,
                    exp_rw, exp_rh, exp_gx, exp_gy, exp_slices);
        g_failures++;
        return;
    }
    // Invariants the splitter depends on: the reference hard-errors on a crop that is not a
    // whole slice, and the cap and the slice size bound every side.
    if (got.refined_size.width % P != 0 || got.refined_size.height % P != 0 ||
        got.refined_size.width > M || got.refined_size.height > M ||
        got.refined_size.width < P || got.refined_size.height < P) {
        std::printf("FAIL %s (%dx%d, %s): refined %dx%d breaks the slice/cap invariants\n",
                    what, w, h, no_upscale ? "flash" : "base",
                    got.refined_size.width, got.refined_size.height);
        g_failures++;
        return;
    }
    std::printf("ok   %s (%dx%d, %s): refined %dx%d grid %dx%d slices %d\n",
                what, w, h, no_upscale ? "flash" : "base",
                got.refined_size.width, got.refined_size.height,
                got.grid_size.width, got.grid_size.height, got.n_slices);
}

int main() {
    // 4:3 and its transpose. This is the class the float32 scale got wrong: 720 * (2048/960)
    // is exactly 1536, and in float32 it lands just above, which buys a fourth slice row.
    expect_sizing("landscape 4:3",  960,  720, false, 2048, 1536, 4, 3);
    expect_sizing("landscape 4:3",  960,  720, true,  1024, 1024, 2, 2);
    expect_sizing("portrait 3:4",   720,  960, false, 1536, 2048, 3, 4);
    expect_sizing("portrait 3:4",   720,  960, true,  1024, 1024, 2, 2);
    expect_sizing("landscape 4:3", 1440, 1080, false, 2048, 1536, 4, 3);
    expect_sizing("landscape 4:3", 1920, 1440, false, 2048, 1536, 4, 3);

    // Small square. Base upscales it to the cap, Flash keeps it at a single slice, which is
    // the case that reaches the single-tile path where the overview is the only image.
    expect_sizing("small square",   256,  256, false, 2048, 2048, 4, 4);
    expect_sizing("small square",   256,  256, true,   512,  512, 1, 1);

    // Exact power-of-two ratios, where float32 and double agree and nothing moved.
    expect_sizing("xga",           1024,  768, false, 2048, 1536, 4, 3);
    expect_sizing("xga",           1024,  768, true,  1024, 1024, 2, 2);
    expect_sizing("vga",            640,  480, false, 2048, 1536, 4, 3);
    expect_sizing("vga",            640,  480, true,  1024,  512, 2, 1);

    // The repo's own test image. Its short side is under one slice, so Flash takes the branch
    // that pins the short side to exactly one slice and enlarges.
    expect_sizing("test-1.jpeg",    640,  488, false, 2048, 2048, 4, 4);
    expect_sizing("test-1.jpeg",    640,  488, true,  1024,  512, 2, 1);

    // Extreme aspect ratios. Both rules cap the long side; the short side can never fall below
    // one slice, and the intermediate products stay in 64-bit.
    expect_sizing("wide panorama",  3000, 1000, false, 2048, 1024, 4, 2);
    expect_sizing("wide panorama",  3000, 1000, true,  2048, 1024, 4, 2);
    expect_sizing("tall panorama",   300, 2000, false,  512, 2048, 1, 4);
    expect_sizing("tall panorama",   300, 2000, true,   512, 2048, 1, 4);
    expect_sizing("hairline",       5000,  100, false, 2048,  512, 4, 1);
    expect_sizing("hairline",       5000,  100, true,  2048,  512, 4, 1);

    // At and above the cap the two rules converge, which is why the DocVQA scans never showed
    // a difference between the checkpoints.
    expect_sizing("at the cap",     2048, 2048, false, 2048, 2048, 4, 4);
    expect_sizing("at the cap",     2048, 2048, true,  2048, 2048, 4, 4);
    expect_sizing("above the cap",  4096, 3072, false, 2048, 1536, 4, 3);
    expect_sizing("above the cap",  4096, 3072, true,  2048, 1536, 4, 3);

    // A degenerate input must not produce a grid: preprocess would otherwise slice nothing and
    // silently send the overview alone.
    const mtmd_idefics3_sizing empty = mtmd_calc_idefics3_sizing(clip_image_size{0, 0}, P, M, false);
    if (empty.refined_size.width != 0 || empty.refined_size.height != 0) {
        std::printf("FAIL zero-size input: refined %dx%d, expected 0x0\n",
                    empty.refined_size.width, empty.refined_size.height);
        g_failures++;
    } else {
        std::printf("ok   zero-size input: refined 0x0\n");
    }

    if (g_failures) {
        std::printf("\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("\nall sizing cases match the reference rule\n");
    return 0;
}
