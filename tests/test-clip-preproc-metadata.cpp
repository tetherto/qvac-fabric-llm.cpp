// QVAC-23075: loader validation for the idefics3-family preprocessing metadata. The sizing rule
// divides by clip.vision.image_size and caps with clip.vision.preproc_image_size, and neither
// degrades gracefully: a zero image_size reaches a GGML_ASSERT at the first image, which aborts
// the process instead of failing the request, and a zero cap makes the refined size {0,0}, so
// the grid is empty and the model silently receives the overview alone. clip_init has to reject
// both while the file is still loading, where it is reportable.
//
// The fixtures are written here rather than committed: each is a metadata-only mmproj, which is
// enough because the check runs after load_hparams and before load_tensors. A file that passes
// validation therefore still fails, on a missing tensor, and that difference in the message is
// what tells the two apart.

#include "clip.h"
#include "clip-impl.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int g_failures = 0;
static std::string g_log;

static void capture_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    g_log += text;
}

struct fixture_params {
    int image_size;
    int preproc_image_size;
    bool no_upscale;
    // Defaults describe the shipped VisionPsy shape; the cases below vary one thing at a time.
    const char * proj_type = "visionpsy";
    bool write_preproc_image_size = true;
    int override_no_upscale = -1; // what the caller passes in clip_context_params
};

// Minimal vision mmproj metadata: everything load_hparams reads as required, plus the two keys
// under test. No tensors.
static std::string write_fixture(const char * name, const fixture_params & p) {
    std::string path = std::string("test-clip-preproc-") + name + ".gguf";

    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", "clip");
    gguf_set_val_str(ctx, KEY_NAME, "VisionPsyNano");
    gguf_set_val_bool(ctx, KEY_HAS_VISION_ENC, true);
    gguf_set_val_str(ctx, KEY_PROJ_TYPE, p.proj_type);

    gguf_set_val_u32(ctx, "clip.vision.embedding_length", 960);
    gguf_set_val_u32(ctx, "clip.vision.projection_dim", 960);
    gguf_set_val_u32(ctx, "clip.vision.feed_forward_length", 2560);
    gguf_set_val_u32(ctx, "clip.vision.block_count", 12);
    gguf_set_val_u32(ctx, "clip.vision.attention.head_count", 12);
    gguf_set_val_f32(ctx, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    gguf_set_val_u32(ctx, KEY_PATCH_SIZE, 16);
    gguf_set_val_u32(ctx, KEY_PROJ_SCALE_FACTOR, 4);

    gguf_set_val_u32(ctx, KEY_IMAGE_SIZE, (uint32_t) p.image_size);
    if (p.write_preproc_image_size) {
        gguf_set_val_u32(ctx, KEY_PREPROC_IMAGE_SIZE, (uint32_t) p.preproc_image_size);
    }
    gguf_set_val_bool(ctx, KEY_PREPROC_NO_UPSCALE, p.no_upscale);

    const float mean_std[3] = { 0.0f, 0.0f, 0.0f };
    gguf_set_arr_data(ctx, KEY_IMAGE_MEAN, GGUF_TYPE_FLOAT32, mean_std, 3);
    const float std_one[3] = { 1.0f, 1.0f, 1.0f };
    gguf_set_arr_data(ctx, KEY_IMAGE_STD, GGUF_TYPE_FLOAT32, std_one, 3);

    gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    return path;
}

// Loads the fixture and returns everything the loader logged.
static std::string try_load(const char * name, const fixture_params & p) {
    const std::string path = write_fixture(name, p);

    clip_context_params cp = {};
    cp.use_gpu = false;
    cp.warmup = false;
    cp.no_alloc = true;
    cp.image_no_upscale = p.override_no_upscale;
    cp.image_tile_mode = CLIP_IMAGE_TILE_MODE_SEQUENTIAL;

    g_log.clear();
    clip_log_set_callback(capture_log, nullptr);
    clip_init_result res = clip_init(path.c_str(), cp);
    clip_log_set_callback(nullptr, nullptr);

    if (res.ctx_v) {
        std::printf("FAIL %s: a metadata-only mmproj must not load\n", name);
        g_failures++;
        clip_free(res.ctx_v);
    }
    if (res.ctx_a) {
        clip_free(res.ctx_a);
    }
    std::remove(path.c_str());
    return g_log;
}

static void expect_rejected(const char * name, const fixture_params & p, const char * needle) {
    const std::string log = try_load(name, p);
    if (log.find(needle) == std::string::npos) {
        std::printf("FAIL %s: expected the load to complain about '%s'\nlog was:\n%s\n",
                    name, needle, log.c_str());
        g_failures++;
        return;
    }
    std::printf("ok   %s: rejected at load with '%s'\n", name, needle);
}

static void expect_passes_validation(const char * name, const fixture_params & p) {
    const std::string log = try_load(name, p);
    // It still fails, on the tensors this fixture does not carry. What must not appear is our
    // own complaint about the sizing metadata.
    if (log.find("slices by image_size") != std::string::npos ||
        log.find("preproc_no_upscale needs") != std::string::npos) {
        std::printf("FAIL %s: valid metadata was rejected by the sizing check\nlog was:\n%s\n",
                    name, log.c_str());
        g_failures++;
        return;
    }
    // And it has to fail for the right reason: reaching load_tensors is what proves the check
    // ran and let it through, rather than something earlier stopping the load first.
    if (log.find("unable to find tensor") == std::string::npos) {
        std::printf("FAIL %s: the load stopped before load_tensors, so the check was never reached\nlog was:\n%s\n",
                    name, log.c_str());
        g_failures++;
        return;
    }
    std::printf("ok   %s: passed the sizing check and failed later, on tensors\n", name);
}

int main() {
    // The shipped shape: 512 slices, 2048 cap. Both variants must get past validation.
    expect_passes_validation("valid-base",  { 512, 2048, false });
    expect_passes_validation("valid-flash", { 512, 2048, true });

    // A cap of exactly one slice is legal, every image becomes a single slice.
    expect_passes_validation("cap-equals-slice", { 512, 512, true });

    // Zero image_size: the divide and the GGML_ASSERT case. Rejected with the flag off too,
    // which is the half that only applied under no-upscale before.
    expect_rejected("zero-image-size-base",  { 0, 2048, false }, "slices by image_size");
    expect_rejected("zero-image-size-flash", { 0, 2048, true },  "slices by image_size");

    // Zero cap: the silent one. Refined size {0,0}, empty grid, overview only.
    expect_rejected("zero-cap-base",  { 512, 0, false }, "slices by image_size");
    expect_rejected("zero-cap-flash", { 512, 0, true },  "slices by image_size");

    // Cap below one slice: std::clamp(val, lo, hi) with lo > hi, undefined behaviour rather
    // than a bad size. Only reachable on the no-upscale path, which is the only one that clamps.
    expect_rejected("cap-below-slice", { 512, 256, true }, "preproc_no_upscale needs");

    // idefics3 keeps loading without the cap key, because the shipped
    // ggml-org/SmolVLM-500M-Instruct-GGUF mmproj has none. It is overview-only in that state, so
    // the loader has to say so instead of either throwing or staying quiet.
    {
        fixture_params p = { 512, 0, false };
        p.proj_type = "idefics3";
        p.write_preproc_image_size = false;
        const std::string log = try_load("idefics3-no-cap", p);
        const bool warned = log.find("encoded as the overview alone") != std::string::npos;
        const bool reached_tensors = log.find("unable to find tensor") != std::string::npos;
        if (!warned || !reached_tensors) {
            std::printf("FAIL idefics3-no-cap: warned=%d reached_tensors=%d\nlog was:\n%s\n",
                        (int) warned, (int) reached_tensors, log.c_str());
            g_failures++;
        } else {
            std::printf("ok   idefics3-no-cap: warns that slicing is off and still loads\n");
        }
    }

    // The caller override. -1 keeps the GGUF value, 0 and 1 are both explicit, and turning it
    // off against a GGUF that turned it on is called out, since that is what a zero-initialized
    // params struct passes.
    {
        fixture_params p = { 512, 2048, true };
        p.override_no_upscale = 0;
        const std::string log = try_load("override-off-against-gguf-on", p);
        if (log.find("overrides clip.vision.preproc_no_upscale=true") == std::string::npos ||
            log.find("preproc_no_upscale: 0 (custom value)") == std::string::npos) {
            std::printf("FAIL override-off-against-gguf-on: expected the override to be announced\nlog was:\n%s\n",
                        log.c_str());
            g_failures++;
        } else {
            std::printf("ok   override-off-against-gguf-on: announced and applied\n");
        }
    }

    {
        fixture_params p = { 512, 2048, false };
        p.override_no_upscale = 1;
        const std::string log = try_load("override-on-against-gguf-off", p);
        if (log.find("preproc_no_upscale: 1 (custom value)") == std::string::npos) {
            std::printf("FAIL override-on-against-gguf-off: expected the override to apply\nlog was:\n%s\n",
                        log.c_str());
            g_failures++;
        } else {
            std::printf("ok   override-on-against-gguf-off: applied\n");
        }
    }

    {
        fixture_params p = { 512, 2048, true };
        p.override_no_upscale = -1;
        const std::string log = try_load("override-unset-keeps-gguf", p);
        if (log.find("(custom value)") != std::string::npos) {
            std::printf("FAIL override-unset-keeps-gguf: -1 must not count as an override\nlog was:\n%s\n",
                        log.c_str());
            g_failures++;
        } else {
            std::printf("ok   override-unset-keeps-gguf: the GGUF value stands\n");
        }
    }

    // A projector that does not use the flag says so rather than silently accepting it.
    {
        fixture_params p = { 512, 2048, false };
        p.proj_type = "gemma3";
        p.write_preproc_image_size = false;
        p.override_no_upscale = 1;
        const std::string log = try_load("override-on-non-idefics3", p);
        if (log.find("only affects idefics3-style preprocessing") == std::string::npos) {
            std::printf("FAIL override-on-non-idefics3: expected the ignore warning\nlog was:\n%s\n",
                        log.c_str());
            g_failures++;
        } else {
            std::printf("ok   override-on-non-idefics3: ignored with a warning\n");
        }
    }

    if (g_failures) {
        std::printf("\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("\npreprocessing metadata is validated at load\n");
    return 0;
}
