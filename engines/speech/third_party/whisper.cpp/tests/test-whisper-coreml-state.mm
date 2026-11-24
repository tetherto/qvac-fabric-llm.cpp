#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "whisper.h"

#include <cstdio>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace {

NSURL * _Nonnull childURL(NSURL * directory, NSString * name) {
    NSURL * result = [directory URLByAppendingPathComponent:name];
    assert(result != nil);
    return result;
}

NSURL * _Nonnull prepareFixture(const char * model_path) {
    NSFileManager * manager = [NSFileManager defaultManager];
    NSString * name = [[NSUUID UUID] UUIDString];
    NSURL * directory = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name]
                                 isDirectory:YES];
    NSError * error = nil;
    assert([manager createDirectoryAtURL:directory withIntermediateDirectories:NO attributes:nil error:&error]);

    NSString * source_path = [NSString stringWithUTF8String:model_path];
    assert(source_path != nil);
    NSURL * model = [NSURL fileURLWithPath:source_path];
    assert([manager copyItemAtURL:model toURL:childURL(directory, @"model.bin") error:&error]);

    NSURL * source = [NSURL fileURLWithPath:@WHISPER_COREML_TEST_MODEL_PATH];
    NSURL * compiled = [MLModel compileModelAtURL:source error:&error];
    assert(compiled != nil);
    assert([manager copyItemAtURL:compiled
                           toURL:childURL(directory, @"model-encoder.mlmodelc")
                           error:&error]);
    assert([manager removeItemAtURL:compiled error:&error]);
    return directory;
}

void assertCpuBackend(whisper_context * ctx) {
    const char * primary = whisper_get_primary_backend_name(ctx);
    assert(primary != nullptr);
    ggml_backend_dev_t device = ggml_backend_dev_by_name(primary);
    assert(device != nullptr);
    const auto type = ggml_backend_dev_type(device);
    assert(type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_ACCEL);
}

void encodeFixture(whisper_context * ctx) {
    constexpr int kMelFramesPerContext = 2;
    constexpr int kThreads = 1;
    const int frames = kMelFramesPerContext * whisper_n_audio_ctx(ctx);
    const int mels = whisper_model_n_mels(ctx);
    const std::vector<float> input(static_cast<size_t>(frames) * mels, 0.0f);
    assert(whisper_set_mel(ctx, input.data(), frames, mels) == 0);
    assert(whisper_encode(ctx, 0, kThreads) == 0);
}

} // namespace

void testWhisperCoremlState(const char * model_path) {
    @autoreleasepool {
        NSURL * directory = prepareFixture(model_path);
        NSURL * model = childURL(directory, @"model.bin");
        auto params = whisper_context_default_params();
        params.use_gpu = false;
        whisper_context * ctx = whisper_init_from_file_with_params([[model path] UTF8String], params);
        assert(ctx != nullptr);
        assert(whisper_encoder_on_coreml(ctx));
        assertCpuBackend(ctx);
        encodeFixture(ctx);
        assert(whisper_encoder_on_coreml(ctx));
        printf("Core ML encoder active with use_gpu=false, ggml backend=%s\n",
               whisper_get_primary_backend_name(ctx));
        whisper_free(ctx);
        ctx = whisper_init_from_file_with_params_no_state([[model path] UTF8String], params);
        assert(ctx != nullptr);
        assert(!whisper_encoder_on_coreml(ctx));
        whisper_free(ctx);
        NSError * error = nil;
        assert([[NSFileManager defaultManager] removeItemAtURL:directory error:&error]);
    }
}
