// llama-server with the Core ML FFN sidecar of the decoder experiment.
// FFN_COREML_ASSETS selects the sidecar (layers.txt and layerN/); the Metal part reads its weights from the -m model.
// Without FFN_COREML_ASSETS this is a plain llama-server.
#define main ffn_coreml_bench_main
#include "decoder-coreml.mm"
#undef main

int llama_server(int argc, char ** argv);

static std::string model_argument(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) { return argv[i + 1]; }
    }
    return {};
}

int main(int argc, char ** argv) {
    const char * assets = std::getenv("FFN_COREML_ASSETS");
    if (!assets) { return llama_server(argc, argv); }
    // the runtime lives for the whole process, like the server's context
    static decoder_state state;
    @autoreleasepool {
        try {
            gpu_resident = std::getenv("FFN_COREML_GPU_RESIDENT") != nullptr;
            require(gpu_resident, "the Core ML server requires FFN_COREML_GPU_RESIDENT");
            ggml_backend_metal_set_custom2(decoder_compute, decoder_resident_compute);
            const std::string model = model_argument(argc, argv);
            require(!model.empty(), "the Core ML server needs the model path in -m");
            llama_backend_init();
            std::ifstream selected(std::filesystem::path(assets) / "layers.txt");
            require(bool(selected), "missing layers.txt in FFN_COREML_ASSETS");
            int layer;
            while (selected >> layer) {
                require(layer >= 0 && layer < 64 && !state.layers.count(layer), "invalid selected layer");
                auto runtime = std::make_unique<decoder_ffn>();
                runtime->init(model, (std::filesystem::path(assets) / ("layer" + std::to_string(layer))).string(), layer, false);
                state.layers.emplace(layer, std::move(runtime));
            }
            require(!state.layers.empty(), "no layers in layers.txt");
            llama_ffn_coreml_set_builder(decoder_build, &state);
            fprintf(stderr, "ffn_coreml: %zu layers from %s\n", state.layers.size(), assets);
        } catch (const std::exception & failure) {
            fprintf(stderr, "ffn_coreml: %s\n", failure.what());
            return 1;
        }
    }
    return llama_server(argc, argv);
}
