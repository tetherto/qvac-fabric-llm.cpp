// llama-server with the Core ML FFN sidecar of the decoder experiment.
// FFN_COREML_ASSETS selects the sidecar (layers.txt and layerN/); the Metal part reads its weights from the -m model.
// With FFN_COREML_OTF the sidecar is written from the -m model at startup instead (see decoder-coreml.mm).
// Without either this is a plain llama-server.
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
    if (!assets && !std::getenv("FFN_COREML_OTF")) { return llama_server(argc, argv); }
    // the runtime lives for the whole process, like the server's context
    static decoder_state state;
    @autoreleasepool {
        try {
            gpu_resident = std::getenv("FFN_COREML_GPU_RESIDENT") != nullptr;
            require(gpu_resident, "the Core ML server requires FFN_COREML_GPU_RESIDENT");
            ggml_backend_metal_set_custom2(decoder_compute, decoder_resident_compute);
            ggml_backend_metal_set_custom(ingraph_cpu, ingraph_execute);
            const std::string model = model_argument(argc, argv);
            require(!model.empty(), "the Core ML server needs the model path in -m");
            llama_backend_init();
            otf.init(model);
            load_layers(state, model, assets ? assets : "");
            set_builders(true, &state);
            fprintf(stderr, "ffn_coreml: %zu layers from %s\n", state.layers.size(), assets ? assets : "the model (load-time export)");
        } catch (const std::exception & failure) {
            fprintf(stderr, "ffn_coreml: %s\n", failure.what());
            return 1;
        }
    }
    return llama_server(argc, argv);
}
