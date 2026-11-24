#include "pocket/flow_lm.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

using tts_cpp::pocket::detail::FlowLM;

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("pocket-invalid-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    auto check = [&](const char * name, const std::function<void(gguf_context *, ggml_context *)> & change,
                     const std::string & expected) {
        auto * f = gguf_init_empty();
        auto * tensors = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        gguf_set_val_str(f, "general.architecture", "pocket-tts-flow-lm");
        gguf_set_val_u32(f, "pocket.schema_version", 1);
        gguf_set_val_str(f, "pocket.flow_type", "lsd");
        gguf_set_val_u32(f, "pocket.dim", 32);
        gguf_set_val_u32(f, "pocket.heads", 4);
        gguf_set_val_u32(f, "pocket.layers", 2);
        gguf_set_val_u32(f, "pocket.ff_dim", 96);
        gguf_set_val_u32(f, "pocket.latent_dim", 8);
        gguf_set_val_u32(f, "pocket.flow_dim", 16);
        gguf_set_val_u32(f, "pocket.flow_depth", 2);
        gguf_set_val_u32(f, "pocket.vocab_size", 64);
        gguf_set_val_f32(f, "pocket.rope_base", 10000);
        gguf_set_val_bool(f, "pocket.bos_before_voice", false);
        change(f, tensors);
        const bool wrote = gguf_write_to_file(f, path.string().c_str(), true);
        gguf_free(f); ggml_free(tensors);
        if (!wrote) throw std::runtime_error("cannot write test GGUF");
        try {
            FlowLM model(path.string());
        } catch (const std::runtime_error & e) {
            if (std::string(e.what()).find(expected) == std::string::npos)
                throw std::runtime_error(std::string(name) + ": wrong failure: " + e.what());
            std::printf("PASS %s\n", name);
            return;
        }
        throw std::runtime_error(std::string(name) + ": accepted invalid GGUF");
    };
    try {
        check("architecture", [](auto * f, auto *) { gguf_set_val_str(f, "general.architecture", "llama"); }, "unsupported general.architecture");
        check("schema type", [](auto * f, auto *) { gguf_set_val_str(f, "pocket.schema_version", "1"); }, "wrong-typed metadata");
        check("future schema", [](auto * f, auto *) { gguf_set_val_u32(f, "pocket.schema_version", 2); }, "invalid metadata");
        check("zero heads", [](auto * f, auto *) { gguf_set_val_u32(f, "pocket.heads", 0); }, "invalid metadata");
        check("non-divisible heads", [](auto * f, auto *) { gguf_set_val_u32(f, "pocket.heads", 3); }, "invalid architecture");
        check("wrong-typed BOS", [](auto * f, auto *) { gguf_set_val_u32(f, "pocket.bos_before_voice", 0); }, "wrong-typed metadata");
        check("wrong-typed rope", [](auto * f, auto *) { gguf_set_val_u32(f, "pocket.rope_base", 10000); }, "wrong-typed metadata");
        check("unsupported sampler", [](auto * f, auto *) { gguf_set_val_str(f, "pocket.flow_type", "flow_matching"); }, "unsupported pocket.flow_type");
        check("missing weights", [](auto *, auto *) {}, "missing tensor: conditioner.embed.weight");
        check("wrong weight shape", [](auto * f, auto * c) {
            auto * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, 31, 65);
            ggml_set_name(t, "conditioner.embed.weight"); gguf_add_tensor(f, t);
        }, "invalid tensor shape/type");
        check("wrong weight type", [](auto * f, auto * c) {
            auto * t = ggml_new_tensor_2d(c, GGML_TYPE_I32, 32, 65);
            ggml_set_name(t, "conditioner.embed.weight"); gguf_add_tensor(f, t);
        }, "invalid tensor shape/type");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::filesystem::remove(path);
        return 1;
    }
    std::filesystem::remove(path);
    return 0;
}
