#include "../src/llama-coreml-ffn-export.h"
#include "llama-coreml-ffn-templates.inc"

#include <cstdio>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

int main() {
    try {
        const _Float16 weights[] = {-2, 0, 2, 1, 0, 0, 0, 0};
        int8_t q[8], scaled[8];
        _Float16 scale[2], half_scale[2];
        otf_quantize(weights, 4, 2, 4, 1, q, scale);
        check(q[0] == -127 && q[1] == 0 && q[2] == 127 && q[3] == 64, "symmetric quantization and round-to-even");
        check(scale[1] == 0 && q[4] == 0 && q[7] == 0, "zero weight channel");
        otf_quantize(weights, 4, 2, 4, 0.5f, scaled, half_scale);
        check(memcmp(q, scaled, sizeof(q)) == 0 && half_scale[0] == _Float16(float(scale[0]) * 0.5f), "folded gate scaling");
        const _Float16 bad[] = {_Float16(INFINITY)};
        bool rejected = false;
        try { otf_quantize(bad, 1, 1, 1, 1, q, scale); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "non-finite weights must be rejected");
        for (auto entry : {std::make_pair(ffn_4608_layout_txt, sizeof(ffn_4608_layout_txt)),
                           std::make_pair(ffn_6144_layout_txt, sizeof(ffn_6144_layout_txt))}) {
            std::istringstream layout(std::string(reinterpret_cast<const char *>(entry.first), entry.second));
            otf_template t;
            t.parse(layout);
            check(t.hidden == 5120 && (t.channels == 4608 || t.channels == 6144), "embedded template dimensions");
            uint64_t next = 64;
            for (const auto & blob : t.blobs) {
                check(blob.offset == next, "embedded blob alignment");
                next += 64 + blob.size;
                next += (64 - next % 64) % 64;
            }
            layout.clear();
            layout.seekg(0);
            t.parse(layout);
            check(t.blobs.size() < 32, "template parsing must replace previous state");
        }
        const auto dir = std::filesystem::current_path() / ("coreml-ffn-export-test-" + std::to_string(getpid()));
        struct cleanup { std::filesystem::path path; ~cleanup() { std::filesystem::remove_all(path); } } cleanup{dir};
        otf_template t;
        t.channels = 2;
        t.hidden = 4;
        t.program = {1, 2, 3};
        t.manifest = "{}";
        t.blobs = {{64, 4, 8, "up", "data", 0, 2, 2, 4}, {192, 1, 4, "up", "scale", 0, 2, 2, 1}};
        otf_layer layer;
        layer.hidden = 4;
        layer.inter = 2;
        layer.up.assign(weights, weights + 8);
        otf_write_package(t, layer, 0, dir);
        std::ifstream input(dir / "Data/com.apple.CoreML/weights/weight.bin", std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        check(bytes.size() == 320 && memcmp(bytes.data() + 128, q, 8) == 0, "package weight bytes");
        check(memcmp(bytes.data() + 256, scale, 4) == 0, "package scale bytes");
        const auto first = otf_cache_dir(dir, (dir / "Data/com.apple.CoreML/model.mlmodel").string(), t);
        t.program[0] = 9;
        check(first != otf_cache_dir(dir, (dir / "Data/com.apple.CoreML/model.mlmodel").string(), t), "template program must invalidate cache");
        printf("Core ML FFN export contracts passed\n");
        return 0;
    } catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
