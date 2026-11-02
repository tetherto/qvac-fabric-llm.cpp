#pragma once

// Load-time export of the FFN sidecar: no conversion step for users. A template made by make_template.py holds the
// int8 Core ML program for one channel count; for each layer and channel range the runtime writes weights/weight.bin
// from the GGUF (per-channel symmetric int8, computed like coremltools so the packages match an offline export),
// compiles the package and keeps the compiled model in a cache keyed by the GGUF and the range.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

struct otf_blob {
    uint64_t    offset; // of the blob metadata in weight.bin
    uint32_t    dtype;  // MILBlob data type: 1 = FP16, 4 = int8
    uint64_t    size;
    std::string role;   // up, gate, down, bias
    std::string kind;   // data, scale, zero
    int         start, end;
    int64_t     rows, cols;
};

struct otf_template {
    std::filesystem::path dir;
    std::vector<otf_blob> blobs;
    int channels = 0;
    int64_t hidden = 0;

    void load(const std::filesystem::path & path) {
        dir = path;
        std::ifstream layout(dir / "layout.txt");
        if (!layout) { throw std::runtime_error("missing layout.txt in " + dir.string()); }
        otf_blob b;
        while (layout >> b.offset >> b.dtype >> b.size >> b.role >> b.kind >> b.start >> b.end >> b.rows >> b.cols) {
            blobs.push_back(b);
            if (b.role == "up" && b.kind == "data") {
                channels = std::max(channels, b.end);
                hidden = b.cols;
            }
        }
        if (blobs.empty() || channels == 0) { throw std::runtime_error("empty template layout in " + dir.string()); }
    }
};

// FFN weights of one layer as FP16, rounded like gguf-py dequantize(...).astype(np.float16)
struct otf_layer {
    int64_t hidden = 0, inter = 0;
    std::vector<_Float16> gate, up; // [inter, hidden]
    std::vector<_Float16> down;     // [hidden, inter]

    void read(const std::string & gguf_path, int layer) {
        ggml_context * meta = nullptr;
        gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), {true, &meta});
        if (!gguf || !meta) { throw std::runtime_error("could not read GGUF header of " + gguf_path); }
        std::ifstream file(gguf_path, std::ios::binary);
        auto load = [&](const char * kind, std::vector<_Float16> & out, int64_t & ne0, int64_t & ne1) {
            const std::string name = "blk." + std::to_string(layer) + ".ffn_" + kind + ".weight";
            const int64_t id = gguf_find_tensor(gguf, name.c_str());
            const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
            if (id < 0 || !t || ggml_n_dims(t) != 2) { throw std::runtime_error("missing matrix " + name); }
            const auto * traits = ggml_get_type_traits(t->type);
            if (!traits->to_float) { throw std::runtime_error("no dequantization for " + name); }
            ne0 = t->ne[0];
            ne1 = t->ne[1];
            const size_t row_bytes = ggml_row_size(t->type, ne0);
            std::vector<char> raw(row_bytes * ne1);
            file.seekg(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id));
            file.read(raw.data(), raw.size());
            if (!file) { throw std::runtime_error("short read of " + name); }
            out.resize(size_t(ne0) * ne1);
            std::vector<float> row(ne0);
            for (int64_t r = 0; r < ne1; ++r) {
                traits->to_float(raw.data() + r * row_bytes, row.data(), ne0);
                for (int64_t c = 0; c < ne0; ++c) { out[size_t(r) * ne0 + c] = _Float16(row[c]); }
            }
        };
        int64_t g0, g1, u0, u1, d0, d1;
        load("gate", gate, g0, g1);
        load("up", up, u0, u1);
        load("down", down, d0, d1);
        gguf_free(gguf);
        ggml_free(meta);
        if (g0 != u0 || g1 != u1 || d0 != g1 || d1 != g0) { throw std::runtime_error("not a dense SwiGLU FFN"); }
        hidden = g0;
        inter = g1;
    }
};

// per-row symmetric int8 as coremltools linear_symmetric per_channel makes it: FP16 arithmetic, round half to even.
// pre scales the FP16 weights first (0.5 for gate: coremltools folds the x/2 of the tanh SiLU into the gate conv)
static void otf_quantize(const _Float16 * src, size_t stride, int64_t rows, int64_t cols, float pre, int8_t * q, _Float16 * s) {
    std::vector<_Float16> v(cols);
    for (int64_t r = 0; r < rows; ++r) {
        float amax = 0;
        for (int64_t c = 0; c < cols; ++c) {
            v[c] = pre == 1 ? src[r * stride + c] : _Float16(float(src[r * stride + c]) * pre);
            amax = std::max(amax, std::fabs(float(v[c])));
        }
        const _Float16 scale = _Float16(amax * 2 / 254);
        s[r] = scale;
        for (int64_t c = 0; c < cols; ++c) {
            const float x = scale == 0 ? 0 : float(_Float16(float(v[c]) / float(scale)));
            q[r * cols + c] = int8_t(std::clamp(std::nearbyint(x), -127.0f, 127.0f));
        }
    }
}

// writes <dst> (an .mlpackage) with FFN channels [c0, c0 + template channels) of the layer
static void otf_write_package(const otf_template & t, const otf_layer & w, int c0, const std::filesystem::path & dst) {
    namespace fs = std::filesystem;
    if (c0 < 0 || c0 + t.channels > w.inter || t.hidden != w.hidden) { throw std::runtime_error("template does not fit the layer"); }
    const fs::path data = dst / "Data/com.apple.CoreML";
    fs::create_directories(data / "weights");
    fs::copy_file(t.dir / "Manifest.json", dst / "Manifest.json", fs::copy_options::overwrite_existing);
    fs::copy_file(t.dir / "model.mlmodel", data / "model.mlmodel", fs::copy_options::overwrite_existing);

    std::ofstream out(data / "weights/weight.bin", std::ios::binary);
    auto pad = [&]() {
        static const char zeros[64] = {};
        const uint64_t at = uint64_t(out.tellp());
        out.write(zeros, (64 - at % 64) % 64);
    };
    char header[64] = {};
    const uint32_t count = uint32_t(t.blobs.size()), version = 2;
    memcpy(header, &count, 4);
    memcpy(header + 4, &version, 4);
    out.write(header, 64);

    // data and scale blobs of one chunk come from one quantization
    std::map<std::pair<std::string, int>, std::pair<std::vector<int8_t>, std::vector<_Float16>>> chunks;
    auto quantized = [&](const otf_blob & b) -> std::pair<std::vector<int8_t>, std::vector<_Float16>> & {
        auto & entry = chunks[{b.role, b.start}];
        if (entry.first.empty()) {
            const int n = b.end - b.start;
            if (b.role == "down") {
                entry.first.resize(size_t(w.hidden) * n);
                entry.second.resize(w.hidden);
                otf_quantize(w.down.data() + c0 + b.start, w.inter, w.hidden, n, 1, entry.first.data(), entry.second.data());
            } else {
                const auto & src = b.role == "gate" ? w.gate : w.up;
                entry.first.resize(size_t(n) * w.hidden);
                entry.second.resize(n);
                otf_quantize(src.data() + size_t(c0 + b.start) * w.hidden, w.hidden, n, w.hidden, b.role == "gate" ? 0.5f : 1, entry.first.data(), entry.second.data());
            }
        }
        return entry;
    };
    for (const auto & b : t.blobs) {
        if (uint64_t(out.tellp()) != b.offset) { throw std::runtime_error("weight.bin layout differs from the template"); }
        char meta[64] = {};
        const uint32_t sentinel = 0xDEADBEEF;
        const uint64_t data_offset = b.offset + 64;
        memcpy(meta, &sentinel, 4);
        memcpy(meta + 4, &b.dtype, 4);
        memcpy(meta + 8, &b.size, 8);
        memcpy(meta + 16, &data_offset, 8);
        out.write(meta, 64);
        if (b.kind == "zero") {
            const std::vector<char> zeros(b.size);
            out.write(zeros.data(), zeros.size());
        } else {
            const auto & q = quantized(b);
            const char * bytes = b.kind == "data" ? (const char *) q.first.data() : (const char *) q.second.data();
            const size_t size = b.kind == "data" ? q.first.size() : q.second.size() * sizeof(_Float16);
            if (size != b.size) { throw std::runtime_error("blob size differs from the template"); }
            out.write(bytes, size);
        }
        pad();
    }
    if (!out) { throw std::runtime_error("could not write " + (data / "weights/weight.bin").string()); }
}

// cache folder for one GGUF and template: the path, size and modification time of the GGUF and the template layout
static std::filesystem::path otf_cache_dir(const std::filesystem::path & root, const std::string & gguf_path, const otf_template & t) {
    struct stat st = {};
    if (stat(gguf_path.c_str(), &st) != 0) { throw std::runtime_error("cannot stat " + gguf_path); }
    std::ifstream layout(t.dir / "layout.txt");
    std::stringstream text;
    text << layout.rdbuf();
    // the template name too: templates for other row counts share the weight layout
    const std::string key = std::filesystem::absolute(gguf_path).string() + "|" + std::to_string(st.st_size) + "|" + std::to_string(st.st_mtime) + "|" +
        t.dir.filename().string() + "|" + text.str();
    char name[32];
    snprintf(name, sizeof(name), "%016zx", std::hash<std::string>{}(key));
    return root / name;
}

#ifdef __OBJC__
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

// compiled model for channels [c0, c0 + template channels) of a layer; exported and compiled on first use
static NSURL * otf_compiled(const otf_template & t, const std::filesystem::path & cache, int layer, int c0,
                            const std::function<const otf_layer &()> & weights, double & export_s, double & compile_s) {
    namespace fs = std::filesystem;
    const fs::path compiled = cache / ("layer" + std::to_string(layer) + "-" + std::to_string(c0) + ".mlmodelc");
    export_s = compile_s = 0;
    if (fs::exists(compiled)) { return [NSURL fileURLWithPath:[NSString stringWithUTF8String:compiled.c_str()]]; }
    fs::create_directories(cache);
    const fs::path package = cache / ("layer" + std::to_string(layer) + "-" + std::to_string(c0) + "." + std::to_string(getpid()) + ".mlpackage");
    auto t0 = std::chrono::steady_clock::now();
    otf_write_package(t, weights(), c0, package);
    auto t1 = std::chrono::steady_clock::now();
    NSError * failure = nil;
    NSURL * url = [MLModel compileModelAtURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:package.c_str()]] error:&failure];
    auto t2 = std::chrono::steady_clock::now();
    fs::remove_all(package);
    if (!url) { throw std::runtime_error(std::string("Core ML compilation failed: ") + (failure ? failure.localizedDescription.UTF8String : "")); }
    // compileModelAtURL writes to a temporary folder; move it into the cache in one step
    const fs::path staging = compiled.string() + "." + std::to_string(getpid()) + ".tmp";
    fs::remove_all(staging);
    std::error_code moved;
    fs::rename(url.path.UTF8String, staging, moved);
    if (moved) {
        fs::copy(url.path.UTF8String, staging, fs::copy_options::recursive);
        fs::remove_all(url.path.UTF8String);
    }
    fs::rename(staging, compiled);
    export_s = std::chrono::duration<double>(t1 - t0).count();
    compile_s = std::chrono::duration<double>(t2 - t1).count();
    return [NSURL fileURLWithPath:[NSString stringWithUTF8String:compiled.c_str()]];
}
#endif
