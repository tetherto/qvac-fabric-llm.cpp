#pragma once

// The generator's view of a source GGUF, read through one open descriptor.
//
// GgufSource opens the file once (O_RDONLY), fstat's that descriptor, parses
// the header through it (gguf_init_from_file_ptr, metadata only) and streams
// every tensor payload through it with qvac_coreml/gguf_stream.h in bounded
// 8 MiB reads. Dequantisation is ggml-base only:
// ggml_get_type_traits(type)->to_float, F32 copied as is; never a ggml-cpu
// symbol (Android builds load ggml-cpu dynamically).
//
// This header needs no ggml: the pipeline (cache.cpp) and the tests see only
// OpenSource; gguf_source.cpp is compiled where ggml is available.

#include "qvac_coreml/recipe.h"
#include "qvac_coreml/slot.h"

#include <memory>
#include <string>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {

// A TensorSource read through one open descriptor whose identity can be
// re-checked when the job opens it and again before it publishes.
class OpenSource : public recipe::TensorSource {
public:
    // fstat of the open descriptor (path is left empty).
    virtual bool stat_now(SourceStat & out) const = 0;
};

// Stat helpers shared by the pipeline, prune and the tests.
bool stat_path(const std::string & path, SourceStat & out);  // follows symlinks
bool stat_fd(int fd, SourceStat & out);
bool same_identity(const SourceStat & a, const SourceStat & b);  // size, mtime_ns, ino, dev

class GgufSource final : public OpenSource {
public:
    // Errors: ENOENT / EACCES / short header -> transient I/O; a file that is
    // not a GGUF -> deterministic.
    static std::unique_ptr<GgufSource> open(const std::string & path, Error & err);
    ~GgufSource() override;

    const recipe::TensorInfo * tensor(const std::string & name) const override;
    Error read_raw(const std::string & name, const std::function<bool(const uint8_t *, size_t)> & sink) override;
    Error read_f32(const std::string & name, std::vector<float> & out, const recipe::RawTap & tap) override;
    bool meta_canonical(const std::string & key, uint32_t & gguf_type, std::vector<uint8_t> & bytes,
                        uint32_t * array_elem_type = nullptr) const override;
    std::string meta_json(const std::string & key) const override;
    bool stat_now(SourceStat & out) const override;

    // Every tensor in file order and every metadata key (qvac-coreml-tool).
    const std::vector<recipe::TensorInfo> & tensors() const;
    std::vector<std::string> meta_keys() const;
    int fd() const;
    // The parsed gguf_context (qvac-coreml-tool copies metadata from it).
    const void * gguf() const;

private:
    struct Impl;
    explicit GgufSource(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace QVAC_COREML_NS
