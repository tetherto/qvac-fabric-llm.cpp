#pragma once

// Content keys of the generation cache: source_digest, artifact_key and
// memo_id. template_sha256 is Template::sha256(); hint_id lives in hints.cpp.
// Every key is SHA-256 over little-endian encodings, identical on every host.

#include "qvac_coreml/recipe.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace digest {

using Hash = std::array<uint8_t, 32>;

struct SourceDigest {
    Hash digest{};
    // SHA-256 of each source tensor's raw payload, for the write-time
    // re-check of the bytes the package was written from.
    std::map<std::string, Hash> tensor_sha256;
    uint32_t tensors = 0;
    uint64_t bytes = 0;  // raw tensor bytes hashed
};

// source_digest = H("qvac-coreml-source\0" || u32le(1) ||
//   for s in sorted(recipe.sources by name):
//     str(s.name) || u32le(ggml_type) || i64le(ne0..ne3) || u64le(nbytes) || raw tensor bytes ||
//   for m in sorted(recipe.meta by key):
//     str(m.key) || u32le(gguf_type) || canonical LE value)
// Streams every source tensor through `src`. A missing tensor or metadata key
// is deterministic (the template no longer matches); read errors pass through.
Error source_digest(const recipe::Recipe & r, recipe::TensorSource & src, SourceDigest & out);

// hex(H("qvac-coreml-artifact\0" || u32le(1) || u32le(generator_version) ||
//       template_sha256 || source_digest || str(options)))[0:32]
std::string artifact_key(uint32_t generator_version, const Hash & template_sha256, const Hash & source_digest,
                         std::string_view options);

// hex(H("qvac-coreml-memo\0" || str(realpath) || u64le(size) || i64le(mtime_ns) || u64le(ino) ||
//       u64le(dev) || template_sha256 || u32le(generator_version) || str(options)))[0:32]
std::string memo_id(std::string_view realpath, uint64_t size, int64_t mtime_ns, uint64_t ino, uint64_t dev,
                    const Hash & template_sha256, uint32_t generator_version, std::string_view options);

// v1 options: sorted "k=v" pairs joined by ";" ("out=fp16").
constexpr const char * DEFAULT_OPTIONS = "out=fp16";

}  // namespace digest
}  // namespace QVAC_COREML_NS
