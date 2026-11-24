#include "digest.h"

#include "sha256.h"

#include <algorithm>
#include <vector>

namespace QVAC_COREML_NS {
namespace digest {

using detail::Sha256;

namespace {

template <size_t N>
void tag(Sha256 & h, const char (&t)[N]) {
    h.update(t, N);  // includes the terminating NUL ("...\0")
}

}  // namespace

Error source_digest(const recipe::Recipe & r, recipe::TensorSource & src, SourceDigest & out) {
    out = SourceDigest{};
    Sha256 h;
    tag(h, "qvac-coreml-source");
    h.u32le(1);
    std::vector<const recipe::Source *> sources;
    for (const recipe::Source & s : r.sources) sources.push_back(&s);
    std::sort(sources.begin(), sources.end(), [](auto * a, auto * b) { return a->name < b->name; });
    for (const recipe::Source * s : sources) {
        const recipe::TensorInfo * ti = src.tensor(s->name);
        if (!ti) return Error::deterministic("source tensor " + s->name + " is absent");
        h.str(s->name);
        h.u32le(ti->type);
        for (int64_t ne : ti->ne) h.i64le(ne);
        h.u64le(ti->nbytes);
        Sha256 th;
        uint64_t n = 0;
        Error e = src.read_raw(s->name, [&](const uint8_t * p, size_t len) {
            h.update(p, len);
            th.update(p, len);
            n += len;
            return true;
        });
        if (!e.ok()) return e;
        if (n != ti->nbytes) {
            return Error::transient("source tensor " + s->name + ": read " + std::to_string(n) + " of " +
                                        std::to_string(ti->nbytes) + " bytes",
                                    "qvac.coreml", 5 /* EIO */);
        }
        out.tensor_sha256[s->name] = th.finish();
        out.bytes += n;
        ++out.tensors;
    }
    std::vector<const recipe::Meta *> meta;
    for (const recipe::Meta & m : r.meta) meta.push_back(&m);
    std::sort(meta.begin(), meta.end(), [](auto * a, auto * b) { return a->key < b->key; });
    for (const recipe::Meta * m : meta) {
        uint32_t type = 0;
        std::vector<uint8_t> bytes;
        if (!src.meta_canonical(m->key, type, bytes)) return Error::deterministic("metadata " + m->key + " is absent");
        h.str(m->key);
        h.u32le(type);
        h.update(bytes.data(), bytes.size());
    }
    out.digest = h.finish();
    return {};
}

std::string artifact_key(uint32_t generator_version, const Hash & template_sha256, const Hash & source_digest,
                         std::string_view options) {
    Sha256 h;
    tag(h, "qvac-coreml-artifact");
    h.u32le(1);
    h.u32le(generator_version);
    h.update(template_sha256.data(), 32);
    h.update(source_digest.data(), 32);
    h.str(options);
    return detail::to_hex(h.finish()).substr(0, 32);
}

std::string memo_id(std::string_view realpath, uint64_t size, int64_t mtime_ns, uint64_t ino, uint64_t dev,
                    const Hash & template_sha256, uint32_t generator_version, std::string_view options) {
    Sha256 h;
    tag(h, "qvac-coreml-memo");
    h.str(realpath);
    h.u64le(size);
    h.i64le(mtime_ns);
    h.u64le(ino);
    h.u64le(dev);
    h.update(template_sha256.data(), 32);
    h.u32le(generator_version);
    h.str(options);
    return detail::to_hex(h.finish()).substr(0, 32);
}

}  // namespace digest
}  // namespace QVAC_COREML_NS
