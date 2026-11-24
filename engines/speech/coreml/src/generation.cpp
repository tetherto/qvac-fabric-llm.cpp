// Generation glue for engines (include/qvac_coreml/generation.h).

#include "qvac_coreml/generation.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

#if defined(QVAC_COREML_GENERATE)
#include "cache.h"
#include "device.h"
#include "digest.h"
#include "failed.h"
#include "generation_internal.h"
#include "gguf_source.h"
#include "templates.h"

#include "nlohmann/json.hpp"

#include <climits>
#include <ctime>
#include <cstdlib>
#include <map>
#include <mutex>

#include <unistd.h>
#if defined(QVAC_COREML_APPLE_COMPILE)
#include "compile.h"
#endif
#endif

namespace QVAC_COREML_NS {
namespace generation {

namespace {

constexpr uint64_t MIB = 1024ull * 1024ull;

uint64_t sat_add(uint64_t a, uint64_t b) { return a > std::numeric_limits<uint64_t>::max() - b ? UINT64_MAX : a + b; }

uint64_t scaled(uint64_t bytes, double factor) {
    const double v = std::ceil(static_cast<double>(bytes) * factor);
    return v >= 1.8e19 ? UINT64_MAX : static_cast<uint64_t>(v);
}

std::atomic<CompileOverride> g_compile_override{nullptr};

}  // namespace

uint64_t resident_bytes(const TemplateMemory & m) {
    uint64_t r = m.runtime_weight_bytes != 0 ? m.runtime_weight_bytes : m.weight_bin_size;
    r = sat_add(r, sat_add(m.io_bytes, m.io_bytes));
    r = sat_add(r, m.state_bytes);
    return sat_add(r, m.workspace_bytes);
}

uint64_t prepare_peak_bytes(const TemplateMemory & m) {
    const uint64_t compile = m.compile_peak_bytes ? *m.compile_peak_bytes : m.weight_bin_size / 2;
    return sat_add(sat_add(m.generation_peak_bytes, 8 * MIB), compile);
}

uint64_t load_peak_bytes(const TemplateMemory & m, ComputeUnits units) {
    std::optional<uint64_t> load;
    switch (units) {
        case ComputeUnits::cpu_and_ane: load = m.load_peak_cpu_and_ane; break;
        case ComputeUnits::all: load = m.load_peak_all; break;
        case ComputeUnits::cpu_and_gpu: load = m.load_peak_cpu_and_gpu; break;
        case ComputeUnits::cpu_only: load = m.load_peak_cpu_only; break;
    }
    return load ? *load : (units == ComputeUnits::cpu_and_ane ? m.weight_bin_size / 4 : m.weight_bin_size);
}

uint64_t transient_bytes(const TemplateMemory & m, ComputeUnits units) {
    return std::max(prepare_peak_bytes(m), load_peak_bytes(m, units));
}

uint64_t envelope_bytes(const TemplateMemory & m, ComputeUnits units) {
    return scaled(sat_add(resident_bytes(m), transient_bytes(m, units)), m.margin > 0 ? m.margin : 1.25);
}

uint64_t ane_compile_reserve_bytes(const TemplateMemory & m, ComputeUnits units) {
    if (!units_include_ane(units)) return 0;
    return m.ane_compile_reserve_bytes ? *m.ane_compile_reserve_bytes : scaled(m.weight_bin_size, 1.6);
}

void set_compile_override(CompileOverride fn) { g_compile_override.store(fn); }

#if defined(QVAC_COREML_GENERATE)

bool built() { return true; }

namespace {

using json = nlohmann::json;

std::optional<uint64_t> opt_u64(const json & j, const char * key) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_number_unsigned()) return std::nullopt;
    return it->get<uint64_t>();
}

uint64_t dtype_bytes(const std::string & dtype) {
    if (dtype == "fp16") return 2;
    if (dtype == "fp64") return 8;
    return 4;  // fp32, int32
}

TemplateMemory memory_of(const recipe::Template & t) {
    TemplateMemory m;
    m.weight_bin_size = t.info().weight_bin_size;
    m.runtime_weight_bytes = m.weight_bin_size;
    m.generation_peak_bytes = t.info().generation_peak_bytes;
    const recipe::TemplateFile * f = t.file("template.json");
    const json j = json::parse(std::string(reinterpret_cast<const char *>(f->data), f->size), nullptr, false);
    if (!j.is_object()) return m;
    m.state_bytes = opt_u64(j, "state_bytes").value_or(0);
    if (auto fns = j.find("functions"); fns != j.end() && fns->is_array()) {
        for (const json & fn : *fns) {
            for (const char * side : {"inputs", "outputs"}) {
                auto io = fn.find(side);
                if (io == fn.end() || !io->is_array()) continue;
                for (const json & x : *io) {
                    uint64_t n = dtype_bytes(x.value("dtype", std::string("fp32")));
                    if (auto shape = x.find("shape"); shape != x.end() && shape->is_array()) {
                        for (const json & d : *shape) {
                            if (d.is_number_unsigned()) n = n * d.get<uint64_t>();
                        }
                    }
                    m.io_bytes = sat_add(m.io_bytes, n);
                }
            }
        }
    }
    auto mem = j.find("memory");
    if (mem == j.end() || !mem->is_object()) return m;
    if (auto v = opt_u64(*mem, "runtime_weight_bytes"); v && *v != 0) m.runtime_weight_bytes = *v;
    m.workspace_bytes = opt_u64(*mem, "workspace_bytes").value_or(0);
    m.compile_peak_bytes = opt_u64(*mem, "compile_peak_bytes");
    m.ane_compile_reserve_bytes = opt_u64(*mem, "ane_compile_reserve_bytes");
    if (auto lp = mem->find("load_peak_bytes"); lp != mem->end() && lp->is_object()) {
        m.load_peak_cpu_and_ane = opt_u64(*lp, "cpu_and_ane");
        m.load_peak_all = opt_u64(*lp, "all");
        m.load_peak_cpu_and_gpu = opt_u64(*lp, "cpu_and_gpu");
        m.load_peak_cpu_only = opt_u64(*lp, "cpu_only");
    }
    if (auto mg = mem->find("margin"); mg != mem->end() && mg->is_number() && mg->get<double>() > 0) {
        m.margin = mg->get<double>();
    }
    if (auto vf = mem->find("validated_for"); vf != mem->end() && vf->is_array()) {
        for (const json & v : *vf) {
            if (!v.is_object() || !v.contains("units") || !v["units"].is_string()) continue;
            if (auto u = parse_compute_units(v["units"].get<std::string>())) m.validated_units.push_back(*u);
        }
    }
    return m;
}

// Templates found through the development override are parsed per directory
// and kept for the process, so PrepareRequest::tmpl stays valid like an
// embedded template's.
std::shared_ptr<const recipe::Template> find_kept(const std::string & id, const std::string & template_dir) {
    if (template_dir.empty() || !template_dir_override_compiled()) return templates::embedded(id);
    static auto * mu = new std::mutex();  // leaked with the map: used after static destructors
    static auto * kept = new std::map<std::string, std::shared_ptr<const recipe::Template>>();
    const std::string key = template_dir + "/" + id;
    std::lock_guard<std::mutex> lk(*mu);
    if (auto it = kept->find(key); it != kept->end()) return it->second;
    std::shared_ptr<const recipe::Template> t = templates::find(id, template_dir);
    if (t) kept->emplace(key, t);
    return t;
}

int64_t unix_now() { return static_cast<int64_t>(std::time(nullptr)); }

// Loads of a rejected artifact retried while its quarantine is busy or after
// another process replaced it, counted by its load failure record.
constexpr int kBusyRequeues = 3;

std::string real_path(const std::string & p) {
    char buf[PATH_MAX];
    return ::realpath(p.c_str(), buf) ? std::string(buf) : p;
}

// <cache>/qvac-coreml-v1/artifacts/<template-id>.<key32>/model.mlmodelc
bool split_artifact_path(const std::string & model_path, std::string & cache_dir, std::string & template_id,
                         std::string & key32) {
    const std::string tail = "/model.mlmodelc";
    if (model_path.size() <= tail.size() || model_path.compare(model_path.size() - tail.size(), tail.size(), tail) != 0) {
        return false;
    }
    const std::string dir = model_path.substr(0, model_path.size() - tail.size());
    const size_t slash = dir.rfind('/');
    if (slash == std::string::npos) return false;
    const std::string name = dir.substr(slash + 1);
    const std::string parent = dir.substr(0, slash);
    const std::string marker = std::string("/") + cache::ROOT + "/artifacts";
    if (parent.size() <= marker.size() || parent.compare(parent.size() - marker.size(), marker.size(), marker) != 0) {
        return false;
    }
    const size_t dot = name.rfind('.');
    if (dot == std::string::npos || name.size() - dot - 1 != 32) return false;
    template_id = name.substr(0, dot);
    key32 = name.substr(dot + 1);
    cache_dir = parent.substr(0, parent.size() - marker.size());
    return true;
}

cache::CompileResult compile_step(const std::string & pkg, const std::string & out) {
    if (CompileOverride fn = g_compile_override.load()) {
        cache::CompileResult r;
        const std::string err = fn(pkg, out);
        if (!err.empty()) r.error = Error::transient(err, "qvac.coreml.test", 1);
        r.os_build = detail::device_os_build();
        r.coremlc = "test";
        return r;
    }
#if defined(QVAC_COREML_APPLE_COMPILE)
    return apple::compile_package(pkg, out);
#else
    cache::CompileResult r;
    r.error = Error::deterministic("this build cannot compile Core ML packages");
    return r;
#endif
}

}  // namespace

std::vector<Candidate> probe(const std::vector<std::string> & template_ids, const std::string & gguf_path,
                             const Resolved & resolved) {
    std::vector<Candidate> out;
    std::unique_ptr<GgufSource> src;
    Error open_error;
    bool opened = false;
    for (const std::string & id : template_ids) {
        Candidate c;
        c.template_id = id;
        std::shared_ptr<const recipe::Template> t;
        try {
            t = find_kept(id, resolved.template_dir);
        } catch (const std::exception & e) {
            c.reason = std::string("template ") + id + " is invalid: " + e.what();
            out.push_back(std::move(c));
            continue;
        }
        if (!t) {
            c.reason = "the library has no template " + id;
            out.push_back(std::move(c));
            continue;
        }
        c.available = true;
        c.tmpl = t.get();
        c.sha256_hex = t->sha256_hex();
        c.default_units = parse_compute_units(t->info().default_compute_units);
        c.memory = memory_of(*t);
        if (!opened) {
            src = GgufSource::open(gguf_path, open_error);
            opened = true;
        }
        if (!src) {
            c.reason = "cannot read the GGUF: " + open_error.message;
        } else {
            const recipe::MatchResult m = recipe::match(*t, *src);
            c.matched = m.ok;
            c.reason = m.reason;
        }
        out.push_back(std::move(c));
    }
    return out;
}

bool source_stat(const std::string & path, SourceStat & out) {
    if (!stat_path(path, out)) return false;
    out.path = real_path(path);
    return true;
}

bool source_stat_fd(int fd, const std::string & path, SourceStat & out) {
    if (fd < 0 || !stat_fd(fd, out)) return false;
    out.path = real_path(path);
    out.changed = false;
    return true;
}

SourceStat loaded_source(const SourceStat & before, bool ok_before, const SourceStat & after, bool ok_after,
                         const std::string & path) {
    SourceStat s = after;
    s.changed = !ok_before || !ok_after || before.changed || after.changed || !same_identity(before, after) ||
                before.path != after.path || real_path(path) != after.path;
    return s;
}

PrepareRequest prepare_request(const Candidate & c, const SourceStat & source, const Resolved & resolved) {
    PrepareRequest r;
    r.template_id = c.template_id;
    r.source = source;
    r.cache_dir = resolved.cache_dir;
    r.options = digest::DEFAULT_OPTIONS;
    r.tmpl = c.tmpl;
    if (source.changed) return r;  // key "": fails on its own, never joins another engine's job
    // Two engines over the same file and template share one job.
    r.key = c.template_id + "\n" + c.sha256_hex + "\n" + resolved.cache_dir + "\n" + source.path + "\n" +
            std::to_string(source.size) + "\n" + std::to_string(source.mtime_ns) + "\n" + std::to_string(source.ino) +
            "\n" + std::to_string(source.dev);
    return r;
}

PrepareResult prepare(const PrepareRequest & req, PrepareContext & ctx) {
    PrepareResult r;
    if (req.tmpl == nullptr) {
        r.error = Error::deterministic("prepare: no template");
        return r;
    }
    const auto * t = static_cast<const recipe::Template *>(req.tmpl);
    cache::PrepareInputs in;
    in.cache_dir = req.cache_dir;
    // The template outlives the job (embedded, or kept by find_kept): a non-owning handle.
    in.tmpl = std::shared_ptr<const recipe::Template>(t, [](const recipe::Template *) {});
    in.expected = req.source;
    in.options = req.options.empty() ? std::string(digest::DEFAULT_OPTIONS) : req.options;
    in.os_build = detail::device_os_build();
    in.compile = &compile_step;
    in.open = [](const std::string & path, Error & err) -> std::unique_ptr<OpenSource> {
        return GgufSource::open(path, err);
    };
    cache::PrepareOutcome o = cache::prepare(in, &ctx);
    return internal::result_of(o);
}

namespace internal {

PrepareResult result_of(cache::PrepareOutcome & o) {
    PrepareResult r;
    switch (o.kind) {
        case cache::PrepareOutcome::Kind::ready:
            r.artifact_path = o.model_path;
            r.artifact_key = o.key32;
            r.cache_hit = o.cache_hit;
            break;
        case cache::PrepareOutcome::Kind::would_block: r.would_block = true; break;
        case cache::PrepareOutcome::Kind::unsupported:
            r.unsupported = o.reason.empty() ? "cache_fs" : o.reason;
            r.error = o.error;
            r.error.cls = ErrorClass::none;
            break;
        case cache::PrepareOutcome::Kind::failed:
            r.error = o.error.ok() ? Error::transient(o.reason.empty() ? "prepare failed" : o.reason) : o.error;
            break;
    }
    return r;
}

}  // namespace internal

bool begin_load(const LoadRequest & req, LoadResult & out) {
    if (req.tmpl == nullptr) return true;  // prebuilt
    std::string cache_dir, template_id, key32;
    if (!split_artifact_path(req.artifact_path, cache_dir, template_id, key32)) return true;
    cache::Layout l(cache_dir);
    // A load failure promoted to deterministic (the same error in another
    // process on this OS build) stops further loads of this (function, units).
    const failed::Verdict v = failed::check(l.failed, key32, detail::device_os_build(), unix_now(), req.function,
                                            to_string(req.units));
    if (v.blocked && v.rec.cls == ErrorClass::deterministic) {
        out.error = v.error;
        return false;
    }
    cache::Adoption a = cache::adopt(l, template_id, key32);
    switch (a.kind) {
        case cache::Adoption::Kind::adopted:
            // Under the shared lease: which publication this load adopted.
            cache::artifact_identity(l, template_id, key32, out.artifact_dev, out.artifact_ino);
            out.lease_fd = a.lease.release_fd();
            return true;
        case cache::Adoption::Kind::would_block: out.would_block = true; return false;
        case cache::Adoption::Kind::missing:
            out.artifact_gone = true;  // the scheduler prepares it again
            out.error = Error::transient("the generated artifact " + req.artifact_path + " is gone");
            return false;
        case cache::Adoption::Kind::invalid:
            out.artifact_gone = true;  // preparation quarantines and regenerates it
            out.error = Error::transient("the generated artifact " + req.artifact_path + " is incomplete");
            return false;
        case cache::Adoption::Kind::error: out.error = a.error; return false;
    }
    return false;
}

void end_load(const LoadRequest & req, LoadResult & out) {
    if (req.tmpl == nullptr) return;
    std::string cache_dir, template_id, key32;
    if (!split_artifact_path(req.artifact_path, cache_dir, template_id, key32)) return;
    const cache::Layout l(cache_dir);
    const std::string units = to_string(req.units);
    if (out.model) {
        cache::touch_used(l, key32);
        failed::clear(l.failed, key32, req.function, units);  // a success ends the failure streak
        return;
    }
    if (out.lease_fd >= 0) {
        ::close(out.lease_fd);
        out.lease_fd = -1;
    }
    if (out.error.ok()) return;
    // The (function, units) load record: a repeat of the same error in
    // another process on this OS build promotes it to deterministic (then
    // begin_load stops loading it), except for resource errors.
    const failed::Record rec = failed::record(l.failed, key32, failed::Stage::load, out.error, template_id,
                                              detail::device_os_build(), unix_now(), req.function, units);
    // The engine rejected an artifact that passed the inventory check (the
    // framework refused the model, or its interface or geometry is wrong):
    // move it to quarantine/ under the key's exclusive lock and send the slot
    // back to preparation, which regenerates it at the same key (once). A
    // resource error (memory, disk, timeout, interruption) keeps the artifact,
    // and so does one another holder of a lease is using.
    const bool rejected = out.rejected || out.error.cls == ErrorClass::deterministic;
    if (!rejected || failed::is_resource(out.error.domain, out.error.code)) return;
    out.rejected = true;
    bool replaced = false;
    const cache::KeyLock::Result q =
        cache::quarantine_if(l, template_id, key32, out.artifact_dev, out.artifact_ino, &replaced, nullptr);
    if (q == cache::KeyLock::Result::ok && !replaced) {
        out.artifact_gone = true;
    } else if ((q == cache::KeyLock::Result::would_block || (q == cache::KeyLock::Result::ok && replaced)) &&
               rec.attempts <= kBusyRequeues) {
        // Another holder's lease (the quarantine waits for it), or another
        // process already replaced the publication (its repair is not
        // quarantined): load again after the scheduler's backoff.
        out.would_block = true;
    }
}

#else  // !QVAC_COREML_GENERATE

bool built() { return false; }

std::vector<Candidate> probe(const std::vector<std::string> & template_ids, const std::string &, const Resolved &) {
    std::vector<Candidate> out;
    for (const std::string & id : template_ids) {
        Candidate c;
        c.template_id = id;
        c.reason = "this build has no Core ML generator";
        out.push_back(std::move(c));
    }
    return out;
}

bool source_stat(const std::string &, SourceStat &) { return false; }

bool source_stat_fd(int, const std::string &, SourceStat &) { return false; }

SourceStat loaded_source(const SourceStat &, bool, const SourceStat & after, bool, const std::string &) {
    SourceStat s = after;
    s.changed = true;
    return s;
}

PrepareRequest prepare_request(const Candidate & c, const SourceStat & source, const Resolved & resolved) {
    PrepareRequest r;
    r.template_id = c.template_id;
    r.source = source;
    r.cache_dir = resolved.cache_dir;
    return r;
}

PrepareResult prepare(const PrepareRequest &, PrepareContext &) {
    PrepareResult r;
    r.error = Error::deterministic("this build has no Core ML generator");
    return r;
}

bool begin_load(const LoadRequest &, LoadResult &) { return true; }
void end_load(const LoadRequest &, LoadResult &) {}

#endif

}  // namespace generation
}  // namespace QVAC_COREML_NS
