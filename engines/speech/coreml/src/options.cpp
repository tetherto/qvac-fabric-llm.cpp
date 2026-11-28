#include "qvac_coreml/options.h"

#include "device.h"

#include "nlohmann/json.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace QVAC_COREML_NS {

const char * to_string(Use v) {
    switch (v) {
        case Use::unset: return "unset";
        case Use::automatic: return "auto";
        case Use::off: return "off";
        case Use::require: return "require";
    }
    return "?";
}

const char * to_string(Generate v) {
    switch (v) {
        case Generate::unset: return "unset";
        case Generate::never: return "never";
        case Generate::background: return "background";
    }
    return "?";
}

const char * to_string(Wait v) {
    switch (v) {
        case Wait::unset: return "unset";
        case Wait::none: return "none";
        case Wait::construct: return "construct";
        case Wait::call: return "call";
    }
    return "?";
}

const char * to_string(Prewarm v) {
    switch (v) {
        case Prewarm::unset: return "unset";
        case Prewarm::none: return "none";
        case Prewarm::minimal: return "minimal";
        case Prewarm::predicted: return "predicted";
        case Prewarm::all: return "all";
    }
    return "?";
}

const char * to_string(ComputeUnits v) {
    switch (v) {
        case ComputeUnits::all: return "all";
        case ComputeUnits::cpu_and_ane: return "cpu_and_ane";
        case ComputeUnits::cpu_and_gpu: return "cpu_and_gpu";
        case ComputeUnits::cpu_only: return "cpu_only";
    }
    return "?";
}

const char * to_string(Source v) {
    switch (v) {
        case Source::explicit_field: return "explicit";
        case Source::engine_env: return "engine_env";
        case Source::shared_env: return "shared_env";
        case Source::default_value: return "default";
    }
    return "?";
}

std::optional<ComputeUnits> parse_compute_units(std::string_view t) {
    if (t == "all") return ComputeUnits::all;
    if (t == "cpu_and_ane") return ComputeUnits::cpu_and_ane;
    if (t == "cpu_and_gpu") return ComputeUnits::cpu_and_gpu;
    if (t == "cpu_only") return ComputeUnits::cpu_only;
    return std::nullopt;
}

bool units_include_ane(ComputeUnits u) { return u == ComputeUnits::all || u == ComputeUnits::cpu_and_ane; }

bool os_at_least(const OsVersion & v, int major, int minor) {
    return v.major > major || (v.major == major && v.minor >= minor);
}

bool template_dir_override_compiled() {
#if defined(QVAC_COREML_DEV)
    return true;
#else
    return false;
#endif
}

namespace {
const char * system_getenv(const char * name) { return std::getenv(name); }
}  // namespace

Resolved resolve_coreml(const Options & o, std::string_view prefix, GetenvFn getenv_fn,
                        std::optional<Platform> platform) {
    if (!getenv_fn) getenv_fn = &system_getenv;
    const Platform plat = platform ? *platform : detail::device_platform();
    const std::string p(prefix);
    auto env = [&](const char * suffix) -> const char * {
        const std::string name = p + suffix;
        return getenv_fn(name.c_str());
    };

    Resolved r;

    // use: kill switch, then explicit field, then <P>_COREML_STRICT, then auto.
    if (env("_COREML_DISABLE") != nullptr) {
        r.use = Use::off;
        r.use_source = Source::engine_env;
        r.disabled_reason = p + "_COREML_DISABLE";
    } else if (o.use != Use::unset) {
        r.use = o.use;
        r.use_source = Source::explicit_field;
        if (r.use == Use::off) r.disabled_reason = "use=off";
    } else if (env("_COREML_STRICT") != nullptr) {
        r.use = Use::require;
        r.use_source = Source::engine_env;
    }

    r.generate = o.generate != Generate::unset ? o.generate : Generate::background;
    r.prewarm = o.prewarm != Prewarm::unset ? o.prewarm : Prewarm::predicted;
    // iOS keeps today's blocking behaviour until an iOS measurement lands.
    r.wait = o.wait != Wait::unset ? o.wait : (plat == Platform::ios ? Wait::call : Wait::none);

    if (o.cache_dir.has_value()) {
        r.cache_dir = *o.cache_dir;
        r.cache_dir_source = Source::explicit_field;
    } else if (const char * c = getenv_fn("QVAC_COREML_CACHE_DIR"); c != nullptr && *c != '\0') {
        r.cache_dir = c;
        r.cache_dir_source = Source::shared_env;
    }

    if (template_dir_override_compiled()) {
        if (o.template_dir.has_value()) {
            r.template_dir = *o.template_dir;
        } else if (const char * d = getenv_fn("QVAC_COREML_TEMPLATE_DIR"); d != nullptr && *d != '\0') {
            r.template_dir = d;
        }
    } else if (o.template_dir.has_value() && !o.template_dir->empty()) {
        r.warnings.push_back("ignoring template_dir: this build has no template override (QVAC_COREML_DEV)");
    }

    if (o.compute_units.has_value() && !o.compute_units->empty()) {
        auto u = parse_compute_units(*o.compute_units);
        if (!u) {
            throw std::invalid_argument("coreml: invalid compute_units='" + *o.compute_units +
                                        "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
        }
        r.compute_units = *u;
        r.units_source = Source::explicit_field;
    } else if (const char * e = env("_COREML_COMPUTE_UNITS"); e != nullptr && *e != '\0') {
        if (auto u = parse_compute_units(e)) {
            r.compute_units = *u;
            r.units_source = Source::engine_env;
        } else {
            r.warnings.push_back("ignoring " + p + "_COREML_COMPUTE_UNITS=" + e +
                                 " (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
        }
    }

    if (o.strict_wait_ms.has_value()) {
        if (*o.strict_wait_ms < 0) throw std::invalid_argument("coreml: strict_wait_ms must be >= 0");
        r.strict_wait_ms = o.strict_wait_ms;
    }
    if (o.memory_budget_bytes.has_value()) {
        if (*o.memory_budget_bytes < 0) throw std::invalid_argument("coreml: memory_budget_bytes must be >= 0");
        r.memory_budget_bytes = o.memory_budget_bytes;
    }
    return r;
}

bool apply_memory_degrade(Resolved & r, const FitVerdict & v) {
    if (!r.memory_budget_bytes.has_value()) return false;  // no budget, no memory-degrade
    if (r.use == Use::off) return false;
    if (!v.coreml_fits && v.ggml_only_fits) {
        r.use = Use::off;
        r.disabled_reason = "memory";
        return true;
    }
    return false;
}

FitReport make_fit_report(const Resolved & r, const FitVerdict & v) {
    FitReport rep;
    rep.verdict = v;
    rep.disabled_reason = r.enabled() ? std::string() : r.disabled_reason;
    rep.fits = r.enabled() ? v.coreml_fits : v.ggml_only_fits;
    return rep;
}

std::string to_json(const FitReport & rep) {
    nlohmann::json j;
    j["fits"] = rep.fits;
    j["coreml"] = {
        {"fits", rep.verdict.coreml_fits},
        {"bytes", rep.verdict.coreml_bytes},
        {"validated", rep.verdict.validated},
        {"disabled_reason", rep.disabled_reason.empty() ? nlohmann::json(nullptr) : nlohmann::json(rep.disabled_reason)},
    };
    j["ggml_only"] = {{"fits", rep.verdict.ggml_only_fits}};
    return j.dump();
}

}  // namespace QVAC_COREML_NS
