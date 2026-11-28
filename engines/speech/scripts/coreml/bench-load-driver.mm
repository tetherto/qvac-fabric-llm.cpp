// bench-load-driver.mm: native driver for scripts/coreml/bench-load-time.py.
// macOS 15+ only; built by `bench-load-time.py build-driver` together with the
// shared runtime sources (coreml/src) in the namespace
// qvac_coreml_bench, so `ttr --mode manager` runs the real runtime scheduler:
// admission gate, memory admission, hints.
//
// Every subcommand prints one JSON object on stdout; diagnostics go to stderr.
//
//   identity
//       host_id / os_build exactly as the runtime computes them (fresh-host.sh
//       smoke: a re-signed copy must report its new identifier).
//   memory [--samples N] [--interval-ms M]
//       the runtime's available-memory reading, the input to memory admission.
//   functions --model X.mlmodelc
//       the function names of a compiled model.
//   load --model X.mlmodelc [--units U] [--function F]... [--predict N] [--keep]
//       loads each function in order in this process: load time, N zero-input
//       predictions, then phys_footprint, resident size, host wired memory and
//       the host's available memory after each step. --keep holds every model (weight-sharing measurement); otherwise
//       each is released first.
//   ttr --mode manager|manual-gated|ungated --route prebuilt|generate
//       --artifact STAGE=PATH:FN[,FN...] ... --work-dir DIR [--cache-dir DIR]
//       [--units U] [--memory-admission] [--timeout-s S]
//       time until every function is loaded and warmed (one zero prediction).
//       route generate takes .mlpackage paths: a stand-in prepare job copies the
//       package (the weight write) and compiles it with compileModelAtURL, as
//       on-device generation will. manager = the runtime's Manager (gated);
//       manual-gated = one thread per queue sharing one mutex (the gate by
//       hand); ungated = the same two threads without the mutex.
//       available_bytes_at_start and footprint_base are read before anything
//       loads. Every model that loads stays resident until the final readings
//       (footprint_end, footprint_max), and models_live counts the models
//       still alive when they are taken.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <libproc.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "bench-ttr.h"
#include "device.h"
#include "nlohmann/json.hpp"
#include "qvac_coreml/manager.h"

namespace qc = QVAC_COREML_NS;
using json = nlohmann::json;

namespace {

// ---- clock, memory, timeline ------------------------------------------------

const auto g_t0 = std::chrono::steady_clock::now();

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_t0).count();
}

uint64_t phys_footprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &n) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}

uint64_t resident_size() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &n) != KERN_SUCCESS) return 0;
    return info.resident_size;
}

// Host-wide wired memory: where the ANE keeps a loaded program's weights,
// outside this process's footprint.
uint64_t host_wired() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &n) != KERN_SUCCESS)
        return 0;
    return (uint64_t) vm.wire_count * (uint64_t) vm_kernel_page_size;
}

uint64_t lifetime_max_footprint() {
    rusage_info_v4 ri{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t *>(&ri)) != 0) return 0;
    return ri.ri_lifetime_max_phys_footprint;
}

struct Span {
    std::string kind;   // prepare | load | warm
    std::string label;  // stage/function
    std::string thread;
    double start_ms;
    double end_ms;
    bool ok;
};

std::mutex g_tl_mu;
std::vector<Span> g_timeline;

void record(const std::string & kind, const std::string & label, double t0, double t1, bool ok) {
    char name[64] = {0};
    pthread_getname_np(pthread_self(), name, sizeof name);
    std::lock_guard<std::mutex> lk(g_tl_mu);
    g_timeline.push_back({kind, label, name, t0, t1, ok});
}

json timeline_json() {
    std::lock_guard<std::mutex> lk(g_tl_mu);
    json a = json::array();
    for (const Span & s : g_timeline) {
        a.push_back({{"kind", s.kind}, {"label", s.label}, {"thread", s.thread},
                     {"start_ms", s.start_ms}, {"end_ms", s.end_ms}, {"ok", s.ok}});
    }
    return a;
}

[[noreturn]] void die(const std::string & msg) {
    fprintf(stderr, "bench-load-driver: %s\n", msg.c_str());
    exit(2);
}

// ---- Core ML helpers ----------------------------------------------------------

MLComputeUnits ml_units(qc::ComputeUnits u) {
    switch (u) {
        case qc::ComputeUnits::cpu_and_ane: return MLComputeUnitsCPUAndNeuralEngine;
        case qc::ComputeUnits::cpu_and_gpu: return MLComputeUnitsCPUAndGPU;
        case qc::ComputeUnits::cpu_only: return MLComputeUnitsCPUOnly;
        case qc::ComputeUnits::all: break;
    }
    return MLComputeUnitsAll;
}

qc::ComputeUnits parse_units(const std::string & s) {
    auto u = qc::parse_compute_units(s);
    if (!u) die("unknown compute units '" + s + "'");
    return *u;
}

std::string ns_error(NSError * e) {
    return e ? std::string(e.localizedDescription.UTF8String ?: "error") : std::string("unknown error");
}

// Loaded runtime models alive in this process. `ttr` reports it beside
// footprint_end, so a model released before the final readings is not counted.
std::atomic<long> g_models_live{0};

struct BenchModel {
    explicit BenchModel(MLModel * m) : model(m) { ++g_models_live; }
    ~BenchModel() { --g_models_live; }
    BenchModel(const BenchModel &) = delete;
    BenchModel & operator=(const BenchModel &) = delete;

    MLModel * model = nil;
};

MLModel * load_model(const std::string & path, const std::string & function, qc::ComputeUnits units,
                     std::string & error) {
    @autoreleasepool {
        MLModelConfiguration * cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = ml_units(units);
        if (!function.empty() && function != "main") cfg.functionName = @(function.c_str());
        NSError * err = nil;
        MLModel * m = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())]
                                        configuration:cfg
                                                error:&err];
        if (!m) error = ns_error(err);
        return m;
    }
}

bool is_mask_name(NSString * name) {
    NSString * l = name.lowercaseString;
    return [l containsString:@"valid"] || [l containsString:@"mask"];
}

// Zero inputs of the declared shapes (masks set to one), so a prediction runs
// the whole graph without real data.
MLDictionaryFeatureProvider * zero_inputs(MLModel * m, std::string & error) {
    NSMutableDictionary<NSString *, MLFeatureValue *> * d = [NSMutableDictionary dictionary];
    for (NSString * name in m.modelDescription.inputDescriptionsByName) {
        MLFeatureDescription * fd = m.modelDescription.inputDescriptionsByName[name];
        MLMultiArrayConstraint * c = fd.multiArrayConstraint;
        if (c == nil) {
            error = std::string("input ") + name.UTF8String + " is not a multi-array";
            return nil;
        }
        NSError * err = nil;
        MLMultiArray * a = [[MLMultiArray alloc] initWithShape:c.shape dataType:c.dataType error:&err];
        if (a == nil) {
            error = ns_error(err);
            return nil;
        }
        const bool one = is_mask_name(name);
        const MLMultiArrayDataType dt = c.dataType;
        [a getMutableBytesWithHandler:^(void * bytes, NSInteger size, NSArray<NSNumber *> * strides) {
          (void) strides;
          memset(bytes, 0, (size_t) size);
          if (!one) return;
          if (dt == MLMultiArrayDataTypeFloat32) {
              float * f = static_cast<float *>(bytes);
              for (NSInteger i = 0; i < size / 4; ++i) f[i] = 1.0f;
          } else if (dt == MLMultiArrayDataTypeFloat16) {
              uint16_t * h = static_cast<uint16_t *>(bytes);
              for (NSInteger i = 0; i < size / 2; ++i) h[i] = 0x3C00;
          } else if (dt == MLMultiArrayDataTypeInt32) {
              int32_t * v = static_cast<int32_t *>(bytes);
              for (NSInteger i = 0; i < size / 4; ++i) v[i] = 1;
          }
        }];
        d[name] = [MLFeatureValue featureValueWithMultiArray:a];
    }
    NSError * err = nil;
    MLDictionaryFeatureProvider * p = [[MLDictionaryFeatureProvider alloc] initWithDictionary:d error:&err];
    if (p == nil) error = ns_error(err);
    return p;
}

bool predict_once(MLModel * m, std::string & error) {
    @autoreleasepool {
        MLDictionaryFeatureProvider * in = zero_inputs(m, error);
        if (in == nil) return false;
        NSError * err = nil;
        id<MLFeatureProvider> out = [m predictionFromFeatures:in error:&err];
        if (out == nil) {
            error = ns_error(err);
            return false;
        }
        return true;
    }
}

std::vector<std::string> function_names(const std::string & path, std::string & error) {
    std::vector<std::string> names;
    @autoreleasepool {
        NSError * err = nil;
        MLModelAsset * asset = [MLModelAsset modelAssetWithURL:[NSURL fileURLWithPath:@(path.c_str())] error:&err];
        if (asset == nil) {
            error = ns_error(err);
            return names;
        }
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block NSArray<NSString *> * listed = nil;
        __block NSError * lerr = nil;
        [asset functionNamesWithCompletionHandler:^(NSArray<NSString *> * fns, NSError * e) {
          listed = fns;
          lerr = e;
          dispatch_semaphore_signal(done);
        }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (listed == nil) {
            error = ns_error(lerr);
            return names;
        }
        for (NSString * n in listed) names.emplace_back(n.UTF8String);
    }
    if (names.empty()) names.emplace_back("main");  // single-function model
    return names;
}

// ---- runtime callbacks (plain function pointers) ------------------------------

qc::LoadResult bench_load_fn(const qc::LoadRequest & req) {
    qc::LoadResult r;
    std::string error;
    const double t0 = now_ms();
    MLModel * m = load_model(req.artifact_path, req.function, req.units, error);
    record("load", req.stage + "/" + req.function, t0, now_ms(), m != nil);
    if (m == nil) {
        r.error = qc::Error::deterministic(error, "MLModel");
        return r;
    }
    auto * bm = new BenchModel(m);
    r.model = std::shared_ptr<void>(bm, [](void * p) { delete static_cast<BenchModel *>(p); });
    return r;
}

qc::Error bench_warm_fn(void * model, const qc::LoadRequest & req) {
    std::string error;
    const double t0 = now_ms();
    const bool ok = predict_once(static_cast<BenchModel *>(model)->model, error);
    record("warm", req.stage + "/" + req.function, t0, now_ms(), ok);
    return ok ? qc::Error{} : qc::Error::deterministic(error, "MLModel");
}

// The stand-in for on-device generation: copy the package (the weight write) and
// compile it on device into a fresh, write-once directory.
bool stand_in_prepare(const std::string & package, const std::string & dir, const std::string & stage,
                      std::string & out_path, std::string & error) {
    @autoreleasepool {
        NSFileManager * fm = NSFileManager.defaultManager;
        NSString * d = @(dir.c_str());
        NSError * err = nil;
        if (![fm createDirectoryAtPath:d withIntermediateDirectories:YES attributes:nil error:&err]) {
            error = ns_error(err);
            return false;
        }
        NSString * pkg = [d stringByAppendingPathComponent:[@(stage.c_str()) stringByAppendingString:@".mlpackage"]];
        if (![fm copyItemAtPath:@(package.c_str()) toPath:pkg error:&err]) {
            error = ns_error(err);
            return false;
        }
        NSURL * compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pkg] error:&err];
        if (compiled == nil) {
            error = ns_error(err);
            return false;
        }
        NSString * dst = [d stringByAppendingPathComponent:[@(stage.c_str()) stringByAppendingString:@".mlmodelc"]];
        if (![fm moveItemAtURL:compiled toURL:[NSURL fileURLWithPath:dst] error:&err]) {
            error = ns_error(err);
            return false;
        }
        [fm removeItemAtPath:pkg error:nil];
        out_path = dst.UTF8String;
        return true;
    }
}

qc::PrepareResult bench_prepare_fn(const qc::PrepareRequest & req, qc::PrepareContext & ctx) {
    qc::PrepareResult r;
    const std::string stage = req.template_id;
    const std::string dir = req.cache_dir + "/" + stage + "-" + std::to_string(getpid());
    const double t0 = now_ms();
    ctx.phase(qc::PreparePhase::generating);
    std::string out, error;
    const bool ok = stand_in_prepare(req.source.path, dir, stage, out, error);
    record("prepare", stage, t0, now_ms(), ok);
    if (!ok) {
        r.error = qc::Error::deterministic(error, "prepare");
        return r;
    }
    r.artifact_path = out;
    r.artifact_key = stage;
    return r;
}

// ---- argument parsing ----------------------------------------------------------

struct Args {
    std::map<std::string, std::vector<std::string>> kv;
    std::vector<std::string> flags;

    std::optional<std::string> get(const std::string & k) const {
        auto it = kv.find(k);
        if (it == kv.end() || it->second.empty()) return std::nullopt;
        return it->second.back();
    }
    std::vector<std::string> all(const std::string & k) const {
        auto it = kv.find(k);
        return it == kv.end() ? std::vector<std::string>{} : it->second;
    }
    bool flag(const std::string & k) const { return std::find(flags.begin(), flags.end(), k) != flags.end(); }
    std::string req(const std::string & k) const {
        auto v = get(k);
        if (!v) die("missing " + k);
        return *v;
    }
};

Args parse(int argc, char ** argv, int from, const std::vector<std::string> & flag_names) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) die("unexpected argument " + k);
        if (std::find(flag_names.begin(), flag_names.end(), k) != flag_names.end()) {
            a.flags.push_back(k);
            continue;
        }
        if (i + 1 >= argc) die(k + " needs a value");
        a.kv[k].push_back(argv[++i]);
    }
    return a;
}

void emit(const json & j) {
    printf("%s\n", j.dump().c_str());
    fflush(stdout);
}

// ---- subcommands ---------------------------------------------------------------

int cmd_identity(const Args &) {
    const qc::OsVersion v = qc::detail::device_os_version();
    emit({{"host_id", qc::detail::device_host_id()},
          {"os_build", qc::detail::device_os_build()},
          {"os_version", std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.patch)},
          {"pid", getpid()}});
    return 0;
}

int cmd_memory(const Args & a) {
    const int samples = std::stoi(a.get("--samples").value_or("1"));
    const int interval = std::stoi(a.get("--interval-ms").value_or("1000"));
    json r = json::array();
    for (int i = 0; i < samples; ++i) {
        if (i) std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        r.push_back({{"t_ms", now_ms()}, {"available_bytes", qc::detail::device_available_memory()}});
    }
    uint64_t memsize = 0;
    size_t len = sizeof memsize;
    sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
    emit({{"memsize_bytes", memsize}, {"low_power", qc::detail::device_low_power()}, {"samples", r}});
    return 0;
}

int cmd_functions(const Args & a) {
    std::string error;
    auto names = function_names(a.req("--model"), error);
    if (!error.empty()) die(error);
    emit({{"model", a.req("--model")}, {"functions", names}});
    return 0;
}

int cmd_load(const Args & a) {
    const std::string model = a.req("--model");
    const qc::ComputeUnits units = parse_units(a.get("--units").value_or("cpu_and_ane"));
    const int predicts = std::stoi(a.get("--predict").value_or("1"));
    const bool keep = a.flag("--keep");
    std::vector<std::string> fns = a.all("--function");
    std::string error;
    if (fns.empty()) fns = function_names(model, error);
    if (!error.empty()) die(error);

    const uint64_t base = phys_footprint();
    const uint64_t available_base = qc::detail::device_available_memory();
    const uint64_t resident_base = resident_size();
    const uint64_t wired_base = host_wired();
    std::vector<MLModel *> held;
    json out = json::array();
    for (const std::string & fn : fns) {
        json rec = {{"function", fn}};
        const double t0 = now_ms();
        std::string err;
        MLModel * m = load_model(model, fn, units, err);
        const double t1 = now_ms();
        rec["load_ms"] = t1 - t0;
        rec["footprint_after_load"] = phys_footprint();
        if (m == nil) {
            rec["error"] = err;
            out.push_back(rec);
            continue;
        }
        json pms = json::array();
        for (int i = 0; i < predicts; ++i) {
            const double p0 = now_ms();
            if (!predict_once(m, err)) {
                rec["error"] = err;
                break;
            }
            pms.push_back(now_ms() - p0);
        }
        rec["predict_ms"] = pms;
        rec["footprint_after_predict"] = phys_footprint();
        rec["available_after_predict"] = qc::detail::device_available_memory();
        rec["resident_after_predict"] = resident_size();
        rec["wired_after_predict"] = host_wired();
        out.push_back(rec);
        if (keep) held.push_back(m);
        m = nil;
    }
    emit({{"model", model},
          {"units", qc::to_string(units)},
          {"keep", keep},
          {"host_id", qc::detail::device_host_id()},
          {"footprint_base", base},
          {"footprint_end", phys_footprint()},
          {"footprint_max", lifetime_max_footprint()},
          {"available_base", available_base},
          {"resident_base", resident_base},
          {"wired_base", wired_base},
          {"functions", out}});
    held.clear();
    return 0;
}

struct ArtifactArg {
    std::string stage;
    std::string path;
    std::vector<std::string> functions;
};

std::vector<ArtifactArg> parse_artifacts(const std::vector<std::string> & specs) {
    std::vector<ArtifactArg> out;
    for (const std::string & s : specs) {
        const size_t eq = s.find('=');
        const size_t colon = s.rfind(':');
        if (eq == std::string::npos || colon == std::string::npos || colon < eq) die("bad --artifact " + s);
        ArtifactArg a{s.substr(0, eq), s.substr(eq + 1, colon - eq - 1), {}};
        std::string fns = s.substr(colon + 1);
        size_t p = 0;
        while (p <= fns.size()) {
            const size_t c = fns.find(',', p);
            std::string f = fns.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!f.empty()) a.functions.push_back(f);
            if (c == std::string::npos) break;
            p = c + 1;
        }
        if (a.functions.empty()) die("no functions in --artifact " + s);
        out.push_back(a);
    }
    return out;
}

uint64_t weight_bytes(const std::string & artifact) {
    // .mlmodelc/weights/weight.bin, or .mlpackage/Data/com.apple.CoreML/weights/weight.bin
    for (const char * rel : {"/weights/weight.bin", "/Data/com.apple.CoreML/weights/weight.bin"}) {
        NSDictionary * attrs = [NSFileManager.defaultManager attributesOfItemAtPath:@((artifact + rel).c_str())
                                                                               error:nil];
        if (attrs) return attrs.fileSize;
    }
    return 0;
}

json ttr_manager(const std::vector<ArtifactArg> & arts, bool generate, qc::ComputeUnits units,
                 const std::string & work, const std::string & cache, bool memory_admission, double timeout_s,
                 double & ready_ms, std::shared_ptr<void> & keep) {
    // Members destroy in reverse order: slots, then the group, then the Manager.
    struct Held {
        std::shared_ptr<qc::Manager> mgr;
        std::shared_ptr<qc::Group> group;
        std::vector<std::shared_ptr<qc::Slot>> slots;
    };
    auto held = std::make_shared<Held>();
    held->mgr = qc::Manager::create();
    auto & mgr = held->mgr;
    auto & slots = held->slots;
    std::vector<qc::GroupRequest> reqs;
    for (const ArtifactArg & a : arts) {
        for (const std::string & fn : a.functions) {
            qc::SlotSpec spec;
            spec.engine = "chatterbox-bench";
            spec.stage = a.stage;
            spec.function = fn;
            spec.units = units;
            spec.load_fn = bench_load_fn;
            spec.warm_fn = bench_warm_fn;
            if (memory_admission) spec.ane_compile_reserve_bytes = (uint64_t) (1.6 * (double) weight_bytes(a.path));
            if (generate) {
                spec.route = qc::Route::generate;
                spec.prepare.key = a.path;
                spec.prepare.template_id = a.stage;
                spec.prepare.source.path = a.path;
                spec.prepare.cache_dir = work;
                spec.prepare_fn = bench_prepare_fn;
            } else {
                spec.route = qc::Route::prebuilt;
                spec.artifact_path = a.path;
            }
            qc::SlotPolicy pol;
            pol.wait = qc::Wait::construct;
            pol.cache_dir = cache;
            auto slot = mgr->make_slot(spec, pol);
            slots.push_back(slot);
            reqs.push_back({slot, qc::Priority::minimal});
        }
    }
    const double t_issue = now_ms();
    held->group = mgr->issue_group(qc::GroupKind::construction, reqs, /*waited=*/true);
    auto & group = held->group;
    // Poll for state transitions (deferred(memory) time, per-slot ready time).
    auto sample = [&](size_t i) {
        const qc::SlotState s = slots[i]->state();
        bench_ttr::Observation o{qc::to_string(s), bench_ttr::Kind::other};
        if (s == qc::SlotState::ready) o.kind = bench_ttr::Kind::ready;
        if (s == qc::SlotState::deferred_memory) o.kind = bench_ttr::Kind::deferred_memory;
        return o;
    };
    const bench_ttr::PollOutcome polled = bench_ttr::poll_slots(
        slots.size(), sample, [&] { return group->finished(); }, now_ms,
        [] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }, t_issue, t_issue + timeout_s * 1000.0);
    const std::vector<double> & ready_at = polled.ready_at;
    const std::vector<double> & deferred_ms = polled.deferred_ms;
    json transitions = json::array();
    for (const bench_ttr::Transition & tr : polled.transitions) {
        transitions.push_back({{"t_ms", tr.t_ms},
                               {"slot", slots[tr.slot]->spec().stage + "/" + slots[tr.slot]->spec().function},
                               {"state", tr.state}});
    }
    qc::PrewarmResult res = group->result();
    ready_ms = bench_ttr::all_ready_ms(ready_at);
    json per_slot = json::array();
    for (size_t i = 0; i < slots.size(); ++i) {
        per_slot.push_back({{"slot", slots[i]->spec().stage + "/" + slots[i]->spec().function},
                            {"ready_ms", ready_at[i]},
                            {"deferred_memory_ms", deferred_ms[i]},
                            {"status", json::parse(qc::to_json(slots[i]->status()))}});
    }
    const qc::Stats st = mgr->stats();
    keep = held;
    return {{"t_issue_ms", t_issue},
            {"group", json::parse(qc::to_json(res))},
            {"slots", per_slot},
            {"transitions", transitions},
            {"stats", {{"loads_started", st.loads_started}, {"loads_completed", st.loads_completed},
                       {"load_failures", st.load_failures}, {"prepares_started", st.prepares_started},
                       {"prepares_completed", st.prepares_completed}, {"requeues", st.requeues},
                       {"hint_writes", st.hint_writes}}}};
}

// Two worker threads (prepare, load) with or without one shared mutex. The
// loaded models go to `keep` so they stay resident until the caller has taken
// its final readings, as the Manager's slots keep theirs.
json ttr_manual(const std::vector<ArtifactArg> & arts, bool generate, qc::ComputeUnits units, const std::string & work,
                bool gated, double & ready_ms, std::shared_ptr<void> & keep) {
    std::vector<bench_ttr::ManualStage> stages;
    for (const ArtifactArg & a : arts) stages.push_back({a.stage, a.path, a.functions});
    const double t_issue = now_ms();
    auto prepare = [&](const std::string & stage, const std::string & path, std::string & out) {
        pthread_setname_np("bench-prepare");
        const double t0 = now_ms();
        std::string error;
        const bool ok = stand_in_prepare(path, work + "/" + stage + "-manual", stage, out, error);
        record("prepare", stage, t0, now_ms(), ok);
        if (!ok) die("prepare " + stage + ": " + error);
        return ok;
    };
    auto load = [&](const std::string & stage, const std::string & path, const std::string & fn) {
        pthread_setname_np("bench-load");
        qc::LoadRequest req;
        req.engine = "chatterbox-bench";
        req.stage = stage;
        req.artifact_path = path;
        req.function = fn;
        req.units = units;
        qc::LoadResult r = bench_load_fn(req);
        if (!r.error.ok() || !bench_warm_fn(r.model.get(), req).ok()) return std::shared_ptr<void>();
        return r.model;
    };
    auto out = std::make_shared<bench_ttr::ManualOutcome>(
        bench_ttr::run_manual(stages, generate, gated, prepare, load, now_ms, t_issue));
    ready_ms = bench_ttr::manual_ready_ms(out->results);
    json slots = json::array();
    for (const bench_ttr::ManualResult & r : out->results)
        slots.push_back({{"slot", r.slot}, {"ready_ms", r.ready_ms}, {"ok", r.ok}});
    keep = out;
    return {{"t_issue_ms", t_issue}, {"slots", slots}};
}

int cmd_ttr(const Args & a) {
    const std::string mode = a.req("--mode");
    const std::string route = a.get("--route").value_or("prebuilt");
    if (route != "prebuilt" && route != "generate") die("--route must be prebuilt or generate");
    const bool generate = route == "generate";
    const qc::ComputeUnits units = parse_units(a.get("--units").value_or("cpu_and_ane"));
    const std::string work = a.req("--work-dir");
    const std::string cache = a.get("--cache-dir").value_or("");
    const double timeout_s = std::stod(a.get("--timeout-s").value_or("1800"));
    auto arts = parse_artifacts(a.all("--artifact"));
    if (arts.empty()) die("no --artifact");
    [NSFileManager.defaultManager createDirectoryAtPath:@(work.c_str()) withIntermediateDirectories:YES
                                             attributes:nil error:nil];

    if (mode != "manager" && mode != "manual-gated" && mode != "ungated")
        die("--mode must be manager, manual-gated or ungated");

    double ready_ms = -1;
    const auto r = bench_ttr::with_readings(
        [] {
            return json{{"footprint_base", phys_footprint()},
                        {"available_bytes_at_start", qc::detail::device_available_memory()}};
        },
        [&](std::shared_ptr<void> & keep) {
            if (mode == "manager")
                return ttr_manager(arts, generate, units, work, cache, a.flag("--memory-admission"), timeout_s,
                                   ready_ms, keep);
            return ttr_manual(arts, generate, units, work, mode == "manual-gated", ready_ms, keep);
        },
        [] {
            return json{{"footprint_end", phys_footprint()},
                        {"footprint_max", lifetime_max_footprint()},
                        {"models_live", g_models_live.load()}};
        });
    json out = {{"mode", mode},
                {"route", route},
                {"units", qc::to_string(units)},
                {"host_id", qc::detail::device_host_id()},
                {"memory_admission", a.flag("--memory-admission")},
                {"time_to_ready_ms", ready_ms},
                {"timeline", timeline_json()},
                {"detail", r.run}};
    for (const json & readings : {r.start, r.end})
        for (const auto & kv : readings.items()) out[kv.key()] = kv.value();
    emit(out);
    return ready_ms < 0 ? 3 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) die("usage: identity | memory | functions | load | ttr (see the file header)");
    pthread_setname_np("bench-main");
    const std::string cmd = argv[1];
    if (cmd == "identity") return cmd_identity(parse(argc, argv, 2, {}));
    if (cmd == "memory") return cmd_memory(parse(argc, argv, 2, {}));
    if (cmd == "functions") return cmd_functions(parse(argc, argv, 2, {}));
    if (cmd == "load") return cmd_load(parse(argc, argv, 2, {"--keep"}));
    if (cmd == "ttr") return cmd_ttr(parse(argc, argv, 2, {"--memory-admission"}));
    die("unknown command " + cmd);
}
