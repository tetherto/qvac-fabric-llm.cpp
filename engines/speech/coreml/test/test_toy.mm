// test-coreml-toy: the generation route end to end on the device, with the
// test-only toy templates (coreml/test/toy-templates) embedded in a
// test library through qvac_coreml_add_to(... TEMPLATES ...), as an engine
// embeds its own:
//
//   embedded table -> twin GGUF -> prepare() (generate, on-device compile,
//   publish, adopt) -> MLModel load -> predict, compared with the same math on
//   the GGUF's weights; artifact.json; user metadata of an on-device-compiled
//   bundle; the hot path on a second prepare(); repair of an unusable artifact
//   (quarantine, regenerate, recompile); two prepares of the same weights into
//   two cache roots compiling at the same time.
//
// Toys: stateless (2 linear layers), multi (two functions sharing weights),
// state (an MLState accumulator). Needs macOS 15 (MLState, functionName, the
// iOS18 opset) and fails, never skips, below it. Prints compile_peak_bytes
// per toy (measurement).
//
//   test-coreml-toy --tool <qvac-coreml-tool> --toys <toy-templates dir> [--only <case>]

#include "cache.h"
#include "compile.h"
#include "device.h"
#include "gguf_source.h"
#include "sha256.h"
#include "templates.h"

#include "generator_test_util.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>

using namespace gtest;

namespace {

std::string g_tool;
std::string g_toys;

constexpr int D_IN = 8, D_HID = 32, D_OUT = 48;
const char * const TOYS[] = {"toy-multi", "toy-state", "toy-stateless"};

std::string sh_quote(const std::string & s) { return "'" + s + "'"; }

bool run_tool(const std::string & args) {
    const std::string cmd = sh_quote(g_tool) + " " + args + " > /dev/null";
    const int st = std::system(cmd.c_str());
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

std::string twin(const TempDir & td, const std::string & id, int seed) {
    const std::string out = td / (id + "-" + std::to_string(seed) + ".gguf");
    const bool ok = run_tool("twin --manifest " + sh_quote(g_toys + "/" + id + "/reference-gguf.json") + " --seed " +
                             std::to_string(seed) + " --types f32 -o " + sh_quote(out));
    CHECK_MSG(ok, "qvac-coreml-tool twin " + id);
    return out;
}

// The f32 tensors of a GGUF as the generator reads them (ggml to_float).
std::vector<float> tensor(const std::string & gguf, const std::string & name) {
    Error err;
    auto src = GgufSource::open(gguf, err);
    std::vector<float> v;
    if (!src) return v;
    Error e = src->read_f32(name, v, nullptr);
    CHECK_MSG(e.ok(), name + ": " + e.message);
    return v;
}

// Round to fp16 and back: the weights the model holds.
std::vector<double> f16w(const std::vector<float> & v) {
    std::vector<double> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = recipe::f16_to_f32(recipe::f32_to_f16(v[i]));
    return r;
}

std::vector<double> input_x() {
    std::vector<double> x(D_IN);
    for (int i = 0; i < D_IN; ++i) x[i] = 0.25 * (i - 3.5);
    return x;
}

// out[h] = sum_i w[h*D_IN+i] x[i] + b[h]   (linear with weight [D_HID, D_IN])
std::vector<double> linear1(const std::vector<double> & w, const std::vector<double> & b, const std::vector<double> & x) {
    std::vector<double> h(D_HID);
    for (int o = 0; o < D_HID; ++o) {
        double s = b[o];
        for (int i = 0; i < D_IN; ++i) s += w[o * D_IN + i] * x[i];
        h[o] = s;
    }
    return h;
}

bool close_to(const std::vector<double> & got, const std::vector<double> & want, const std::string & what) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "      %s: %zu values, expected %zu\n", what.c_str(), got.size(), want.size());
        return false;
    }
    double scale = 1e-3;
    for (double w : want) scale = std::max(scale, std::fabs(w));
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]));
    const bool ok = worst <= 2e-2 * scale;
    if (!ok) std::fprintf(stderr, "      %s: max |delta| %g, scale %g\n", what.c_str(), worst, scale);
    return ok;
}

struct Prepared {
    cache::PrepareOutcome out;
    std::shared_ptr<const recipe::Template> tmpl;
};

cache::PrepareInputs inputs_for(const std::string & cache_dir, std::shared_ptr<const recipe::Template> t,
                                const std::string & gguf) {
    cache::PrepareInputs in;
    in.cache_dir = cache_dir;
    in.tmpl = std::move(t);
    stat_path(gguf, in.expected);
    in.expected.path = gguf;
    in.os_build = detail::device_os_build();
    in.compile = apple::compile_package;
    in.open = [](const std::string & p, Error & e) -> std::unique_ptr<OpenSource> { return GgufSource::open(p, e); };
    return in;
}

MLModel * load(const std::string & mlmodelc, const char * function, MLComputeUnits units, std::string * err_out) {
    @autoreleasepool {
        MLModelConfiguration * cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = units;
        if (function) {
            if (@available(macOS 15.0, *)) cfg.functionName = @(function);
        }
        NSError * err = nil;
        MLModel * m = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(mlmodelc.c_str())]
                                        configuration:cfg
                                                error:&err];
        if (!m && err_out) *err_out = err ? std::string(err.localizedDescription.UTF8String) : "no model";
        return m;
    }
}

MLMultiArray * array(const std::vector<double> & v, MLMultiArrayDataType dt) {
    NSError * err = nil;
    MLMultiArray * a = [[MLMultiArray alloc] initWithShape:@[ @1, @(v.size()) ] dataType:dt error:&err];
    for (size_t i = 0; i < v.size(); ++i) a[i] = @(v[i]);
    return a;
}

std::vector<double> values(MLMultiArray * a) {
    std::vector<double> v;
    if (!a) return v;
    for (NSInteger i = 0; i < a.count; ++i) v.push_back(a[i].doubleValue);
    return v;
}

API_AVAILABLE(macos(15.0))
std::vector<double> predict(MLModel * m, MLMultiArray * x, MLState * state = nil) {
    NSError * err = nil;
    MLDictionaryFeatureProvider * in =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x" : [MLFeatureValue featureValueWithMultiArray:x]}
                                                          error:&err];
    id<MLFeatureProvider> out = state ? [m predictionFromFeatures:in usingState:state error:&err]
                                      : [m predictionFromFeatures:in error:&err];
    CHECK_MSG(out != nil, err ? std::string(err.localizedDescription.UTF8String) : "prediction failed");
    return out ? values([out featureValueForName:@"y"].multiArrayValue) : std::vector<double>{};
}

std::string user_meta(MLModel * m, NSString * key) {
    NSDictionary * md = m.modelDescription.metadata[MLModelCreatorDefinedKey];
    NSString * v = md[key];
    return v ? std::string(v.UTF8String) : std::string();
}

// The prediction checks of one toy against its GGUF's weights.
API_AVAILABLE(macos(15.0))
void check_predictions(const std::string & id, const std::string & mlmodelc, const std::string & gguf) {
    const auto w1 = f16w(tensor(gguf, "toy.w1"));
    const auto b1 = f16w(tensor(gguf, "toy.b1"));
    const auto x = input_x();
    std::string err;
    if (id == "toy-stateless") {
        const auto w2 = f16w(tensor(gguf, "toy.w2"));  // numpy [D_HID, D_OUT]; the exporter uses w2.T
        const auto b2 = f16w(tensor(gguf, "toy.b2"));
        auto h = linear1(w1, b1, x);
        for (double & v : h) v = std::max(0.0, v);
        std::vector<double> y(D_OUT);
        for (int o = 0; o < D_OUT; ++o) {
            double s = recipe::f16_to_f32(recipe::f32_to_f16(0.5f * static_cast<float>(b2[o])));
            for (int k = 0; k < D_HID; ++k) s += w2[k * D_OUT + o] * h[k];
            y[o] = s;
        }
        MLModel * m = load(mlmodelc, nullptr, MLComputeUnitsCPUOnly, &err);
        CHECK_MSG(m != nil, "load cpu_only: " + err);
        if (!m) return;
        CHECK(close_to(predict(m, array(x, MLMultiArrayDataTypeFloat32)), y, id + " cpu_only"));
        CHECK(user_meta(m, @"qvac.toy.kind") == "stateless");
        MLModel * all = load(mlmodelc, nullptr, MLComputeUnitsAll, &err);
        CHECK_MSG(all != nil, "load all: " + err);
        if (all) CHECK(close_to(predict(all, array(x, MLMultiArrayDataTypeFloat32)), y, id + " all"));
    } else if (id == "toy-multi") {
        const auto s = f16w(tensor(gguf, "toy.s"));
        const auto h = linear1(w1, b1, x);
        std::vector<double> alt(D_HID);
        for (int o = 0; o < D_HID; ++o) alt[o] = h[o] * s[o];
        MLModel * m = load(mlmodelc, "main", MLComputeUnitsCPUOnly, &err);
        CHECK_MSG(m != nil, "load main: " + err);
        MLModel * a = load(mlmodelc, "alt", MLComputeUnitsCPUOnly, &err);
        CHECK_MSG(a != nil, "load alt: " + err);
        if (!m || !a) return;
        CHECK(close_to(predict(m, array(x, MLMultiArrayDataTypeFloat32)), h, id + " main"));
        CHECK(close_to(predict(a, array(x, MLMultiArrayDataTypeFloat32)), alt, id + " alt"));
        CHECK(user_meta(a, @"qvac.toy.kind") == "multi");
    } else {
        const auto h = linear1(w1, b1, x);
        std::vector<double> h2(D_HID);
        for (int o = 0; o < D_HID; ++o) h2[o] = 2 * h[o];
        MLModel * m = load(mlmodelc, nullptr, MLComputeUnitsCPUOnly, &err);
        CHECK_MSG(m != nil, "load state: " + err);
        if (!m) return;
        MLState * st = [m newState];
        CHECK(st != nil);
        MLMultiArray * xa = array(x, MLMultiArrayDataTypeFloat16);
        CHECK(close_to(predict(m, xa, st), h, id + " step 1"));
        CHECK(close_to(predict(m, xa, st), h2, id + " step 2 (state carried)"));
        MLState * fresh = [m newState];
        CHECK(close_to(predict(m, xa, fresh), h, id + " fresh state"));
        CHECK(user_meta(m, @"qvac.toy.kind") == "state");
    }
}

// ---- cases -------------------------------------------------------------------------------

void case_os_floor() {
    const OsVersion v = detail::device_os_version();
    CHECK_MSG(os_at_least(v, 15, 0), "test-coreml-toy needs macOS 15 (MLState, functionName, iOS18 opset); this is " +
                                         std::to_string(v.major) + "." + std::to_string(v.minor));
    std::printf("os %d.%d.%d build %s\n", v.major, v.minor, v.patch, detail::device_os_build().c_str());
}

void case_embedded_table() {
    CHECK((templates::embedded_ids() == std::vector<std::string>{"toy-multi", "toy-state", "toy-stateless"}));
    size_t n = 0;
    const templates::EmbeddedFile * f = templates::embedded_files(n);
    CHECK(n == 12);  // 4 files per toy, no literals.bin, no reference-gguf.json
    for (size_t i = 0; i < n; ++i) {
        const std::string name = f[i].name;
        CHECK(name != "reference-gguf.json");
        const std::string want = read_bytes(g_toys + "/" + f[i].id + "/" + name);
        const size_t size = static_cast<size_t>(f[i].end - f[i].begin);
        CHECK_MSG(size == want.size() && std::memcmp(f[i].begin, want.data(), size) == 0,
                  std::string(f[i].id) + "/" + name + ": embedded bytes differ from the source tree");
        CHECK_MSG(detail::to_hex(detail::sha256(f[i].begin, size)) == f[i].sha256_hex,
                  std::string(f[i].id) + "/" + name + ": table sha256 differs from the embedded bytes");
        CHECK(reinterpret_cast<uintptr_t>(f[i].begin) % 16 == 0);
    }
    for (const char * id : TOYS) {
        auto e = templates::embedded(id);
        CHECK_MSG(e != nullptr, id);
        if (!e) continue;
        auto d = recipe::Template::from_dir(g_toys + "/" + id);
        CHECK_MSG(e->sha256_hex() == d->sha256_hex(), std::string(id) + ": template_sha256 embedded vs source");
        CHECK(e == templates::embedded(id));                // parsed once
        CHECK(templates::find(id) == e);                    // no override: the embedded one
        auto o = templates::find(id, g_toys);               // override (compiled into this library)
        CHECK(o != nullptr && o != e && o->sha256_hex() == e->sha256_hex());
    }
    CHECK(templates::embedded("nope") == nullptr);
    CHECK(templates::find("nope", g_toys) == nullptr);
    CHECK(template_dir_override_compiled());
}

void case_compile_naming() {
    // compileModelAtURL leaves the bundle at $TMPDIR/<package stem>.mlmodelc;
    // the pipeline's packages are named qvac-coreml-<key>-<pid>-<random>.mlpackage,
    // so a killed compile leaves a qvac-coreml-*.mlmodelc there, which routine
    // prune sweeps. Then the move (rename, and the copy fallback).
    TempDir td;
    auto t = templates::embedded("toy-stateless");
    const std::string gguf = twin(td, "toy-stateless", 5);
    Error err;
    auto src = GgufSource::open(gguf, err);
    CHECK(src != nullptr);
    if (!t || !src) return;
    const std::string key = "0123456789abcdef0123456789abcdef";
    const std::string pkg = td / ("qvac-coreml-" + key + ".mlpackage");
    Error we = cache::write_package(*t, *src, pkg);
    CHECK_MSG(we.ok(), we.message);
    const std::string out = td / "model.mlmodelc";
    cache::CompileResult r = apple::compile_package(pkg, out);
    CHECK_MSG(r.error.ok(), r.error.message);
    char tmp_real[PATH_MAX], got_real[PATH_MAX];
    std::string tmpdir = NSTemporaryDirectory().UTF8String;
    const std::string want = std::string(::realpath(tmpdir.c_str(), tmp_real) ? tmp_real : tmpdir.c_str()) +
                             "/qvac-coreml-" + key + ".mlmodelc";
    const std::string parent = r.tmp_output.substr(0, r.tmp_output.rfind('/'));
    const std::string got = std::string(::realpath(parent.c_str(), got_real) ? got_real : parent.c_str()) +
                            r.tmp_output.substr(r.tmp_output.rfind('/'));
    CHECK_MSG(got == want, "compiled to " + r.tmp_output + ", expected " + want);
    CHECK(!fs::exists(r.tmp_output));  // moved
    CHECK(fs::exists(out + "/model.mil") && fs::exists(out + "/coremldata.bin") && fs::exists(out + "/weights/weight.bin"));
    CHECK(r.os_build == detail::device_os_build() && !r.os_build.empty());
    CHECK_MSG(!r.coremlc.empty() && std::isdigit(static_cast<unsigned char>(r.coremlc[0])), "coremlc: " + r.coremlc);
    CHECK(apple::coremlc_version("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"1\"}, "
                                 "{\"coremlc-version\", \"3500.32.1\"}})]") == "3500.32.1");
    CHECK(apple::coremlc_version("program(1.3)\n{") == "");

    // Copy fallback: the bundle arrives whole and the source is removed.
    const std::string copied = td / "copied.mlmodelc";
    Error ce = apple::move_bundle(out, copied, true);
    CHECK_MSG(ce.ok(), ce.message);
    CHECK(!fs::exists(out) && fs::exists(copied + "/weights/weight.bin"));
    CHECK(cache::inventory_of(td.path, copied).size() >= 3);
    Error me = apple::move_bundle(td / "absent.mlmodelc", td / "x.mlmodelc", false);
    CHECK(!me.ok() && me.code == ENOENT);

    // A package that does not exist fails transiently with ENOENT, as an I/O
    // error (stage prepare) that the failure records never promote.
    cache::CompileResult missing = apple::compile_package(td / "absent.mlpackage", td / "y.mlmodelc");
    CHECK(!missing.error.ok() && missing.error.cls == ErrorClass::transient && missing.error.code == ENOENT);
    CHECK(missing.stage == failed::Stage::prepare);
}

void case_generate_load_predict() {
    if (@available(macOS 15.0, *)) {
        TempDir td;
        const std::string cache_dir = td / "cache";
        for (const char * id : TOYS) {
            auto t = templates::embedded(id);
            CHECK_MSG(t != nullptr, id);
            if (!t) continue;
            const std::string gguf = twin(td, id, 11);
            cache::PrepareOutcome o = cache::prepare(inputs_for(cache_dir, t, gguf));
            CHECK_MSG(o.kind == cache::PrepareOutcome::Kind::ready, std::string(id) + ": " + o.error.message);
            if (o.kind != cache::PrepareOutcome::Kind::ready) continue;
            CHECK(!o.cache_hit && o.lease.held());
            std::printf("compile_peak_bytes %s %llu (compile %lld ms, write %lld ms, digest %lld ms)\n", id,
                        static_cast<unsigned long long>(o.compile_peak_bytes), static_cast<long long>(o.compile_ms),
                        static_cast<long long>(o.write_ms), static_cast<long long>(o.digest_ms));

            // artifact.json
            auto a = cache::read_artifact_json(o.artifact_dir);
            CHECK(a.has_value());
            if (a) {
                CHECK(a->complete && a->template_id == id && a->artifact_key == o.key32);
                CHECK(a->os_build == detail::device_os_build());
                CHECK(!a->coremlc.empty());
                CHECK(a->export_kind == t->info().export_kind && a->functions == t->info().functions);
                bool mil = false, data = false, weights = false;
                for (const auto & e : a->inventory) {
                    mil |= e.path == "model.mlmodelc/model.mil";
                    data |= e.path == "model.mlmodelc/coremldata.bin";
                    weights |= e.path == "model.mlmodelc/weights/weight.bin";
                }
                CHECK_MSG(mil && data && weights, std::string(id) + ": inventory lacks a compiled file");
            }
            CHECK(cache::validate(o.artifact_dir) == cache::Validity::ok);
            check_predictions(id, o.model_path, gguf);

            // Hot path: the same source hits its memo; nothing is generated.
            o.lease.release();
            cache::PrepareOutcome h = cache::prepare(inputs_for(cache_dir, t, gguf));
            CHECK(h.kind == cache::PrepareOutcome::Kind::ready && h.cache_hit && h.memo_hit);
            CHECK(h.artifact_dir == o.artifact_dir && h.compile_ms == 0 && h.write_ms == 0);
        }
        // A GGUF of another toy does not match: deterministic, nothing generated.
        auto t = templates::embedded("toy-stateless");
        const std::string other = twin(td, "toy-multi", 12);
        cache::PrepareOutcome m = cache::prepare(inputs_for(cache_dir, t, other));
        CHECK(m.kind == cache::PrepareOutcome::Kind::failed && m.error.cls == ErrorClass::deterministic);
    } else {
        CHECK_MSG(false, "macOS 15 required");
    }
}

void case_quarantine_repair() {
    if (@available(macOS 15.0, *)) {
        TempDir td;
        const std::string cache_dir = td / "cache";
        auto t = templates::embedded("toy-stateless");
        if (!t) {
            CHECK(false);
            return;
        }
        const std::string gguf = twin(td, "toy-stateless", 21);
        cache::Layout l(cache_dir);
        cache::PrepareOutcome o = cache::prepare(inputs_for(cache_dir, t, gguf));
        CHECK(o.kind == cache::PrepareOutcome::Kind::ready);
        if (o.kind != cache::PrepareOutcome::Kind::ready) return;
        const std::string dir = o.artifact_dir;
        o.lease.release();

        // 1. The inventory check fails on the hit (a truncated compiled file):
        //    prepare() quarantines and regenerates at the same key.
        {
            const std::string f = dir + "/model.mlmodelc/coremldata.bin";
            ::truncate(f.c_str(), 3);
            CHECK(cache::validate(dir) == cache::Validity::invalid);
            cache::PrepareOutcome r = cache::prepare(inputs_for(cache_dir, t, gguf));
            CHECK_MSG(r.kind == cache::PrepareOutcome::Kind::ready, r.error.message);
            CHECK(!r.cache_hit && r.key32 == o.key32 && r.artifact_dir == dir);
            CHECK(count_files(l.quarantine) >= 1);
            CHECK(cache::validate(dir) == cache::Validity::ok);
            check_predictions("toy-stateless", r.model_path, gguf);
        }

        // 2. The load reports an invalid model although the inventory passes
        //    (model.mil overwritten at the same size): with no lease held the
        //    engine quarantines it; the next prepare() regenerates, and the
        //    fresh model.mil (new mtime) is compiled again.
        {
            const std::string mil = dir + "/model.mlmodelc/model.mil";
            struct stat before {};
            ::stat(mil.c_str(), &before);
            write_bytes(mil, std::string(static_cast<size_t>(before.st_size), 'x'));
            CHECK(cache::validate(dir) == cache::Validity::ok);
            cache::PrepareOutcome hit = cache::prepare(inputs_for(cache_dir, t, gguf));
            CHECK(hit.kind == cache::PrepareOutcome::Kind::ready && hit.cache_hit);
            std::string err;
            MLModel * bad = load(hit.model_path, nullptr, MLComputeUnitsCPUOnly, &err);
            CHECK_MSG(bad == nil, "a model.mil of 'x' loaded");
            hit.lease.release();
            Error qe;
            CHECK(cache::quarantine(l, "toy-stateless", hit.key32, &qe) == cache::KeyLock::Result::ok);
            CHECK(!fs::exists(dir));
            cache::PrepareOutcome r = cache::prepare(inputs_for(cache_dir, t, gguf));
            CHECK_MSG(r.kind == cache::PrepareOutcome::Kind::ready, r.error.message);
            CHECK(!r.cache_hit && r.artifact_dir == dir);
            struct stat after {};
            ::stat(mil.c_str(), &after);
            CHECK(after.st_mtimespec.tv_sec > before.st_mtimespec.tv_sec ||
                  (after.st_mtimespec.tv_sec == before.st_mtimespec.tv_sec &&
                   after.st_mtimespec.tv_nsec > before.st_mtimespec.tv_nsec));
            check_predictions("toy-stateless", r.model_path, gguf);

            // A leased artifact is never moved aside.
            Error be;
            CHECK(cache::quarantine(l, "toy-stateless", r.key32, &be) == cache::KeyLock::Result::would_block);
            CHECK(fs::exists(dir));
        }
    } else {
        CHECK_MSG(false, "macOS 15 required");
    }
}

void case_concurrent_prepare() {
    // Two prepares of identical weights into two cache roots at once. Each
    // root has its own key lock, so neither waits for the other, and Core ML
    // compiles both packages into the one user temp directory. In every round
    // the two compiles are released together, and each prepare must compile,
    // move and publish a bundle of its own that loads and predicts.
    if (@available(macOS 15.0, *)) {
        TempDir td;
        auto t = templates::embedded("toy-stateless");
        CHECK(t != nullptr);
        if (!t) return;
        const std::string gguf = twin(td, "toy-stateless", 31);
        std::string key;
        constexpr int ROUNDS = 12;
        for (int round = 0; round < ROUNDS; ++round) {
            std::mutex mu;
            std::condition_variable cv;
            int arrived = 0;
            bool together = true;
            auto compile = [&](const std::string & pkg, const std::string & out) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    ++arrived;
                    cv.notify_all();
                    together &= cv.wait_for(lk, std::chrono::seconds(30), [&] { return arrived == 2; });
                }
                return apple::compile_package(pkg, out);
            };
            cache::PrepareOutcome o[2];
            std::thread th[2];
            for (int i = 0; i < 2; ++i) {
                cache::PrepareInputs in =
                    inputs_for(td / ("cache-" + std::to_string(round) + "-" + std::to_string(i)), t, gguf);
                in.compile = compile;
                th[i] = std::thread([&o, i, in] { o[i] = cache::prepare(in); });
            }
            for (std::thread & x : th) x.join();
            CHECK(together);
            for (int i = 0; i < 2; ++i) {
                const std::string what = "round " + std::to_string(round) + ", prepare " + std::to_string(i);
                CHECK_MSG(o[i].kind == cache::PrepareOutcome::Kind::ready, what + ": " + o[i].error.message);
                if (o[i].kind != cache::PrepareOutcome::Kind::ready) continue;
                CHECK(!o[i].cache_hit);
                CHECK_MSG(cache::validate(o[i].artifact_dir) == cache::Validity::ok, what);
                check_predictions("toy-stateless", o[i].model_path, gguf);
            }
            CHECK(o[0].key32 == o[1].key32 && o[0].artifact_dir != o[1].artifact_dir);
            key = o[0].key32;
        }
        // Every compiled bundle was moved out of the user temp directory.
        int left = 0;
        std::error_code ec;
        for (fs::directory_iterator it(NSTemporaryDirectory().UTF8String, ec), end; !ec && it != end;
             it.increment(ec)) {
            left += it->path().filename().string().rfind("qvac-coreml-" + key, 0) == 0;
        }
        CHECK_MSG(!key.empty() && left == 0, std::to_string(left) + " compile output(s) of the key left behind");
    } else {
        CHECK_MSG(false, "macOS 15 required");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--tool") g_tool = argv[i + 1];
        if (std::string(argv[i]) == "--toys") g_toys = argv[i + 1];
    }
    if (g_tool.empty() || g_toys.empty()) {
        std::fprintf(stderr, "usage: test-coreml-toy --tool <qvac-coreml-tool> --toys <toy-templates dir>\n");
        return 2;
    }
    return run_cases(argc, argv,
                     {{"os-floor", case_os_floor},
                      {"embedded-table", case_embedded_table},
                      {"compile-naming", case_compile_naming},
                      {"generate-load-predict", case_generate_load_predict},
                      {"quarantine-repair", case_quarantine_repair},
                      {"concurrent-prepare", case_concurrent_prepare}},
                     "test-coreml-toy");
}
