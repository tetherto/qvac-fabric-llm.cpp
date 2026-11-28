// test-coreml-cache: the content-addressed cache with a fake compile step
// (the Apple compile is not portable). Per-test temp
// dirs; forked processes for the cross-process cases. Portable: Linux and
// macOS. Every case runs; there is no skip path.

#include "generator_test_util.h"

#include "digest.h"
#include "failed.h"
#include "hints.h"
#if defined(QVAC_COREML_HAVE_GGML)
#include "generation_internal.h"
#include "qvac_coreml/generation.h"
#endif

#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>

using namespace gtest;
using namespace std::chrono_literals;
using cache::PrepareOutcome;

namespace {

// ---- processes -------------------------------------------------------------------------

struct Pipe {
    int fd[2] = {-1, -1};
    Pipe() {
        if (::pipe(fd) != 0) std::exit(1);
    }
    ~Pipe() {
        for (int f : fd) {
            if (f >= 0) ::close(f);
        }
    }
    void send(char c) { (void) !::write(fd[1], &c, 1); }
    // The next byte, or 0 after `ms`.
    char recv(int ms = 20000) {
        pollfd p{fd[0], POLLIN, 0};
        if (::poll(&p, 1, ms) <= 0) return 0;
        char c = 0;
        return ::read(fd[0], &c, 1) == 1 ? c : 0;
    }
};

// Runs `fn` in a child; its exit code is fn's result, or 1 after a failed CHECK.
pid_t spawn(const std::function<int()> & fn) {
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        g_failures = 0;
        const int rc = fn();
        std::fflush(stderr);
        ::_exit(g_failures ? 1 : rc);
    }
    return pid;
}

int reap(pid_t pid) {
    int st = 0;
    if (::waitpid(pid, &st, 0) != pid) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
}

// ---- environment -----------------------------------------------------------------------------

TemplateSpec cache_spec() {
    TemplateSpec s;
    s.id = "toy-dac";
    s.decls = "source \"w\" f32 [4,8]\nsource \"b\" f32 [8]\nmeta \"general.architecture\" str\n";
    s.items = {blob("fp16", 32, "f16(t(\"w\"))"), blob("fp16", 8, "f16(mul(t(\"b\"), 2))")};
    s.match = {{"architecture", "toy"}};
    return s;
}

std::vector<TT> tensors(float seed) {
    std::vector<float> w(32), b(8);
    for (size_t i = 0; i < w.size(); ++i) w[i] = seed + 0.125f * static_cast<float>(i);
    for (size_t i = 0; i < b.size(); ++i) b[i] = seed - 0.5f * static_cast<float>(i);
    return {TT{"w", T_F32, {4, 8}, w}, TT{"b", T_F32, {8}, b}};
}

const std::vector<TM> kMeta = {{"general.architecture", "str", "toy"}};

struct Env {
    TempDir td;
    std::string cache;
    std::shared_ptr<const recipe::Template> tpl;

    Env() {
        cache = td / "cache";
        tpl = make_template(td / "tpl", cache_spec());
        g_compiles = 0;
        g_compile_override = nullptr;
        g_compile_gate = nullptr;
        g_before_read = nullptr;
        g_compile_marker_dir.clear();
    }

    std::string source(const std::string & name, float seed = 1.0f) {
        const std::string p = td / name;
        write_source_file(p, tensors(seed), kMeta);
        return p;
    }

    cache::PrepareInputs inputs(const std::string & path, const std::string & os_build = "TEST1") const {
        cache::PrepareInputs in;
        in.cache_dir = cache;
        in.tmpl = tpl;
        CHECK(stat_path(path, in.expected));
        in.expected.path = path;
        in.os_build = os_build;
        in.compile = fake_compile;
        in.open = FileSource::open;
        return in;
    }

    PrepareOutcome prepare(const std::string & path, const std::string & os_build = "TEST1") const {
        return cache::prepare(inputs(path, os_build));
    }

    // Retries would_block like the scheduler's re-queue (with a short backoff).
    PrepareOutcome prepare_retry(const std::string & path, int max_ms = 20000) const {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
        for (;;) {
            PrepareOutcome o = prepare(path);
            if (o.kind != PrepareOutcome::Kind::would_block || std::chrono::steady_clock::now() > end) return o;
            std::this_thread::sleep_for(5ms);
        }
    }

    cache::Catalog catalog() const {
        cache::Catalog c;
        c.templates = {{tpl->id(), tpl->sha256_hex()}};
        return c;
    }

    cache::Layout layout() const { return cache::Layout(cache); }

    // Every routine prune sweeps this test's own temp dir, never the host's
    // $TMPDIR (where the Apple compile's in-flight qvac-coreml-*.mlmodelc outputs live).
    cache::PrunePolicy routine(int64_t later_s = 0) const {
        cache::PrunePolicy p = cache::PrunePolicy::routine();
        p.tmp_dir = td / "tmp";
        if (later_s != 0) p.now_s = detail::unix_now_s() + later_s;
        return p;
    }

    int artifacts() const { return count_files(layout().artifacts); }
    int memos() const { return static_cast<int>(cache::all_memos(layout()).size()); }
    // Every memo points at a valid artifact.
    bool memos_consistent() const {
        const cache::Layout l = layout();
        for (const cache::MemoFile & m : cache::all_memos(l)) {
            if (!m.memo) return false;
            if (cache::validate(l.artifact_dir(m.memo->template_id, m.key32)) != cache::Validity::ok) return false;
        }
        return true;
    }
};

bool ready(const PrepareOutcome & o) { return o.kind == PrepareOutcome::Kind::ready && o.lease.held(); }

int64_t now_s() { return detail::unix_now_s(); }

constexpr int64_t kDay = 24 * 3600;

// ---- cases ----------------------------------------------------------------------------------

void case_golden_keys() {
    const std::string spec = "spec";
    const std::string spec_hex = detail::to_hex(detail::sha256(spec.data(), spec.size()));
    const std::string recipe_txt = "mlrecipe 1\ntemplate golden\nspec_sha256 " + spec_hex +
                                   "\nweight_bin_size 192\nblobs 1\n"
                                   "source \"b\" f16 [2]\nsource \"w\" f32 [2,2]\nmeta \"toy.n\" u32\n"
                                   "meta \"general.architecture\" str\n"
                                   "blob 64 fp16 4 f16(t(\"w\"))\nend\n";
    const std::string tjson = R"({"schema":1,"id":"golden","weight_bin_size":192,"functions":[{"name":"main"}]})";
    const std::string manifest = "{}";
    auto file = [](const char * n, const std::string & b) {
        recipe::TemplateFile f;
        f.name = n;
        f.data = reinterpret_cast<const uint8_t *>(b.data());
        f.size = b.size();
        return f;
    };
    auto t = recipe::Template::from_files({file("template.json", tjson), file("recipe.txt", recipe_txt),
                                           file("model.mlmodel", spec), file("Manifest.json", manifest)});
    // Expected values come from an independent Python implementation of the keys.
    CHECK(t->sha256_hex() == "6be7a8664b55e85239cce95152728b8fbe7c5e5a26dd44ea865707b8fbd90d9e");
    MemSource src({TT{"w", T_F32, {2, 2}, {1, 2, 3, 4}}, TT{"b", T_F16, {2}, {0.5f, -1.0f}}},
                  {{"toy.n", "u32", 7}, {"general.architecture", "str", "toy"}});
    digest::SourceDigest dg;
    CHECK(digest::source_digest(t->recipe(), src, dg).ok());
    CHECK(detail::to_hex(dg.digest) == "958d2e755219e86ee59a0d2457e918eae1b457e35a2984ea3a08b692f0d08959");
    CHECK(dg.tensors == 2 && dg.bytes == 20);
    CHECK(digest::artifact_key(GENERATOR_VERSION, t->sha256(), dg.digest, "out=fp16") ==
          "3bf6324bc2800e1fc7b84aa991e7cf70");
    CHECK(digest::memo_id("/models/a.gguf", 1234, 1700000000123456789LL, 42, 7, t->sha256(), GENERATOR_VERSION,
                          "out=fp16") == "ad46d5f5a7bab0f449b9433e87008867");
    // Each key changes with each of its inputs.
    CHECK(digest::artifact_key(GENERATOR_VERSION + 1, t->sha256(), dg.digest, "out=fp16") !=
          "3bf6324bc2800e1fc7b84aa991e7cf70");
    CHECK(digest::artifact_key(GENERATOR_VERSION, t->sha256(), dg.digest, "out=fp32") !=
          "3bf6324bc2800e1fc7b84aa991e7cf70");
    CHECK(digest::memo_id("/models/a.gguf", 1234, 1700000000123456790LL, 42, 7, t->sha256(), GENERATOR_VERSION,
                          "out=fp16") != "ad46d5f5a7bab0f449b9433e87008867");
    MemSource other({TT{"w", T_F32, {2, 2}, {1, 2, 3, 5}}, TT{"b", T_F16, {2}, {0.5f, -1.0f}}},
                    {{"toy.n", "u32", 7}, {"general.architecture", "str", "toy"}});
    digest::SourceDigest dg2;
    CHECK(digest::source_digest(t->recipe(), other, dg2).ok() && dg2.digest != dg.digest);
}

void case_generate_and_artifact_json() {
    Env env;
    const std::string a = env.source("a.gguf");
    PrepareOutcome o = env.prepare(a);
    CHECK_MSG(ready(o), o.error.message);
    CHECK(!o.cache_hit && !o.memo_hit);
    CHECK(g_compiles == 1);
    CHECK(o.artifact_dir == env.layout().artifact_dir("toy-dac", o.key32));
    CHECK(cache::validate(o.artifact_dir) == cache::Validity::ok);
    const auto info = cache::read_artifact_json(o.artifact_dir);
    CHECK(info.has_value());
    if (info) {
        CHECK(info->complete && info->artifact_key == o.key32 && info->template_id == "toy-dac");
        CHECK(info->template_sha256 == env.tpl->sha256_hex() && info->generator_version == GENERATOR_VERSION);
        CHECK(info->options == "out=fp16" && info->os_build == "TEST1" && info->export_kind == "toy-v1");
        CHECK(info->source_tensors == 2 && info->source_bytes == (32 + 8) * 4);
        CHECK(info->inventory.size() == 4);
        CHECK(info->inventory[0].path == "model.mlmodelc/analytics/coremldata.bin");
        CHECK(info->functions == std::vector<std::string>({"main"}));
    }
    // The compiled weight.bin is the generator's: MILBlob v2 with the recipe's layout.
    const std::string wb = read_bytes(o.model_path + "/weights/weight.bin");
    CHECK(wb.size() == env.tpl->recipe().weight_bin_size);
    auto idx = milblob::read_index(reinterpret_cast<const uint8_t *>(wb.data()), wb.size());
    CHECK(idx && idx->size() == 2 && (*idx)[0].offset == 64 && (*idx)[1].offset == 192);
    uint16_t h0;
    std::memcpy(&h0, wb.data() + 128, 2);
    CHECK(h0 == recipe::f32_to_f16(1.0f));
    // Memo, lock, no staging leftovers.
    CHECK(env.memos() == 1);
    const auto mf = cache::find_memo(env.layout(), o.memo_id);
    CHECK(mf && mf->memo && mf->memo->artifact_key == o.key32 && mf->memo->path == a);
    CHECK(fs::exists(env.layout().lock_path(o.key32)));
    CHECK(count_files(env.layout().staging) == 0);
    // generate() via the tool path writes the same weight.bin.
    MemSource ms(tensors(1.0f), kMeta);
    CHECK(cache::write_package(*env.tpl, ms, env.td / "pkg.mlpackage").ok());
    CHECK(read_bytes(env.td / "pkg.mlpackage/Data/com.apple.CoreML/weights/weight.bin") == wb);
    CHECK(read_bytes(env.td / "pkg.mlpackage/Manifest.json") == read_bytes(env.td / "tpl/Manifest.json"));
}

// The generation peak covers the spec patch pass too: a patch expression
// whose interpreter peak exceeds every blob expression's sets it.
void case_patch_pass_peak() {
    TempDir td;
    TemplateSpec spec = cache_spec();
    spec.decls = "source \"w\" f32 [4,8]\nsource \"b\" f32 [8]\nsource \"big\" f32 [4096]\n"
                 "meta \"general.architecture\" str\n";
    spec.items.push_back(patch_(16, "fp16", 4096, "f16(mul(t(\"big\"), 2))"));
    spec.spec_bytes = std::string(16 + 4096 * 2 + 16, '\x01');
    std::string sentinel;
    for (int i = 0; i < 4096 * 2; ++i) sentinel += "01";
    spec.extra["patches"] = json::array({{{"spec_offset", 16}, {"len", 4096 * 2}, {"sentinel_hex", sentinel}}});
    auto tpl = make_template(td / "tpl", spec);
    std::vector<TT> ts = tensors(1.0f);
    ts.push_back(TT{"big", T_F32, {4096}, std::vector<float>(4096, 0.25f)});
    MemSource ms(ts, kMeta);
    recipe::EvalStats blobs, patches;
    CHECK(recipe::evaluate(tpl->recipe(), ms, [](const recipe::BlobOut &) { return Error{}; }, &blobs).ok());
    std::vector<uint8_t> patched;
    const recipe::TemplateFile * f = tpl->file("model.mlmodel");
    CHECK(recipe::patched_spec(tpl->recipe(), ms, f->data, f->size, patched, nullptr, &patches).ok());
    // t("big") and its product are live together, plus the spec copy.
    CHECK(patches.peak_bytes >= 2 * 4096 * 4 + f->size);
    CHECK(patches.peak_bytes > blobs.peak_bytes);
    recipe::EvalStats st;
    CHECK(cache::write_package(*tpl, ms, td / "pkg.mlpackage", &st).ok());
    CHECK_MSG(st.peak_bytes == patches.peak_bytes,
              std::to_string(st.peak_bytes) + " vs patch pass " + std::to_string(patches.peak_bytes));
    CHECK(read_bytes(td / "pkg.mlpackage/Data/com.apple.CoreML/model.mlmodel") ==
          std::string(patched.begin(), patched.end()));
}

void case_memo_hot_path() {
    Env env;
    const std::string a = env.source("a.gguf");
    { PrepareOutcome o = env.prepare(a); CHECK(ready(o)); }
    const int reads = g_file_raw_reads;
    PrepareOutcome o = env.prepare(a);
    CHECK(ready(o) && o.cache_hit && o.memo_hit);
    CHECK(g_file_raw_reads == reads);  // fstat, memo scan, adopt: no hashing
    CHECK(g_compiles == 1);
    // Same bytes, new mtime: a new memo, the same artifact, no regeneration.
    o.lease.release();
    std::this_thread::sleep_for(20ms);
    write_source_file(a, tensors(1.0f), kMeta);
    PrepareOutcome o2 = env.prepare(a);
    CHECK(ready(o2) && o2.cache_hit && !o2.memo_hit && o2.key32 == o.key32 && o2.memo_id != o.memo_id);
    CHECK(g_compiles == 1);
    CHECK(env.memos() == 2);
    // Different bytes: a different key. (Linux file timestamps are
    // tick-coarse: a same-size rewrite within one tick keeps the stat tuple.)
    std::this_thread::sleep_for(20ms);
    write_source_file(a, tensors(2.0f), kMeta);
    PrepareOutcome o3 = env.prepare(a);
    CHECK(ready(o3) && !o3.cache_hit && o3.key32 != o.key32);
    CHECK(g_compiles == 2);
}

void case_publish_race() {
    Env env;
    const std::string a = env.source("a.gguf");
    g_compile_marker_dir = env.td / "compiles";
    fs::create_directories(g_compile_marker_dir);
    Pipe start;
    std::vector<pid_t> kids;
    for (int i = 0; i < 8; ++i) {
        kids.push_back(spawn([&] {
            start.recv();  // all eight start together
            PrepareOutcome o = env.prepare_retry(a);
            CHECK_MSG(ready(o), o.error.message);
            CHECK(cache::validate(o.artifact_dir) == cache::Validity::ok);
            return 0;
        }));
    }
    for (int i = 0; i < 8; ++i) start.send('g');
    int failures = 0;
    for (pid_t k : kids) failures += reap(k) != 0;
    CHECK(failures == 0);
    CHECK(env.artifacts() == 1);
    CHECK(count_files(g_compile_marker_dir) == 1);  // one process generated; seven adopted
    CHECK(env.memos() == 1);
    CHECK(count_files(env.layout().staging) == 0);
    CHECK(env.memos_consistent());
}

void case_recheck_under_lock() {
    // Another process publishes the key between this job's adoption check
    // and its LOCK_EX (here: inside the pre-flight step). The re-check under
    // EX adopts the winner's artifact instead of generating a second one.
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf");  // same bytes, another path
    cache::PrepareInputs in = env.inputs(b);
    bool nested = false;
    in.fs_probe = [&](const std::string & dir, cache::FsFacts & f) {
        if (!nested) {
            nested = true;
            PrepareOutcome oa = env.prepare(a);
            CHECK(ready(oa) && !oa.cache_hit && g_compiles == 1);
            // A failure record left at the key: the re-check under EX clears it.
            failed::record(env.layout().failed, oa.key32, failed::Stage::compile,
                           Error::transient("x", "com.apple.CoreML", 3), "toy-dac", "TEST1", now_s());
        }
        return cache::default_fs_probe(dir, f);
    };
    PrepareOutcome o = cache::prepare(in);
    CHECK(nested);
    CHECK(ready(o) && o.cache_hit);
    CHECK(g_compiles == 1);
    CHECK(env.artifacts() == 1 && env.memos() == 2);
    CHECK(!fs::exists(failed::record_path(env.layout().failed, o.key32)));
}

void case_hit_adoption() {
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf");  // same bytes, another file
    std::string key;
    {
        PrepareOutcome oa = env.prepare(a);
        CHECK(ready(oa) && !oa.cache_hit);
        key = oa.key32;
        PrepareOutcome ob = env.prepare(b);  // B hits K and writes B -> K
        CHECK(ready(ob) && ob.cache_hit && !ob.memo_hit && ob.key32 == key);
    }  // both unload (leases released)
    CHECK(env.memos() == 2 && g_compiles == 1);
    fs::remove(a);
    // Exactly the SDK's call after deleting model A.
    cache::PruneReport r = cache::prune(env.cache, cache::PrunePolicy::scoped({a}), env.catalog());
    CHECK(r.deleted_artifacts.empty());
    CHECK(r.deleted_memos.size() == 1);
    CHECK(env.artifacts() == 1 && env.memos() == 1);
    const int reads = g_file_raw_reads;
    PrepareOutcome ob = env.prepare(b);
    CHECK(ready(ob) && ob.cache_hit && ob.memo_hit && ob.key32 == key);
    CHECK(g_compiles == 1 && g_file_raw_reads == reads);  // no regeneration, no hashing
}

// Hit adoption whose memo write fails (I/O): no adoption, no lease; nothing
// may count as a reference that is not on disk. Blocking the memo's name with
// a directory makes the rename fail for every user, root included.
void case_hit_adoption_memo_io_failure() {
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf");  // same bytes, another file
    const cache::Layout l = env.layout();
    std::string key;
    {
        PrepareOutcome oa = env.prepare(a);
        CHECK(ready(oa));
        key = oa.key32;
    }
    SourceStat st;
    CHECK(stat_path(b, st));
    const std::string real = fs::canonical(b).string();
    const std::string memo_b = l.memo_path(digest::memo_id(real, st.size, st.mtime_ns, st.ino, st.dev,
                                                           env.tpl->sha256(), GENERATOR_VERSION,
                                                           digest::DEFAULT_OPTIONS),
                                           key);
    fs::create_directories(memo_b + "/blocker");
    PrepareOutcome ob = env.prepare(b);
    CHECK(ob.kind == PrepareOutcome::Kind::failed && ob.error.cls == ErrorClass::transient && !ob.lease.held());
    int readable = 0;
    for (const cache::MemoFile & m : cache::all_memos(l)) readable += m.memo ? 1 : 0;
    CHECK(readable == 1 && fs::is_directory(memo_b));  // only A's memo
    // No lease is left behind: prune can take EX on the key.
    {
        cache::KeyLock ex;
        CHECK(cache::KeyLock::take(l, key, cache::KeyLock::Mode::exclusive, ex) == cache::KeyLock::Result::ok);
    }
    // Once the write can succeed, B adopts and its memo protects K from A's scoped prune.
    fs::remove_all(memo_b);
    {
        PrepareOutcome ok = env.prepare(b);
        CHECK(ready(ok) && ok.cache_hit && ok.key32 == key && fs::is_regular_file(memo_b));
    }
    // A memo that is present but unreadable is rewritten, not trusted.
    write_bytes(memo_b, R"({"schema":1})");
    {
        PrepareOutcome ok = env.prepare(b);
        CHECK(ready(ok) && ok.cache_hit && !ok.memo_hit);
        std::ifstream in(memo_b);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto m = cache::memo_from_json(text);
        CHECK(m && m->artifact_key == key && m->path == real);
    }
    fs::remove(a);
    cache::PruneReport r = cache::prune(env.cache, cache::PrunePolicy::scoped({a}), env.catalog());
    CHECK(r.deleted_artifacts.empty() && env.artifacts() == 1);
    CHECK(g_compiles == 1);
}

// A prune deletes the memo between the hot path's lookup and its adoption (the
// artifact stays): the adoption re-checks the memo under SH and writes it back,
// so a ready source always has its reference on disk.
void case_memo_hit_recheck() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    std::string key, memo_file;
    {
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o));
        key = o.key32;
        memo_file = l.memo_path(o.memo_id, key);
    }
    CHECK(fs::is_regular_file(memo_file));
    cache::PrepareInputs in = env.inputs(a);
    int hits = 0;
    in.after_memo_lookup = [&] {
        ++hits;
        fs::remove(memo_file);
    };
    const int reads = g_file_raw_reads;
    PrepareOutcome o = cache::prepare(in);
    CHECK(hits == 1 && ready(o) && o.memo_hit && o.key32 == key);
    CHECK(g_file_raw_reads == reads && g_compiles == 1);  // still the hot path
    CHECK(fs::is_regular_file(memo_file) && env.memos() == 1 && env.memos_consistent());
}

// The same failure under EX: the re-check finds the artifact another process
// published, and the memo write fails; prepare fails transiently and holds no lock.
void case_recheck_memo_io_failure() {
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf");
    const cache::Layout l = env.layout();
    cache::PrepareInputs in = env.inputs(b);
    SourceStat st;
    CHECK(stat_path(b, st));
    const std::string memo_id = digest::memo_id(fs::canonical(b).string(), st.size, st.mtime_ns, st.ino, st.dev,
                                                env.tpl->sha256(), GENERATOR_VERSION, digest::DEFAULT_OPTIONS);
    std::string key;
    bool nested = false;
    in.fs_probe = [&](const std::string & dir, cache::FsFacts & f) {
        if (!nested) {
            nested = true;
            PrepareOutcome oa = env.prepare(a);
            CHECK(ready(oa));
            key = oa.key32;
            fs::create_directories(l.memo_path(memo_id, key) + "/blocker");
        }
        return cache::default_fs_probe(dir, f);
    };
    PrepareOutcome o = cache::prepare(in);
    CHECK(nested && o.kind == PrepareOutcome::Kind::failed && o.error.cls == ErrorClass::transient);
    CHECK(!o.lease.held() && g_compiles == 1);
    cache::KeyLock ex;
    CHECK(cache::KeyLock::take(l, key, cache::KeyLock::Mode::exclusive, ex) == cache::KeyLock::Result::ok);
}

void case_relocation() {
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf");
    std::string key;
    {
        PrepareOutcome oa = env.prepare(a);
        PrepareOutcome ob = env.prepare(b);
        CHECK(ready(oa) && ready(ob));
        key = oa.key32;
    }
    fs::create_directories(env.td / "moved");
    const std::string b2 = env.td / "moved/b.gguf";
    fs::rename(b, b2);  // B moves and is not reloaded
    fs::remove(a);
    cache::PruneReport r = cache::prune(env.cache, cache::PrunePolicy::scoped({a}), env.catalog());
    CHECK(r.deleted_artifacts.empty());  // B's dead memo is outside the scope
    CHECK(r.deleted_memos.size() == 1 && env.memos() == 1);
    PrepareOutcome o = env.prepare(b2);
    CHECK(ready(o) && o.cache_hit && !o.memo_hit && o.key32 == key);
    CHECK(g_compiles == 1);
    CHECK(env.memos() == 2);  // the new path's memo
    const auto mf = cache::find_memo(env.layout(), o.memo_id);
    CHECK(mf && mf->memo && mf->memo->path == b2);
}

void case_scope() {
    Env env;
    const std::string a = env.source("a.gguf");
    std::string key;
    {
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o));
        key = o.key32;
    }
    // A memo-less artifact (its memo removed by hand).
    for (const cache::MemoFile & m : cache::all_memos(env.layout())) fs::remove(m.file);
    cache::PruneReport r = cache::prune(env.cache, cache::PrunePolicy::scoped({env.td / "elsewhere.gguf"}), env.catalog());
    CHECK(r.deleted_artifacts.empty() && env.artifacts() == 1);
    r = cache::prune(env.cache, cache::PrunePolicy::scoped({a}), env.catalog());  // no memo: never scoped
    CHECK(r.deleted_artifacts.empty() && env.artifacts() == 1);
    r = cache::prune(env.cache, env.routine(), env.catalog());
    CHECK(r.deleted_artifacts.empty() && env.artifacts() == 1);  // inside the grace period
    cache::PrunePolicy later = env.routine();
    later.now_s = now_s() + 31 * kDay;
    r = cache::prune(env.cache, later, env.catalog());
    CHECK(r.deleted_artifacts.size() == 1 && env.artifacts() == 0);
    CHECK(!fs::exists(env.layout().used_path(key)));
    CHECK(fs::exists(env.layout().lock_path(key)));  // locks/ is never pruned

    // The scoped positive case: the only memo is dead and in scope.
    const std::string c = env.source("c.gguf", 3.0f);
    { CHECK(ready(env.prepare(c))); }
    fs::remove(c);
    r = cache::prune(env.cache, cache::PrunePolicy::scoped({env.td.path}), env.catalog());  // a directory scope
    CHECK(r.deleted_artifacts.size() == 1 && env.artifacts() == 0 && env.memos() == 0);
}

// routine removes compile outputs (qvac-coreml-*.mlmodelc) older than an hour
// from its tmp_dir only; Env::routine() keeps every test off the host's $TMPDIR.
void case_tmp_outputs() {
    Env env;
    const std::string tmp = env.td / "tmp";
    const std::string out = tmp + "/qvac-coreml-1234.abcd.mlmodelc";
    const std::vector<std::string> other = {tmp + "/qvac-coreml-1234.mlpackage", tmp + "/other.mlmodelc",
                                            tmp + "/qvac-coreml-gen-x"};
    for (const std::string & d : other) fs::create_directories(d);
    write_bytes(out + "/coremldata.bin", "x");
    // A compile output outside this test's tmp_dir (the host's $TMPDIR): never touched.
    TempDir host;
    const std::string host_out = host / "qvac-coreml-9999.ffff.mlmodelc";
    fs::create_directories(host_out);
    cache::PrunePolicy r0 = env.routine();
    CHECK(r0.tmp_dir == tmp && !r0.now_s);
    fs::create_directories(env.layout().root);  // prune returns early without a cache root
    CHECK(cache::prune(env.cache, r0, env.catalog()).tmp == 0 && fs::exists(out));  // in flight
    CHECK(cache::prune(env.cache, env.routine(2 * 3600), env.catalog()).tmp == 1 && !fs::exists(out));
    for (const std::string & d : other) CHECK(fs::exists(d));
    CHECK(fs::exists(host_out));
}

// Core ML compiles <dir>/<stem>.mlpackage into <user temp>/<stem>.mlmodelc, a
// path every process and every cache directory shares. Two prepares of the
// same weights into two cache roots hold different key locks and compile at
// the same time (both threads of one process here, so the pid alone cannot
// tell them apart): each must hand the compile its own package stem. The
// stem keeps the qvac-coreml-<key> prefix, so routine prune still sweeps the
// leftover of a compile that was killed.
void case_package_stem_per_invocation() {
    Env env;
    const std::string a = env.source("a.gguf");
    std::mutex mu;
    std::condition_variable cv;
    int arrived = 0;
    bool together = true;
    std::vector<std::string> pkgs;
    auto compile = [&](const std::string & pkg, const std::string & out) {
        {
            std::unique_lock<std::mutex> lk(mu);
            pkgs.push_back(pkg);
            ++arrived;
            cv.notify_all();
            // Both are compiling at once: neither prepare waits for the other's lock.
            together &= cv.wait_for(lk, 10s, [&] { return arrived == 2; });
        }
        return fake_compile(pkg, out);
    };
    PrepareOutcome o[2];
    std::thread th[2];
    for (int i = 0; i < 2; ++i) {
        cache::PrepareInputs in = env.inputs(a);
        in.cache_dir = env.td / ("cache-" + std::to_string(i));
        in.compile = compile;
        th[i] = std::thread([&o, i, in] { o[i] = cache::prepare(in); });
    }
    for (std::thread & t : th) t.join();
    CHECK(together);
    CHECK_MSG(ready(o[0]) && ready(o[1]), o[0].error.message + " / " + o[1].error.message);
    CHECK(o[0].key32 == o[1].key32 && o[0].artifact_dir != o[1].artifact_dir);
    for (const PrepareOutcome & x : o) CHECK(cache::validate(x.artifact_dir) == cache::Validity::ok);
    CHECK(pkgs.size() == 2);
    if (pkgs.size() != 2) return;
    auto stem = [](const std::string & pkg) { return fs::path(pkg).stem().string(); };
    CHECK_MSG(stem(pkgs[0]) != stem(pkgs[1]), "both compiles used the package stem " + stem(pkgs[0]));
    for (const std::string & p : pkgs) {
        CHECK(fs::path(p).extension() == ".mlpackage");
        CHECK_MSG(stem(p).rfind("qvac-coreml-" + o[0].key32, 0) == 0, p);
    }

    // The bundle a killed compile leaves at <user temp>/<stem>.mlmodelc,
    // including the _<UUID> name Core ML picks when that path is taken.
    const std::string tmp = env.td / "tmp";
    const std::string uuid = "0F9E1D2C-3B4A-5968-7766-554433221100";
    const std::vector<std::string> left = {tmp + "/" + stem(pkgs[0]) + ".mlmodelc",
                                           tmp + "/" + stem(pkgs[1]) + "_" + uuid + ".mlmodelc"};
    for (const std::string & d : left) write_bytes(d + "/coremldata.bin", "x");
    fs::create_directories(env.layout().root);
    CHECK(cache::prune(env.cache, env.routine(), env.catalog()).tmp == 0);  // in flight
    CHECK(cache::prune(env.cache, env.routine(2 * 3600), env.catalog()).tmp == 2);
    for (const std::string & d : left) CHECK(!fs::exists(d));
}

void case_used_age() {
    // routine measures age from used/<key32> (last successful load).
    Env env;
    const std::string a = env.source("a.gguf");
    std::string key;
    {
        PrepareOutcome o = env.prepare(a);
        key = o.key32;
    }
    fs::remove(a);  // the memo is dead: only age protects the artifact
    cache::touch_used(env.layout(), key);
    cache::PrunePolicy p = env.routine();
    p.now_s = now_s() + 29 * kDay;
    CHECK(cache::prune(env.cache, p, env.catalog()).deleted_artifacts.empty());
    p.now_s = now_s() + 31 * kDay;
    CHECK(cache::prune(env.cache, p, env.catalog()).deleted_artifacts.size() == 1);
}

void case_superseded_and_retired() {
    for (int variant = 0; variant < 3; ++variant) {
        Env env;
        const std::string a = env.source("a.gguf");
        { CHECK(ready(env.prepare(a))); }
        cache::Catalog c = env.catalog();
        if (variant == 0) c.templates[0].second = std::string(64, 'f');  // superseded template
        if (variant == 1) c.generator_version = GENERATOR_VERSION + 1;   // new generator
        if (variant == 2) {                                              // retired id
            c.templates.clear();
            c.retired_ids = {"toy-dac"};
        }
        // The file still exists, but the memo is dead: collected only after the grace period.
        CHECK(cache::prune(env.cache, env.routine(), c).deleted_artifacts.empty());
        cache::PrunePolicy later = env.routine();
        later.now_s = now_s() + 31 * kDay;
        cache::PruneReport r = cache::prune(env.cache, later, c);
        CHECK_MSG(r.deleted_artifacts.size() == 1 && env.artifacts() == 0 && env.memos() == 0,
                  "variant " + std::to_string(variant));
        // The current catalog would have kept it: the live memo protects it.
        Env env2;
        const std::string a2 = env2.source("a.gguf");
        { CHECK(ready(env2.prepare(a2))); }
        CHECK(cache::prune(env2.cache, later, env2.catalog()).deleted_artifacts.empty());
    }
}

void case_dead_memos_of_kept_artifacts() {
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf");
    {
        PrepareOutcome oa = env.prepare(a);
        PrepareOutcome ob = env.prepare(b);
        CHECK(ready(oa) && ready(ob));
    }
    fs::remove(a);
    cache::PruneReport r = cache::prune(env.cache, env.routine(), env.catalog());
    CHECK(r.deleted_memos.empty() && env.memos() == 2);  // young dead memo stays
    cache::PrunePolicy later = env.routine();
    later.now_s = now_s() + 31 * kDay;
    r = cache::prune(env.cache, later, env.catalog());
    CHECK(r.deleted_artifacts.empty() && env.artifacts() == 1);  // B is live
    CHECK(r.deleted_memos.size() == 1 && env.memos() == 1);
    // Orphan memos (artifact gone, e.g. after a crash) are collected too.
    const cache::Layout l = env.layout();
    cache::Memo m = *cache::all_memos(l)[0].memo;
    m.artifact_key = std::string(32, 'a');
    m.path = env.td / "gone.gguf";
    CHECK(cache::write_memo(l, std::string(32, 'b'), m).ok());
    CHECK(env.memos() == 2);
    r = cache::prune(env.cache, later, env.catalog());
    CHECK(r.deleted_memos.size() == 1 && env.memos() == 1);
}

void case_unknown_ids() {
    Env env;
    const std::string a = env.source("a.gguf");
    std::string key;
    {
        PrepareOutcome o = env.prepare(a);
        key = o.key32;
    }
    const cache::Layout l = env.layout();
    // Another addon's artifact, memo, failure record and quarantine entry.
    const std::string other_key = std::string(31, 'c') + "d";
    fs::create_directories(l.artifact_dir("other-tpl", other_key) + "/model.mlmodelc");
    cache::Memo m = *cache::all_memos(l)[0].memo;
    m.template_id = "other-tpl";
    m.artifact_key = other_key;
    m.path = env.td / "deleted.gguf";
    CHECK(cache::write_memo(l, std::string(32, 'e'), m).ok());
    m.artifact_key = std::string(32, '9');  // and an orphan one (its artifact is gone)
    CHECK(cache::write_memo(l, std::string(32, 'f'), m).ok());
    failed::record(l.failed, other_key, failed::Stage::prepare, Error::transient("x"), "other-tpl", "TEST1",
                   now_s() - 10 * kDay);
    fs::create_directories(l.quarantine + "/other-tpl." + other_key + ".123");
    const std::string other_staging = l.staging + "/" + other_key + ".123.abcd";
    fs::create_directories(other_staging);
    write_bytes(other_staging + "/" + cache::STAGING_OWNER_FILE, "other-tpl");
    fs::remove(a);
    cache::PrunePolicy later = env.routine();
    later.now_s = now_s() + 31 * kDay;
    for (const cache::PrunePolicy & p :
         {later, cache::PrunePolicy::scoped({env.td / "deleted.gguf", a}), cache::PrunePolicy::all()}) {
        cache::prune(env.cache, p, env.catalog());
        CHECK(fs::exists(l.artifact_dir("other-tpl", other_key)));
        CHECK(fs::exists(l.memo_path(std::string(32, 'e'), other_key)));
        CHECK(fs::exists(l.memo_path(std::string(32, 'f'), std::string(32, '9'))));
        CHECK(fs::exists(failed::record_path(l.failed, other_key)));
        CHECK(fs::exists(l.quarantine + "/other-tpl." + other_key + ".123"));
        CHECK(fs::exists(other_staging));
    }
    CHECK(!fs::exists(l.artifact_dir("toy-dac", key)));  // ours went
}

// Staging carries its owner's template id; prune removes only staging owned
// by this library (embedded or retired ids), lease-free, and (routine) > 1 h old.
void case_staging_ownership() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    std::string seen_owner;
    g_compile_gate = [&] {
        for (const auto & e : fs::directory_iterator(l.staging)) {
            std::ifstream in((e.path() / cache::STAGING_OWNER_FILE).string());
            std::getline(in, seen_owner);
        }
    };
    {
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o));
    }
    g_compile_gate = nullptr;
    CHECK(seen_owner == "toy-dac");
    auto make = [&](const std::string & name, const char * owner) {
        const std::string d = l.staging + "/" + name;
        fs::create_directories(d + "/qvac-coreml-x.mlpackage");
        if (owner) write_bytes(d + "/" + cache::STAGING_OWNER_FILE, owner);
        return d;
    };
    const std::string k1(32, '1'), k2(32, '2'), k3(32, '3'), k4(32, '4'), k5(32, '5');
    auto ours = [&] { return make(k1 + ".1.a", "toy-dac"); };
    auto retired = [&] { return make(k2 + ".2.b", "old-tpl"); };
    const std::string unknown = make(k3 + ".3.c", "other-tpl");
    const std::string no_owner = make(k4 + ".4.d", nullptr);
    const std::string junk = make("not-a-key.5", "toy-dac");
    const std::string busy = make(k5 + ".6.e", "toy-dac");
    std::string d1 = ours(), d2 = retired();
    cache::KeyLock lease;
    CHECK(cache::KeyLock::take(l, k5, cache::KeyLock::Mode::shared, lease) == cache::KeyLock::Result::ok);
    cache::Catalog cat = env.catalog();
    cat.retired_ids = {"old-tpl"};
    auto survivors = [&] {
        return fs::exists(unknown) && fs::exists(no_owner) && fs::exists(junk);
    };
    cache::PrunePolicy fresh = env.routine();
    cache::prune(env.cache, fresh, cat);  // nothing is older than 1 h
    CHECK(fs::exists(d1) && fs::exists(d2) && fs::exists(busy) && survivors());
    cache::prune(env.cache, cache::PrunePolicy::scoped({a}), cat);  // scoped never touches staging
    CHECK(fs::exists(d1) && fs::exists(d2) && fs::exists(busy) && survivors());
    cache::PrunePolicy later = fresh;
    later.now_s = now_s() + 2 * 3600;
    CHECK(cache::prune(env.cache, later, cat).staging == 2);
    CHECK(!fs::exists(d1) && !fs::exists(d2) && fs::exists(busy) && survivors());
    d1 = ours();
    d2 = retired();
    CHECK(cache::prune(env.cache, cache::PrunePolicy::all(), cat).staging == 2);
    CHECK(!fs::exists(d1) && !fs::exists(d2) && fs::exists(busy) && survivors());
    lease.release();
    CHECK(cache::prune(env.cache, cache::PrunePolicy::all(), cat).staging == 1);
    CHECK(!fs::exists(busy) && survivors());
}

// prune(all) removes the hints of this library's artifacts (embedded or
// retired ids), never another library's, a prebuilt sidecar's, another
// cache's or an unattributable one, and skips a leased key.
void case_hint_ownership() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    std::string dir;
    {
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o));
        dir = o.artifact_dir;
    }
    const std::string k2(32, '2'), k3(32, '3'), k6(32, '6'), k7(32, '7');
    auto hint = [&](const std::string & name, const std::string & artifact) {
        if (!artifact.empty()) fs::create_directories(artifact);
        detail::HintRecord rec;
        rec.warm_load_ms = 10;
        rec.last_ok = detail::iso8601_utc(now_s());
        rec.artifact = artifact;
        CHECK(detail::write_hint(l.hints + "/" + name + ".json", rec));
        return l.hints + "/" + name + ".json";
    };
    const std::string owned = hint("owned", dir + "/model.mlmodelc");
    const std::string retired = hint("retired", l.artifact_dir("old-tpl", k2));
    const std::string busy = hint("busy", l.artifact_dir("toy-dac", k7) + "/model.mlmodelc");
    const std::vector<std::string> keep = {
        hint("unknown", l.artifact_dir("other-tpl", k3) + "/model.mlmodelc"),
        hint("prebuilt", env.td / "models/s3gen.mlmodelc"),
        hint("other-cache", env.td / ("elsewhere/qvac-coreml-v1/artifacts/toy-dac." + k6)),
        hint("unattributed", ""),
    };
    CHECK(detail::read_hint(owned)->artifact.find("/artifacts/toy-dac.") != std::string::npos);
    cache::Catalog cat = env.catalog();
    cat.retired_ids = {"old-tpl"};
    cache::KeyLock lease;
    CHECK(cache::KeyLock::take(l, k7, cache::KeyLock::Mode::shared, lease) == cache::KeyLock::Result::ok);
    auto kept = [&] {
        bool all = true;
        for (const std::string & h : keep) all = all && fs::exists(h);
        return all;
    };
    cache::PrunePolicy later = env.routine();
    later.now_s = now_s() + 400 * kDay;
    for (const cache::PrunePolicy & p : {later, cache::PrunePolicy::scoped({a, dir})}) {
        CHECK(cache::prune(env.cache, p, cat).hints == 0);  // only `all` removes hints
    }
    CHECK(fs::exists(owned) && fs::exists(retired) && fs::exists(busy) && kept());
    CHECK(cache::prune(env.cache, cache::PrunePolicy::all(), cat).hints == 2);
    CHECK(!fs::exists(owned) && !fs::exists(retired) && fs::exists(busy) && kept());
    lease.release();
    CHECK(cache::prune(env.cache, cache::PrunePolicy::all(), cat).hints == 1);
    CHECK(!fs::exists(busy) && kept() && fs::is_directory(l.hints));
}

void case_prune_vs_leases() {
    Env env;
    const std::string a = env.source("a.gguf");
    // In-process lease.
    PrepareOutcome o = env.prepare(a);
    CHECK(ready(o));
    cache::PruneReport r = cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
    CHECK_MSG(r.busy_artifacts.size() == 1 && r.deleted_artifacts.empty() && env.memos() == 1,
              "busy " + std::to_string(r.busy_artifacts.size()) + " deleted " +
                  std::to_string(r.deleted_artifacts.size()) + " memos " + std::to_string(env.memos()) +
                  " lease fd held " + std::to_string(o.lease.held()));
    std::string why;
    CHECK_MSG(cache::validate(o.artifact_dir, &why) == cache::Validity::ok, why + " " + o.artifact_dir);
    o.lease.release();

    // `all` concurrent with a load in another process.
    Pipe loaded, go;
    pid_t kid = spawn([&] {
        PrepareOutcome k = env.prepare_retry(a);
        CHECK(ready(k));
        loaded.send('l');
        go.recv();  // hold the model (lease) until told
        CHECK(cache::validate(k.artifact_dir) == cache::Validity::ok);
        return 0;
    });
    CHECK(loaded.recv() == 'l');
    r = cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
    CHECK(r.busy_artifacts.size() == 1 && r.deleted_artifacts.empty());
    cache::PrunePolicy later = env.routine();
    later.now_s = now_s() + 365 * kDay;
    fs::remove(a);  // dead memo, old: routine would collect it, but it is leased
    r = cache::prune(env.cache, later, env.catalog());
    CHECK(r.busy_artifacts.size() == 1 && r.deleted_artifacts.empty());
    go.send('g');
    CHECK(reap(kid) == 0);
    r = cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
    CHECK(r.deleted_artifacts.size() == 1 && env.artifacts() == 0 && env.memos() == 0);
}

void case_adoption_vs_prune_race() {
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf", 2.0f);
    const std::string c = env.source("c.gguf", 3.0f);
    { CHECK(ready(env.prepare(c))); }
    fs::remove(c);  // gives scoped prunes something to delete
    Pipe start;
    std::vector<pid_t> kids;
    for (int i = 0; i < 4; ++i) {
        kids.push_back(spawn([&, i] {
            start.recv();
            for (int it = 0; it < 25; ++it) {
                PrepareOutcome o = env.prepare_retry((it + i) % 2 ? a : b);
                CHECK_MSG(ready(o), o.error.message);
                if (!ready(o)) return 1;
                // Under the lease the artifact cannot go away.
                std::this_thread::sleep_for(1ms);
                CHECK(cache::validate(o.artifact_dir) == cache::Validity::ok);
                CHECK(fs::exists(env.layout().memo_path(o.memo_id, o.key32)));
            }
            return 0;
        }));
    }
    for (int i = 0; i < 2; ++i) {
        kids.push_back(spawn([&, i] {
            start.recv();
            for (int it = 0; it < 25; ++it) {
                if ((it + i) % 2) {
                    cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
                } else {
                    cache::prune(env.cache, cache::PrunePolicy::scoped({c}), env.catalog());
                }
                std::this_thread::sleep_for(1ms);
            }
            return 0;
        }));
    }
    for (size_t i = 0; i < kids.size(); ++i) start.send('g');
    int failures = 0;
    for (pid_t k : kids) failures += reap(k) != 0;
    CHECK(failures == 0);
    CHECK(env.memos_consistent());  // no memo points at a deleted artifact
    CHECK(count_files(env.layout().staging) == 0);
}

void case_failure_records() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    // Compile failure: transient, attempts 1.
    g_compile_override = [] {
        cache::CompileResult r;
        r.error = Error::transient("compiler rejected the model", "com.apple.CoreML", 3);
        return r;
    };
    PrepareOutcome o = env.prepare(a);
    CHECK(o.kind == PrepareOutcome::Kind::failed && o.error.code == 3);
    const std::string key = o.key32;
    auto rec = failed::read(failed::record_path(l.failed, key));
    CHECK(rec && rec->cls == ErrorClass::transient && rec->attempts == 1 && rec->template_id == "toy-dac");
    CHECK(env.artifacts() == 0 && count_files(l.staging) == 0);
    // This process backs off: no second compile.
    o = env.prepare(a);
    CHECK(o.kind == PrepareOutcome::Kind::failed && g_compiles == 1);
    // The next process retries; the same (domain, code) on the same os_build promotes.
    pid_t kid = spawn([&] {
        PrepareOutcome k = env.prepare(a);
        CHECK(k.kind == PrepareOutcome::Kind::failed && g_compiles == 2);
        return 0;
    });
    CHECK(reap(kid) == 0);
    rec = failed::read(failed::record_path(l.failed, key));
    CHECK(rec && rec->cls == ErrorClass::deterministic && rec->attempts == 2);
    // Deterministic: never retried on this os_build, not even by another process ...
    kid = spawn([&] {
        PrepareOutcome k = env.prepare(a);
        CHECK(k.kind == PrepareOutcome::Kind::failed && k.error.cls == ErrorClass::deterministic && g_compiles == 1);
        return 0;
    });
    CHECK(reap(kid) == 0);
    // ... and never expires; a new os_build may try again.
    CHECK(failed::check(l.failed, key, "TEST1", now_s() + 400 * kDay).blocked);
    g_compile_override = nullptr;
    o = env.prepare(a, "TEST2");
    CHECK(ready(o) && g_compiles == 2);
}

// A success in between clears the record: promotion needs two consecutive
// failures, so fail (A) -> success (B) -> damage -> fail (C) stays transient
// and a healthy process D still retries.
void case_success_clears_failure() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    auto rejecting = [] {
        cache::CompileResult r;
        r.error = Error::transient("compiler rejected the model", "com.apple.CoreML", 3);
        return r;
    };
    g_compile_override = rejecting;
    PrepareOutcome o = env.prepare(a);  // A
    CHECK(o.kind == PrepareOutcome::Kind::failed);
    const std::string key = o.key32;
    const std::string rec_path = failed::record_path(l.failed, key);
    CHECK(fs::exists(rec_path));
    pid_t kid = spawn([&] {  // B
        g_compile_override = nullptr;
        CHECK(ready(env.prepare(a)));
        return 0;
    });
    CHECK(reap(kid) == 0);
    CHECK(!fs::exists(rec_path));
    fs::resize_file(l.artifact_dir("toy-dac", key) + "/model.mlmodelc/weights/weight.bin", 10);
    kid = spawn([&] {  // C
        g_compile_override = rejecting;
        CHECK(env.prepare(a).kind == PrepareOutcome::Kind::failed);
        return 0;
    });
    CHECK(reap(kid) == 0);
    auto rec = failed::read(rec_path);
    CHECK(rec && rec->cls == ErrorClass::transient && rec->attempts == 1);
    kid = spawn([&] {  // D
        g_compile_override = nullptr;
        PrepareOutcome k = env.prepare(a);
        CHECK(ready(k) && !k.cache_hit);
        return 0;
    });
    CHECK(reap(kid) == 0);
    CHECK(!fs::exists(rec_path));
    // The load lane clears a per-function record after a successful load.
    const std::string fn = failed::record_path(l.failed, key, "frames_256", "cpu_and_ane");
    failed::record(l.failed, key, failed::Stage::load, Error::transient("x", "com.apple.CoreML", 0), "toy-dac", "TEST1",
                   now_s(), "frames_256", "cpu_and_ane");
    CHECK(fs::exists(fn));
    failed::clear(l.failed, key, "frames_256", "cpu_and_ane");
    CHECK(!fs::exists(fn));
}

// A's compile fails; A stops right after its generation attempt released the
// key's EX. B generates successfully in that window. When A resumes, its
// failure must not land after B's success: no record survives, a later single
// failure (C) stays transient and a healthy process (D) generates.
void case_failure_publication_race() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    auto rejecting = [] {
        cache::CompileResult r;
        r.error = Error::transient("compiler rejected the model", "com.apple.CoreML", 3);
        return r;
    };
    Pipe unlocked, go;
    pid_t kid_a = spawn([&] {  // A
        g_compile_override = rejecting;
        cache::PrepareInputs in = env.inputs(a);
        in.after_key_unlock = [&] {
            unlocked.send('u');
            go.recv();
        };
        CHECK(cache::prepare(in).kind == PrepareOutcome::Kind::failed);
        return 0;
    });
    CHECK(unlocked.recv() == 'u');
    pid_t kid_b = spawn([&] {  // B, while A is between its unlock and its return
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o) && !o.cache_hit);
        return 0;
    });
    CHECK(reap(kid_b) == 0);
    go.send('g');
    CHECK(reap(kid_a) == 0);
    CHECK(env.artifacts() == 1);
    const std::string key =
        fs::directory_iterator(l.artifacts)->path().filename().string().substr(std::string("toy-dac.").size());
    const std::string rec_path = failed::record_path(l.failed, key);
    CHECK(!fs::exists(rec_path));  // B's success is the last word
    // C: one failure after the artifact is damaged stays transient.
    fs::resize_file(l.artifact_dir("toy-dac", key) + "/model.mlmodelc/weights/weight.bin", 10);
    pid_t kid_c = spawn([&] {
        g_compile_override = rejecting;
        CHECK(env.prepare(a).kind == PrepareOutcome::Kind::failed);
        return 0;
    });
    CHECK(reap(kid_c) == 0);
    auto rec = failed::read(rec_path);
    CHECK(rec && rec->cls == ErrorClass::transient && rec->attempts == 1);
    // D: a healthy process retries and generates.
    pid_t kid_d = spawn([&] {
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o) && !o.cache_hit && o.key32 == key);
        return 0;
    });
    CHECK(reap(kid_d) == 0);
    CHECK(!fs::exists(rec_path));
}

// A routine prune reads an expired failure record and decides to delete it.
// Before it acts, a preparation of the same key fails again and replaces the
// record. The prune must keep the replacement: it holds this process's
// backoff, or (after a second process failed the same way) the
// deterministic suppression. A key someone holds is skipped, not waited for.
void case_prune_vs_fresh_failure_record() {
    auto rejecting = [] {
        cache::CompileResult r;
        r.error = Error::transient("compiler rejected the model", "com.apple.CoreML", 3);
        return r;
    };
    // An expired transient record for the key of `a`: a compile failure eight days ago.
    auto expired_record = [&](Env & env, const std::string & a) {
        cache::PrepareInputs in = env.inputs(a);
        in.now_s = now_s() - 8 * kDay;
        PrepareOutcome o = cache::prepare(in);
        CHECK(o.kind == PrepareOutcome::Kind::failed);
        auto rec = failed::read(failed::record_path(env.layout().failed, o.key32));
        CHECK(rec && failed::expired(*rec, now_s()));
        return o.key32;
    };
    {
        // The replacement is this process's own failure: its backoff survives.
        Env env;
        const std::string a = env.source("a.gguf");
        g_compile_override = rejecting;
        const std::string key = expired_record(env, a);
        const std::string rec_path = failed::record_path(env.layout().failed, key);
        CHECK(g_compiles == 1);
        cache::PrunePolicy p = env.routine();
        int scans = 0;
        p.after_failed_scan = [&] {
            ++scans;
            CHECK(env.prepare(a).kind == PrepareOutcome::Kind::failed);  // the expired record does not block
        };
        cache::PruneReport r = cache::prune(env.cache, p, env.catalog());
        CHECK(scans == 1 && g_compiles == 2);
        CHECK_MSG(r.failed == 0 && fs::exists(rec_path), "the prune deleted the fresh record");
        auto rec = failed::read(rec_path);
        CHECK(rec && !failed::expired(*rec, now_s()) && rec->process == failed::process_token());
        PrepareOutcome o = env.prepare(a);
        CHECK(o.kind == PrepareOutcome::Kind::failed && g_compiles == 2);  // backing off: no third compile
    }
    {
        // Two other processes fail the same way meanwhile: the second promotes
        // the record, and the promotion survives the prune.
        Env env;
        const std::string a = env.source("a.gguf");
        g_compile_override = rejecting;
        const std::string key = expired_record(env, a);
        const cache::Layout l = env.layout();
        const std::string rec_path = failed::record_path(l.failed, key);
        cache::PrunePolicy p = env.routine();
        int scans = 0;
        p.after_failed_scan = [&] {
            ++scans;
            for (int i = 0; i < 2; ++i) {
                pid_t kid = spawn([&] {
                    CHECK(env.prepare(a).kind == PrepareOutcome::Kind::failed);
                    return 0;
                });
                CHECK(reap(kid) == 0);
            }
        };
        cache::PruneReport r = cache::prune(env.cache, p, env.catalog());
        CHECK(scans == 1);
        CHECK_MSG(r.failed == 0 && fs::exists(rec_path), "the prune deleted the promoted record");
        auto rec = failed::read(rec_path);
        CHECK(rec && rec->cls == ErrorClass::deterministic && rec->attempts == 2);
        const int compiles = g_compiles;
        PrepareOutcome o = env.prepare(a);
        CHECK(o.kind == PrepareOutcome::Kind::failed && o.error.cls == ErrorClass::deterministic &&
              g_compiles == compiles);

        // While the key is held, prune leaves its record alone and returns
        // at once; the next prune deletes it.
        cache::KeyLock held;
        CHECK(cache::KeyLock::take(l, key, cache::KeyLock::Mode::exclusive, held) == cache::KeyLock::Result::ok);
        r = cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
        CHECK_MSG(r.failed == 0 && fs::exists(rec_path), "the prune deleted a record of a held key");
        held.release();
        r = cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
        CHECK(r.failed == 1 && !fs::exists(rec_path));
    }
}

// The process token tells a retry in this process from another process's
// failure. Threads of a new process that ask for it at the same moment (or
// record failures for different keys at once) all see one complete token,
// and it is that process's own: a child after fork never keeps its parent's.
void case_process_token_first_use() {
    TempDir td;
    const std::string parent = failed::process_token();
    auto well_formed = [](const std::string & tok) {
        const std::string prefix = std::to_string(::getpid()) + "-";
        return tok.size() == prefix.size() + 16 && tok.compare(0, prefix.size(), prefix) == 0;
    };
    CHECK(well_formed(parent));
    constexpr int kRounds = 100;
    constexpr int kThreads = 8;
    int bad_rounds = 0;
    for (int round = 0; round < kRounds; ++round) {
        // Even rounds ask for the token; odd rounds record two compile
        // failures per key, which must stay transient (one process).
        const bool via_records = round % 2 != 0;
        const std::string dir = td / ("failed-" + std::to_string(round));
        pid_t kid = spawn([&] {
            std::atomic<int> waiting{0};
            std::atomic<bool> go{false};
            std::vector<std::string> got(kThreads);
            std::vector<std::thread> threads;
            const Error err = Error::transient("compiler rejected the model", "com.apple.CoreML", 3);
            auto key_of = [](int i) { return std::string(31, 'a') + static_cast<char>('0' + i); };
            for (int i = 0; i < kThreads; ++i) {
                threads.emplace_back([&, i] {
                    waiting.fetch_add(1);
                    while (!go.load()) {
                    }
                    if (via_records) {
                        for (int n = 0; n < 2; ++n) {
                            failed::record(dir, key_of(i), failed::Stage::compile, err, "toy-dac", "TEST1", now_s());
                        }
                    } else {
                        got[i] = failed::process_token();
                    }
                });
            }
            while (waiting.load() < kThreads) {
            }
            go.store(true);
            for (std::thread & t : threads) t.join();
            const std::string mine = failed::process_token();
            CHECK_MSG(well_formed(mine), "'" + mine + "'");
            CHECK(mine != parent);
            for (int i = 0; i < kThreads; ++i) {
                if (via_records) {
                    auto rec = failed::read(failed::record_path(dir, key_of(i)));
                    CHECK_MSG(rec && rec->process == mine && rec->attempts == 2 && rec->cls == ErrorClass::transient,
                              rec ? "'" + rec->process + "' " + to_string(rec->cls) : std::string("no record"));
                } else {
                    CHECK_MSG(got[i] == mine, "'" + got[i] + "' vs '" + mine + "'");
                }
            }
            return 0;
        });
        bad_rounds += reap(kid) != 0;
    }
    CHECK_MSG(bad_rounds == 0, std::to_string(bad_rounds) + " of " + std::to_string(kRounds) + " rounds");
    CHECK(failed::process_token() == parent);
}

// An I/O error around the compile (the CompileFn reports stage prepare: a
// missing package, a failed move or copy) stays transient however often other
// processes repeat it; a compiler rejection still promotes.
void case_compile_io_failure_stays_transient() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    const std::pair<std::string, int64_t> io_errors[] = {{"NSPOSIXErrorDomain", EACCES}, {"NSCocoaErrorDomain", 513}};
    for (const auto & [domain, code] : io_errors) {
        g_compile_override = [domain = domain, code = code] {
            cache::CompileResult r;
            r.error = Error::transient("move the compiled bundle: permission denied", domain, code);
            r.stage = failed::Stage::prepare;
            return r;
        };
        g_compiles = 0;
        PrepareOutcome o = env.prepare(a);
        CHECK(o.kind == PrepareOutcome::Kind::failed && o.error.code == code);
        const std::string rec_path = failed::record_path(l.failed, o.key32);
        auto rec = failed::read(rec_path);
        CHECK_MSG(rec && rec->cls == ErrorClass::transient && rec->attempts == 1, domain);
        // A second process hits the same error: still transient, attempts 2.
        pid_t kid = spawn([&] {
            PrepareOutcome k = env.prepare(a);
            CHECK(k.kind == PrepareOutcome::Kind::failed && k.error.cls == ErrorClass::transient && g_compiles == 2);
            return 0;
        });
        CHECK(reap(kid) == 0);
        rec = failed::read(rec_path);
        CHECK_MSG(rec && rec->cls == ErrorClass::transient && rec->attempts == 2 && rec->domain == domain &&
                      rec->code == code,
                  domain);
        // The next process retries; with the environment repaired it generates.
        kid = spawn([&] {
            g_compile_override = nullptr;
            PrepareOutcome k = env.prepare(a);
            CHECK(ready(k) && !k.cache_hit && g_compiles == 2);
            return 0;
        });
        CHECK(reap(kid) == 0);
        CHECK(!fs::exists(rec_path));
        cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
        CHECK(env.artifacts() == 0);
    }
    // A Core ML rejection (stage compile, the default) from two processes promotes.
    g_compile_override = [] {
        cache::CompileResult r;
        r.error = Error::transient("compiler rejected the model", "com.apple.CoreML", 3);
        return r;
    };
    PrepareOutcome o = env.prepare(a);
    CHECK(o.kind == PrepareOutcome::Kind::failed);
    pid_t kid = spawn([&] {
        CHECK(env.prepare(a).kind == PrepareOutcome::Kind::failed);
        return 0;
    });
    CHECK(reap(kid) == 0);
    auto rec = failed::read(failed::record_path(l.failed, o.key32));
    CHECK(rec && rec->cls == ErrorClass::deterministic && rec->attempts == 2);
    g_compile_override = nullptr;
}

void case_resource_codes_never_promote() {
    TempDir td;
    const std::string dir = td / "failed";
    const std::string key(32, '1');
    const Error oom = Error::transient("out of memory", "NSPOSIXErrorDomain", ENOMEM);
    CHECK(failed::is_resource("NSPOSIXErrorDomain", ENOMEM));
    CHECK(failed::is_resource("NSCocoaErrorDomain", 4097));  // XPC interrupted
    CHECK(failed::is_resource(failed::ERROR_DOMAIN, failed::TIMEOUT));
    CHECK(!failed::is_resource("com.apple.CoreML", 3));
    failed::record(dir, key, failed::Stage::compile, oom, "toy-dac", "TEST1", now_s());
    for (int i = 0; i < 3; ++i) {
        pid_t kid = spawn([&] {
            failed::record(dir, key, failed::Stage::compile, oom, "toy-dac", "TEST1", now_s());
            return 0;
        });
        CHECK(reap(kid) == 0);
    }
    auto rec = failed::read(failed::record_path(dir, key));
    CHECK(rec && rec->cls == ErrorClass::transient && rec->attempts == 4);
    // Another code resets the count; a different os_build does too.
    failed::record(dir, key, failed::Stage::compile, Error::transient("x", "com.apple.CoreML", 3), "toy-dac", "TEST1",
                   now_s());
    rec = failed::read(failed::record_path(dir, key));
    CHECK(rec && rec->attempts == 1 && rec->cls == ErrorClass::transient);
    // In-process backoff: 30 s doubling to 5 min; another process may retry.
    CHECK(failed::backoff_s(1) == 30 && failed::backoff_s(2) == 60 && failed::backoff_s(5) == 300 &&
          failed::backoff_s(9) == 300);
    const int64_t t0 = rec->last_at;
    CHECK(failed::check(dir, key, "TEST1", t0 + 10).blocked);
    CHECK(!failed::check(dir, key, "TEST1", t0 + 31).blocked);
    pid_t kid = spawn([&] {
        CHECK(!failed::check(dir, key, "TEST1", t0 + 10).blocked);
        return 0;
    });
    CHECK(reap(kid) == 0);
    // Transient records expire after 7 days.
    CHECK(!failed::expired(*rec, t0 + 7 * kDay) && failed::expired(*rec, t0 + 7 * kDay + 1));
}

void case_load_failure_per_function() {
    Env env;
    const std::string a = env.source("a.gguf");
    std::string key;
    {
        PrepareOutcome o = env.prepare(a);
        key = o.key32;
    }
    const cache::Layout l = env.layout();
    const Error bad = Error::transient("load failed", "com.apple.CoreML", 0);
    failed::record(l.failed, key, failed::Stage::load, bad, "toy-dac", "TEST1", now_s(), "frames_256", "cpu_and_ane");
    pid_t kid = spawn([&] {
        failed::record(l.failed, key, failed::Stage::load, bad, "toy-dac", "TEST1", now_s(), "frames_256",
                       "cpu_and_ane");
        return 0;
    });
    CHECK(reap(kid) == 0);
    const std::string p = failed::record_path(l.failed, key, "frames_256", "cpu_and_ane");
    CHECK(p != failed::record_path(l.failed, key) && p != failed::record_path(l.failed, key, "frames_512", "cpu_and_ane"));
    auto rec = failed::read(p);
    CHECK(rec && rec->cls == ErrorClass::deterministic && rec->function == "frames_256" && rec->units == "cpu_and_ane");
    CHECK(failed::check(l.failed, key, "TEST1", now_s(), "frames_256", "cpu_and_ane").blocked);
    // Siblings stay loadable: other widths, other units, and the artifact itself.
    CHECK(!failed::check(l.failed, key, "TEST1", now_s(), "frames_512", "cpu_and_ane").blocked);
    CHECK(!failed::check(l.failed, key, "TEST1", now_s(), "frames_256", "cpu_and_gpu").blocked);
    CHECK(!failed::check(l.failed, key, "TEST1", now_s()).blocked);
    PrepareOutcome o = env.prepare(a);
    CHECK(ready(o) && o.cache_hit);
}

void case_deterministic_prepare_failure() {
    Env env;
    const std::string a = env.td / "inf.gguf";
    auto t = tensors(1.0f);
    t[0].values[3] = 1e30f;  // overflows fp16: non-finite weight
    write_source_file(a, t, kMeta);
    PrepareOutcome o = env.prepare(a);
    CHECK(o.kind == PrepareOutcome::Kind::failed && o.error.cls == ErrorClass::deterministic);
    CHECK(o.error.message.find("non-finite") != std::string::npos);
    auto rec = failed::read(failed::record_path(env.layout().failed, o.key32));
    CHECK(rec && rec->cls == ErrorClass::deterministic);
    CHECK(g_compiles == 0 && env.artifacts() == 0 && count_files(env.layout().staging) == 0);
    const int reads = g_file_raw_reads;
    o = env.prepare(a);
    CHECK(o.kind == PrepareOutcome::Kind::failed && o.error.cls == ErrorClass::deterministic);
    CHECK(g_file_raw_reads - reads == 2);  // digested again (no memo), but not regenerated
    // Expired transient records and our failure records go with prune.
    cache::PrunePolicy p = cache::PrunePolicy::all();
    CHECK(cache::prune(env.cache, p, env.catalog()).failed == 1);
}

void case_quarantine() {
    Env env;
    const std::string a = env.source("a.gguf");
    std::string key, dir;
    {
        PrepareOutcome o = env.prepare(a);
        key = o.key32;
        dir = o.artifact_dir;
        // Leased: cannot be quarantined.
        CHECK(cache::quarantine(env.layout(), "toy-dac", key) == cache::KeyLock::Result::would_block);
    }
    // Damage the published artifact: the inventory check fails on the next hit.
    fs::resize_file(dir + "/model.mlmodelc/weights/weight.bin", 10);
    std::string why;
    CHECK(cache::validate(dir, &why) == cache::Validity::invalid && why.find("weight.bin") != std::string::npos);
    PrepareOutcome o = env.prepare(a);
    CHECK(ready(o) && !o.cache_hit && o.key32 == key);  // regenerated at the same key
    CHECK(g_compiles == 2);
    CHECK(cache::validate(dir) == cache::Validity::ok);
    CHECK(count_files(env.layout().quarantine) == 1);
    o.lease.release();
    // An artifact.json that is incomplete is also unusable.
    auto info = cache::read_artifact_json(dir);
    info->complete = false;
    write_bytes(dir + "/artifact.json", cache::to_json(*info));
    CHECK(cache::validate(dir) == cache::Validity::invalid);
    CHECK(cache::quarantine(env.layout(), "toy-dac", key) == cache::KeyLock::Result::ok);
    CHECK(!fs::exists(dir) && count_files(env.layout().quarantine) == 2);
    cache::PruneReport r = cache::prune(env.cache, env.routine(), env.catalog());
    CHECK(r.quarantine == 2 && count_files(env.layout().quarantine) == 0);
}

// A prune killed between moving an artifact to its trash and removing it: the
// next routine (or `all`) prune removes the trash; another library's stays.
void case_prune_trash() {
    Env env;
    const cache::Layout l = env.layout();
    const std::string k(32, 'c');
    auto leftover = [&](const std::string & tid) {
        const std::string t = cache::prune_trash_path(l, tid + "." + k);
        CHECK(fs::path(t).parent_path() == fs::path(l.quarantine));
        write_bytes(t + "/model.mlmodelc/weights/weight.bin", "partly deleted");
        return t;
    };
    cache::PrunePolicy routine = env.routine();
    for (const cache::PrunePolicy & p : {routine, cache::PrunePolicy::all()}) {
        const std::string mine = leftover("toy-dac");
        const std::string theirs = leftover("other-tpl");
        cache::PruneReport r = cache::prune(env.cache, p, env.catalog());
        CHECK(r.quarantine == 1 && !fs::exists(mine) && fs::exists(theirs));
        fs::remove_all(theirs);
    }
    // A real prune leaves no trash behind.
    const std::string a = env.source("a.gguf");
    CHECK(ready(env.prepare(a)));
    cache::PruneReport r = cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
    CHECK(r.deleted_artifacts.size() == 1 && count_files(l.quarantine) == 0);
}

// Syntactically valid JSON with wrong-typed fields (another addon's file in a
// shared cache, or damage): every reader treats it as unreadable, nothing
// throws, an unreadable artifact is repaired and prune runs to completion.
void case_malformed_records() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    std::string key, dir, memo_file;
    {
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o));
        key = o.key32;
        dir = o.artifact_dir;
        memo_file = l.memo_path(o.memo_id, key);
    }
    CHECK(!cache::memo_from_json(R"({"schema":"1"})"));
    CHECK(!cache::memo_from_json(R"({"schema":null,"path":"/x"})"));
    for (const char * bad : {R"({"schema":null})", R"({"schema":"1"})", R"({"schema":1,"complete":"yes"})",
                             R"({"schema":1,"complete":true,"artifact_key":"k",)"
                             R"("template":{"id":"toy-dac","sha256":"s"},"generator_version":1,"inventory":7})"}) {
        write_bytes(dir + "/artifact.json", bad);
        CHECK(!cache::read_artifact_json(dir));
        CHECK(cache::validate(dir) == cache::Validity::invalid);
    }
    write_bytes(memo_file, R"({"schema":"1"})");
    const std::string rec = failed::record_path(l.failed, key);
    fs::create_directories(l.failed);
    for (const char * bad : {R"({"schema":1,"class":"transient","attempts":"two"})", R"({"schema":[1]})",
                             R"({"schema":1,"class":7})", R"({"schema":1,"class":"deterministic","code":"3"})"}) {
        write_bytes(rec, bad);
        CHECK(!failed::read(rec));
        CHECK(!failed::check(l.failed, key, "TEST1", now_s()).blocked);
    }
    // prepare: the unreadable memo is a miss, the record is no record, the
    // unreadable artifact is quarantined and regenerated at the same key.
    PrepareOutcome o = env.prepare(a);
    CHECK(ready(o) && !o.cache_hit && o.key32 == key && g_compiles == 2);
    CHECK(count_files(l.quarantine) == 1 && cache::validate(dir) == cache::Validity::ok);
    o.lease.release();
    // Prune in every mode with wrong-typed memos, artifact.json and records.
    const std::string other = std::string(32, 'c');
    write_bytes(l.memo_path(std::string(32, 'd'), other), R"({"schema":null})");
    write_bytes(l.memo_path(std::string(32, 'e'), key), R"({"schema":"1","template_id":3})");
    write_bytes(failed::record_path(l.failed, other), R"({"schema":1,"class":"transient","attempts":"two"})");
    fs::create_directories(l.artifact_dir("toy-dac", other));
    write_bytes(l.artifact_dir("toy-dac", other) + "/artifact.json", R"({"schema":null})");
    cache::PrunePolicy routine = env.routine();
    routine.now_s = now_s() + 400 * kDay;
    cache::prune(env.cache, routine, env.catalog());
    cache::prune(env.cache, cache::PrunePolicy::scoped({a}), env.catalog());
    cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog());
    CHECK(env.artifacts() == 0);
}

void case_toctou() {
    Env env;
    const std::string a = env.source("a.gguf");
    // (a) The file changed after the engine loaded it.
    cache::PrepareInputs in = env.inputs(a);
    std::this_thread::sleep_for(20ms);
    write_source_file(a, tensors(1.0f), kMeta);  // same bytes, new mtime
    PrepareOutcome o = cache::prepare(in);
    CHECK(o.kind == PrepareOutcome::Kind::failed && o.reason == "source_changed" &&
          o.error.cls == ErrorClass::transient);
    CHECK(env.artifacts() == 0 && g_compiles == 0);
    // (b) The bytes change between the digest and the write.
    // Same size, inode and (restored) mtime: only the write-time digest
    // recheck can tell, not the final fstat.
    in = env.inputs(a);
    int reads = 0;
    g_before_read = [&](const std::string &) {
        if (++reads != 3) return;  // the first read of the write phase
        write_source_file(a, tensors(5.0f), kMeta);
        struct timespec ts[2];
        ts[0].tv_sec = ts[1].tv_sec = static_cast<time_t>(in.expected.mtime_ns / 1000000000LL);
        ts[0].tv_nsec = ts[1].tv_nsec = static_cast<long>(in.expected.mtime_ns % 1000000000LL);
        ::utimensat(AT_FDCWD, a.c_str(), ts, 0);
        SourceStat now;
        CHECK(stat_path(a, now) && same_identity(now, in.expected));
    };
    o = cache::prepare(in);
    g_before_read = nullptr;
    CHECK(o.kind == PrepareOutcome::Kind::failed && o.reason == "source_changed");
    CHECK(o.error.code == failed::SOURCE_CHANGED && o.error.cls == ErrorClass::transient);
    CHECK(env.artifacts() == 0 && g_compiles == 0);  // nothing published under the stale key
    CHECK(count_files(env.layout().staging) == 0);
    auto rec = failed::read(failed::record_path(env.layout().failed, o.key32));
    CHECK(rec && rec->cls == ErrorClass::transient && rec->code == failed::SOURCE_CHANGED);
    // The next load (new stat tuple, new bytes) generates normally.
    o = env.prepare(a);
    CHECK(ready(o) && !o.cache_hit);
}

void case_loaded_source() {
    Env env;
    const std::string a = env.source("a.gguf");
    const std::string b = env.source("b.gguf", 2.0f);
    SourceStat before, after, other;
    CHECK(generation::source_stat(a, before) && generation::source_stat(a, after) &&
          generation::source_stat(b, other));
    // The file the engine loaded, unchanged across its load, asked for by its own path.
    CHECK(!generation::loaded_source(before, true, after, true, a).changed);
    // Replaced during the load, a stat missing, or generation asked for another file.
    CHECK(generation::loaded_source(before, true, other, true, b).changed);
    CHECK(generation::loaded_source(before, false, after, true, a).changed);
    CHECK(generation::loaded_source(before, true, after, false, a).changed);
    CHECK(generation::loaded_source(before, true, after, true, b).changed);
    // prepare fails a changed source with source_changed and generates nothing,
    // even when the file at its path still has the recorded identity.
    cache::PrepareInputs in = env.inputs(a);
    in.expected.changed = true;
    PrepareOutcome o = cache::prepare(in);
    CHECK(o.kind == PrepareOutcome::Kind::failed && o.reason == "source_changed");
    CHECK(o.error.code == failed::SOURCE_CHANGED && o.error.cls == ErrorClass::transient);
    CHECK(env.artifacts() == 0 && env.memos() == 0 && g_compiles == 0);
    // Identity from the descriptor the engine read through: a file swapped in
    // at the path after the open (and restored later) does not change it, and
    // prepare refuses the path while it holds the other file.
    {
        const int fd = ::open(a.c_str(), O_RDONLY | O_CLOEXEC);
        CHECK(fd >= 0);
        const std::string held = a + ".held";
        CHECK(::rename(a.c_str(), held.c_str()) == 0 && ::rename(b.c_str(), a.c_str()) == 0);  // B at the path
        SourceStat by_fd;
        CHECK(generation::source_stat_fd(fd, a, by_fd) && same_identity(by_fd, before) && !by_fd.changed);
        cache::PrepareInputs swapped = env.inputs(a);
        swapped.expected = by_fd;
        swapped.expected.path = a;
        PrepareOutcome so = cache::prepare(swapped);
        CHECK(so.kind == PrepareOutcome::Kind::failed && so.reason == "source_changed");
        CHECK(::rename(a.c_str(), b.c_str()) == 0 && ::rename(held.c_str(), a.c_str()) == 0);  // A back
        SourceStat again;
        CHECK(generation::source_stat_fd(fd, a, again) && same_identity(again, before));
        swapped.expected = again;
        swapped.expected.path = a;
        CHECK(ready(cache::prepare(swapped)));
        ::close(fd);
        CHECK(!generation::source_stat_fd(-1, a, again));
    }
    // Its request never joins another engine's job for the same file.
    generation::Candidate c;
    c.template_id = env.tpl->id();
    c.sha256_hex = env.tpl->sha256_hex();
    Resolved r;
    r.cache_dir = env.cache;
    CHECK(!generation::prepare_request(c, after, r).key.empty());
    CHECK(generation::prepare_request(c, generation::loaded_source(before, true, other, true, b), r).key.empty());
}

void case_preflight() {
    Env env;
    const std::string a = env.source("a.gguf");
    cache::PrepareInputs in = env.inputs(a);
    in.fs_probe = [](const std::string &, cache::FsFacts & f) {
        f.local = false;
        f.fstype = "nfs";
        f.free_bytes = 1ull << 40;
        return true;
    };
    PrepareOutcome o = cache::prepare(in);
    CHECK(o.kind == PrepareOutcome::Kind::unsupported && o.reason == "cache_fs");
    cache::FsFacts real;
    CHECK(cache::default_fs_probe(env.cache, real) && cache::fs_supported(real));
    in.fs_probe = [real](const std::string &, cache::FsFacts & f) {
        f = real;
        f.free_bytes = 100;  // below 2 x weight_bin_size
        return true;
    };
    o = cache::prepare(in);
    CHECK(o.kind == PrepareOutcome::Kind::failed && o.reason == "no_space" && o.error.cls == ErrorClass::transient);
    CHECK(g_compiles == 0);
    // Recorded under EX (ordered against a success), as a resource code.
    auto rec = failed::read(failed::record_path(env.layout().failed, o.key32));
    CHECK(rec && rec->code == failed::NO_SPACE && rec->cls == ErrorClass::transient);
}

// ---- EWOULDBLOCK re-queue through the scheduler ----------------------------------------------

cache::PrepareInputs g_sched_inputs;

PrepareResult sched_prepare(const PrepareRequest &, PrepareContext & ctx) {
    PrepareOutcome o = cache::prepare(g_sched_inputs, &ctx);
    PrepareResult r;
    r.would_block = o.kind == PrepareOutcome::Kind::would_block;
    r.cache_hit = o.cache_hit;
    r.artifact_key = o.key32;
    r.artifact_path = o.model_path;
    if (o.kind == PrepareOutcome::Kind::failed || o.kind == PrepareOutcome::Kind::unsupported) r.error = o.error;
    return r;
}

LoadResult sched_load(const LoadRequest &) {
    LoadResult r;
    r.model = std::make_shared<int>(1);
    return r;
}

bool wait_state(const std::shared_ptr<Slot> & s, SlotState want, int ms = 20000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (s->state() == want) return true;
        std::this_thread::sleep_for(5ms);
    }
    return false;
}

void case_would_block_requeue() {
    Env env;
    const std::string a = env.source("a.gguf");
    // Cache level: a key lock held by another process is would_block, never a wait.
    MemSource ms(tensors(1.0f), kMeta);
    digest::SourceDigest dg;
    CHECK(digest::source_digest(env.tpl->recipe(), ms, dg).ok());
    const std::string key = digest::artifact_key(GENERATOR_VERSION, env.tpl->sha256(), dg.digest, "out=fp16");
    Pipe held, go;
    // Another process generating this key: it holds LOCK_EX from before the
    // write until after the publish (its compile is held open here).
    pid_t kid = spawn([&] {
        g_compile_gate = [&] {
            held.send('h');
            go.recv();
        };
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o) && !o.cache_hit);
        return 0;
    });
    CHECK(held.recv() == 'h');
    const auto t0 = std::chrono::steady_clock::now();
    PrepareOutcome o = env.prepare(a);
    CHECK(o.kind == PrepareOutcome::Kind::would_block);
    CHECK(std::chrono::steady_clock::now() - t0 < 5s);

    // Scheduler level: the job releases the gate, re-queues with backoff, and
    // other jobs run; once the other process publishes, the retry adopts.
    g_sched_inputs = env.inputs(a);
    Config cfg;
    cfg.backoff_initial = 20ms;
    cfg.backoff_max = 100ms;
    cfg.pin_image = false;
    cfg.available_memory = [] { return uint64_t{1} << 40; };
    auto mgr = Manager::create(cfg);
    SlotSpec gen;
    gen.engine = "toy";
    gen.stage = "dac";
    gen.units = ComputeUnits::cpu_only;
    gen.route = Route::generate;
    gen.prepare.key = "toy-dac:" + a;
    gen.prepare.template_id = "toy-dac";
    gen.prepare_fn = sched_prepare;
    gen.load_fn = sched_load;
    SlotSpec other = gen;
    other.stage = "other";
    other.route = Route::prebuilt;
    other.artifact_path = env.td / "prebuilt.mlmodelc";
    other.prepare_fn = nullptr;
    SlotPolicy pol;
    auto s1 = mgr->make_slot(gen, pol);
    auto s2 = mgr->make_slot(other, pol);
    s1->request(Priority::now);
    s2->request(Priority::minimal);  // lower priority: runs only because s1 yields the gate
    CHECK(wait_state(s2, SlotState::ready));
    CHECK(s1->state() == SlotState::queued);
    CHECK(mgr->stats().requeues >= 1);
    go.send('g');
    CHECK(reap(kid) == 0);
    CHECK(wait_state(s1, SlotState::ready));
    const SlotStatus st = s1->status();
    CHECK(st.route == StatusRoute::cached && st.artifact_key == key);
    CHECK(g_compiles == 0);  // this process never generated
}

// The hint every normal load writes (Runtime::record_hint, never staled) names
// <cache>/artifacts/<id>.<key>/model.mlmodelc, so prune(all) owns and removes it.
void case_scheduler_hint_pruned() {
    Env env;
    const std::string a = env.source("a.gguf");
    const cache::Layout l = env.layout();
    g_sched_inputs = env.inputs(a);
    std::string hint_path, key;
    {
        Config cfg;
        cfg.pin_image = false;
        cfg.available_memory = [] { return uint64_t{1} << 40; };
        auto mgr = Manager::create(cfg);
        SlotSpec gen;
        gen.engine = "toy";
        gen.stage = "dac";
        gen.units = ComputeUnits::cpu_only;
        gen.route = Route::generate;
        gen.prepare.key = "toy-dac:" + a;
        gen.prepare.template_id = "toy-dac";
        gen.prepare_fn = sched_prepare;
        gen.load_fn = sched_load;
        SlotPolicy pol;
        pol.cache_dir = env.cache;
        auto s = mgr->make_slot(gen, pol);
        CHECK(!fs::exists(l.hints));  // the first load starts with no hint file
        s->request(Priority::now);
        CHECK(wait_state(s, SlotState::ready));
        key = s->status().artifact_key;
        hint_path = s->hint_path();
    }
    CHECK(key.size() == 32 && !hint_path.empty());
    CHECK(fs::path(hint_path).parent_path() == fs::path(l.hints));
    const auto rec = detail::read_hint(hint_path);
    CHECK(rec && rec->cold_load_ms.has_value() && !rec->stale_until_s.has_value());
    CHECK(rec && rec->artifact == fs::canonical(l.artifact_dir("toy-dac", key) + "/model.mlmodelc").string());
    CHECK(cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog()).hints == 1);
    CHECK(!fs::exists(hint_path));
}

#if defined(QVAC_COREML_HAVE_GGML)
// ---- the generation glue through the scheduler ---------------------------------------------

std::function<void()> g_before_glue_load;

PrepareResult glue_prepare(const PrepareRequest &, PrepareContext & ctx) {
    PrepareOutcome o = cache::prepare(g_sched_inputs, &ctx);
    return generation::internal::result_of(o);
}

// Loads like an engine's LoadFn: under the lease the glue's begin_load takes.
LoadResult glue_load(const LoadRequest & req) {
    if (g_before_glue_load) g_before_glue_load();
    LoadResult r;
    if (!generation::begin_load(req, r)) return r;
    r.model = std::make_shared<int>(1);
    generation::end_load(req, r);
    return r;
}

// The engine rejects the first `g_glue_rejects` loads: the framework refused
// the model (LoadResult::rejected), with an error of class
// `g_glue_reject_class` (a framework error is transient for the engine).
std::atomic<int> g_glue_rejects{0};
std::atomic<int> g_glue_loads{0};
ErrorClass g_glue_reject_class = ErrorClass::transient;

LoadResult glue_load_rejecting(const LoadRequest & req) {
    LoadResult r;
    if (!generation::begin_load(req, r)) return r;
    if (g_glue_loads++ < g_glue_rejects) {
        r.error = Error{g_glue_reject_class, "com.apple.CoreML", 0, "the engine rejected the model"};
        r.rejected = true;
    } else {
        r.model = std::make_shared<int>(1);
    }
    generation::end_load(req, r);
    return r;
}

std::shared_ptr<Slot> glue_slot(const std::shared_ptr<Manager> & mgr, const Env & env, const std::string & a,
                                LoadFn load = glue_load) {
    SlotSpec gen;
    gen.engine = "toy";
    gen.stage = "dac";
    gen.units = ComputeUnits::cpu_only;
    gen.route = Route::generate;
    gen.tmpl = env.tpl.get();
    gen.prepare.key = "toy-dac:" + a;
    gen.prepare.template_id = "toy-dac";
    gen.prepare_fn = glue_prepare;
    gen.load_fn = load;
    SlotPolicy pol;
    pol.cache_dir = env.cache;
    return mgr->make_slot(gen, pol);
}

std::shared_ptr<Manager> glue_manager() {
    Config cfg;
    cfg.backoff_initial = 20ms;
    cfg.backoff_max = 100ms;
    cfg.pin_image = false;
    cfg.available_memory = [] { return uint64_t{1} << 40; };
    return Manager::create(cfg);
}

// A prune between the artifact's publication and its load (the preparation
// released its lease) deletes it; the load reports it gone and the slot goes
// back to preparation, which regenerates it, instead of failing.
void case_generated_artifact_pruned_before_load() {
    Env env;
    const std::string a = env.source("a.gguf");
    g_sched_inputs = env.inputs(a);
    int pruned = 0;
    g_before_glue_load = [&] {
        if (pruned++ == 0) {
            CHECK(cache::prune(env.cache, cache::PrunePolicy::all(), env.catalog()).deleted_artifacts.size() == 1);
        }
    };
    {
        auto mgr = glue_manager();
        auto s = glue_slot(mgr, env, a);
        s->request(Priority::now);
        CHECK_MSG(wait_state(s, SlotState::ready), s->status().error);
        CHECK(g_compiles == 2);  // generated, pruned, generated again
        CHECK(env.artifacts() == 1 && env.memos() == 1);
    }
    g_before_glue_load = nullptr;

    // An artifact that went incomplete is likewise prepared again.
    const std::string b = env.source("b.gguf", 2.0f);
    PrepareOutcome o = env.prepare(b);
    CHECK(ready(o));
    o.lease.release();
    LoadRequest req;
    req.tmpl = env.tpl.get();
    req.artifact_path = o.model_path;
    fs::remove_all(o.model_path);
    LoadResult r;
    CHECK(!generation::begin_load(req, r));
    CHECK(r.artifact_gone && r.lease_fd < 0 && !r.error.ok());
}

// One load of `req` through the glue that the framework refuses with
// (cls, domain, code): what a LoadFn does with a rejected generated artifact.
LoadResult reject_once(const LoadRequest & req, ErrorClass cls, const std::string & domain, int64_t code) {
    LoadResult r;
    if (!generation::begin_load(req, r)) return r;
    r.error = Error{cls, domain, code, "the framework refused the model"};
    r.rejected = true;
    generation::end_load(req, r);
    return r;
}

// A cached artifact the framework refuses (it passed the inventory check) is
// quarantined under EX and regenerated at the same key, once; the error class
// the engine reported stays on the slot. The (function, units) load record
// counts the failures: the same error in another process promotes it, and
// begin_load then refuses to load. A resource error never quarantines and
// never promotes.
void case_generated_rejected_hit_quarantined() {
    Env env;
    const std::string a = env.source("a.gguf");
    g_sched_inputs = env.inputs(a);
    { CHECK(ready(env.prepare(a))); }  // a cache hit for the slot below
    CHECK(g_compiles == 1);
    const cache::Layout l = env.layout();
    {
        // A framework rejection classed transient (as the engine reports it).
        g_glue_rejects = 1;
        g_glue_loads = 0;
        auto mgr = glue_manager();
        auto s = glue_slot(mgr, env, a, &glue_load_rejecting);
        s->request(Priority::now);
        CHECK_MSG(wait_state(s, SlotState::ready), s->status().error);
        CHECK(g_compiles == 2 && g_glue_loads == 2);
        CHECK(count_files(l.quarantine) == 1);  // the rejected copy, for prune
        CHECK(env.artifacts() == 1 && s->status().route == StatusRoute::generated);
    }
    std::string key;
    {
        g_glue_rejects = 100;
        g_glue_loads = 0;
        auto mgr = glue_manager();
        auto s = glue_slot(mgr, env, a, &glue_load_rejecting);
        s->request(Priority::now);
        CHECK(wait_state(s, SlotState::failed));
        CHECK(g_glue_loads == 2 && g_compiles == 3);  // one regeneration, then failed
        const SlotStatus st = s->status();
        CHECK(st.error == "the engine rejected the model" && st.error_class == ErrorClass::transient);
    }
    g_glue_rejects = 0;
    // The load record: two failures in this process, still transient.
    PrepareOutcome o = env.prepare(a);
    CHECK(ready(o));
    o.lease.release();
    key = o.key32;
    LoadRequest req;
    req.tmpl = env.tpl.get();
    req.artifact_path = o.model_path;
    req.units = ComputeUnits::cpu_and_ane;  // the slots above loaded with cpu_only
    CHECK(failed::read(failed::record_path(l.failed, key, req.function, "cpu_only")));
    const std::string rec_path = failed::record_path(l.failed, key, req.function, to_string(req.units));
    CHECK(!failed::read(rec_path));  // another (function, units) than the slots above
    reject_once(req, ErrorClass::transient, "com.apple.CoreML", 0);
    auto rec = failed::read(rec_path);
    CHECK(rec && rec->cls == ErrorClass::transient && rec->attempts == 1);
    // The same rejection in another process promotes the record.
    { CHECK(ready(env.prepare(a))); }
    pid_t kid = spawn([&] {
        reject_once(req, ErrorClass::transient, "com.apple.CoreML", 0);
        return 0;
    });
    CHECK(reap(kid) == 0);
    rec = failed::read(rec_path);
    CHECK(rec && rec->cls == ErrorClass::deterministic && rec->attempts == 2);
    // Then nothing loads it: begin_load refuses with the record's class.
    { CHECK(ready(env.prepare(a))); }
    LoadResult blocked;
    CHECK(!generation::begin_load(req, blocked) && blocked.error.cls == ErrorClass::deterministic &&
          blocked.lease_fd < 0 && !blocked.artifact_gone);
    // A resource error, in two processes: never quarantined, never promoted.
    const size_t quarantined = count_files(l.quarantine);
    req.units = ComputeUnits::cpu_and_gpu;
    const std::string res_path = failed::record_path(l.failed, key, req.function, to_string(req.units));
    LoadResult first = reject_once(req, ErrorClass::transient, "NSPOSIXErrorDomain", ENOMEM);
    kid = spawn([&] {
        reject_once(req, ErrorClass::transient, "NSPOSIXErrorDomain", ENOMEM);
        return 0;
    });
    CHECK(reap(kid) == 0);
    rec = failed::read(res_path);
    CHECK(!first.artifact_gone && count_files(l.quarantine) == quarantined && env.artifacts() == 1);
    CHECK(rec && rec->cls == ErrorClass::transient && rec->attempts == 2);
}

// Quarantine under contention. While another holder keeps a shared lease the
// quarantine of a rejected artifact cannot run: the load is requeued, not
// failed, and once the lease is gone the next rejection quarantines it. If
// another process repaired the entry after this load released its lease, the
// repair (a new publication) is not quarantined: the load is requeued and
// loads it.
std::function<void(LoadResult &, int)> g_contention;  // runs inside the load, after begin_load
std::atomic<int> g_contention_rejects{0};

LoadResult glue_load_contention(const LoadRequest & req) {
    LoadResult r;
    if (!generation::begin_load(req, r)) return r;
    const int n = ++g_glue_loads;
    if (g_contention) g_contention(r, n);
    if (n <= g_contention_rejects) {
        r.error = Error{ErrorClass::transient, "com.apple.CoreML", 0, "the engine rejected the model"};
        r.rejected = true;
    } else {
        r.model = std::make_shared<int>(1);
    }
    generation::end_load(req, r);
    return r;
}

void case_generated_rejected_quarantine_contention() {
    {
        Env env;
        const std::string a = env.source("a.gguf");
        g_sched_inputs = env.inputs(a);
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o));
        const std::string key = o.key32;
        o.lease.release();
        const cache::Layout l = env.layout();
        cache::KeyLock other;  // another holder's lease
        Error le;
        CHECK(cache::KeyLock::take(l, key, cache::KeyLock::Mode::shared, other, &le) == cache::KeyLock::Result::ok);
        g_glue_loads = 0;
        g_contention_rejects = 2;
        g_contention = [&](LoadResult &, int n) {
            if (n == 2) other.release();  // the other holder is done before the second load's end
        };
        auto mgr = glue_manager();
        auto s = glue_slot(mgr, env, a, &glue_load_contention);
        s->request(Priority::now);
        CHECK_MSG(wait_state(s, SlotState::ready), s->status().error);
        CHECK(g_glue_loads == 3);                // busy -> requeued, rejected -> quarantined, regenerated -> loaded
        CHECK(count_files(l.quarantine) == 1 && g_compiles == 2);
        g_contention = nullptr;
    }
    {
        Env env;
        const std::string a = env.source("a.gguf");
        g_sched_inputs = env.inputs(a);
        PrepareOutcome o = env.prepare(a);
        CHECK(ready(o));
        const std::string key = o.key32;
        const std::string tid = env.tpl->id();
        o.lease.release();
        const cache::Layout l = env.layout();
        g_glue_loads = 0;
        g_contention_rejects = 1;
        g_contention = [&](LoadResult & r, int n) {
            if (n != 1) return;
            // This load is done with its lease; another process repairs the
            // entry before this one takes the exclusive lock.
            ::close(r.lease_fd);
            r.lease_fd = -1;
            CHECK(cache::quarantine(l, tid, key, nullptr) == cache::KeyLock::Result::ok);
            CHECK(ready(env.prepare(a)));
        };
        auto mgr = glue_manager();
        auto s = glue_slot(mgr, env, a, &glue_load_contention);
        s->request(Priority::now);
        CHECK_MSG(wait_state(s, SlotState::ready), s->status().error);
        CHECK(g_glue_loads == 2);                                // requeued, then loads the repair
        CHECK(count_files(l.quarantine) == 1 && g_compiles == 2);  // the repair is not quarantined
        g_contention = nullptr;
    }
}

// An unsupported cache volume: the slot is unsupported(cache_fs), not failed.
void case_generated_unsupported_cache_fs() {
    Env env;
    const std::string a = env.source("a.gguf");
    g_sched_inputs = env.inputs(a);
    g_sched_inputs.fs_probe = [](const std::string &, cache::FsFacts & f) {
        f.local = false;
        f.fstype = "nfs";
        f.free_bytes = 1ull << 40;
        return true;
    };
    auto mgr = glue_manager();
    auto s = glue_slot(mgr, env, a);
    s->request(Priority::now);
    CHECK(wait_state(s, SlotState::unsupported));
    const SlotStatus st = s->status();
    CHECK(st.reason == "cache_fs" && st.error_class == ErrorClass::none);
    CHECK(state_label(st.state, st.reason) == "unsupported(cache_fs)");
    CHECK(g_compiles == 0);
}
#endif

}  // namespace

int main(int argc, char ** argv) {
    ::signal(SIGPIPE, SIG_IGN);
    return run_cases(argc, argv,
                     {
                         {"golden-keys", case_golden_keys},
                         {"generate-and-artifact-json", case_generate_and_artifact_json},
                         {"patch-pass-peak", case_patch_pass_peak},
                         {"memo-hot-path", case_memo_hot_path},
                         {"publish-race", case_publish_race},
                         {"recheck-under-lock", case_recheck_under_lock},
                         {"hit-adoption", case_hit_adoption},
                         {"hit-adoption-memo-io-failure", case_hit_adoption_memo_io_failure},
                         {"memo-hit-recheck", case_memo_hit_recheck},
                         {"recheck-memo-io-failure", case_recheck_memo_io_failure},
                         {"relocation", case_relocation},
                         {"scope", case_scope},
                         {"tmp-outputs", case_tmp_outputs},
                         {"package-stem-per-invocation", case_package_stem_per_invocation},
                         {"used-age", case_used_age},
                         {"superseded-and-retired", case_superseded_and_retired},
                         {"dead-memos-of-kept-artifacts", case_dead_memos_of_kept_artifacts},
                         {"unknown-ids", case_unknown_ids},
                         {"staging-ownership", case_staging_ownership},
                         {"hint-ownership", case_hint_ownership},
                         {"prune-vs-leases", case_prune_vs_leases},
                         {"adoption-vs-prune-race", case_adoption_vs_prune_race},
                         {"failure-records", case_failure_records},
                         {"success-clears-failure", case_success_clears_failure},
                         {"failure-publication-race", case_failure_publication_race},
                         {"compile-io-failure-stays-transient", case_compile_io_failure_stays_transient},
                         {"prune-vs-fresh-failure-record", case_prune_vs_fresh_failure_record},
                         {"process-token-first-use", case_process_token_first_use},
                         {"resource-codes-never-promote", case_resource_codes_never_promote},
                         {"load-failure-per-function", case_load_failure_per_function},
                         {"deterministic-prepare-failure", case_deterministic_prepare_failure},
                         {"quarantine", case_quarantine},
                         {"prune-trash", case_prune_trash},
                         {"malformed-records", case_malformed_records},
                         {"toctou", case_toctou},
                         {"loaded-source", case_loaded_source},
                         {"preflight", case_preflight},
                         {"would-block-requeue", case_would_block_requeue},
                         {"scheduler-hint-pruned", case_scheduler_hint_pruned},
#if defined(QVAC_COREML_HAVE_GGML)
                         {"generated-artifact-pruned-before-load", case_generated_artifact_pruned_before_load},
                         {"generated-unsupported-cache-fs", case_generated_unsupported_cache_fs},
                         {"generated-rejected-hit-quarantined", case_generated_rejected_hit_quarantined},
                         {"generated-rejected-quarantine-contention", case_generated_rejected_quarantine_contention},
#endif
                     },
                     "test-coreml-cache");
}
