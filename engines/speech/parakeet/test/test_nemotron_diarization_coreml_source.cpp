// test-nemotron-diarization-coreml-source: a generated diarizer is prepared
// only from the file whose weights the model loaded. The model path names A
// before and after the load, but B (another file with the same contents and
// shapes) at the moment the GGUF is opened, so the model holds B's weights.
// Two stats of the path around the load both see A; the identity of the one
// descriptor the weights were read through is B's. The diarizers must not
// generate from A: the prepare job fails with source_changed and the cache
// holds no artifact. Loaded again with nothing swapped, the minimal capacity
// is generated and ready. Then that cached bundle's coremldata.bin is
// overwritten with garbage of the same size (it still passes the inventory
// check; a garbage model.mil crashes Core ML itself, so it is not used): the
// framework refuses it, the bundle is quarantined and generated again, and
// the capacity is ready on the generated route. Last, a model whose GGUF is
// written in place while its weights are read (same inode, one tensor byte
// changed) loads while another model of the same file is still compiling its
// capacity: its identity is marked changed after the last read, so it never
// joins that job (whose bundle comes from the bytes before the write) and
// fails source_changed.
//
// Usage: test-nemotron-diarization-coreml-source <q8_0 GGUF> <work dir> [template id]
// The GGUF is Nemotron 3 Diarization or Sortformer v2.1 (both have diarizer
// templates); `template id` is its minimal capacity's template
// (nemotron-diar-f358 by default, sortformer-diar-f410 for Sortformer v2.1).
// Never skips: every check needs Core ML generation (macOS 15, the generator
// built in).

#include "coreml/parakeet_coreml_sidecars.h"
#include "parakeet_ctc.h"
#include "qvac_coreml/generation.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

constexpr int kThreads = 4;
constexpr int kGpuLayers = 99;

bool g_passed = true;
fs::path g_dir;
std::string g_template = "nemotron-diar-f358";  // the minimal capacity's template
std::string g_stage = "diarizer-358";           // its status stage

void check(bool condition, const char * label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label);
    g_passed = g_passed && condition;
}

// "open": the path names B; "read": the path names A again.
void swap_around_open(const char * point) {
    std::error_code ec;
    if (std::string(point) == "open") {
        fs::rename(g_dir / "m.gguf", g_dir / "a.gguf", ec);
        fs::rename(g_dir / "b.gguf", g_dir / "m.gguf", ec);
    } else if (std::string(point) == "read") {
        fs::rename(g_dir / "m.gguf", g_dir / "b.gguf", ec);
        fs::rename(g_dir / "a.gguf", g_dir / "m.gguf", ec);
    }
    if (ec) std::printf("swap at %s: %s\n", point, ec.message().c_str());
}

// Holds the first model's compile open until released; then fails it.
std::atomic<bool> g_compile_entered{false};
std::atomic<bool> g_compile_release{false};
std::string held_compile(const std::string &, const std::string &) {
    g_compile_entered = true;
    while (!g_compile_release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return "compile held by the test";
}

// "opened": flip the low bit of the byte in the middle of the file, inside
// the tensor data (the file changes in place: same inode and size, one weight
// and the mtime differ) while the weights are read.
void touch_in_place(const char * point) {
    if (std::string(point) != "opened") return;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // a later mtime
    std::error_code ec;
    const std::streamoff middle = static_cast<std::streamoff>(fs::file_size(g_dir / "m.gguf", ec) / 2);
    std::fstream f(g_dir / "m.gguf", std::ios::in | std::ios::out | std::ios::binary);
    char c = 0;
    f.seekg(middle);
    f.read(&c, 1);
    c = static_cast<char>(c ^ 1);
    f.seekp(middle);
    f.write(&c, 1);
    f.flush();
}

std::string slot_json(const std::string & status, const char * stage_name) {
    const size_t at = status.find(std::string("\"stage\":\"") + stage_name + "\"");
    if (at == std::string::npos) return std::string();
    const size_t from = status.rfind("{\"artifact_key\"", at);
    return status.substr(from, at - from + 60);
}

std::string load_status(const fs::path & cache) {
    parakeet::CoremlOptions options;
    options.cache_dir = cache.string();
    options.wait = parakeet::CoremlOptions::Wait::construct;
    parakeet::ParakeetCtcModel model;
    const int rc = parakeet::load_from_gguf((g_dir / "m.gguf").string(), model, kThreads, kGpuLayers, false, "auto",
                                            options);
    check(rc == 0, "the model loads");
    return rc == 0 ? parakeet::model_coreml_status_json(model) : std::string();
}

bool has_artifacts(const fs::path & cache) {
    std::error_code ec;
    const fs::path artifacts = cache / "qvac-coreml-v1" / "artifacts";
    return fs::exists(artifacts, ec) && !fs::is_empty(artifacts, ec);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <work dir> [template id]\n", argv[0]);
        return 2;
    }
    g_dir = argv[2];
    if (argc == 4) {
        g_template = argv[3];
        const size_t f = g_template.rfind("-f");
        if (f == std::string::npos || std::atoi(g_template.c_str() + f + 2) <= 0) {
            std::fprintf(stderr, "template id %s does not end in -f<capacity>\n", argv[3]);
            return 2;
        }
        g_stage = "diarizer-" + std::to_string(std::atoi(g_template.c_str() + f + 2));
    }
    std::error_code ec;
    fs::remove_all(g_dir, ec);
    fs::create_directories(g_dir);
    fs::copy_file(argv[1], g_dir / "m.gguf", ec);
    fs::copy_file(argv[1], g_dir / "b.gguf", ec);
    if (ec) {
        std::fprintf(stderr, "cannot stage the GGUF: %s\n", ec.message().c_str());
        return 1;
    }

    parakeet::set_gguf_load_observer(&swap_around_open);
    const std::string swapped = load_status(g_dir / "cache-swapped");
    parakeet::set_gguf_load_observer(nullptr);
    std::printf("swapped: %s\n", swapped.c_str());
    check(swapped.find("source_changed") != std::string::npos,
          "the path named another file when the weights were opened: source_changed");
    check(swapped.find("\"state\":\"ready\"") == std::string::npos, "no generated diarizer is ready");
    check(!has_artifacts(g_dir / "cache-swapped"), "nothing is generated from the file the path names now");

    const std::string plain = load_status(g_dir / "cache-plain");
    std::printf("plain: %s\n", plain.c_str());
    check(plain.find("source_changed") == std::string::npos && plain.find("\"state\":\"ready\"") != std::string::npos &&
              has_artifacts(g_dir / "cache-plain"),
          "nothing swapped: the minimal capacity is generated and ready");

    // A cached bundle the framework refuses.
    fs::path mil;
    const fs::path artifacts = g_dir / "cache-plain" / "qvac-coreml-v1" / "artifacts";
    for (const auto & e : fs::recursive_directory_iterator(artifacts, ec)) {
        if (e.path().filename() == "coremldata.bin" && e.path().parent_path().filename() == "model.mlmodelc" &&
            e.path().string().find(g_template + ".") != std::string::npos) {
            mil = e.path();
        }
    }
    check(!mil.empty(), "the minimal capacity's generated bundle is in the cache");
    if (!mil.empty()) {
        const uintmax_t size = fs::file_size(mil, ec);
        {
            std::ofstream out(mil, std::ios::binary | std::ios::trunc);
            const std::string garbage(static_cast<size_t>(size), '#');
            out.write(garbage.data(), static_cast<std::streamsize>(garbage.size()));
        }
        check(fs::file_size(mil, ec) == size, "coremldata.bin corrupted at the same size");
        const std::string repaired = load_status(g_dir / "cache-plain");
        std::printf("repaired: %s\n", repaired.c_str());
        int quarantined = 0;
        for (const auto & e : fs::directory_iterator(g_dir / "cache-plain" / "qvac-coreml-v1" / "quarantine", ec)) {
            (void) e;
            ++quarantined;
        }
        const std::string stage = slot_json(repaired, g_stage.c_str());
        check(quarantined == 1, "the refused bundle is quarantined (one entry)");
        check(stage.find("\"state\":\"ready\"") != std::string::npos &&
                  stage.find("\"route\":\"generated\"") != std::string::npos,
              "the minimal capacity is generated again and ready");
    }

    // A file written in place during the upload, beside a running job.
    {
        parakeet::pcm::generation::set_compile_override(&held_compile);
        parakeet::CoremlOptions options;
        options.cache_dir = (g_dir / "cache-join").string();
        parakeet::ParakeetCtcModel first;
        check(parakeet::load_from_gguf((g_dir / "m.gguf").string(), first, kThreads, kGpuLayers, false, "auto",
                                       options) == 0,
              "the first model loads");
        for (int i = 0; i < 6000 && !g_compile_entered; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        check(g_compile_entered, "the first model's job is compiling (held open)");
        parakeet::set_gguf_load_observer(&touch_in_place);
        parakeet::ParakeetCtcModel second;
        check(parakeet::load_from_gguf((g_dir / "m.gguf").string(), second, kThreads, kGpuLayers, false, "auto",
                                       options) == 0,
              "the second model loads");
        parakeet::set_gguf_load_observer(nullptr);
        // Its own job waits behind the held one (one Core ML job at a time);
        // joined, it would share the running job's phase.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const std::string waiting = slot_json(parakeet::model_coreml_status_json(second), g_stage.c_str());
        std::printf("second model, %s, while the first compiles: %s\n", g_stage.c_str(), waiting.c_str());
        check(waiting.find("\"state\":\"compiling\"") == std::string::npos,
              "the second model did not join the first model's running job");
        g_compile_release = true;
        std::string second_minimal;
        for (int i = 0; i < 3000; ++i) {
            second_minimal = slot_json(parakeet::model_coreml_status_json(second), g_stage.c_str());
            if (second_minimal.find("\"state\":\"failed\"") != std::string::npos) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::printf("second model, %s: %s\n", g_stage.c_str(), second_minimal.c_str());
        check(second_minimal.find("\"state\":\"failed\"") != std::string::npos &&
                  second_minimal.find("source_changed") != std::string::npos,
              "written during its upload: source_changed, not the running job's bundle");
        parakeet::pcm::generation::set_compile_override(nullptr);
    }

    fs::remove_all(g_dir, ec);
    std::printf("%s\n", g_passed ? "PASS" : "FAIL");
    return g_passed ? 0 : 1;
}
