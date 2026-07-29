#include "ggml-vector-index.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

extern "C" void ggml_vec_index_test_set_oom_countdown(int64_t countdown);
extern "C" void ggml_vec_index_test_set_write_fail_after(int64_t bytes);
extern "C" void ggml_vec_index_test_set_truncate_fail(int fail);
extern "C" void ggml_vec_index_test_set_data_fsync_fail(int fail);
extern "C" void ggml_vec_index_test_set_parent_fsync_fail(int fail);
extern "C" void ggml_vec_index_test_set_delta_append_wait_target(int target);
extern "C" int ggml_vec_index_test_get_delta_append_waiters(void);
extern "C" void ggml_vec_index_test_set_load_with_delta_pause_ms(int pause_ms);
extern "C" void ggml_vec_index_test_reset_delta_tail_scan_count(void);
extern "C" int64_t ggml_vec_index_test_get_delta_tail_scan_count(void);
extern "C" void ggml_vec_index_test_reset_state_crc_scan_count(void);
extern "C" int64_t ggml_vec_index_test_get_state_crc_scan_count(void);
extern "C" int ggml_vec_index_test_get_load_with_delta_waiters(void);

namespace {

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

std::string unique_temp_path(const std::string & filename) {
    static std::atomic<uint64_t> counter{ 0 };
    const std::filesystem::path path(filename);
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string unique_name =
        path.stem().string() + "-" +
        std::to_string(now) + "-" +
        std::to_string(counter.fetch_add(1)) +
        path.extension().string();
    return (std::filesystem::temp_directory_path() / unique_name).string();
}

void reset_fault_hooks() {
    ggml_vec_index_test_set_oom_countdown(-1);
    ggml_vec_index_test_set_write_fail_after(-1);
    ggml_vec_index_test_set_truncate_fail(0);
    ggml_vec_index_test_set_data_fsync_fail(0);
    ggml_vec_index_test_set_parent_fsync_fail(0);
    ggml_vec_index_test_set_delta_append_wait_target(0);
    ggml_vec_index_test_set_load_with_delta_pause_ms(0);
    ggml_vec_index_test_reset_delta_tail_scan_count();
    ggml_vec_index_test_reset_state_crc_scan_count();
}

std::vector<uint8_t> read_file_bytes(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open());
    const auto size = std::filesystem::file_size(path);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) {
        f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        CHECK(f.gcount() == static_cast<std::streamsize>(bytes.size()));
    }
    return bytes;
}

std::string temp_file_prefix(const std::string & path) {
    return std::filesystem::path(path).filename().string() + ".tmp.";
}

void remove_temp_siblings(const std::string & path) {
    const std::filesystem::path p(path);
    const std::filesystem::path parent =
        p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
    const std::string prefix = temp_file_prefix(path);
    for (const auto & entry : std::filesystem::directory_iterator(parent)) {
        const std::string name = entry.path().filename().string();
        if (name.compare(0, prefix.size(), prefix) == 0) {
            std::filesystem::remove(entry.path());
        }
    }
}

void expect_no_temp_siblings(const std::string & path) {
    const std::filesystem::path p(path);
    const std::filesystem::path parent =
        p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
    const std::string prefix = temp_file_prefix(path);
    for (const auto & entry : std::filesystem::directory_iterator(parent)) {
        const std::string name = entry.path().filename().string();
        CHECK(name.compare(0, prefix.size(), prefix) != 0);
    }
}

void write_marker_file(const std::string & path) {
    std::ofstream f(path, std::ios::binary);
    CHECK(f.is_open());
    f << "1";
    CHECK(f.good());
}

bool wait_for_path(const std::string & path, int timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::filesystem::exists(path)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::filesystem::exists(path);
}

#ifdef _WIN32
std::wstring utf8_to_wide_checked(const std::string & value) {
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, nullptr, 0);
    CHECK(size > 0);
    std::vector<wchar_t> buffer(static_cast<size_t>(size));
    CHECK(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, buffer.data(), size) == size);
    return std::wstring(buffer.data());
}

std::wstring windows_quote_arg(const std::wstring & value) {
    if (!value.empty() && value.find_first_of(L" \t\"") == std::wstring::npos) {
        return value;
    }

    std::wstring quoted = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') {
            ++backslashes;
        } else if (c == L'"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(c);
            backslashes = 0;
        } else {
            quoted.append(backslashes, L'\\');
            backslashes = 0;
            quoted.push_back(c);
        }
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}
#endif

std::string delta_writer_description(
        const std::string & snapshot_path,
        const std::string & delta_path,
        const std::string & start_path,
        const std::string & ready_path,
        uint64_t id) {
    return "snapshot=" + snapshot_path +
        " delta=" + delta_path +
        " start=" + start_path +
        " ready=" + ready_path +
        " id=" + std::to_string(id);
}

#ifdef _WIN32
using DeltaWriterProcess = PROCESS_INFORMATION;
#else
using DeltaWriterProcess = pid_t;
#endif

#ifdef _WIN32
DeltaWriterProcess spawn_windows_process(const std::vector<std::wstring> & args) {
    std::wstring command_line;
    for (const std::wstring & arg : args) {
        if (!command_line.empty()) {
            command_line.push_back(L' ');
        }
        command_line += windows_quote_arg(arg);
    }
    std::vector<wchar_t> mutable_command_line(command_line.begin(), command_line.end());
    mutable_command_line.push_back(L'\0');
    STARTUPINFOW startup = {};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {};
    CHECK(CreateProcessW(
        nullptr,
        mutable_command_line.data(),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &startup,
        &process) != 0);
    CloseHandle(process.hThread);
    process.hThread = nullptr;
    return process;
}
#endif

DeltaWriterProcess spawn_delta_writer_process(
        const char * self_path,
        const std::string & snapshot_path,
        const std::string & delta_path,
        const std::string & start_path,
        const std::string & ready_path,
        uint64_t id) {
    const std::string id_arg = std::to_string(id);
#ifdef _WIN32
    const std::vector<std::wstring> args = {
        utf8_to_wide_checked(self_path),
        L"--delta-writer",
        utf8_to_wide_checked(snapshot_path),
        utf8_to_wide_checked(delta_path),
        utf8_to_wide_checked(start_path),
        utf8_to_wide_checked(ready_path),
        utf8_to_wide_checked(id_arg),
    };
    return spawn_windows_process(args);
#else
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        execl(
            self_path,
            self_path,
            "--delta-writer",
            snapshot_path.c_str(),
            delta_path.c_str(),
            start_path.c_str(),
            ready_path.c_str(),
            id_arg.c_str(),
            static_cast<char *>(nullptr));
        _exit(127);
    }
    return pid;
#endif
}

int wait_delta_writer_process(DeltaWriterProcess process) {
#ifdef _WIN32
    if (WaitForSingleObject(process.hProcess, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(process.hProcess);
        return -1;
    }
    DWORD exit_code = 0;
    if (GetExitCodeProcess(process.hProcess, &exit_code) == 0) {
        CloseHandle(process.hProcess);
        return -1;
    }
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
#else
    int status = 0;
    while (waitpid(process, &status, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
#endif
}

int run_delta_writer_child(
        const char * snapshot_path,
        const char * delta_path,
        const char * start_path,
        const char * ready_path,
        uint64_t id) {
    auto * idx = ggml_vec_index_load(snapshot_path);
    if (idx == nullptr) {
        return 2;
    }

    write_marker_file(ready_path);
    if (!wait_for_path(start_path, 5000)) {
        ggml_vec_index_free(idx);
        return 3;
    }

    const float scale = static_cast<float>((id % 17) + 1) * 0.125f;
    const std::array<float, 4> vector = {
        scale,
        scale + 0.25f,
        scale + 0.5f,
        scale + 0.75f,
    };
    const int status = ggml_vec_index_add_logged(
        idx, vector.data(), 1, &id, delta_path);
    ggml_vec_index_free(idx);
    return status == GGML_VEC_INDEX_OK ? 0 : 4;
}

#ifdef _WIN32
int run_argv_check_child(int argc, char ** argv) {
    if (argc != 5) {
        return 5;
    }
    if (std::string(argv[2]) != "argument with spaces") {
        return 6;
    }
    if (std::string(argv[3]) != "quote\"inside") {
        return 7;
    }
    if (std::string(argv[4]) != "trailing slash \\") {
        return 8;
    }
    return 0;
}

void test_windows_command_line_quoting(const char * self_path) {
    const std::vector<std::wstring> args = {
        utf8_to_wide_checked(self_path),
        L"--argv-check",
        L"argument with spaces",
        L"quote\"inside",
        L"trailing slash \\",
    };
    CHECK(wait_delta_writer_process(spawn_windows_process(args)) == 0);
}
#endif

void test_hardlink_delta_appends() {
    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = { 701, 702 };
    const std::array<float, 4> vector_a = { 0.5f, 0.25f, 0.0f, 0.0f };
    const std::array<float, 4> vector_b = { 0.0f, 0.25f, 0.5f, 0.0f };
    const uint64_t id_a = 703;
    const uint64_t id_b = 704;

    const std::string snapshot_path =
        unique_temp_path("ggml-vector-index-hardlink-base.tvim");
    const std::string delta_path =
        unique_temp_path("ggml-vector-index-hardlink-log.tvid");
    const std::string alias_path =
        unique_temp_path("ggml-vector-index-hardlink-alias.tvid");

    auto * base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(
        base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(base);

    auto * writer_a = ggml_vec_index_load(snapshot_path.c_str());
    auto * writer_b = ggml_vec_index_load(snapshot_path.c_str());
    CHECK(writer_a != nullptr);
    CHECK(writer_b != nullptr);

    int status_a = GGML_VEC_INDEX_E_INTERNAL;
    int status_b = GGML_VEC_INDEX_E_INTERNAL;
    ggml_vec_index_test_set_delta_append_wait_target(2);
    std::thread thread_a([&]() {
        status_a = ggml_vec_index_add_logged(
            writer_a, vector_a.data(), 1, &id_a, delta_path.c_str());
    });
    const auto wait_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (ggml_vec_index_test_get_delta_append_waiters() == 0 &&
           std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::yield();
    }
    CHECK(ggml_vec_index_test_get_delta_append_waiters() == 1);

    std::error_code hardlink_ec;
    std::filesystem::create_hard_link(delta_path, alias_path, hardlink_ec);
    CHECK(!hardlink_ec);
    std::thread thread_b([&]() {
        status_b = ggml_vec_index_add_logged(
            writer_b, vector_b.data(), 1, &id_b, alias_path.c_str());
    });
    thread_a.join();
    thread_b.join();
    reset_fault_hooks();

    CHECK((status_a == GGML_VEC_INDEX_OK && status_b == GGML_VEC_INDEX_E_IO) ||
          (status_a == GGML_VEC_INDEX_E_IO && status_b == GGML_VEC_INDEX_OK));

    if (status_a == GGML_VEC_INDEX_OK) {
        CHECK(ggml_vec_index_remove_logged(writer_a, id_a, alias_path.c_str()) == GGML_VEC_INDEX_OK);
    } else {
        CHECK(ggml_vec_index_remove_logged(writer_b, id_b, delta_path.c_str()) == GGML_VEC_INDEX_OK);
    }

    auto * replayed = ggml_vec_index_load_with_delta(
        snapshot_path.c_str(), delta_path.c_str());
    CHECK(replayed != nullptr);
    CHECK(ggml_vec_index_len(replayed) == 2);
    CHECK(ggml_vec_index_contains(replayed, id_a) == 0);
    CHECK(ggml_vec_index_contains(replayed, id_b) == 0);
    ggml_vec_index_free(replayed);
    ggml_vec_index_free(writer_a);
    ggml_vec_index_free(writer_b);

    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(alias_path);
    std::filesystem::remove(delta_path + ".lock");
    std::filesystem::remove(alias_path + ".lock");
}

void test_cross_process_delta_appends(const char * self_path, bool use_hardlink_alias) {
    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = { 601, 602 };
    const uint64_t child_id_a = 603;
    const uint64_t child_id_b = 604;

    const std::filesystem::path temp_dir =
        unique_temp_path(use_hardlink_alias ? "ggml vector index process hardlink dir" :
                                             "ggml vector index process dir");
    std::error_code setup_ec;
    std::filesystem::remove_all(temp_dir, setup_ec);
    CHECK(std::filesystem::create_directories(temp_dir));

    const std::string snapshot_path =
        (temp_dir / "snapshot path with spaces.tvim").string();
    const std::string delta_path =
        (temp_dir / "delta log with spaces.tvid").string();
    const std::string alias_path =
        (temp_dir / "delta alias with spaces.tvid").string();
    const std::string start_path =
        (temp_dir / "start marker with spaces").string();
    const std::string ready_path_a =
        (temp_dir / "ready marker a with spaces").string();
    const std::string ready_path_b =
        (temp_dir / "ready marker b with spaces").string();

    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(alias_path);
    std::filesystem::remove(delta_path + ".lock");
    std::filesystem::remove(alias_path + ".lock");
    std::filesystem::remove(start_path);
    std::filesystem::remove(ready_path_a);
    std::filesystem::remove(ready_path_b);

    auto * base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(
        base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(base);

    std::string second_delta_path = delta_path;
    if (use_hardlink_alias) {
        {
            std::ofstream empty_delta(delta_path, std::ios::binary);
            CHECK(empty_delta.is_open());
        }
        std::error_code hardlink_ec;
        std::filesystem::create_hard_link(delta_path, alias_path, hardlink_ec);
        CHECK(!hardlink_ec);
        second_delta_path = alias_path;
    }

    int status_a = -1;
    int status_b = -1;
    const DeltaWriterProcess process_a = spawn_delta_writer_process(
        self_path, snapshot_path, delta_path, start_path, ready_path_a, child_id_a);
    const DeltaWriterProcess process_b = spawn_delta_writer_process(
        self_path, snapshot_path, second_delta_path, start_path, ready_path_b, child_id_b);
    const bool ready =
        wait_for_path(ready_path_a, 5000) &&
        wait_for_path(ready_path_b, 5000);
    write_marker_file(start_path);
    status_a = wait_delta_writer_process(process_a);
    status_b = wait_delta_writer_process(process_b);

    if (!ready ||
        !((status_a == 0 && status_b == 4) ||
          (status_a == 4 && status_b == 0))) {
        std::fprintf(
            stderr,
            "cross-process delta append failed: ready=%d status_a=%d status_b=%d\nchild_a: %s\nchild_b: %s\n",
            ready ? 1 : 0,
            status_a,
            status_b,
            delta_writer_description(snapshot_path, delta_path, start_path, ready_path_a, child_id_a).c_str(),
            delta_writer_description(snapshot_path, second_delta_path, start_path, ready_path_b, child_id_b).c_str());
    }
    CHECK(ready);
    CHECK((status_a == 0 && status_b == 4) ||
          (status_a == 4 && status_b == 0));
    CHECK(std::filesystem::exists(delta_path + ".lock"));
    if (use_hardlink_alias) {
        CHECK(std::filesystem::exists(alias_path + ".lock"));
    }

    auto * replayed = ggml_vec_index_load_with_delta(
        snapshot_path.c_str(), delta_path.c_str());
    CHECK(replayed != nullptr);
    CHECK(ggml_vec_index_len(replayed) == 3);
    CHECK(ggml_vec_index_contains(replayed, base_ids[0]) == 1);
    CHECK(ggml_vec_index_contains(replayed, base_ids[1]) == 1);
    CHECK(ggml_vec_index_contains(replayed, child_id_a) == (status_a == 0 ? 1 : 0));
    CHECK(ggml_vec_index_contains(replayed, child_id_b) == (status_b == 0 ? 1 : 0));
    ggml_vec_index_free(replayed);

    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(alias_path);
    std::filesystem::remove(delta_path + ".lock");
    std::filesystem::remove(alias_path + ".lock");
    std::filesystem::remove(start_path);
    std::filesystem::remove(ready_path_a);
    std::filesystem::remove(ready_path_b);
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_dir, cleanup_ec);
}

void test_quantized_logged_faults(int bit_width) {
    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = {
        static_cast<uint64_t>(800 + bit_width),
        static_cast<uint64_t>(900 + bit_width),
    };
    const std::array<float, 4> logged_vector = { 0.25f, -0.5f, 0.75f, -1.0f };
    const std::array<float, 4> extra_vector = { -0.125f, 0.375f, -0.625f, 0.875f };

    const std::string suffix = std::to_string(bit_width);
    const std::string snapshot_path =
        unique_temp_path("ggml-vector-index-quant-fault-base-" + suffix + ".tvim");
    const std::string delta_path =
        unique_temp_path("ggml-vector-index-quant-fault-log-" + suffix + ".tvid");
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(
        idx, base_vectors.data(), static_cast<int>(base_ids.size()), base_ids.data()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);

    const uint64_t failed_initial_id = static_cast<uint64_t>(1000 + bit_width);
    ggml_vec_index_test_set_write_fail_after(8);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &failed_initial_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, failed_initial_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 2);
    if (std::filesystem::exists(delta_path)) {
        CHECK(std::filesystem::file_size(delta_path) == 0);
    }

    const uint64_t logged_id = static_cast<uint64_t>(1100 + bit_width);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(ggml_vec_index_len(idx) == 3);
    const std::vector<uint8_t> old_delta = read_file_bytes(delta_path);

    const uint64_t failed_rollback_id = static_cast<uint64_t>(1200 + bit_width);
    ggml_vec_index_test_set_write_fail_after(8);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &failed_rollback_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_INTERNAL);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, failed_rollback_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 3);
    CHECK(read_file_bytes(delta_path) == old_delta);

    ggml_vec_index_test_set_write_fail_after(8);
    CHECK(ggml_vec_index_remove_logged(idx, logged_id, delta_path.c_str()) ==
          GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(read_file_bytes(delta_path) == old_delta);

    CHECK(ggml_vec_index_remove_logged(idx, base_ids[0], delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(idx, base_ids[0]) == 0);
    CHECK(ggml_vec_index_len(idx) == 2);

    auto * replayed = ggml_vec_index_load_with_delta(snapshot_path.c_str(), delta_path.c_str());
    CHECK(replayed != nullptr);
    CHECK(ggml_vec_index_bit_width(replayed) == bit_width);
    CHECK(ggml_vec_index_len(replayed) == 2);
    CHECK(ggml_vec_index_contains(replayed, base_ids[0]) == 0);
    CHECK(ggml_vec_index_contains(replayed, base_ids[1]) == 1);
    CHECK(ggml_vec_index_contains(replayed, logged_id) == 1);

    ggml_vec_index_free(replayed);
    ggml_vec_index_free(idx);
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");
}

void test_torn_delta_header_recovery() {
    const std::string snapshot_path =
        unique_temp_path("ggml-vector-index-torn-header-base.tvim");
    const std::string delta_path =
        unique_temp_path("ggml-vector-index-torn-header-log.tvid");
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");

    const std::array<float, 2> base_vector = { 1.0f, 0.0f };
    const uint64_t base_id = 1201;
    auto * base = ggml_vec_index_create(/*dim=*/2, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vector.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(base);

    {
        std::ofstream torn(delta_path, std::ios::binary);
        CHECK(torn.is_open());
        torn.write("TVI", 3);
        CHECK(torn.good());
    }

    auto * loaded = ggml_vec_index_load_with_delta(snapshot_path.c_str(), delta_path.c_str());
    CHECK(loaded != nullptr);
    const std::array<float, 2> logged_vector = { 0.0f, 1.0f };
    const uint64_t logged_id = 1202;
    CHECK(ggml_vec_index_add_logged(
        loaded, logged_vector.data(), 1, &logged_id, delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(loaded, logged_id) == 1);
    ggml_vec_index_free(loaded);

    auto * replayed = ggml_vec_index_load_with_delta(snapshot_path.c_str(), delta_path.c_str());
    CHECK(replayed != nullptr);
    CHECK(ggml_vec_index_contains(replayed, base_id) == 1);
    CHECK(ggml_vec_index_contains(replayed, logged_id) == 1);
    ggml_vec_index_free(replayed);

    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");
}

void test_delta_append_fault_windows() {
    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = { 1301, 1302 };
    const std::array<float, 4> logged_vector = { 0.0f, 0.0f, 1.0f, 0.0f };
    const std::array<float, 4> extra_vector = { 0.0f, 0.0f, 0.0f, 1.0f };

    const std::string snapshot_path =
        unique_temp_path("ggml-vector-index-append-window-base.tvim");
    const std::string delta_path =
        unique_temp_path("ggml-vector-index-append-window-log.tvid");
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);

    const uint64_t logged_id = 1303;
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> old_delta = read_file_bytes(delta_path);

    const uint64_t torn_payload_id = 1304;
    ggml_vec_index_test_set_write_fail_after(56);
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &torn_payload_id, delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, torn_payload_id) == 0);
    CHECK(read_file_bytes(delta_path) == old_delta);

    ggml_vec_index_test_set_write_fail_after(56);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &torn_payload_id, delta_path.c_str()) == GGML_VEC_INDEX_E_INTERNAL);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, torn_payload_id) == 0);
    CHECK(std::filesystem::file_size(delta_path) > old_delta.size());

    auto * torn_replayed = ggml_vec_index_load_with_delta(snapshot_path.c_str(), delta_path.c_str());
    CHECK(torn_replayed != nullptr);
    CHECK(ggml_vec_index_contains(torn_replayed, torn_payload_id) == 0);
    ggml_vec_index_free(torn_replayed);

    const uint64_t healed_id = 1305;
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &healed_id, delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(idx, healed_id) == 1);

    const uint64_t data_fsync_id = 1306;
    ggml_vec_index_test_set_data_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &data_fsync_id, delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, data_fsync_id) == 1);
    const uint64_t blocked_after_data_fsync_id = 1307;
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &blocked_after_data_fsync_id, delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_contains(idx, blocked_after_data_fsync_id) == 0);
    CHECK(ggml_vec_index_remove_logged(idx, logged_id, delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(ggml_vec_index_compact_delta(idx, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_E_IO);

    auto * data_fsync_replayed = ggml_vec_index_load_with_delta(snapshot_path.c_str(), delta_path.c_str());
    CHECK(data_fsync_replayed != nullptr);
    CHECK(ggml_vec_index_contains(data_fsync_replayed, data_fsync_id) == 1);
    ggml_vec_index_test_set_data_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_remove_logged(data_fsync_replayed, data_fsync_id, delta_path.c_str()) ==
          GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(data_fsync_replayed, data_fsync_id) == 0);
    CHECK(ggml_vec_index_add_logged(
        data_fsync_replayed, extra_vector.data(), 1, &blocked_after_data_fsync_id, delta_path.c_str()) ==
          GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_contains(data_fsync_replayed, blocked_after_data_fsync_id) == 0);
    CHECK(ggml_vec_index_compact_delta(
        data_fsync_replayed, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    ggml_vec_index_free(data_fsync_replayed);

    ggml_vec_index_free(idx);
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");
}

void test_compact_delta_rejects_symlinks() {
#ifndef _WIN32
    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = { 1401, 1402 };
    const std::array<float, 4> logged_vector = { 0.0f, 0.0f, 1.0f, 0.0f };
    const std::array<float, 4> extra_vector = { 0.0f, 0.0f, 0.0f, 1.0f };

    const std::string snapshot_path =
        unique_temp_path("ggml-vector-index-symlink-compact-base.tvim");
    const std::string snapshot_link =
        unique_temp_path("ggml-vector-index-symlink-compact-base-link.tvim");
    const std::string delta_path =
        unique_temp_path("ggml-vector-index-symlink-compact-log.tvid");
    const std::string delta_link =
        unique_temp_path("ggml-vector-index-symlink-compact-log-link.tvid");
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(snapshot_link);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_link);
    std::filesystem::remove(delta_path + ".lock");
    std::filesystem::remove(delta_link + ".lock");

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> old_snapshot = read_file_bytes(snapshot_path);

    const uint64_t logged_id = 1403;
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) == GGML_VEC_INDEX_OK);

    std::error_code symlink_ec;
    std::filesystem::create_symlink(snapshot_path, snapshot_link, symlink_ec);
    if (symlink_ec) {
        ggml_vec_index_free(idx);
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(delta_path);
        std::filesystem::remove(delta_path + ".lock");
        return;
    }
    std::filesystem::create_symlink(delta_path, delta_link, symlink_ec);
    if (symlink_ec) {
        ggml_vec_index_free(idx);
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(snapshot_link);
        std::filesystem::remove(delta_path);
        std::filesystem::remove(delta_path + ".lock");
        return;
    }

    CHECK(ggml_vec_index_compact_delta(
        idx, snapshot_link.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(read_file_bytes(snapshot_path) == old_snapshot);
    CHECK(std::filesystem::is_symlink(std::filesystem::symlink_status(snapshot_link)));

    CHECK(ggml_vec_index_compact_delta(
        idx, snapshot_path.c_str(), delta_link.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(read_file_bytes(snapshot_path) == old_snapshot);
    CHECK(std::filesystem::is_symlink(std::filesystem::symlink_status(delta_link)));

    const uint64_t extra_id = 1404;
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &extra_id, delta_path.c_str()) == GGML_VEC_INDEX_OK);
    auto * replayed = ggml_vec_index_load_with_delta(snapshot_path.c_str(), delta_path.c_str());
    CHECK(replayed != nullptr);
    CHECK(ggml_vec_index_len(replayed) == 4);
    CHECK(ggml_vec_index_contains(replayed, logged_id) == 1);
    CHECK(ggml_vec_index_contains(replayed, extra_id) == 1);

    ggml_vec_index_free(replayed);
    ggml_vec_index_free(idx);
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(snapshot_link);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_link);
    std::filesystem::remove(delta_path + ".lock");
    std::filesystem::remove(delta_link + ".lock");
#endif
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 7 && std::string(argv[1]) == "--delta-writer") {
        const uint64_t id = std::strtoull(argv[6], nullptr, 10);
        return run_delta_writer_child(argv[2], argv[3], argv[4], argv[5], id);
    }
#ifdef _WIN32
    if (argc >= 2 && std::string(argv[1]) == "--argv-check") {
        return run_argv_check_child(argc, argv);
    }
#endif

    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = { 101, 102 };

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(
        idx, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);

    {
        const std::array<float, 8> new_vectors = {
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        };
        const std::array<uint64_t, 2> new_ids = { 201, 202 };

        ggml_vec_index_test_set_oom_countdown(0);
        CHECK(ggml_vec_index_add(
            idx, new_vectors.data(), 2, new_ids.data()) == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();
        CHECK(ggml_vec_index_len(idx) == 2);

        // Fail before the second map insertion, after the first was committed.
        ggml_vec_index_test_set_oom_countdown(3);
        CHECK(ggml_vec_index_add(
            idx, new_vectors.data(), 2, new_ids.data()) == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();
        CHECK(ggml_vec_index_len(idx) == 2);
        CHECK(ggml_vec_index_contains(idx, new_ids[0]) == 0);
        CHECK(ggml_vec_index_contains(idx, new_ids[1]) == 0);
    }

    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> ids{};
        ggml_vec_index_test_set_oom_countdown(0);
        CHECK(ggml_vec_index_search(
            idx, base_vectors.data(), 1, /*k=*/1,
            scores.data(), ids.data()) == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();
    }

    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> ids{};
        CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
            scores.data(), ids.data()) == GGML_VEC_INDEX_OK);

        ggml_vec_index_test_set_oom_countdown(0);
        CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
              == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();

        CHECK(ggml_vec_index_search_ivf(
            idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
            scores.data(), ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ids[0] == base_ids[0]);
    }

    const std::string path =
        unique_temp_path("ggml-vector-index-fault-test.tvim");
    std::filesystem::remove(path);
    remove_temp_siblings(path);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> old_snapshot = read_file_bytes(path);

    const std::array<float, 4> extra_vector = { 0.5f, 0.5f, 0.5f, 0.5f };
    const uint64_t extra_id = 301;
    CHECK(ggml_vec_index_add(
        idx, extra_vector.data(), 1, &extra_id) == GGML_VEC_INDEX_OK);

    ggml_vec_index_test_set_oom_countdown(0);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_E_OOM);
    reset_fault_hooks();
    CHECK(read_file_bytes(path) == old_snapshot);
    expect_no_temp_siblings(path);

    // The first checkpoint passes; the second fails after temp creation.
    ggml_vec_index_test_set_oom_countdown(1);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_E_OOM);
    reset_fault_hooks();
    CHECK(read_file_bytes(path) == old_snapshot);
    expect_no_temp_siblings(path);

    ggml_vec_index_test_set_write_fail_after(40);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(read_file_bytes(path) == old_snapshot);
    expect_no_temp_siblings(path);

    ggml_vec_index_test_set_oom_countdown(0);
    CHECK(ggml_vec_index_load(path.c_str()) == nullptr);
    reset_fault_hooks();

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 2);
    ggml_vec_index_free(loaded);

    {
        const std::string parent_fsync_path =
            unique_temp_path("ggml-vector-index-parent-fsync-test.tvim");
        std::filesystem::remove(parent_fsync_path);

        auto * parent_fsync_idx = ggml_vec_index_create(dim, /*bit_width=*/32);
        CHECK(parent_fsync_idx != nullptr);
        CHECK(ggml_vec_index_add(
            parent_fsync_idx, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(parent_fsync_idx, parent_fsync_path.c_str()) ==
              GGML_VEC_INDEX_OK);
        const std::vector<uint8_t> before_parent_fsync =
            read_file_bytes(parent_fsync_path);

        const uint64_t parent_fsync_id = 302;
        CHECK(ggml_vec_index_add(
            parent_fsync_idx, extra_vector.data(), 1, &parent_fsync_id) ==
            GGML_VEC_INDEX_OK);
        ggml_vec_index_test_set_parent_fsync_fail(1);
        CHECK(ggml_vec_index_write(parent_fsync_idx, parent_fsync_path.c_str()) ==
              GGML_VEC_INDEX_E_NOT_DURABLE);
        reset_fault_hooks();
        CHECK(read_file_bytes(parent_fsync_path) != before_parent_fsync);

        auto * parent_fsync_loaded = ggml_vec_index_load(parent_fsync_path.c_str());
        CHECK(parent_fsync_loaded != nullptr);
        CHECK(ggml_vec_index_len(parent_fsync_loaded) == 3);
        CHECK(ggml_vec_index_contains(parent_fsync_loaded, parent_fsync_id) == 1);
        ggml_vec_index_free(parent_fsync_loaded);
        ggml_vec_index_free(parent_fsync_idx);
        std::filesystem::remove(parent_fsync_path);
    }
    {
        const std::string tail_cache_snapshot_path =
            unique_temp_path("ggml-vector-index-tail-cache-oom-base.tvim");
        const std::string tail_cache_delta_path =
            unique_temp_path("ggml-vector-index-tail-cache-oom-log.tvid");
        std::filesystem::remove(tail_cache_snapshot_path);
        std::filesystem::remove(tail_cache_delta_path);
        std::filesystem::remove(tail_cache_delta_path + ".lock");

        auto * tail_cache_idx = ggml_vec_index_create(dim, /*bit_width=*/32);
        CHECK(tail_cache_idx != nullptr);
        CHECK(ggml_vec_index_add(
            tail_cache_idx, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(tail_cache_idx, tail_cache_snapshot_path.c_str()) ==
              GGML_VEC_INDEX_OK);

        const uint64_t tail_cache_id = 303;
        ggml_vec_index_test_set_oom_countdown(3);
        CHECK(ggml_vec_index_add_logged(
            tail_cache_idx, extra_vector.data(), 1,
            &tail_cache_id, tail_cache_delta_path.c_str()) == GGML_VEC_INDEX_OK);
        reset_fault_hooks();
        CHECK(ggml_vec_index_contains(tail_cache_idx, tail_cache_id) == 1);

        auto * tail_cache_replayed = ggml_vec_index_load_with_delta(
            tail_cache_snapshot_path.c_str(), tail_cache_delta_path.c_str());
        CHECK(tail_cache_replayed != nullptr);
        CHECK(ggml_vec_index_contains(tail_cache_replayed, tail_cache_id) == 1);
        ggml_vec_index_free(tail_cache_replayed);
        ggml_vec_index_free(tail_cache_idx);
        std::filesystem::remove(tail_cache_snapshot_path);
        std::filesystem::remove(tail_cache_delta_path);
        std::filesystem::remove(tail_cache_delta_path + ".lock");
    }

    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::string delta_path =
        unique_temp_path("ggml-vector-index-fault-test.tvid");
    std::filesystem::remove(delta_path);

    const std::array<float, 4> logged_vector = { 0.0f, 0.0f, 1.0f, 0.0f };
    const uint64_t logged_id = 401;
    ggml_vec_index_test_set_write_fail_after(8);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, logged_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 3);
    if (std::filesystem::exists(delta_path)) {
        CHECK(std::filesystem::file_size(delta_path) == 0);
    }

    const uint64_t allowed_id = base_ids[0];
    std::array<float, 1> logged_scores{};
    std::array<uint64_t, 1> logged_out_ids{};
    auto * stale_filter = ggml_vec_index_filter_create(idx, &allowed_id, 1);
    CHECK(stale_filter != nullptr);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
          == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(ggml_vec_index_search_prepared_filtered(
        idx, stale_filter, base_vectors.data(), 1, /*k=*/1,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(
        idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(stale_filter);
    const std::vector<uint8_t> old_delta = read_file_bytes(delta_path);
    const std::string wrong_delta_path =
        unique_temp_path("ggml-vector-index-wrong-zero-log.tvid");
    CHECK(ggml_vec_index_add_logged(idx, nullptr, 0, nullptr, wrong_delta_path.c_str()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_add_logged(idx, nullptr, 0, nullptr, delta_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    std::filesystem::remove(wrong_delta_path);

    auto * rollback_filter = ggml_vec_index_filter_create(idx, &allowed_id, 1);
    CHECK(rollback_filter != nullptr);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
          == GGML_VEC_INDEX_OK);
    const uint64_t failed_internal_id = 402;
    ggml_vec_index_test_set_write_fail_after(8);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &failed_internal_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_INTERNAL);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, failed_internal_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 4);
    CHECK(read_file_bytes(delta_path) == old_delta);
    CHECK(ggml_vec_index_search_prepared_filtered(
        idx, rollback_filter, base_vectors.data(), 1, /*k=*/1,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(logged_out_ids[0] == allowed_id);
    CHECK(ggml_vec_index_search_ivf(
        idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(logged_out_ids[0] == allowed_id);
    ggml_vec_index_filter_free(rollback_filter);

    auto * committed_add_filter = ggml_vec_index_filter_create(idx, &allowed_id, 1);
    CHECK(committed_add_filter != nullptr);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
          == GGML_VEC_INDEX_OK);
    const uint64_t committed_add_id = 403;
    ggml_vec_index_test_set_parent_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &committed_add_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_NOT_DURABLE);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, committed_add_id) == 1);
    CHECK(ggml_vec_index_len(idx) == 5);
    CHECK(ggml_vec_index_search_prepared_filtered(
        idx, committed_add_filter, base_vectors.data(), 1, /*k=*/1,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(
        idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(committed_add_filter);
    const std::vector<uint8_t> delta_after_committed_add = read_file_bytes(delta_path);

    const uint64_t blocked_after_indeterminate_id = 404;
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1,
        &blocked_after_indeterminate_id, delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_contains(idx, blocked_after_indeterminate_id) == 0);
    CHECK(ggml_vec_index_remove_logged(idx, logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(ggml_vec_index_compact_delta(idx, path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    CHECK(read_file_bytes(delta_path) == delta_after_committed_add);

    const std::string committed_add_snapshot_path =
        unique_temp_path("ggml-vector-index-committed-add-base.tvim");
    const std::string committed_add_delta_path =
        unique_temp_path("ggml-vector-index-committed-add-log.tvid");
    std::filesystem::remove(committed_add_snapshot_path);
    std::filesystem::remove(committed_add_delta_path);
    std::filesystem::remove(committed_add_delta_path + ".lock");

    auto * committed_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(committed_base != nullptr);
    CHECK(ggml_vec_index_add(
        committed_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(committed_base, committed_add_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    const uint64_t committed_replay_id = 601;
    ggml_vec_index_test_set_parent_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        committed_base, extra_vector.data(), 1,
        &committed_replay_id, committed_add_delta_path.c_str()) ==
        GGML_VEC_INDEX_E_NOT_DURABLE);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(committed_base, committed_replay_id) == 1);
    const uint64_t committed_replay_blocked_id = 603;
    CHECK(ggml_vec_index_add_logged(
        committed_base, extra_vector.data(), 1,
        &committed_replay_blocked_id, committed_add_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_contains(committed_base, committed_replay_blocked_id) == 0);
    auto * committed_replayed = ggml_vec_index_load_with_delta(
        committed_add_snapshot_path.c_str(), committed_add_delta_path.c_str());
    CHECK(committed_replayed != nullptr);
    CHECK(ggml_vec_index_len(committed_replayed) == 3);
    CHECK(ggml_vec_index_contains(committed_replayed, committed_replay_id) == 1);
    const uint64_t committed_replay_after_reload_id = 604;
    CHECK(ggml_vec_index_add_logged(
        committed_replayed, logged_vector.data(), 1,
        &committed_replay_after_reload_id, committed_add_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(committed_replayed, committed_replay_after_reload_id) == 1);
    ggml_vec_index_free(committed_replayed);
    ggml_vec_index_free(committed_base);
    std::filesystem::remove(committed_add_snapshot_path);
    std::filesystem::remove(committed_add_delta_path);
    std::filesystem::remove(committed_add_delta_path + ".lock");

    const std::string committed_remove_snapshot_path =
        unique_temp_path("ggml-vector-index-committed-remove-base.tvim");
    const std::string committed_remove_delta_path =
        unique_temp_path("ggml-vector-index-committed-remove-log.tvid");
    std::filesystem::remove(committed_remove_snapshot_path);
    std::filesystem::remove(committed_remove_delta_path);
    std::filesystem::remove(committed_remove_delta_path + ".lock");

    auto * committed_remove_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(committed_remove_base != nullptr);
    CHECK(ggml_vec_index_add(
        committed_remove_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(committed_remove_base, committed_remove_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    ggml_vec_index_test_set_parent_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_remove_logged(
        committed_remove_base, base_ids[0], committed_remove_delta_path.c_str()) == GGML_VEC_INDEX_E_NOT_DURABLE);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(committed_remove_base, base_ids[0]) == 0);
    CHECK(ggml_vec_index_add_logged(
        committed_remove_base, extra_vector.data(), 1,
        &committed_replay_blocked_id, committed_remove_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    auto * committed_remove_replayed = ggml_vec_index_load_with_delta(
        committed_remove_snapshot_path.c_str(), committed_remove_delta_path.c_str());
    CHECK(committed_remove_replayed != nullptr);
    CHECK(ggml_vec_index_contains(committed_remove_replayed, base_ids[0]) == 0);
    CHECK(ggml_vec_index_contains(committed_remove_replayed, base_ids[1]) == 1);
    ggml_vec_index_free(committed_remove_replayed);
    ggml_vec_index_free(committed_remove_base);
    std::filesystem::remove(committed_remove_snapshot_path);
    std::filesystem::remove(committed_remove_delta_path);
    std::filesystem::remove(committed_remove_delta_path + ".lock");

    const std::string compact_parent_snapshot_path =
        unique_temp_path("ggml-vector-index-compact-parent-fsync-base.tvim");
    const std::string compact_parent_delta_path =
        unique_temp_path("ggml-vector-index-compact-parent-fsync-log.tvid");
    std::filesystem::remove(compact_parent_snapshot_path);
    std::filesystem::remove(compact_parent_delta_path);
    std::filesystem::remove(compact_parent_delta_path + ".lock");

    auto * compact_parent = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(compact_parent != nullptr);
    CHECK(ggml_vec_index_add(
        compact_parent, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(compact_parent, compact_parent_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    const uint64_t compact_parent_id = 602;
    CHECK(ggml_vec_index_add_logged(
        compact_parent, extra_vector.data(), 1,
        &compact_parent_id, compact_parent_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_test_set_parent_fsync_fail(1);
    CHECK(ggml_vec_index_compact_delta(
        compact_parent,
        compact_parent_snapshot_path.c_str(),
        compact_parent_delta_path.c_str()) == GGML_VEC_INDEX_E_PARTIAL_COMPACT);
    reset_fault_hooks();
    CHECK(std::filesystem::file_size(compact_parent_delta_path) > 16);
    auto * compact_parent_replayed = ggml_vec_index_load_with_delta(
        compact_parent_snapshot_path.c_str(), compact_parent_delta_path.c_str());
    CHECK(compact_parent_replayed != nullptr);
    CHECK(ggml_vec_index_len(compact_parent_replayed) == 3);
    CHECK(ggml_vec_index_contains(compact_parent_replayed, compact_parent_id) == 1);
    ggml_vec_index_free(compact_parent_replayed);
    ggml_vec_index_free(compact_parent);
    std::filesystem::remove(compact_parent_snapshot_path);
    std::filesystem::remove(compact_parent_delta_path);
    std::filesystem::remove(compact_parent_delta_path + ".lock");

    const std::string cached_tail_snapshot_path =
        unique_temp_path("ggml-vector-index-cached-tail-base.tvim");
    const std::string cached_tail_delta_path =
        unique_temp_path("ggml-vector-index-cached-tail-log.tvid");
    std::filesystem::remove(cached_tail_snapshot_path);
    std::filesystem::remove(cached_tail_delta_path);
    std::filesystem::remove(cached_tail_delta_path + ".lock");

    auto * cached_tail = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(cached_tail != nullptr);
    CHECK(ggml_vec_index_add(
        cached_tail, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(cached_tail, cached_tail_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    const uint64_t cached_tail_id_a = 701;
    const uint64_t cached_tail_id_b = 702;
    ggml_vec_index_test_reset_delta_tail_scan_count();
    ggml_vec_index_test_reset_state_crc_scan_count();
    CHECK(ggml_vec_index_add_logged(
        cached_tail, logged_vector.data(), 1,
        &cached_tail_id_a, cached_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add_logged(
        cached_tail, extra_vector.data(), 1,
        &cached_tail_id_b, cached_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_remove_logged(
        cached_tail, base_ids[0], cached_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_test_get_delta_tail_scan_count() == 0);
    CHECK(ggml_vec_index_test_get_state_crc_scan_count() == 0);

    auto * cached_tail_replayed = ggml_vec_index_load_with_delta(
        cached_tail_snapshot_path.c_str(), cached_tail_delta_path.c_str());
    CHECK(cached_tail_replayed != nullptr);
    CHECK(ggml_vec_index_len(cached_tail_replayed) == 3);
    CHECK(ggml_vec_index_contains(cached_tail_replayed, base_ids[0]) == 0);
    CHECK(ggml_vec_index_contains(cached_tail_replayed, cached_tail_id_a) == 1);
    CHECK(ggml_vec_index_contains(cached_tail_replayed, cached_tail_id_b) == 1);
    ggml_vec_index_free(cached_tail_replayed);
    ggml_vec_index_free(cached_tail);
    std::filesystem::remove(cached_tail_snapshot_path);
    std::filesystem::remove(cached_tail_delta_path);
    std::filesystem::remove(cached_tail_delta_path + ".lock");

    const std::string stale_tail_snapshot_path =
        unique_temp_path("ggml-vector-index-stale-tail-base.tvim");
    const std::string stale_tail_delta_path =
        unique_temp_path("ggml-vector-index-stale-tail-log.tvid");
    std::filesystem::remove(stale_tail_snapshot_path);
    std::filesystem::remove(stale_tail_delta_path);
    std::filesystem::remove(stale_tail_delta_path + ".lock");

    auto * stale_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(stale_base != nullptr);
    CHECK(ggml_vec_index_add(
        stale_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(stale_base, stale_tail_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    ggml_vec_index_free(stale_base);

    auto * stale_writer = ggml_vec_index_load(stale_tail_snapshot_path.c_str());
    CHECK(stale_writer != nullptr);
    const uint64_t stale_tail_id_a = 801;
    CHECK(ggml_vec_index_add_logged(
        stale_writer, logged_vector.data(), 1,
        &stale_tail_id_a, stale_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    const uint64_t stale_tail_size = std::filesystem::file_size(stale_tail_delta_path);

    auto * fresh_writer = ggml_vec_index_load_with_delta(
        stale_tail_snapshot_path.c_str(), stale_tail_delta_path.c_str());
    CHECK(fresh_writer != nullptr);
    CHECK(ggml_vec_index_compact_delta(
        fresh_writer,
        stale_tail_snapshot_path.c_str(),
        stale_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    const uint64_t stale_tail_id_b = 802;
    CHECK(ggml_vec_index_add_logged(
        fresh_writer, extra_vector.data(), 1,
        &stale_tail_id_b, stale_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(std::filesystem::file_size(stale_tail_delta_path) == stale_tail_size);

    const uint64_t stale_tail_rejected_id = 803;
    CHECK(ggml_vec_index_add_logged(
        stale_writer, logged_vector.data(), 1,
        &stale_tail_rejected_id, stale_tail_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_contains(stale_writer, stale_tail_rejected_id) == 0);

    auto * stale_tail_replayed = ggml_vec_index_load_with_delta(
        stale_tail_snapshot_path.c_str(), stale_tail_delta_path.c_str());
    CHECK(stale_tail_replayed != nullptr);
    CHECK(ggml_vec_index_len(stale_tail_replayed) == 4);
    CHECK(ggml_vec_index_contains(stale_tail_replayed, stale_tail_id_a) == 1);
    CHECK(ggml_vec_index_contains(stale_tail_replayed, stale_tail_id_b) == 1);
    CHECK(ggml_vec_index_contains(stale_tail_replayed, stale_tail_rejected_id) == 0);
    ggml_vec_index_free(stale_tail_replayed);
    ggml_vec_index_free(fresh_writer);
    ggml_vec_index_free(stale_writer);
    std::filesystem::remove(stale_tail_snapshot_path);
    std::filesystem::remove(stale_tail_delta_path);
    std::filesystem::remove(stale_tail_delta_path + ".lock");

    const std::string stale_compact_snapshot_path =
        unique_temp_path("ggml-vector-index-stale-compact-base.tvim");
    const std::string stale_compact_delta_path =
        unique_temp_path("ggml-vector-index-stale-compact-log.tvid");
    std::filesystem::remove(stale_compact_snapshot_path);
    std::filesystem::remove(stale_compact_delta_path);
    std::filesystem::remove(stale_compact_delta_path + ".lock");

    auto * stale_compact_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(stale_compact_base != nullptr);
    CHECK(ggml_vec_index_add(
        stale_compact_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(stale_compact_base, stale_compact_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    ggml_vec_index_free(stale_compact_base);

    auto * stale_compactor = ggml_vec_index_load(stale_compact_snapshot_path.c_str());
    auto * current_writer = ggml_vec_index_load(stale_compact_snapshot_path.c_str());
    CHECK(stale_compactor != nullptr);
    CHECK(current_writer != nullptr);
    const uint64_t stale_compact_id = 804;
    CHECK(ggml_vec_index_add_logged(
        current_writer, extra_vector.data(), 1,
        &stale_compact_id, stale_compact_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_compact_delta(
        stale_compactor,
        stale_compact_snapshot_path.c_str(),
        stale_compact_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);

    auto * stale_compact_replayed = ggml_vec_index_load_with_delta(
        stale_compact_snapshot_path.c_str(), stale_compact_delta_path.c_str());
    CHECK(stale_compact_replayed != nullptr);
    CHECK(ggml_vec_index_len(stale_compact_replayed) == 3);
    CHECK(ggml_vec_index_contains(stale_compact_replayed, stale_compact_id) == 1);
    ggml_vec_index_free(stale_compact_replayed);
    ggml_vec_index_free(current_writer);
    ggml_vec_index_free(stale_compactor);
    std::filesystem::remove(stale_compact_snapshot_path);
    std::filesystem::remove(stale_compact_delta_path);
    std::filesystem::remove(stale_compact_delta_path + ".lock");

    const std::string load_compact_snapshot_path =
        unique_temp_path("ggml-vector-index-load-compact-base.tvim");
    const std::string load_compact_delta_path =
        unique_temp_path("ggml-vector-index-load-compact-log.tvid");
    std::filesystem::remove(load_compact_snapshot_path);
    std::filesystem::remove(load_compact_delta_path);
    std::filesystem::remove(load_compact_delta_path + ".lock");

    auto * load_compact_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(load_compact_base != nullptr);
    CHECK(ggml_vec_index_add(
        load_compact_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(load_compact_base, load_compact_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    const uint64_t load_compact_id = 901;
    CHECK(ggml_vec_index_add_logged(
        load_compact_base, logged_vector.data(), 1,
        &load_compact_id, load_compact_delta_path.c_str()) == GGML_VEC_INDEX_OK);

    auto * load_compact_writer = ggml_vec_index_load_with_delta(
        load_compact_snapshot_path.c_str(), load_compact_delta_path.c_str());
    CHECK(load_compact_writer != nullptr);

    ggml_vec_index_t * concurrent_loaded = nullptr;
    ggml_vec_index_test_set_load_with_delta_pause_ms(250);
    std::thread load_thread([&]() {
        concurrent_loaded = ggml_vec_index_load_with_delta(
            load_compact_snapshot_path.c_str(), load_compact_delta_path.c_str());
    });
    const auto load_wait_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (ggml_vec_index_test_get_load_with_delta_waiters() == 0 &&
           std::chrono::steady_clock::now() < load_wait_deadline) {
        std::this_thread::yield();
    }
    CHECK(ggml_vec_index_test_get_load_with_delta_waiters() == 1);
    CHECK(ggml_vec_index_compact_delta(
        load_compact_writer,
        load_compact_snapshot_path.c_str(),
        load_compact_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    load_thread.join();
    reset_fault_hooks();
    CHECK(concurrent_loaded != nullptr);
    CHECK(ggml_vec_index_len(concurrent_loaded) == 3);
    CHECK(ggml_vec_index_contains(concurrent_loaded, load_compact_id) == 1);

    auto * load_compact_replayed = ggml_vec_index_load_with_delta(
        load_compact_snapshot_path.c_str(), load_compact_delta_path.c_str());
    CHECK(load_compact_replayed != nullptr);
    CHECK(ggml_vec_index_len(load_compact_replayed) == 3);
    CHECK(ggml_vec_index_contains(load_compact_replayed, load_compact_id) == 1);
    ggml_vec_index_free(load_compact_replayed);
    ggml_vec_index_free(concurrent_loaded);
    ggml_vec_index_free(load_compact_writer);
    ggml_vec_index_free(load_compact_base);
    std::filesystem::remove(load_compact_snapshot_path);
    std::filesystem::remove(load_compact_delta_path);
    std::filesystem::remove(load_compact_delta_path + ".lock");

    test_quantized_logged_faults(/*bit_width=*/8);
    test_quantized_logged_faults(/*bit_width=*/4);
    test_torn_delta_header_recovery();
    test_delta_append_fault_windows();
    test_compact_delta_rejects_symlinks();

    const std::string shared_snapshot_path =
        unique_temp_path("ggml-vector-index-shared-delta-base.tvim");
    const std::string shared_delta_path =
        unique_temp_path("ggml-vector-index-shared-delta-log.tvid");
    std::filesystem::remove(shared_snapshot_path);
    std::filesystem::remove(shared_delta_path);
    std::filesystem::remove(shared_delta_path + ".lock");

    auto * shared_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(shared_base != nullptr);
    CHECK(ggml_vec_index_add(
        shared_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(shared_base, shared_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(shared_base);

    auto * shared_a = ggml_vec_index_load(shared_snapshot_path.c_str());
    auto * shared_b = ggml_vec_index_load(shared_snapshot_path.c_str());
    CHECK(shared_a != nullptr);
    CHECK(shared_b != nullptr);
    const uint64_t shared_id_a = 501;
    const uint64_t shared_id_b = 502;
    std::atomic<bool> start_shared_appends{ false };
    int status_a = GGML_VEC_INDEX_E_INTERNAL;
    int status_b = GGML_VEC_INDEX_E_INTERNAL;
    ggml_vec_index_test_set_delta_append_wait_target(2);
    std::thread thread_a([&]() {
        while (!start_shared_appends.load()) {
            std::this_thread::yield();
        }
        status_a = ggml_vec_index_add_logged(
            shared_a, logged_vector.data(), 1, &shared_id_a, shared_delta_path.c_str());
    });
    std::thread thread_b([&]() {
        while (!start_shared_appends.load()) {
            std::this_thread::yield();
        }
        status_b = ggml_vec_index_add_logged(
            shared_b, extra_vector.data(), 1, &shared_id_b, shared_delta_path.c_str());
    });
    start_shared_appends.store(true);
    thread_a.join();
    thread_b.join();
    reset_fault_hooks();

    CHECK((status_a == GGML_VEC_INDEX_OK && status_b == GGML_VEC_INDEX_E_IO) ||
          (status_a == GGML_VEC_INDEX_E_IO && status_b == GGML_VEC_INDEX_OK));
    CHECK(ggml_vec_index_contains(shared_a, shared_id_a) ==
          (status_a == GGML_VEC_INDEX_OK ? 1 : 0));
    CHECK(ggml_vec_index_contains(shared_b, shared_id_b) ==
          (status_b == GGML_VEC_INDEX_OK ? 1 : 0));

    auto * shared_replayed = ggml_vec_index_load_with_delta(
        shared_snapshot_path.c_str(), shared_delta_path.c_str());
    CHECK(shared_replayed != nullptr);
    CHECK(ggml_vec_index_len(shared_replayed) == 3);
    CHECK(ggml_vec_index_contains(shared_replayed, base_ids[0]) == 1);
    CHECK(ggml_vec_index_contains(shared_replayed, base_ids[1]) == 1);
    CHECK(ggml_vec_index_contains(shared_replayed, shared_id_a) ==
          (status_a == GGML_VEC_INDEX_OK ? 1 : 0));
    CHECK(ggml_vec_index_contains(shared_replayed, shared_id_b) ==
          (status_b == GGML_VEC_INDEX_OK ? 1 : 0));

    ggml_vec_index_free(shared_replayed);
    ggml_vec_index_free(shared_a);
    ggml_vec_index_free(shared_b);
    std::filesystem::remove(shared_snapshot_path);
    std::filesystem::remove(shared_delta_path);
    std::filesystem::remove(shared_delta_path + ".lock");

    test_hardlink_delta_appends();
#ifdef _WIN32
    test_windows_command_line_quoting(argv[0]);
#endif
    test_cross_process_delta_appends(argv[0], /*use_hardlink_alias=*/false);
    test_cross_process_delta_appends(argv[0], /*use_hardlink_alias=*/true);

    ggml_vec_index_free(idx);
    std::filesystem::remove(path);
    std::filesystem::remove(delta_path);

    std::printf("test-vector-index-faults: OK\n");
    return 0;
}
