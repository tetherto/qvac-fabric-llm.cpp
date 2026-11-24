// Model-free checks that the fixture harnesses reject corrupt output.
#include "parity_test.h"

#include <filesystem>
#include <limits>

namespace {

constexpr double kTolerance = 2e-3;
int failures = 0;

void check(bool passed, const char * message) {
    if (passed) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

template <typename T>
npy_array reference(const std::vector<T> & values, const char * dtype) {
    npy_array ref;
    ref.shape = {(int64_t) values.size()};
    ref.dtype = dtype;
    ref.data.resize(values.size() * sizeof(T));
    if (!values.empty()) std::memcpy(ref.data.data(), values.data(), ref.data.size());
    return ref;
}

void test_float_parity() {
    const std::vector<float> values = {1.0f, -2.0f, 0.5f};
    auto ref = reference(values, "<f4");
    auto passes = [&](const std::vector<float> & got) {
        return tts_test::check_f32("float gate", got, {(int64_t) got.size()}, ref, kTolerance);
    };
    check(passes(values), "identical output passes");
    check(passes({1.001f, -2.0f, 0.5f}), "rounding below tolerance passes");
    check(!passes({1.1f, -2.0f, 0.5f}), "numerical regression fails");
    check(!passes({1.0f, -2.0f}), "matching prefix with missing tail fails");
    check(!passes({1.0f, -2.0f, 0.5f, 0.0f}), "extra output fails");
    check(!passes({}), "empty output fails");
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
        check(!passes({invalid, -2.0f, 0.5f}), "non-finite output fails");
        ref = reference<float>({invalid, -2.0f, 0.5f}, "<f4");
        check(!passes(values), "non-finite reference fails");
    }
    ref = reference(values, "<i4");
    check(!passes(values), "wrong reference dtype fails");
    ref = reference(values, "<f4");
    ref.shape = {1, 3};
    check(!passes(values), "same element count with wrong shape fails");
    ref = reference(values, "<f4");
    ref.data.pop_back();
    check(!passes(values), "short reference storage fails");
    ref = reference<float>({}, "<f4");
    check(!passes({}), "two empty arrays fail");
    ref = reference<float>({0.0f}, "<f4");
    check(passes({0.0f}), "finite zero reference passes");
    check(!passes({1.0f}), "zero reference does not hide error");
}

void test_token_parity() {
    const std::vector<int32_t> tokens = {1, 23, 456};
    auto ref = reference(tokens, "<i4");
    auto passes = [&](const std::vector<int32_t> & got) {
        return tts_test::check_i32("token gate", got, {(int64_t) got.size()}, ref);
    };
    check(passes(tokens), "identical tokens pass");
    check(!passes({1, 23, 457}), "one wrong token fails");
    check(!passes({1, 23}), "matching token prefix fails");
    check(!passes({1, 23, 456, 0}), "extra token fails");
    check(!passes({}), "empty tokens fail");
    ref.dtype = "<f4";
    check(!passes(tokens), "wrong token dtype fails");
}

void expect_truncated_file(const std::filesystem::path & path, uintmax_t size, const char * field) {
    std::filesystem::resize_file(path, size);
    try {
        const auto ref = npy_load(path.string());
        const bool passed = tts_test::check_f32("truncated file", {1.0f, -2.0f, 0.0f},
                                                {3}, ref, kTolerance);
        check(!passed, "a zero-filled truncated reference must not pass parity");
        check(false, "loading a truncated file must throw");
    } catch (const std::runtime_error & error) {
        check(std::string(error.what()).find(std::string("truncated ") + field) != std::string::npos,
              "truncation identifies the incomplete field");
    }
}

void test_npy_payload_reads(const std::filesystem::path & path) {
    const std::vector<float> values = {1.0f, -2.0f, 0.0f};
    npy_save_f32(path.string(), {3}, values.data());
    const auto full_size = std::filesystem::file_size(path);
    check(tts_test::check_f32("complete file", values, {3}, npy_load(path.string()), kTolerance),
          "complete on-disk reference including a zero tail passes");
    expect_truncated_file(path, full_size - 1, "payload");
    expect_truncated_file(path, full_size - sizeof(float), "payload");
    expect_truncated_file(path, full_size - values.size() * sizeof(float), "payload");
    expect_truncated_file(path, full_size - values.size() * sizeof(float) - 1, "header");
    expect_truncated_file(path, 9, "header length");
    expect_truncated_file(path, 7, "version");
    expect_truncated_file(path, 5, "magic");
    npy_save_f32(path.string(), {0}, values.data());
    check(npy_load(path.string()).data.empty(), "complete empty payload remains readable");
    std::filesystem::remove(path);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s REFERENCE.npy\n", argv[0]); return 1; }
    test_float_parity();
    test_token_parity();
    test_npy_payload_reads(argv[1]);
    return failures == 0 ? 0 : 1;
}
