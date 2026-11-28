#include "bench_wav.h"

#include <cstdio>
#include <iterator>
#include <limits>

static bool check(bool ok, const char * message) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", message);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string path = argv[1];
    std::string error;
    bool ok = check(bench_write_wav(path, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f}, 24000, error),
                    "write valid PCM");
    std::ifstream in(path, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), {});
    in.close();
    const std::vector<unsigned char> expected = {
        'R','I','F','F',46,0,0,0,'W','A','V','E','f','m','t',' ',16,0,0,0,
        1,0,1,0,192,93,0,0,128,187,0,0,2,0,16,0,'d','a','t','a',10,0,0,0,
        1,128,0,192,0,0,0,64,255,127};
    ok &= check(bytes == expected, "PCM16 header, little-endian samples and clipping");
    ok &= check(!bench_write_wav(path, {}, 24000, error), "reject empty output");
    ok &= check(!bench_write_wav(path, {0.0f}, 0, error), "reject invalid sample rate");
    ok &= check(!bench_write_wav(path, {std::numeric_limits<float>::quiet_NaN()}, 24000, error),
                "reject NaN");
    ok &= check(!bench_write_wav(path, {std::numeric_limits<float>::infinity()}, 24000, error),
                "reject infinity");
    std::ifstream preserved(path, std::ios::binary);
    const std::vector<unsigned char> after((std::istreambuf_iterator<char>(preserved)), {});
    preserved.close();
    ok &= check(after == expected, "invalid PCM must not truncate existing output");
    // A regular file cannot be used as a parent directory on any supported OS.
    ok &= check(!bench_write_wav(path + "/output.wav", {0.0f}, 24000, error), "open failure");
#ifdef __linux__
    ok &= check(!bench_write_wav("/dev/full", {0.0f}, 24000, error), "buffered write/close failure");
#endif
    std::remove(path.c_str());
    return ok ? 0 : 1;
}
