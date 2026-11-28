#include "pocket/reference_audio.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using tts_cpp::pocket::detail::ReferenceAudio;
namespace fs = std::filesystem;
void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F fn, const char * message) {
    try { fn(); } catch (const std::invalid_argument &) { return; }
    throw std::runtime_error(message);
}
void wav(const fs::path & path, unsigned format, unsigned channels, unsigned rate,
         unsigned bits, unsigned frames, const std::vector<unsigned char> & data) {
    std::ofstream f(path, std::ios::binary);
    auto put = [&](uint64_t value, int n) { for (int i = 0; i < n; ++i) f.put(char(value >> (8*i))); };
    const auto size = uint64_t(frames)*channels*(bits/8);
    f << "RIFF"; put(36+size, 4); f << "WAVEfmt "; put(16, 4);
    put(format, 2); put(channels, 2); put(rate, 4); put(uint64_t(rate)*channels*(bits/8), 4);
    put(channels*(bits/8), 2); put(bits, 2); f << "data"; put(size, 4);
    f.write(reinterpret_cast<const char *>(data.data()), data.size());
}
int main() {
    const auto dir = fs::temp_directory_path()/(
        "pocket-reference-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{dir};
    const auto path = dir/"reference.wav";
    try {
        wav(path, 1, 2, 24000, 16, 2, {0,64,0,0,0,128,0,64});
        ReferenceAudio audio(path.string());
        check(audio.frames() == 2 && audio.sample_rate() == 24000 && audio.file_bytes() == 52, "wrong metadata");
        const auto samples = audio.read_mono();
        check(samples.size() == 2 && samples[0] == 0.25f && samples[1] == -0.25f, "wrong stereo downmix");
        check(audio.read_mono() == samples, "second read changed PCM");
#ifndef _WIN32
        // A path replacement after validation cannot substitute unchecked data.
        fs::rename(path, dir/"original.wav");
        wav(path, 1, 1, 24000, 16, 1, {0,0});
        check(audio.read_mono() == samples, "decode reopened replaced path");
#endif
        wav(path, 1, 1, 24000, 16, 1000000000, {0,0});
        rejects([&] { ReferenceAudio invalid(path.string()); }, "oversized frame count accepted");
        wav(path, 1, 1, 24000, 16, 100, {0,0});
        rejects([&] { ReferenceAudio invalid(path.string()); }, "truncated data accepted");
        wav(path, 1, 1, 4000, 16, 1, {0,0});
        rejects([&] { ReferenceAudio invalid(path.string()); }, "unsupported rate accepted");
        wav(path, 1, 65, 24000, 8, 1, std::vector<unsigned char>(65));
        rejects([&] { ReferenceAudio invalid(path.string()); }, "excess channels accepted");
        wav(path, 3, 1, 24000, 32, 1, {0,0,128,127}); // infinity
        ReferenceAudio floating(path.string());
        rejects([&] { floating.read_mono(); }, "non-finite float PCM accepted");
        // RF64's sample count overrides its tiny physical data chunk.
        {
            std::ofstream f(path, std::ios::binary);
            const std::string hex = "52463634ffffffff57415645647336341c0000004a000000000000000200"
                "000000000000000000400000000000000000666d74201000000001000100"
                "c05d000080bb00000200100064617461ffffffff0000";
            for (size_t i = 0; i < hex.size(); i += 2) f.put(char(std::stoul(hex.substr(i, 2), nullptr, 16)));
        }
        rejects([&] { ReferenceAudio invalid(path.string()); }, "RF64 allocation bomb accepted");
        std::cout << "Pocket bounded WAV header, decode, downmix, repeat and malformed-input tests passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
