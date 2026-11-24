#pragma once

// I/O and argument validation for the mm3-replay parity driver, separated from
// the CLI so the success and failure paths are unit-testable without model
// weights. Every writer reports failure instead of silently producing no
// artifact.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

// Rejects missing, empty, and non-element-aligned files: a truncated recording
// must fail the replay instead of silently dropping trailing bytes, and an
// empty forced-code file must not degrade replay mode into sampled generation.
template <typename T>
static bool mm3_replay_read_raw(const std::string & path, std::vector<T> & data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || (size_t) bytes % sizeof(T) != 0) {
        return false;
    }
    file.seekg(0);
    data.resize((size_t) bytes / sizeof(T));
    file.read(reinterpret_cast<char *>(data.data()), bytes);
    return file.good();
}

template <typename T>
static bool mm3_replay_write_raw(const std::string & path, const T * data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.write(reinterpret_cast<const char *>(data), (std::streamsize) (count * sizeof(T)));
    file.flush();
    return file.good();
}

static void mm3_replay_interleave_pcm(const std::vector<float> & planar, int64_t samples,
                                      std::vector<int16_t> & pcm) {
    pcm.resize((size_t) samples * 2);
    for (int64_t i = 0; i < samples; ++i) {
        const float l = planar[(size_t) i];
        const float r = planar[(size_t) (samples + i)];
        pcm[(size_t) i * 2]     = (int16_t) (std::max(-1.0f, std::min(1.0f, l)) * 32767.0f);
        pcm[(size_t) i * 2 + 1] = (int16_t) (std::max(-1.0f, std::min(1.0f, r)) * 32767.0f);
    }
}

static bool mm3_replay_write_wav(const std::string & path, const std::vector<float> & planar,
                                 int64_t samples, int rate) {
    std::vector<int16_t> pcm;
    mm3_replay_interleave_pcm(planar, samples, pcm);
    const uint32_t data_bytes = (uint32_t) (pcm.size() * sizeof(int16_t));
    const uint32_t byte_rate  = (uint32_t) rate * 2 * 2;
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    uint32_t u32;
    uint16_t u16;
    file.write("RIFF", 4);
    u32 = 36 + data_bytes;      file.write((char *) &u32, 4);
    file.write("WAVEfmt ", 8);
    u32 = 16;                   file.write((char *) &u32, 4);
    u16 = 1;                    file.write((char *) &u16, 2);
    u16 = 2;                    file.write((char *) &u16, 2);
    u32 = (uint32_t) rate;      file.write((char *) &u32, 4);
    u32 = byte_rate;            file.write((char *) &u32, 4);
    u16 = 4;                    file.write((char *) &u16, 2);
    u16 = 16;                   file.write((char *) &u16, 2);
    file.write("data", 4);
    file.write((char *) &data_bytes, 4);
    file.write((const char *) pcm.data(), data_bytes);
    file.flush();
    return file.good();
}

static bool mm3_replay_mode_is_supported(const std::string & mode) {
    return mode == "full" || mode == "replay" || mode == "condcheck";
}

static bool mm3_replay_prepare_output_dir(const std::string & path, std::string * error) {
    std::error_code code;
    std::filesystem::create_directories(path, code);
    if (code) {
        if (error) {
            *error = "cannot create output directory '" + path + "': " + code.message();
        }
        return false;
    }
    if (!std::filesystem::is_directory(path, code) || code) {
        if (error) {
            *error = "output path '" + path + "' is not a directory";
        }
        return false;
    }
    return true;
}
