#include "wav_reader.h"

#include <cstdint>
#include <cstring>

namespace tts_cpp::acestep {

static constexpr uint16_t WAV_PCM_FORMAT      = 1;
static constexpr uint16_t WAV_PCM_BITS        = 16;
static constexpr uint16_t WAV_MONO_CHANNELS   = 1;
static constexpr uint16_t WAV_STEREO_CHANNELS = 2;
static constexpr float    PCM16_SCALE         = 32768.0f;

struct WavFormat {
    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint32_t sample_rate = 0;
};

template <typename T>
static bool read_value(FILE * file, T & value) {
    return fread(&value, sizeof(T), 1, file) == 1;
}

static bool read_tag(FILE * file, char (&tag)[4]) {
    return fread(tag, 1, sizeof(tag), file) == sizeof(tag);
}

static bool tag_equals(const char (&tag)[4], const char * expected) {
    return memcmp(tag, expected, sizeof(tag)) == 0;
}

static bool skip_bytes(FILE * file, uint32_t bytes) {
    return bytes == 0 || fseek(file, (long) bytes, SEEK_CUR) == 0;
}

static bool read_riff_header(FILE * file) {
    char riff[4];
    char wave[4];
    uint32_t size = 0;
    return read_tag(file, riff) && read_value(file, size) && read_tag(file, wave) &&
           tag_equals(riff, "RIFF") && tag_equals(wave, "WAVE");
}

static bool read_format_chunk(FILE * file, uint32_t size, WavFormat & format) {
    constexpr uint32_t PCM_FORMAT_SIZE = 16;
    if (size < PCM_FORMAT_SIZE) return false;

    uint32_t byte_rate = 0;
    uint16_t block_align = 0;
    if (!read_value(file, format.format) ||
        !read_value(file, format.channels) ||
        !read_value(file, format.sample_rate) ||
        !read_value(file, byte_rate) ||
        !read_value(file, block_align) ||
        !read_value(file, format.bits)) {
        return false;
    }
    return skip_bytes(file, size - PCM_FORMAT_SIZE);
}

static bool wav_format_supported(const WavFormat & format) {
    return format.format == WAV_PCM_FORMAT && format.bits == WAV_PCM_BITS &&
           (format.channels == WAV_MONO_CHANNELS || format.channels == WAV_STEREO_CHANNELS);
}

static std::vector<float> convert_pcm16_to_stereo(const std::vector<int16_t> & input,
                                                  int frames, int channels) {
    std::vector<float> output((size_t) frames * WAV_STEREO_CHANNELS);
    for (int frame = 0; frame < frames; ++frame) {
        const float left = input[(size_t) frame * channels] / PCM16_SCALE;
        const float right =
            channels == WAV_STEREO_CHANNELS ? input[(size_t) frame * channels + 1] / PCM16_SCALE : left;
        output[(size_t) frame * WAV_STEREO_CHANNELS] = left;
        output[(size_t) frame * WAV_STEREO_CHANNELS + 1] = right;
    }
    return output;
}

static WavReadResult read_data_chunk(FILE * file, uint32_t size, const WavFormat & format) {
    if (!wav_format_supported(format)) {
        return { {}, 0, 0, "WAV must be PCM16 mono or stereo" };
    }

    const size_t sample_count = size / sizeof(int16_t);
    std::vector<int16_t> input(sample_count);
    const size_t samples_read = fread(input.data(), sizeof(int16_t), sample_count, file);
    if (samples_read != sample_count || sample_count % format.channels != 0) {
        return { {}, 0, 0, "invalid WAV data chunk" };
    }
    const int frames = (int) (samples_read / format.channels);
    return {
        convert_pcm16_to_stereo(input, frames, format.channels),
        frames,
        (int) format.sample_rate,
        {}
    };
}

static WavReadResult read_wav_chunks(FILE * file) {
    WavFormat format;
    while (!feof(file)) {
        char tag[4];
        uint32_t size = 0;
        if (!read_tag(file, tag) || !read_value(file, size)) break;

        if (tag_equals(tag, "fmt ")) {
            if (!read_format_chunk(file, size, format)) {
                return { {}, 0, 0, "invalid WAV format chunk" };
            }
        } else if (tag_equals(tag, "data")) {
            return read_data_chunk(file, size, format);
        } else if (!skip_bytes(file, size)) {
            return { {}, 0, 0, "invalid WAV chunk" };
        }

        if ((size & 1u) != 0 && !skip_bytes(file, 1)) {
            return { {}, 0, 0, "invalid WAV padding" };
        }
    }
    return { {}, 0, 0, "WAV has no data chunk" };
}

WavReadResult read_pcm16_wav(FILE * file) {
    if (!file) return { {}, 0, 0, "cannot open WAV" };
    rewind(file);
    if (!read_riff_header(file)) return { {}, 0, 0, "file is not RIFF/WAVE" };
    return read_wav_chunks(file);
}

WavReadResult load_pcm16_wav(const char * path) {
    FILE * file = fopen(path, "rb");
    if (!file) return { {}, 0, 0, "cannot open WAV" };
    WavReadResult result = read_pcm16_wav(file);
    fclose(file);
    return result;
}

} // namespace tts_cpp::acestep
