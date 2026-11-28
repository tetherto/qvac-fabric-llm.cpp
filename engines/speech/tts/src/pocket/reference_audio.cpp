#include "pocket/reference_audio.h"
#define DR_WAV_IMPLEMENTATION
#define DRWAV_API static
#define DRWAV_PRIVATE static
#include "dr_wav.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace tts_cpp::pocket::detail {
struct ReferenceAudio::Impl {
    std::ifstream input;
    uint64_t bytes = 0, cursor = 0;
    drwav wav{};
    bool initialized = false;
    explicit Impl(const std::string & path) : input(path, std::ios::binary | std::ios::ate) {
        if (!input) throw std::invalid_argument("cannot open reference WAV");
        const auto end = input.tellg();
        if (end <= 0 || end > 64*1024*1024)
            throw std::invalid_argument("reference WAV must be at most 64 MiB on disk");
        bytes = uint64_t(end); input.seekg(0);
        // Do not give dr_wav a tell callback: it silently clamps truncated
        // data chunks to file size. Preserve the declared size so our header
        // validation can reject truncated input before allocating PCM.
        initialized = drwav_init(&wav, read, seek, nullptr, this, nullptr);
        if (!initialized) throw std::invalid_argument("cannot read reference WAV header");
    }
    ~Impl() { if (initialized) drwav_uninit(&wav); }
    static size_t read(void * user, void * data, size_t size) {
        auto & self = *static_cast<Impl *>(user);
        size = size_t(std::min(uint64_t(size), self.bytes-self.cursor));
        self.input.read(static_cast<char *>(data), std::streamsize(size));
        const auto count = size_t(self.input.gcount()); self.cursor += count;
        return count;
    }
    static drwav_bool32 seek(void * user, int offset, drwav_seek_origin origin) {
        auto & self = *static_cast<Impl *>(user);
        const int64_t base = origin == DRWAV_SEEK_SET ? 0 :
            origin == DRWAV_SEEK_CUR ? int64_t(self.cursor) : int64_t(self.bytes);
        const int64_t target = base+offset;
        if (target < 0 || uint64_t(target) > self.bytes) return DRWAV_FALSE;
        self.input.clear(); self.input.seekg(std::streamoff(target));
        if (!self.input) return DRWAV_FALSE;
        self.cursor = uint64_t(target); return DRWAV_TRUE;
    }
};
ReferenceAudio::ReferenceAudio(const std::string & path) : impl_(new Impl(path)) {
    const auto & wav = impl_->wav;
    const auto frames = wav.totalPCMFrameCount;
    const bool pcm = wav.translatedFormatTag == DR_WAVE_FORMAT_PCM &&
        (wav.bitsPerSample == 8 || wav.bitsPerSample == 16 || wav.bitsPerSample == 24 || wav.bitsPerSample == 32);
    const bool floating = wav.translatedFormatTag == DR_WAVE_FORMAT_IEEE_FLOAT &&
        (wav.bitsPerSample == 32 || wav.bitsPerSample == 64);
    // Bound all operands before multiplying header-controlled counts.
    if ((!pcm && !floating) || wav.sampleRate < 8000 || wav.sampleRate > 192000 ||
        !frames || frames > uint64_t(wav.sampleRate)*30 || !wav.channels || wav.channels > 64 ||
        wav.dataChunkDataPos > impl_->bytes || wav.dataChunkDataSize > impl_->bytes-wav.dataChunkDataPos ||
        frames*wav.channels*(wav.bitsPerSample/8) > wav.dataChunkDataSize)
        throw std::invalid_argument("reference must be an uncompressed WAV of at most 30 seconds with valid data bounds");
}
ReferenceAudio::~ReferenceAudio() = default;
uint64_t ReferenceAudio::file_bytes() const { return impl_->bytes; }
uint64_t ReferenceAudio::frames() const { return impl_->wav.totalPCMFrameCount; }
int ReferenceAudio::sample_rate() const { return int(impl_->wav.sampleRate); }
std::vector<float> ReferenceAudio::read_mono() {
    auto & wav = impl_->wav;
    if (!drwav_seek_to_pcm_frame(&wav, 0)) throw std::invalid_argument("cannot seek reference WAV");
    std::vector<float> mono(static_cast<size_t>(frames()));
    const size_t block = 4096;
    std::vector<float> interleaved(block*wav.channels);
    for (size_t start = 0; start < mono.size();) {
        const size_t count = std::min(block, mono.size()-start);
        if (drwav_read_pcm_frames_f32(&wav, count, interleaved.data()) != count)
            throw std::invalid_argument("truncated reference WAV");
        for (size_t i = 0; i < count; ++i) {
            double sum = 0;
            for (size_t channel = 0; channel < wav.channels; ++channel) {
                const float value = interleaved[i*wav.channels+channel];
                if (!std::isfinite(value)) throw std::invalid_argument("non-finite reference WAV sample");
                sum += value;
            }
            mono[start+i] = float(sum/wav.channels);
        }
        start += count;
    }
    return mono;
}
}
