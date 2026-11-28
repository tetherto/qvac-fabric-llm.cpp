#pragma once

#include "moss/transcribe_model.h"

#include <complex>
#include <cstddef>
#include <vector>

namespace parakeet::moss::detail {

class TranscribeFft {
public:
    explicit TranscribeFft(int size);

    std::vector<std::complex<float>> transform(const std::vector<float> & input) const;

private:
    void recurse(const std::complex<float> * input, int stride, int count, std::complex<float> * output) const;
    void dft(const std::complex<float> * input, int stride, int count, std::complex<float> * output) const;
    std::complex<float> dft_bin(const std::complex<float> * input, int stride, int count, int bin) const;
    void butterflies(std::complex<float> * output, int count) const;
    std::complex<float> twiddle(long long index, int count) const;

    int size_;
    std::vector<std::complex<float>> table_;
};

class TranscribeMel {
public:
    TranscribeMel(const TranscribeAudioConfig & config, std::vector<float> filters);

    std::vector<float> chunk(const float * samples, size_t count) const;
    int frames() const;
    int bins() const;

private:
    std::vector<float> padded_chunk(const float * samples, size_t count) const;
    std::vector<float> windowed_frame(const std::vector<float> & padded, int frame) const;
    std::vector<float> power_frame(const std::vector<float> & padded, int frame) const;
    std::vector<float> power_spectrogram(const std::vector<float> & padded) const;
    void project_frame(const std::vector<float> & power, int frame, std::vector<float> & mel) const;
    std::vector<float> mel_spectrogram(const std::vector<float> & power) const;

    TranscribeAudioConfig config_;
    std::vector<float> filters_;
    std::vector<float> window_;
    TranscribeFft fft_;
};

size_t transcribe_chunk_count(const TranscribeAudioConfig & config, size_t samples);
int transcribe_chunk_tokens(const TranscribeConfig & config, size_t samples);
int transcribe_audio_tokens(const TranscribeConfig & config, size_t samples);
void normalize_log_mel(std::vector<float> & mel);

} // namespace parakeet::moss::detail
