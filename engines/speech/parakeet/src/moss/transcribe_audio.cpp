#include "moss/transcribe_audio.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace parakeet::moss::detail {
namespace {

constexpr double TWO_PI = 6.283185307179586476925286766559;
constexpr float MEL_FLOOR = 1e-10f;
constexpr float DYNAMIC_RANGE = 8.0f;
constexpr float LOG_OFFSET = 4.0f;
constexpr float LOG_SCALE = 4.0f;

std::vector<std::complex<float>> twiddle_table(int size) {
    std::vector<std::complex<float>> table((size_t) size);
    for (int i = 0; i < size; ++i) {
        const double angle = -TWO_PI * (double) i / (double) size;
        table[(size_t) i] = {(float) std::cos(angle), (float) std::sin(angle)};
    }
    return table;
}

std::vector<float> periodic_hann(int size) {
    std::vector<float> window((size_t) size);
    for (int i = 0; i < size; ++i) {
        window[(size_t) i] = (float) (0.5 - 0.5 * std::cos(TWO_PI * (double) i / (double) size));
    }
    return window;
}

size_t reflect(long long index, size_t length) {
    if (index < 0) {
        return (size_t) -index;
    }
    if ((size_t) index >= length) {
        return (size_t) (2 * (long long) length - 2 - index);
    }
    return (size_t) index;
}

std::vector<float> reflect_pad(const std::vector<float> & chunk, long long pad) {
    std::vector<float> padded(chunk.size() + 2 * (size_t) pad);
    for (size_t i = 0; i < padded.size(); ++i) {
        padded[i] = chunk[reflect((long long) i - pad, chunk.size())];
    }
    return padded;
}

std::vector<float> spectrum_power(const std::vector<std::complex<float>> & spectrum, int bins) {
    std::vector<float> power((size_t) bins);
    for (size_t i = 0; i < power.size(); ++i) {
        power[i] = std::norm(spectrum[i]);
    }
    return power;
}

float band_energy(const float * filter, const float * spectrum, int bins) {
    double energy = 0.0;
    for (int bin = 0; bin < bins; ++bin) {
        energy += (double) filter[bin] * spectrum[bin];
    }
    return (float) energy;
}

float log_mel(float energy) {
    return std::log10(std::max(energy, MEL_FLOOR));
}

float clamp_and_scale(float value, float floor) {
    return (std::max(value, floor) + LOG_OFFSET) / LOG_SCALE;
}

} // namespace

TranscribeFft::TranscribeFft(int size) : size_(size), table_(twiddle_table(size)) {}

std::complex<float> TranscribeFft::twiddle(long long index, int count) const {
    return table_[(size_t) ((index % count) * (size_ / count))];
}

std::complex<float> TranscribeFft::dft_bin(const std::complex<float> * input, int stride, int count, int bin) const {
    std::complex<float> sum = 0.0f;
    for (int j = 0; j < count; ++j) {
        sum += input[(size_t) j * stride] * twiddle((long long) j * bin, count);
    }
    return sum;
}

void TranscribeFft::dft(const std::complex<float> * input, int stride, int count, std::complex<float> * output) const {
    for (int bin = 0; bin < count; ++bin) {
        output[bin] = dft_bin(input, stride, count, bin);
    }
}

void TranscribeFft::butterflies(std::complex<float> * output, int count) const {
    const int half = count / 2;
    for (int k = 0; k < half; ++k) {
        const std::complex<float> even = output[k];
        const std::complex<float> odd = twiddle(k, count) * output[k + half];
        output[k] = even + odd;
        output[k + half] = even - odd;
    }
}

void TranscribeFft::recurse(const std::complex<float> * input, int stride, int count,
                            std::complex<float> * output) const {
    if (count % 2 != 0) {
        dft(input, stride, count, output);
        return;
    }
    recurse(input, stride * 2, count / 2, output);
    recurse(input + stride, stride * 2, count / 2, output + count / 2);
    butterflies(output, count);
}

std::vector<std::complex<float>> TranscribeFft::transform(const std::vector<float> & input) const {
    const std::vector<std::complex<float>> complex_input(input.begin(), input.end());
    std::vector<std::complex<float>> output((size_t) size_);
    recurse(complex_input.data(), 1, size_, output.data());
    return output;
}

TranscribeMel::TranscribeMel(const TranscribeAudioConfig & config, std::vector<float> filters)
    : config_(config), filters_(std::move(filters)), window_(periodic_hann(config.n_fft)), fft_(config.n_fft) {
    if (filters_.size() != (size_t) config_.n_mels * bins()) {
        throw std::runtime_error("moss transcribe: mel filter bank does not match the audio geometry");
    }
}

int TranscribeMel::frames() const {
    return config_.chunk_frames;
}

int TranscribeMel::bins() const {
    return config_.n_fft / 2 + 1;
}

std::vector<float> TranscribeMel::padded_chunk(const float * samples, size_t count) const {
    const size_t length = (size_t) config_.chunk_samples;
    std::vector<float> chunk(length, 0.0f);
    std::copy(samples, samples + std::min(count, length), chunk.begin());
    return reflect_pad(chunk, config_.n_fft / 2);
}

std::vector<float> TranscribeMel::windowed_frame(const std::vector<float> & padded, int frame) const {
    std::vector<float> windowed((size_t) config_.n_fft);
    const size_t offset = (size_t) frame * config_.hop_length;
    for (size_t i = 0; i < windowed.size(); ++i) {
        windowed[i] = padded[offset + i] * window_[i];
    }
    return windowed;
}

std::vector<float> TranscribeMel::power_frame(const std::vector<float> & padded, int frame) const {
    return spectrum_power(fft_.transform(windowed_frame(padded, frame)), bins());
}

std::vector<float> TranscribeMel::power_spectrogram(const std::vector<float> & padded) const {
    std::vector<float> power((size_t) frames() * bins());
    for (int frame = 0; frame < frames(); ++frame) {
        const std::vector<float> row = power_frame(padded, frame);
        std::copy(row.begin(), row.end(), power.begin() + (std::ptrdiff_t) ((size_t) frame * bins()));
    }
    return power;
}

void TranscribeMel::project_frame(const std::vector<float> & power, int frame, std::vector<float> & mel) const {
    const float * spectrum = power.data() + (size_t) frame * bins();
    for (int band = 0; band < config_.n_mels; ++band) {
        const float * filter = filters_.data() + (size_t) band * bins();
        mel[(size_t) band * frames() + frame] = log_mel(band_energy(filter, spectrum, bins()));
    }
}

std::vector<float> TranscribeMel::mel_spectrogram(const std::vector<float> & power) const {
    std::vector<float> mel((size_t) config_.n_mels * frames());
    for (int frame = 0; frame < frames(); ++frame) {
        project_frame(power, frame, mel);
    }
    return mel;
}

std::vector<float> TranscribeMel::chunk(const float * samples, size_t count) const {
    std::vector<float> mel = mel_spectrogram(power_spectrogram(padded_chunk(samples, count)));
    normalize_log_mel(mel);
    return mel;
}

void normalize_log_mel(std::vector<float> & mel) {
    if (mel.empty()) {
        return;
    }
    const float floor = *std::max_element(mel.begin(), mel.end()) - DYNAMIC_RANGE;
    std::transform(mel.begin(), mel.end(), mel.begin(), [floor](float value) {
        return clamp_and_scale(value, floor);
    });
}

size_t transcribe_chunk_count(const TranscribeAudioConfig & config, size_t samples) {
    const size_t length = (size_t) config.chunk_samples;
    return (samples + length - 1) / length;
}

int transcribe_chunk_tokens(const TranscribeConfig & config, size_t samples) {
    if (samples == 0) {
        return 0;
    }
    return (int) ((samples - 1) / (size_t) config.samples_per_token() + 1);
}

int transcribe_audio_tokens(const TranscribeConfig & config, size_t samples) {
    const size_t length = (size_t) config.audio.chunk_samples;
    const size_t chunks = transcribe_chunk_count(config.audio, samples);
    if (chunks == 0) {
        return 0;
    }
    const size_t tail = samples - (chunks - 1) * length;
    return (int) (chunks - 1) * transcribe_chunk_tokens(config, length) + transcribe_chunk_tokens(config, tail);
}

} // namespace parakeet::moss::detail
