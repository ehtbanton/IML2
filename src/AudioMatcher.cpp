#define _USE_MATH_DEFINES
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "AudioMatcher.h"
#include "SpectrogramCache.h"
#include <fftw3.h>
#include <iostream>
#include <fstream>
#include <windows.h>
#include <ctime>
#include <cmath>
#include <SFML/Audio.hpp>

// Function to normalize audio to a consistent RMS level
void normalizeAudio(std::vector<float>& samples, float targetRMS = 0.1f) {
    // Calculate current RMS
    float sumSquares = 0.0f;
    for (const float& sample : samples) {
        sumSquares += sample * sample;
    }

    if (samples.size() > 0) {
        float currentRMS = std::sqrt(sumSquares / static_cast<float>(samples.size()));

        // Avoid division by very small numbers
        if (currentRMS > 1e-6f) {
            // Calculate scaling factor
            float scale = targetRMS / currentRMS;

            // Apply scaling
            for (float& sample : samples) {
                sample *= scale;
            }
        }
    }
}

// AudioAnalyzer implementations
const std::vector<std::vector<float>>& AudioAnalyzer::getReferenceSpectogram() const {
    return reference_spectogram;
}

void AudioAnalyzer::setReferenceSpectogram(const std::vector<std::vector<float>>& data) {
    reference_spectogram = data;
}

AudioAnalyzer::AudioAnalyzer() : fft_in(nullptr), fft_out(nullptr), fft_plan(nullptr) {
    initializeFFT();
}

AudioAnalyzer::~AudioAnalyzer() {
    if (fft_plan) fftw_destroy_plan(fft_plan);
    if (fft_in) fftw_free(fft_in);
    if (fft_out) fftw_free(fft_out);
}

void AudioAnalyzer::initializeFFT() {
    // Initialize FFTW
    fft_in = fftw_alloc_real(FFT_SIZE);
    fft_out = fftw_alloc_complex(FFT_SIZE / 2 + 1);
    fft_plan = fftw_plan_dft_r2c_1d(static_cast<int>(FFT_SIZE), fft_in, fft_out, FFTW_MEASURE);

    // Create Hanning window
    window_function.resize(FFT_SIZE);
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        window_function[i] = 0.5 * (1.0 - cos(2.0 * M_PI * static_cast<double>(i) / (static_cast<double>(FFT_SIZE) - 1.0)));
    }
}

bool AudioAnalyzer::analyzeFile(const std::string& filepath) {
    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile(filepath)) {
        std::cerr << "Failed to load audio file: " << filepath << std::endl;
        return false;
    }

    const sf::Int16* samples = buffer.getSamples();
    size_t sampleCount = buffer.getSampleCount();
    unsigned int channels = buffer.getChannelCount();
    unsigned int sampleRate = buffer.getSampleRate(); // Get sample rate from buffer

    std::vector<float> monoSamples;
    monoSamples.reserve(sampleCount / channels);

    // Convert to mono and normalize to float range [-1.0, 1.0]
    for (size_t i = 0; i < sampleCount; i += channels) {
        float sum = 0.0f;
        for (unsigned int ch = 0; ch < channels; ++ch) {
            sum += static_cast<float>(samples[i + ch]);
        }
        monoSamples.push_back(sum / (static_cast<float>(channels) * 32768.0f));
    }

    // Normalize the entire audio file to a consistent RMS level
    normalizeAudio(monoSamples);

    reference_spectogram.clear();
    const float scale = 1.0f; // Reduced from 10.0f since we're already normalized

    // Process the entire audio file
    for (size_t i = 0; i + FFT_SIZE <= monoSamples.size(); i += HOP_SIZE) {
        for (size_t j = 0; j < FFT_SIZE; ++j) {
            fft_in[j] = static_cast<double>(monoSamples[i + j] * window_function[j] * scale);
        }

        fftw_execute(fft_plan);

        std::vector<float> magnitudes(FFT_SIZE / 2);
        for (size_t j = 0; j < FFT_SIZE / 2; ++j) {
            float real = static_cast<float>(fft_out[j][0]);
            float imag = static_cast<float>(fft_out[j][1]);
            magnitudes[j] = std::sqrt(real * real + imag * imag) / (static_cast<float>(FFT_SIZE) * 0.5f); // Adjusted normalization
        }

        reference_spectogram.push_back(magnitudes);
    }

    std::cout << "Static Spectrogram: Processed " << reference_spectogram.size() << " frames (approx. "
        << (static_cast<double>(reference_spectogram.size() * HOP_SIZE) / sampleRate) << " seconds)" << std::endl;

    return true;
}

// AudioMatcher implementation
AudioMatcher::AudioMatcher(const std::string& audio_file)
    : audio_path(audio_file)
    , cache_path(audio_file + ".cache")
    , is_initialized(false)
    , match_attempts(0)
    , match_hits(0)
    , average_confidence(0.0f)
    , last_stats_time(std::chrono::steady_clock::now())
{
    // Create cache object
    cache = new SpectrogramCache();

    // Set fingerprinter debug level
    fingerprinter.setDebugLevel(AudioFingerprinter::DebugLevel::Basic);

    std::cout << "Audio Matcher initialized with file: " << audio_file << std::endl;
}

AudioMatcher::~AudioMatcher() {
    delete cache;
}

const std::vector<std::vector<float>>& AudioMatcher::getStaticSpectrogram() const {
    return cache->spectogram_data;
}

bool AudioMatcher::needsUpdate() {
    // Check if cache file exists
    std::ifstream cache_test(cache_path);
    if (!cache_test.good()) {
        return true;
    }
    cache_test.close();

    // Check if audio file exists
    std::ifstream audio_test(audio_path);
    if (!audio_test.good()) {
        return false;  // Audio file doesn't exist, can't update
    }
    audio_test.close();

    // Get file modification times using Windows API
    WIN32_FILE_ATTRIBUTE_DATA audioAttrib, cacheAttrib;
    if (!GetFileAttributesExA(audio_path.c_str(), GetFileExInfoStandard, &audioAttrib) ||
        !GetFileAttributesExA(cache_path.c_str(), GetFileExInfoStandard, &cacheAttrib)) {
        return true;  // If we can't get attributes, assume we need to update
    }

    ULARGE_INTEGER audioTime, cacheTime;
    audioTime.LowPart = audioAttrib.ftLastWriteTime.dwLowDateTime;
    audioTime.HighPart = audioAttrib.ftLastWriteTime.dwHighDateTime;
    cacheTime.LowPart = cacheAttrib.ftLastWriteTime.dwLowDateTime;
    cacheTime.HighPart = cacheAttrib.ftLastWriteTime.dwHighDateTime;

    return audioTime.QuadPart > cacheTime.QuadPart;
}

bool AudioMatcher::initialize() {
    if (is_initialized) return true;

    // Check if audio file exists
    std::ifstream audio_test(audio_path);
    if (!audio_test.good()) {
        std::cerr << "Audio file not found: " << audio_path << std::endl;
        return false;
    }
    audio_test.close();

    bool needs_analysis = needsUpdate();

    if (needs_analysis) {
        std::cout << "Analyzing audio file..." << std::endl;

        if (!analyzer.analyzeFile(audio_path)) {
            std::cerr << "Failed to analyze audio file" << std::endl;
            return false;
        }

        // Store analysis results in cache
        cache->timestamp = std::time(nullptr);
        cache->spectogram_data = analyzer.getReferenceSpectogram();

        if (!cache->serialize(cache_path)) {
            std::cerr << "Failed to save cache file" << std::endl;
            return false;
        }

        std::cout << "Analysis complete and cached" << std::endl;
    }
    else {
        std::cout << "Loading cached analysis..." << std::endl;

        if (!cache->deserialize(cache_path)) {
            std::cerr << "Failed to load cache file, will re-analyze" << std::endl;
            return initialize();
        }
    }

    // Initialize the fingerprinter with the spectral data
    fingerprinter.setReferenceData(cache->spectogram_data);

    // Run self-test to ensure the fingerprinter is working correctly
    std::cout << "Running fingerprinter self-test..." << std::endl;
    if (!runSelfTest()) {
        std::cerr << "Warning: Fingerprinter self-test failed" << std::endl;
        // Continue anyway, might still work
    }

    is_initialized = true;
    return true;
}

// Returns {position, confidence}
std::pair<size_t, float> AudioMatcher::findMatchWithConfidence(const std::vector<float>& current_magnitudes) {
    std::lock_guard<std::mutex> lock(matcher_mutex);

    if (!is_initialized) {
        std::cerr << "AudioMatcher not initialized" << std::endl;
        return { 0, 0.0f };
    }

    match_attempts++;

    auto result = fingerprinter.findMatchWithConfidence(current_magnitudes);

    // Update statistics
    if (result.second > 0.2f) {
        match_hits++;
        average_confidence = average_confidence * 0.95f + result.second * 0.05f;
    }

    // Periodically print stats
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 10) {
        printStats();
        last_stats_time = now;
    }

    return result;
}

// Legacy method for backward compatibility
size_t AudioMatcher::findMatch(const std::vector<float>& current_magnitudes) {
    return findMatchWithConfidence(current_magnitudes).first;
}

double AudioMatcher::getTimestamp(size_t position) const {
    return position * (static_cast<double>(512) / 48000.0);  // HOP_SIZE / SAMPLE_RATE
}

bool AudioMatcher::runSelfTest() {
    return fingerprinter.runSelfTest();
}

void AudioMatcher::printStats() {
    float hit_rate = 0.0f;
    if (match_attempts > 0) {
        hit_rate = static_cast<float>(match_hits) / static_cast<float>(match_attempts) * 100.0f;
    }

    std::cout << "--- Audio Matcher Stats ---" << std::endl;
    std::cout << "  Match attempts: " << match_attempts << std::endl;
    std::cout << "  Match hits: " << match_hits << " (" << hit_rate << "%)" << std::endl;
    std::cout << "  Average confidence: " << average_confidence << std::endl;
}