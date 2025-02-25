#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include "AudioFingerprinter.h"
#include "SpectrogramCache.h"

// AudioAnalyzer class definition
class AudioAnalyzer {
public:
    AudioAnalyzer();
    ~AudioAnalyzer();

    // Remove inline implementations to avoid "already has a body" errors
    const std::vector<std::vector<float>>& getReferenceSpectogram() const;
    void setReferenceSpectogram(const std::vector<std::vector<float>>& data);

    bool analyzeFile(const std::string& filepath);

private:
    static const size_t FFT_SIZE = 1024;
    static const size_t HOP_SIZE = 512;
    std::vector<double> window_function;

    // FFTW variables
    double* fft_in;
    fftw_complex* fft_out;
    fftw_plan fft_plan;

    // Store frequency data for the reference audio
    std::vector<std::vector<float>> reference_spectogram;

    void initializeFFT();
};

class AudioMatcher {
private:
    AudioAnalyzer analyzer;
    AudioFingerprinter fingerprinter;
    SpectrogramCache* cache;
    std::string cache_path;
    std::string audio_path;
    bool is_initialized;
    mutable std::mutex matcher_mutex;

    // Statistics
    std::atomic<size_t> match_attempts;
    std::atomic<size_t> match_hits;
    std::atomic<float> average_confidence;
    std::chrono::steady_clock::time_point last_stats_time;

    bool needsUpdate();

public:
    AudioMatcher(const std::string& audio_file);
    ~AudioMatcher();

    // Get reference spectrogram
    const std::vector<std::vector<float>>& getStaticSpectrogram() const;

    // Initialize analyzer and fingerprinter
    bool initialize();

    // Find match with confidence
    std::pair<size_t, float> findMatchWithConfidence(const std::vector<float>& current_magnitudes);

    // Legacy method for backward compatibility
    size_t findMatch(const std::vector<float>& current_magnitudes);

    // Convert frame position to time in seconds
    double getTimestamp(size_t position) const;

    // Run self-test
    bool runSelfTest();

    // Print statistics about the matching process - just declaration
    void printStats();
};