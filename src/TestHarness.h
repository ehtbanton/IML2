#pragma once

// First define these to prevent Windows macro conflicts
#define NOMINMAX 
#define WIN32_LEAN_AND_MEAN

// Include SFML before Windows headers if needed
#include <SFML/System.hpp>

// Standard library headers
#include <vector>
#include <utility>
#include <string>

// Forward declarations
class AudioMatcher;
class TestSignalGenerator;

class AudioMatchTestHarness {
private:
    AudioMatcher* matcher;
    TestSignalGenerator* testGen;
    bool useTestSignal;

    // Test parameters
    std::vector<size_t> testPositions;
    std::vector<float> confidenceValues;
    size_t testCounter;
    bool runningTest;

public:
    AudioMatchTestHarness(AudioMatcher* m, bool useTestSignal = false);
    ~AudioMatchTestHarness();

    // Start automated test sequence
    void startTest();

    // Stop test and report results
    void stopTest();

    // Process one test frame
    std::pair<size_t, float> processTestFrame();

    // Get test frame for external use
    std::vector<float> getTestFrame();

    bool isRunningTest() const;
};

// Test signal generator
class TestSignalGenerator {
private:
    static const size_t FFT_SIZE = 1024;
    static const size_t HOP_SIZE = 512;
    static const size_t SAMPLE_RATE = 48000;
    static const size_t CHANNEL_COUNT = 1;  // Mono

    std::vector<float> samples;
    std::vector<std::vector<float>> spectrogram;
    size_t position;

    // Generate different test signals
    std::vector<float> generateChirp(float startFreq, float endFreq, float duration);
    std::vector<float> generateNoise(float duration, float amplitude = 0.1f);
    std::vector<float> generateSineWave(float freq, float duration, float amplitude = 0.5f);
    std::vector<float> generateMelody(float duration);

    // Apply a window function to sample frames
    void applyHannWindow(double* output, const std::vector<float>& input, size_t start, size_t size);

    // Process samples to generate spectrogram
    void processSpectrogram();

public:
    TestSignalGenerator();

    // Generate test signal with various patterns
    void generateTestSignal(float duration = 30.0f);

    // Get current frame of spectrogram
    std::vector<float> getCurrentFrame();

    // Reset position to beginning
    void reset();

    // Get entire spectrogram
    const std::vector<std::vector<float>>& getSpectrogram() const;

    // Save samples to WAV file for reference
    bool saveToWAV(const std::string& filename);
};