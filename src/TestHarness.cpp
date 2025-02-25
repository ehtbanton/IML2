#define _USE_MATH_DEFINES
#include "TestHarness.h"
#include "AudioMatcher.h"
#include <fftw3.h>
#include <SFML/Audio.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

// TestSignalGenerator implementation
TestSignalGenerator::TestSignalGenerator() : position(0) {
    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));
}

std::vector<float> TestSignalGenerator::generateChirp(float startFreq, float endFreq, float duration) {
    size_t numSamples = static_cast<size_t>(duration * SAMPLE_RATE);
    std::vector<float> result(numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / SAMPLE_RATE;
        float phase = 2.0f * static_cast<float>(M_PI) * (startFreq * t +
            (endFreq - startFreq) * t * t / (2.0f * duration));
        result[i] = std::sin(phase) * 0.5f;  // 0.5 amplitude
    }

    return result;
}

std::vector<float> TestSignalGenerator::generateNoise(float duration, float amplitude) {
    size_t numSamples = static_cast<size_t>(duration * SAMPLE_RATE);
    std::vector<float> result(numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        float randVal = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        result[i] = randVal * amplitude;
    }

    return result;
}

std::vector<float> TestSignalGenerator::generateSineWave(float freq, float duration, float amplitude) {
    size_t numSamples = static_cast<size_t>(duration * SAMPLE_RATE);
    std::vector<float> result(numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / SAMPLE_RATE;
        result[i] = std::sin(2.0f * static_cast<float>(M_PI) * freq * t) * amplitude;
    }

    return result;
}

std::vector<float> TestSignalGenerator::generateMelody(float duration) {
    std::vector<float> result;

    // Define some notes (frequencies in Hz)
    float notes[] = { 261.63f, 293.66f, 329.63f, 349.23f, 392.00f, 440.00f, 493.88f, 523.25f };
    float noteDuration = 0.2f;  // 200ms per note

    size_t numNotes = static_cast<size_t>(duration / noteDuration);
    for (size_t i = 0; i < numNotes; ++i) {
        float note = notes[i % 8];
        auto noteSound = generateSineWave(note, noteDuration, 0.4f);
        result.insert(result.end(), noteSound.begin(), noteSound.end());
    }

    return result;
}

void TestSignalGenerator::applyHannWindow(double* output, const std::vector<float>& input, size_t start, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        double window = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
        output[i] = static_cast<double>(input[start + i]) * window;
    }
}

void TestSignalGenerator::processSpectrogram() {
    // Initialize FFTW
    double* fft_in = fftw_alloc_real(FFT_SIZE);
    fftw_complex* fft_out = fftw_alloc_complex(FFT_SIZE / 2 + 1);
    fftw_plan fft_plan = fftw_plan_dft_r2c_1d(FFT_SIZE, fft_in, fft_out, FFTW_ESTIMATE);

    spectrogram.clear();

    for (size_t i = 0; i + FFT_SIZE <= samples.size(); i += HOP_SIZE) {
        // Apply window function
        applyHannWindow(fft_in, samples, i, FFT_SIZE);

        // Execute FFT
        fftw_execute(fft_plan);

        // Calculate magnitudes
        std::vector<float> magnitudes(FFT_SIZE / 2);
        for (size_t j = 0; j < FFT_SIZE / 2; ++j) {
            float real = static_cast<float>(fft_out[j][0]);
            float imag = static_cast<float>(fft_out[j][1]);
            magnitudes[j] = std::sqrt(real * real + imag * imag) / (FFT_SIZE * 2.0f);
        }

        spectrogram.push_back(magnitudes);
    }

    // Clean up FFTW
    fftw_destroy_plan(fft_plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
}

void TestSignalGenerator::generateTestSignal(float duration) {
    samples.clear();

    // Create a complex test signal with different patterns
    auto chirpUp = generateChirp(100.0f, 10000.0f, 5.0f);
    auto chirpDown = generateChirp(10000.0f, 100.0f, 5.0f);
    auto noise = generateNoise(5.0f);
    auto sine440 = generateSineWave(440.0f, 5.0f);
    auto sine880 = generateSineWave(880.0f, 5.0f);
    auto melody = generateMelody(5.0f);

    // Combine patterns into one test signal
    samples.insert(samples.end(), chirpUp.begin(), chirpUp.end());
    samples.insert(samples.end(), sine440.begin(), sine440.end());
    samples.insert(samples.end(), chirpDown.begin(), chirpDown.end());
    samples.insert(samples.end(), noise.begin(), noise.end());
    samples.insert(samples.end(), melody.begin(), melody.end());
    samples.insert(samples.end(), sine880.begin(), sine880.end());

    // Process to create spectrogram
    processSpectrogram();

    std::cout << "Generated test signal: " << samples.size() << " samples, "
        << spectrogram.size() << " spectrogram frames" << std::endl;
}

std::vector<float> TestSignalGenerator::getCurrentFrame() {
    if (spectrogram.empty()) {
        return std::vector<float>();
    }

    std::vector<float> frame = spectrogram[position];
    position = (position + 1) % spectrogram.size();
    return frame;
}

void TestSignalGenerator::reset() {
    position = 0;
}

const std::vector<std::vector<float>>& TestSignalGenerator::getSpectrogram() const {
    return spectrogram;
}

bool TestSignalGenerator::saveToWAV(const std::string& filename) {
    sf::SoundBuffer buffer;

    // Convert float samples to Int16
    std::vector<sf::Int16> int16Samples(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        int16Samples[i] = static_cast<sf::Int16>(samples[i] * 32767.0f);
    }

    if (!buffer.loadFromSamples(int16Samples.data(), int16Samples.size(),
        CHANNEL_COUNT, SAMPLE_RATE)) {
        std::cerr << "Failed to create sound buffer" << std::endl;
        return false;
    }

    if (!buffer.saveToFile(filename)) {
        std::cerr << "Failed to save sound buffer to " << filename << std::endl;
        return false;
    }

    std::cout << "Saved test signal to " << filename << std::endl;
    return true;
}

// AudioMatchTestHarness implementation
AudioMatchTestHarness::AudioMatchTestHarness(AudioMatcher* m, bool useTestSignal)
    : matcher(m), useTestSignal(useTestSignal), testCounter(0), runningTest(false)
{
    testGen = new TestSignalGenerator();

    if (useTestSignal) {
        testGen->generateTestSignal(30.0f);
        testGen->saveToWAV("test_signal.wav");
    }
}

AudioMatchTestHarness::~AudioMatchTestHarness() {
    delete testGen;
}

void AudioMatchTestHarness::startTest() {
    if (runningTest) return;

    // Clear previous test results
    testPositions.clear();
    confidenceValues.clear();
    testCounter = 0;

    // Reset test generator
    testGen->reset();

    runningTest = true;
    std::cout << "Starting automated test sequence..." << std::endl;
}

void AudioMatchTestHarness::stopTest() {
    if (!runningTest) return;

    runningTest = false;

    // Calculate statistics
    if (testPositions.size() < 2) {
        std::cout << "Test too short to generate statistics." << std::endl;
        return;
    }

    float avgConfidence = 0.0f;
    for (float conf : confidenceValues) {
        avgConfidence += conf;
    }
    avgConfidence /= confidenceValues.size();

    // Calculate position consistency
    bool consistentPositions = true;
    size_t lastPos = testPositions[0];
    for (size_t i = 1; i < testPositions.size(); i++) {
        size_t expectedAdvance = 1; // Should advance by 1 frame per test
        size_t actualAdvance = 0;

        if (testPositions[i] >= lastPos) {
            actualAdvance = testPositions[i] - lastPos;
        }
        else {
            // Handle wrap-around or jump backward
            actualAdvance = 0;  // Consider it as no advance
        }

        if (std::abs(static_cast<int>(actualAdvance) - static_cast<int>(expectedAdvance)) > 5) {
            consistentPositions = false;
            std::cout << "Position jump detected at test " << i << ": from "
                << lastPos << " to " << testPositions[i] << std::endl;
        }

        lastPos = testPositions[i];
    }

    std::cout << "Test complete. Results:" << std::endl;
    std::cout << "  Tests run: " << testPositions.size() << std::endl;
    std::cout << "  Average confidence: " << avgConfidence << std::endl;
    std::cout << "  Position consistency: " << (consistentPositions ? "GOOD" : "POOR") << std::endl;
}

std::pair<size_t, float> AudioMatchTestHarness::processTestFrame() {
    if (!runningTest) {
        return { 0, 0.0f };
    }

    // Get test frame
    std::vector<float> testFrame = testGen->getCurrentFrame();
    if (testFrame.empty()) {
        stopTest();
        return { 0, 0.0f };
    }

    // Run matcher
    auto result = matcher->findMatchWithConfidence(testFrame);

    // Record results
    testPositions.push_back(result.first);
    confidenceValues.push_back(result.second);
    testCounter++;

    // Log progress
    if (testCounter % 30 == 0) {
        std::cout << "Test progress: " << testCounter << " frames, current position: "
            << result.first << ", confidence: " << result.second << std::endl;
    }

    return result;
}

std::vector<float> AudioMatchTestHarness::getTestFrame() {
    return testGen->getCurrentFrame();
}

bool AudioMatchTestHarness::isRunningTest() const {
    return runningTest;
}