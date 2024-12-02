// AudioAnalyzer.cpp
#include "AudioAnalyzer.hpp"
#include <iostream>
#include <algorithm>

AudioAnalyzer::AudioAnalyzer(size_t bufferSize)
    : audioCapture(std::make_unique<WasapiCapture>(bufferSize))
    , processBuffer(bufferSize, 0.0f)
    , fftBuffer(FFT_SIZE)
    , windowFunction(FFT_SIZE)
    , currentVolume(0.0f)
    , currentCentroid(0.0f)
    , volumeSmoothing(0.8f)
    , centroidSmoothing(0.7f)
{
    initializeWindowFunction();
}

AudioAnalyzer::~AudioAnalyzer() {
    stop();
}

void AudioAnalyzer::initializeWindowFunction() {
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        windowFunction[i] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / (FFT_SIZE - 1))));
    }
}

bool AudioAnalyzer::start() {
    return audioCapture->start();
}

void AudioAnalyzer::stop() {
    audioCapture->stop();
}

void AudioAnalyzer::update() {
    if (audioCapture->getBuffer(processBuffer)) {
        float rmsLevel = calculateRMS();
        calculateFFT();
        float centroid = calculateSpectralCentroid();

        currentVolume = smoothValue(currentVolume, rmsLevel, volumeSmoothing);
        currentCentroid = smoothValue(currentCentroid, centroid, centroidSmoothing);

        normalizeValue(currentVolume);
        normalizeValue(currentCentroid);
    }
}

float AudioAnalyzer::calculateRMS() {
    float sum = 0.0f;
    for (const auto& sample : processBuffer) {
        sum += sample * sample;
    }
    return std::sqrt(sum / static_cast<float>(processBuffer.size()));
}

void AudioAnalyzer::calculateFFT() {
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        if (i < processBuffer.size()) {
            fftBuffer[i] = std::complex<float>(processBuffer[i] * windowFunction[i], 0.0f);
        }
        else {
            fftBuffer[i] = std::complex<float>(0.0f, 0.0f);
        }
    }

    size_t n = FFT_SIZE;
    for (size_t k = 2; k <= n; k *= 2) {
        for (size_t j = 0; j < n; j += k) {
            for (size_t i = j; i < j + k / 2; ++i) {
                size_t im = i + k / 2;
                float angle = -2.0f * static_cast<float>(M_PI) * static_cast<float>(i - j) / static_cast<float>(k);
                std::complex<float> w(std::cos(angle), std::sin(angle));
                std::complex<float> t = fftBuffer[i];
                std::complex<float> u = fftBuffer[im] * w;
                fftBuffer[i] = t + u;
                fftBuffer[im] = t - u;
            }
        }
    }
}

float AudioAnalyzer::calculateSpectralCentroid() {
    float weightedSum = 0.0f;
    float magnitudeSum = 0.0f;

    for (size_t i = 0; i < FFT_SIZE / 2; ++i) {
        float magnitude = std::abs(fftBuffer[i]);
        float frequency = static_cast<float>(i * audioCapture->getSampleRate()) / static_cast<float>(FFT_SIZE);

        weightedSum += magnitude * frequency;
        magnitudeSum += magnitude;
    }

    if (magnitudeSum > 0.0f) {
        return weightedSum / (magnitudeSum * static_cast<float>(audioCapture->getSampleRate() / 2));
    }
    return 0.0f;
}

void AudioAnalyzer::normalizeValue(float& value, float minValue, float maxValue) {
    value = std::max<float>(minValue, std::min<float>(value, maxValue));
}

float AudioAnalyzer::smoothValue(float current, float target, float smoothing) {
    return current * smoothing + target * (1.0f - smoothing);
}