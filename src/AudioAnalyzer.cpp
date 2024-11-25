// AudioAnalyzer.cpp
#include "AudioAnalyzer.hpp"
#include <iostream>

AudioAnalyzer::AudioAnalyzer() 
    : time(0.0f)
    , frequency(2.0f)  // oscillations per second
    , amplitude(1.0f)
    , currentVolume(0.0f)
    , currentCentroid(0.0f)
    , volumeSmoothing(0.8f)
    , centroidSmoothing(0.7f)
{
}

bool AudioAnalyzer::start() {
    std::cout << "Test signal generator started" << std::endl;
    return true;
}

void AudioAnalyzer::stop() {
    // Nothing to stop in test mode
}

void AudioAnalyzer::update() {
    // Generate test signals
    generateTestSignal();
    
    // Debug output
    std::cout << "Volume: " << currentVolume << " Centroid: " << currentCentroid << std::endl;
}

void AudioAnalyzer::generateTestSignal() {
    // Update time
    time += 1.0f / SAMPLE_RATE;
    
    // Generate oscillating volume (0.2 to 1.0)
    float targetVolume = 0.2f + 0.8f * (0.5f + 0.5f * std::sin(2.0f * M_PI * frequency * time));
    
    // Generate oscillating frequency for color (0.0 to 1.0)
    // Using a different frequency for variety
    float targetCentroid = 0.5f + 0.5f * std::sin(2.0f * M_PI * (frequency * 0.7f) * time);
    
    // Smooth the values
    currentVolume = smoothValue(currentVolume, targetVolume, volumeSmoothing);
    currentCentroid = smoothValue(currentCentroid, targetCentroid, centroidSmoothing);
    
    // Normalize values
    normalizeValue(currentVolume);
    normalizeValue(currentCentroid);
}

void AudioAnalyzer::normalizeValue(float& value, float minValue, float maxValue) {
    value = std::max(minValue, std::min(value, maxValue));
}

float AudioAnalyzer::smoothValue(float current, float target, float smoothing) {
    return current * smoothing + target * (1.0f - smoothing);
}