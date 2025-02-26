#define _USE_MATH_DEFINES
#include "StaticSpectrogramImpl.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

StaticSpectrogramImpl::StaticSpectrogramImpl(sf::RenderWindow& win, const std::vector<std::vector<float>>& data,
    const std::string& titleText, const sf::Vector2f& position)
    : SpectrogramBase(win, titleText, position), spectrogramData(data)
{
    // Set default size
    spectrogramSize = sf::Vector2f(800, 600);

    // Initialize UI elements before setting vertex array
    initializeUI();

    // Check if we have data before resizing vertices
    size_t numFrames = spectrogramData.size();
    size_t bins = FFT_SIZE / 2;

    if (numFrames > 0) {
        // Set primitive type for vertex array
        vertices.setPrimitiveType(sf::Quads);

        // Calculate total vertex count needed
        size_t totalVertices = numFrames * bins * 4;

        // Resize vertex array
        vertices.resize(totalVertices);

        // Update vertex positions
        updateVertexPositions();
    }
    else {
        // Initialize with empty vertex array
        vertices.setPrimitiveType(sf::Quads);
        vertices.resize(0);
    }
}

void StaticSpectrogramImpl::initializeUI() {
    // Use the spectrogramSize directly
    spectrogramBackground.setSize(spectrogramSize);
    spectrogramBackground.setPosition(spectrogramPosition);
    spectrogramBackground.setFillColor(sf::Color(20, 20, 20));
    spectrogramBackground.setOutlineColor(sf::Color::White);
    spectrogramBackground.setOutlineThickness(1.0f);

    // Initialize position indicator
    positionIndicator.setSize(sf::Vector2f(2.0f, spectrogramSize.y));
    positionIndicator.setFillColor(sf::Color(255, 0, 0, 200));
    positionIndicator.setPosition(spectrogramPosition.x, spectrogramPosition.y);

    positionText = createText(font, "0:00", 14,
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y + spectrogramSize.y + 15.0f));
    positionText.setFillColor(sf::Color::Red);

    confidenceText = createText(font, "Confidence: 0%", 14,
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y + spectrogramSize.y + 15.0f));
    confidenceText.setFillColor(sf::Color::Red);

    // Title text
    titleText = createText(font, title, 16,
        sf::Vector2f(spectrogramPosition.x + spectrogramSize.x / 2 - 80,
            spectrogramPosition.y - 30.0f));
    titleText.setFillColor(sf::Color::White);

    frequencyLabels.clear();
    float sampleRate = 48000.0f;
    float freqPerBin = sampleRate / static_cast<float>(FFT_SIZE);
    const float minFreq = 10.0f;
    const float maxFreq = 24000.0f;
    int numFreqLabels = 8;
    for (int i = 0; i < numFreqLabels; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(numFreqLabels - 1);
        float freq = minFreq * std::pow(maxFreq / minFreq, t);
        std::stringstream ss;
        if (freq >= 1000.0f)
            ss << std::fixed << std::setprecision(1) << (freq / 1000.0f) << " kHz";
        else
            ss << std::fixed << std::setprecision(0) << freq << " Hz";

        float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
        float yPos = spectrogramPosition.y + spectrogramSize.y * (1.0f - yRatio);
        frequencyLabels.push_back(createText(font, ss.str(), 10, sf::Vector2f(spectrogramPosition.x - 55.0f, yPos - 5.0f)));
    }

    // Create time labels along the horizontal axis.
    timeLabels.clear();
    size_t numFrames = spectrogramData.size();
    float secondsPerFrame = 512.0f / 48000.0f;
    float totalTime = static_cast<float>(numFrames) * secondsPerFrame;

    // Corrected loop for time labels (fixed to 30 seconds)
    for (int t = 0; t <= 30; t += 5) {
        float tRatio = static_cast<float>(t) / 30.0f; // Use a fixed 30-second window
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << t << " s";
        float xPos = spectrogramPosition.x + spectrogramSize.x * tRatio;
        timeLabels.push_back(createText(font, ss.str(), 10, sf::Vector2f(xPos - 15.0f, spectrogramPosition.y + spectrogramSize.y + 10.0f)));
    }
}

void StaticSpectrogramImpl::updateVertexPositions() {
    size_t numFrames = spectrogramData.size();
    size_t bins = FFT_SIZE / 2;

    // Skip if no data
    if (numFrames == 0) {
        return;
    }

    // Check if vertex array has been properly sized
    if (vertices.getVertexCount() != numFrames * bins * 4) {
        vertices.resize(numFrames * bins * 4);
    }

    float frameWidth = spectrogramSize.x / static_cast<float>(numFrames);
    float sampleRate = 48000.0f;
    float freqPerBin = sampleRate / static_cast<float>(FFT_SIZE);
    const float minFreq = 10.0f;
    const float maxFreq = 24000.0f;

    for (size_t x = 0; x < numFrames; ++x) {
        for (size_t y = 0; y < bins; ++y) {
            size_t idx = (x * bins + y) * 4;

            // Skip if index is out of bounds
            if (idx + 3 >= vertices.getVertexCount()) {
                continue;
            }

            float xpos = spectrogramPosition.x + static_cast<float>(x) * frameWidth;
            float freq = static_cast<float>(y) * freqPerBin;
            freq = std::max(minFreq, std::min(maxFreq, freq));
            float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
            float ypos = spectrogramPosition.y + spectrogramSize.y * (1.0f - yRatio);

            float nextFreq = static_cast<float>(y + 1) * freqPerBin;
            nextFreq = std::max(minFreq, std::min(maxFreq, nextFreq));
            float nextYRatio = std::log10(nextFreq / minFreq) / std::log10(maxFreq / minFreq);
            float nextYpos = spectrogramPosition.y + spectrogramSize.y * (1.0f - nextYRatio);

            vertices[idx].position = sf::Vector2f(xpos, ypos);
            vertices[idx + 1].position = sf::Vector2f(xpos + frameWidth, ypos);
            vertices[idx + 2].position = sf::Vector2f(xpos + frameWidth, nextYpos);
            vertices[idx + 3].position = sf::Vector2f(xpos, nextYpos);
        }
    }
}

void StaticSpectrogramImpl::draw() {
    // Draw background elements
    window.draw(spectrogramBackground);
    window.draw(titleText);

    // Always draw labels
    for (const auto& label : frequencyLabels) {
        window.draw(label);
    }
    for (const auto& label : timeLabels) {
        window.draw(label);
    }

    // Draw position indicator if needed
    if (showPositionIndicator) {
        window.draw(positionIndicator);
    }

    // Skip spectrum drawing if no data
    if (spectrogramData.empty()) {
        return;
    }

    // Get dimensions
    size_t numFrames = spectrogramData.size();
    size_t bins = FFT_SIZE / 2;

    // CRITICAL FIX: Ensure vertex array is properly sized
    size_t requiredVertices = numFrames * bins * 4;
    if (vertices.getVertexCount() != requiredVertices) {
        vertices.resize(requiredVertices);
        updateVertexPositions();
    }

    // Only continue if we have vertices to draw
    if (vertices.getVertexCount() == 0) {
        return;
    }

    // Update colors for each vertex with careful bounds checking
    for (size_t x = 0; x < numFrames; ++x) {
        // Skip if frame index is out of range
        if (x >= spectrogramData.size()) {
            continue;
        }

        const auto& frame = spectrogramData[x];
        if (frame.empty()) {
            continue;
        }

        // Apply normalization (using the same normalization settings as the live version)
        std::vector<float> normalized = normalizeSpectrum(frame);

        // Determine how many bins we can safely process
        size_t maxBins = std::min(bins, normalized.size());

        for (size_t y = 0; y < maxBins; ++y) {
            // Calculate vertex index
            size_t idx = (x * bins + y) * 4;

            // Skip if vertex index is out of range
            if (idx + 3 >= vertices.getVertexCount()) {
                break;
            }

            // Get color with frequency-dependent adjustments
            float normalizedFreq = static_cast<float>(y) / bins;
            sf::Color color = getColor(normalized[y]);

            // Call with proper scope
            this->matchStaticSpectrogramColor(color, normalizedFreq);

            // Update vertex colors
            for (int i = 0; i < 4; ++i) {
                vertices[idx + i].color = color;
            }
        }
    }

    // Draw the spectrum data
    window.draw(vertices);
}

void StaticSpectrogramImpl::handleResize() {
    initializeUI();
    updateVertexPositions();
}