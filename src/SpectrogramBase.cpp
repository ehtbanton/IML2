#define _USE_MATH_DEFINES
#include "SpectrogramBase.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

SpectrogramBase::SpectrogramBase(sf::RenderWindow& win, const std::string& titleText, const sf::Vector2f& position)
    : window(win),
    vertices(sf::Quads),
    title(titleText),
    spectrogramPosition(position),
    showPositionIndicator(false)
{
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
        std::cerr << "Failed to load font" << std::endl;
    }
}

sf::Color SpectrogramBase::getColor(float magnitude) {
    // Convert magnitude to decibels (logarithmic scale)
    float db = 20.0f * std::log10(magnitude + 1e-9f);

    // Use more aggressive normalization to match the static spectrogram
    // This makes the colors more intense, with more reds and yellows
    float normalized = (db + 60.0f) / 55.0f; // Adjusted parameters
    normalized = std::max(0.0f, std::min(1.0f, normalized));

    // Apply gamma correction to enhance mid-range values - match the static display
    normalized = std::pow(normalized, 0.5f); // Lower gamma = brighter colors

    // Create color map identical to static spectrogram
    if (normalized < 0.2f) {
        // Dark Blue to Blue (keep lower intensity as is)
        float t = normalized / 0.2f;
        return sf::Color(0, 0, static_cast<sf::Uint8>(64 + 191 * t));
    }
    else if (normalized < 0.4f) {
        // Blue to Cyan
        float t = (normalized - 0.2f) / 0.2f;
        return sf::Color(0, static_cast<sf::Uint8>(255 * t), 255);
    }
    else if (normalized < 0.6f) {
        // Cyan to Green
        float t = (normalized - 0.4f) / 0.2f;
        return sf::Color(0, 255, static_cast<sf::Uint8>(255 * (1.0f - t)));
    }
    else if (normalized < 0.7f) { // Compress this range for more yellows and reds
        // Green to Yellow
        float t = (normalized - 0.6f) / 0.1f;
        return sf::Color(static_cast<sf::Uint8>(255 * t), 255, 0);
    }
    else {
        // Yellow to Red - expanded to create more reds like in the static spectrogram
        float t = (normalized - 0.7f) / 0.3f;
        t = std::min(1.0f, t); // Ensure we don't exceed 1.0
        return sf::Color(255, static_cast<sf::Uint8>(255 * (1.0f - t)), 0);
    }
}

void SpectrogramBase::applyFrequencyBoosting(std::vector<float>& magnitudes) {
    // Apply non-linear boosting to different frequency ranges
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        float normalizedFreq = static_cast<float>(i) / magnitudes.size();

        if (normalizedFreq < 0.1f) {
            // Boost low frequencies (like in static spectrogram)
            magnitudes[i] *= 1.6f;
        }
        else if (normalizedFreq < 0.3f) {
            // Boost low-mid frequencies
            magnitudes[i] *= 1.4f;
        }
        else if (normalizedFreq < 0.6f) {
            // Boost mid frequencies
            magnitudes[i] *= 1.2f;
        }
        // Higher frequencies match better already
    }
}

void SpectrogramBase::matchStaticSpectrogramColor(sf::Color& color, float normalizedFreq) {
    // Apply frequency-dependent color adjustment to match static spectrogram
    // Static spectrogram is more saturated in certain frequency ranges

    if (normalizedFreq < 0.2f) {
        // Boost saturation in bass range
        color.r = std::min(255, static_cast<int>(color.r * 1.2f));
        color.g = std::min(255, static_cast<int>(color.g * 1.2f));
    }
    else if (normalizedFreq > 0.7f) {
        // Slightly reduce brightness in high frequencies to match static
        color.g = static_cast<sf::Uint8>(color.g * 0.9f);
        color.b = static_cast<sf::Uint8>(color.b * 0.9f);
    }
}

std::vector<float> SpectrogramBase::normalizeSpectrum(const std::vector<float>& magnitudes) {
    // Make a copy for modification
    std::vector<float> normalized = magnitudes;

    // Apply consistent amplification - use the same value for BOTH spectrograms
    // This increases the values to make colors more vibrant
    const float AMPLIFICATION = 5.0f; // Higher value = more bright colors

    for (float& val : normalized) {
        // Amplify the values to match the static spectrogram's intensity
        val *= AMPLIFICATION;
    }

    // Apply frequency-dependent boosting to match the static spectrogram
    applyFrequencyBoosting(normalized);

    return normalized;
}

sf::Text SpectrogramBase::createText(const sf::Font& font, const std::string& content, unsigned int size, const sf::Vector2f& pos) {
    sf::Text text;
    text.setFont(font);
    text.setString(content);
    text.setCharacterSize(size);
    text.setFillColor(sf::Color::White);
    text.setPosition(pos);
    return text;
}

void SpectrogramBase::updatePositionIndicator(double seconds, float confidence) {
    showPositionIndicator = true;

    // Position along X axis based on seconds into the song
    float xRatio = std::min(1.0f, static_cast<float>(seconds / 30.0f)); // Scale to 30 second window
    float xPos = spectrogramPosition.x + xRatio * spectrogramSize.x;

    // Update position indicator line - just a vertical line with no text
    positionIndicator.setPosition(xPos, spectrogramPosition.y);
    positionIndicator.setSize(sf::Vector2f(2.0f, spectrogramSize.y));

    // Color based on confidence
    sf::Uint8 alpha = static_cast<sf::Uint8>(std::min(1.0f, confidence) * 255);
    positionIndicator.setFillColor(sf::Color(255, 0, 0, alpha));
}

void SpectrogramBase::setPosition(const sf::Vector2f& position) {
    spectrogramPosition = position;
    handleResize();
}

void SpectrogramBase::setSize(const sf::Vector2f& size) {
    spectrogramSize = size;
    handleResize();
}