#pragma once

// First define these to prevent Windows macro conflicts
#define NOMINMAX 
#define WIN32_LEAN_AND_MEAN

// SFML headers MUST come before Windows headers
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

// Now it's safe to include Windows headers
#include <windows.h>

// Standard library headers
#include <vector>
#include <deque>
#include <mutex>
#include <chrono>

// Base class for both static and live spectrograms
class SpectrogramBase {
protected:
    static const size_t FFT_SIZE = 1024;

    sf::RenderWindow& window;
    sf::VertexArray vertices;
    sf::Font font;
    sf::RectangleShape spectrogramBackground;
    sf::Vector2f spectrogramPosition;
    sf::Vector2f spectrogramSize;
    std::vector<sf::Text> frequencyLabels;
    std::vector<sf::Text> timeLabels;

    // Position indicator
    sf::RectangleShape positionIndicator;
    sf::Text positionText;
    sf::Text confidenceText;
    sf::Text titleText;
    bool showPositionIndicator;

    // Title for this spectrogram
    std::string title;

    sf::Color getColor(float magnitude);
    sf::Text createText(const sf::Font& font, const std::string& content, unsigned int size, const sf::Vector2f& pos);

    // Add normalization function to ensure consistency
    std::vector<float> normalizeSpectrum(const std::vector<float>& magnitudes);

    // New methods for color matching
    void matchStaticSpectrogramColor(sf::Color& color, float normalizedFreq);
    void applyFrequencyBoosting(std::vector<float>& magnitudes);

public:
    SpectrogramBase(sf::RenderWindow& win, const std::string& title, const sf::Vector2f& position);
    virtual ~SpectrogramBase() {}

    virtual void draw() = 0;
    virtual void handleResize() = 0;
    virtual void updatePositionIndicator(double seconds, float confidence);
    void setPosition(const sf::Vector2f& position);
    void setSize(const sf::Vector2f& size);
};