#pragma once

// Forward declare the original class names first
class Spectrogram;
class StaticSpectrogram;

// Now include the implementation headers
#include "LiveSpectrogramImpl.h"
#include "StaticSpectrogramImpl.h"

// Define the original classes with proper inheritance
class Spectrogram : public LiveSpectrogramImpl {
public:
    // Inherit constructor
    Spectrogram(sf::RenderWindow& win, const std::string& title, const sf::Vector2f& position)
        : LiveSpectrogramImpl(win, title, position) {
    }

    // Explicitly inherit all needed methods
    using LiveSpectrogramImpl::processSamples;
    using LiveSpectrogramImpl::draw;
    using LiveSpectrogramImpl::handleResize;
    using LiveSpectrogramImpl::updatePositionIndicator;
    using LiveSpectrogramImpl::setPosition;
    using LiveSpectrogramImpl::setSize;
    using LiveSpectrogramImpl::getCurrentMagnitudes;
    using LiveSpectrogramImpl::getTimeWindow;
    using LiveSpectrogramImpl::resetState;
};

class StaticSpectrogram : public StaticSpectrogramImpl {
public:
    // Inherit constructor
    StaticSpectrogram(sf::RenderWindow& win, const std::vector<std::vector<float>>& data,
        const std::string& title, const sf::Vector2f& position)
        : StaticSpectrogramImpl(win, data, title, position) {
    }

    // Explicitly inherit all needed methods
    using StaticSpectrogramImpl::draw;
    using StaticSpectrogramImpl::handleResize;
    using StaticSpectrogramImpl::updatePositionIndicator;
    using StaticSpectrogramImpl::setPosition;
    using StaticSpectrogramImpl::setSize;
};