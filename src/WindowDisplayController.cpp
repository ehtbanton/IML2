// WindowDisplayController.cpp
#include "WindowDisplayController.hpp"
#include <iostream>

WindowDisplayController::WindowDisplayController(sf::RenderWindow& win)
    : window(win)
    , lastVolume(0.0)
    , volumeChangeAccumulator(0.0)
    , lastHue(0.0) {}

double WindowDisplayController::clamp(double value, double min, double max) {
    return std::max(min, std::min(value, max));
}

sf::Color WindowDisplayController::HSVtoRGB(double hue, double saturation, double value) {
    hue = clamp(hue);
    saturation = clamp(saturation);
    value = clamp(value);

    double h = hue * 6.0;
    int i = static_cast<int>(h);
    double f = h - i;
    double p = value * (1.0 - saturation);
    double q = value * (1.0 - saturation * f);
    double t = value * (1.0 - saturation * (1.0 - f));

    double r, g, b;
    switch (i % 6) {
    case 0: r = value; g = t; b = p; break;
    case 1: r = q; g = value; b = p; break;
    case 2: r = p; g = value; b = t; break;
    case 3: r = p; g = q; b = value; break;
    case 4: r = t; g = p; b = value; break;
    case 5: r = value; g = p; b = q; break;
    default: r = 0; g = 0; b = 0; break;
    }

    return sf::Color(
        static_cast<sf::Uint8>(r * 255),
        static_cast<sf::Uint8>(g * 255),
        static_cast<sf::Uint8>(b * 255)
    );
}

void WindowDisplayController::updateDisplay(double spectralCentroid, double volume) {
    // Detect sudden volume changes (pulses)
    double volumeChange = volume - lastVolume;
    lastVolume = volume;

    // Accumulate recent volume changes with decay
    volumeChangeAccumulator *= 0.7; // Decay factor
    volumeChangeAccumulator += std::max(0.0, volumeChange * 5.0);
    volumeChangeAccumulator = clamp(volumeChangeAccumulator, 0.0, 1.0);

    // SUPER sensitive spectral centroid processing
    // First, apply non-linear scaling to emphasize small changes
    double nonLinearCentroid = std::pow(spectralCentroid, 0.4); // Make small changes more significant

    // Multiply the range dramatically (10x instead of 4x)
    double expandedCentroid = nonLinearCentroid * 10.0;

    // Add some additional modulation based on the rate of change
    static double lastRawCentroid = spectralCentroid;
    double centroidChange = std::abs(spectralCentroid - lastRawCentroid);
    lastRawCentroid = spectralCentroid;

    // Add extra rotation based on how quickly the centroid is changing
    expandedCentroid += centroidChange * 20.0;

    // Wrap to keep in 0-1 range
    expandedCentroid = fmod(expandedCentroid, 1.0);

    // Use very light smoothing to maintain responsiveness
    double targetHue = expandedCentroid;
    lastHue = lastHue * 0.3 + targetHue * 0.7; // More weight on new values

    // Calculate final hue with additional modulation
    double hue = fmod(lastHue + volumeChangeAccumulator * 0.2, 1.0);

    // Calculate brightness using both steady volume and sudden changes
    double baseValue = std::pow(volume * 2.5, 0.7);
    double pulseValue = volumeChangeAccumulator * 0.5;
    double value = clamp(baseValue + pulseValue, 0.15, 1.0);

    // Dynamic saturation - fuller colors during loud moments
    double saturation = clamp(0.8 + volumeChangeAccumulator * 0.2, 0.8, 1.0);

    // Convert to RGB color
    sf::Color color = HSVtoRGB(hue, saturation, value);

    // Debug output (every 60 frames or so)
    static int frameCount = 0;
    if (++frameCount % 60 == 0) {
        std::cout << "Color values - Raw Centroid: " << spectralCentroid
            << " -> NonLinear: " << nonLinearCentroid
            << " -> Expanded: " << expandedCentroid
            << " -> Final Hue: " << hue
            << " Change Rate: " << centroidChange
            << std::endl;
    }

    // Clear window with new color
    window.clear(color);
    window.display();
}