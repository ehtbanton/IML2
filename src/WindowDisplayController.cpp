// WindowDisplayController.cpp
#include "WindowDisplayController.hpp"
#include <iostream>

WindowDisplayController::WindowDisplayController(sf::RenderWindow& win)
    : window(win)
    , lastVolume(0.0)
    , volumeChangeAccumulator(0.0)
    , lastHue(0.0)
{
}

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
    volumeChangeAccumulator *= params.volumeDecayRate;
    volumeChangeAccumulator += std::max(0.0, volumeChange * params.volumeAmplification);
    volumeChangeAccumulator = clamp(volumeChangeAccumulator, 0.0, 1.0);

    // Spectral centroid processing
    double nonLinearCentroid = std::pow(spectralCentroid, params.spectralPower);
    double expandedCentroid = nonLinearCentroid * params.spectralMultiplier;

    // Add modulation based on rate of change
    static double lastRawCentroid = spectralCentroid;
    double centroidChange = std::abs(spectralCentroid - lastRawCentroid);
    lastRawCentroid = spectralCentroid;

    expandedCentroid += centroidChange * params.changeMultiplier;
    expandedCentroid = fmod(expandedCentroid, 1.0);

    // Smooth hue changes
    double targetHue = expandedCentroid;
    lastHue = lastHue * params.hueSmoothing + targetHue * (1.0 - params.hueSmoothing);

    // Calculate final hue with pulse modulation
    double hue = fmod(lastHue + volumeChangeAccumulator * 0.2, 1.0);

    // Calculate brightness
    double baseValue = std::pow(volume * params.volumeMultiplier, params.volumePower);
    double pulseValue = volumeChangeAccumulator * params.pulseMultiplier;
    double value = clamp(baseValue + pulseValue, params.minBrightness, params.maxBrightness);

    // Calculate saturation
    double saturation = clamp(
        params.baseSaturation + volumeChangeAccumulator * params.pulseSaturation,
        params.baseSaturation,
        1.0
    );

    // Convert to RGB and update display
    sf::Color color = HSVtoRGB(hue, saturation, value);
    window.clear(color);
    window.display();

    // Debug output
    static int frameCount = 0;
    if (++frameCount % 60 == 0) {
        std::cout << "Spectral: " << spectralCentroid
            << " -> NonLinear: " << nonLinearCentroid
            << " -> Expanded: " << expandedCentroid
            << " Change: " << centroidChange
            << " Volume: " << volume
            << " Pulse: " << volumeChangeAccumulator
            << std::endl;
    }
}