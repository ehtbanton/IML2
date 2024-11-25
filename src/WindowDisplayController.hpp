// WindowDisplayController.hpp
#ifndef WINDOW_DISPLAY_CONTROLLER_HPP
#define WINDOW_DISPLAY_CONTROLLER_HPP

#include <SFML/Graphics.hpp>
#include <cmath>

class WindowDisplayController {
private:
    sf::RenderWindow& window;

    // State tracking
    double lastVolume;
    double volumeChangeAccumulator;
    double lastHue;

    // Tuning parameters
    struct {
        // Volume pulse detection
        double volumeDecayRate = 0.7;      // How quickly volume pulses fade (0-1)
        double volumeAmplification = 5.0;   // How much to amplify sudden volume changes

        // Spectral sensitivity
        double spectralPower = 0.4;        // Non-linear scaling power
        double spectralMultiplier = 10.0;  // How much to multiply spectral changes
        double changeMultiplier = 20.0;    // Additional rotation from spectral changes

        // Color smoothing
        double hueSmoothing = 0.3;         // Color change smoothing factor

        // Brightness settings
        double volumePower = 0.7;          // Non-linear volume scaling
        double volumeMultiplier = 2.5;     // Base volume amplification
        double pulseMultiplier = 0.5;      // Pulse brightness effect
        double minBrightness = 0.15;       // Minimum brightness
        double maxBrightness = 1.0;        // Maximum brightness

        // Saturation settings
        double baseSaturation = 0.8;       // Base color saturation
        double pulseSaturation = 0.2;      // Pulse saturation effect
    } params;

    // Helper functions
    double clamp(double value, double min = 0.0, double max = 1.0);
    sf::Color HSVtoRGB(double hue, double saturation, double value);

public:
    WindowDisplayController(sf::RenderWindow& win);
    void updateDisplay(double spectralCentroid, double volume);
};

#endif // WINDOW_DISPLAY_CONTROLLER_HPP