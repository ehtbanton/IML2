// WindowDisplayController.cpp
#include "WindowDisplayController.hpp"

WindowDisplayController::WindowDisplayController(sf::RenderWindow& win)
    : window(win) {}

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

void WindowDisplayController::updateDisplay(double input1, double input2) {
    // Map first input to hue (0-1 range)
    double hue = clamp(input1);

    // Map second input to brightness/value (0-1 range)
    double brightness = clamp(input2);

    // Convert to RGB color (using constant saturation for vibrant colors)
    sf::Color color = HSVtoRGB(hue, 1.0, brightness);

    // Clear window with new color
    window.clear(color);
    window.display();
}