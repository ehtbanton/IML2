// WindowDisplayController.hpp
#ifndef WINDOW_DISPLAY_CONTROLLER_HPP
#define WINDOW_DISPLAY_CONTROLLER_HPP

#include <SFML/Graphics.hpp>
#include <cmath>

class WindowDisplayController {
private:
    sf::RenderWindow& window;

    // Helper function to clamp values between min and max
    double clamp(double value, double min = 0.0, double max = 1.0);

    // Convert HSV to RGB color
    sf::Color HSVtoRGB(double hue, double saturation, double value);

public:
    // Constructor
    WindowDisplayController(sf::RenderWindow& win);

    // Update window color based on two inputs
    void updateDisplay(double input1, double input2);
};

#endif // WINDOW_DISPLAY_CONTROLLER_HPP