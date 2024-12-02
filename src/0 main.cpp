// main.cpp
#include "WindowDisplayController.hpp"
#include "AudioAnalyzer.hpp"
#include <iostream>
#include <SFML/Graphics.hpp>

int main() {
    // Create window with standard size
    const unsigned int WINDOW_WIDTH = 800;
    const unsigned int WINDOW_HEIGHT = 600;
    const std::string WINDOW_TITLE = "Audio Reactive Display";

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_TITLE);
    WindowDisplayController displayController(window);

    // Create audio analyzer with default buffer size (2048)
    AudioAnalyzer audioAnalyzer;

    // Start audio analysis
    if (!audioAnalyzer.start()) {
        std::cerr << "Failed to start audio capture!" << std::endl;
        return -1;
    }

    std::cout << "Starting main loop - Play some audio to see the effect!" << std::endl;
    std::cout << "Close the window to exit." << std::endl;

    // Target frame rate
    const sf::Time frameTime = sf::seconds(1.0f / 60.0f);
    sf::Clock clock;

    // Main loop
    while (window.isOpen()) {
        // Handle window events
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                }
            }
        }

        // Update audio analysis
        audioAnalyzer.update();

        // Get normalized values (0.0 to 1.0)
        float spectralCentroid = audioAnalyzer.getSpectralCentroid();
        float volume = audioAnalyzer.getVolume();

        // Update display using audio parameters
        displayController.updateDisplay(spectralCentroid, volume);

        // Frame rate limiting
        sf::Time elapsed = clock.getElapsedTime();
        if (elapsed < frameTime) {
            sf::sleep(frameTime - elapsed);
        }
        clock.restart();
    }

    // Cleanup
    audioAnalyzer.stop();
    return 0;
}