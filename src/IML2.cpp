#include "WindowDisplayController.hpp"
#include "AudioAnalyzer.hpp"
#include <iostream>

int main() {
    // Create window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Audio Reactive Display");
    WindowDisplayController displayController(window);
    AudioAnalyzer audioAnalyzer;

    // Start audio analysis
    if (!audioAnalyzer.start()) {
        std::cout << "Failed to start audio capture!" << std::endl;
        return -1;
    }

    std::cout << "Starting main loop - Play some audio to see the effect!" << std::endl;

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Update audio analysis
        audioAnalyzer.update();

        // Get normalized values (0.0 to 1.0)
        float spectralCentroid = audioAnalyzer.getSpectralCentroid();
        float volume = audioAnalyzer.getVolume();

        // Debug output every 60 frames
        static int frameCount = 0;
        if (++frameCount % 60 == 0) {
            std::cout << "\rAudio - Centroid: " << spectralCentroid
                << " Volume: " << volume << std::flush;
        }

        // Update display using audio parameters
        displayController.updateDisplay(spectralCentroid, volume);

        // Small sleep to prevent excessive CPU usage
        sf::sleep(sf::milliseconds(16)); // ~60 fps
    }

    audioAnalyzer.stop();
    return 0;
}