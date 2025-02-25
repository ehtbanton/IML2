#define _USE_MATH_DEFINES
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

// Include SFML headers first, before any Windows headers
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

// Then Windows and other system headers
#include <windows.h>
#include <fftw3.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>

// Finally, project-specific headers
#include "AudioFingerprinter.h"
#include "AudioMatcher.h"
#include "SpectrogramVisualizer.h"
#include "AudioCapture.h"
#include "TestHarness.h"
#include "SpectrogramCache.h"

int main() {
    // Initialize FFTW threads for better performance
    fftw_init_threads();
    fftw_plan_with_nthreads(4);  // Use 4 threads for parallel execution

    // Create a single SFML window for both spectrograms
    sf::RenderWindow window(sf::VideoMode(1280, 720), "Audio Position Matcher", sf::Style::Default | sf::Style::Resize);
    window.setFramerateLimit(60);

    // Get the path to the executable directory and determine the audio file path
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string exePath(path);
    std::string exeDir = exePath.substr(0, exePath.find_last_of("\\/"));
    std::string audioPath = exeDir + "\\beethoven.wav";

    std::cout << "Looking for audio file at: " << audioPath << std::endl;

    // Initialize the AudioMatcher
    AudioMatcher matcher(audioPath);
    if (!matcher.initialize()) {
        std::cerr << "Failed to initialize audio matcher" << std::endl;
        std::cerr << "Please ensure the file exists at: " << audioPath << std::endl;
        return -1;
    }

    // Retrieve the static spectrogram data from the matcher
    const auto& staticSpectroData = matcher.getStaticSpectrogram();

    // Create positions for the spectrograms (left and right)
    sf::Vector2f livePos(50.0f, 50.0f);
    sf::Vector2f staticPos(675.0f, 50.0f);

    // Create an instance of our static spectrogram display using the static data
    StaticSpectrogram staticSpectrogram(window, staticSpectroData, "Static Spectrogram (Reference)", staticPos);

    // Initialize the live spectrogram
    Spectrogram liveSpectrogram(window, "Real-time Spectrogram (Live Capture)", livePos);

    // Initialize the audio capture
    AudioCapture capture;
    capture.setSpectrogram(&liveSpectrogram);
    if (!capture.start()) {
        std::cerr << "Failed to start audio capture" << std::endl;
        std::cerr << "Make sure audio devices are properly configured" << std::endl;
        return -1;
    }

    // Initialize test harness
    AudioMatchTestHarness testHarness(&matcher);

    sf::Clock debugTimer;
    sf::Clock testTimer;

    // Create UI elements for test controls
    sf::Font font;
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
        std::cerr << "Failed to load font, using default" << std::endl;
    }

    sf::RectangleShape testButton(sf::Vector2f(120.0f, 40.0f));
    testButton.setPosition(580.0f, 670.0f);
    testButton.setFillColor(sf::Color(60, 60, 180));
    testButton.setOutlineColor(sf::Color::White);
    testButton.setOutlineThickness(1.0f);

    sf::Text testButtonText("Run Test", font, 16);
    testButtonText.setPosition(600.0f, 680.0f);
    testButtonText.setFillColor(sf::Color::White);

    // Status text
    sf::Text statusText("Ready - Click 'Run Test' to check matching accuracy", font, 18);
    statusText.setPosition(50.0f, 670.0f);
    statusText.setFillColor(sf::Color::White);

    std::cout << "Application initialized successfully. Monitoring audio..." << std::endl;

    // Main loop: poll events and draw both spectrograms in one window
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
            {
                window.close();
            }
            else if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0, 0, static_cast<float>(event.size.width), static_cast<float>(event.size.height));
                window.setView(sf::View(visibleArea));

                // Recalculate positions for spectrograms
                livePos = sf::Vector2f(50.0f, 50.0f);
                staticPos = sf::Vector2f(static_cast<float>(event.size.width) / 2.0f + 25.0f, 50.0f);

                liveSpectrogram.setPosition(livePos);
                staticSpectrogram.setPosition(staticPos);

                liveSpectrogram.handleResize();
                staticSpectrogram.handleResize();
            }
            else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    // Check if test button was clicked
                    sf::Vector2f mousePos(static_cast<float>(event.mouseButton.x),
                        static_cast<float>(event.mouseButton.y));

                    if (testButton.getGlobalBounds().contains(mousePos)) {
                        if (!testHarness.isRunningTest()) {
                            testHarness.startTest();
                            testButtonText.setString("Stop Test");
                            statusText.setString("Test running...");
                        }
                        else {
                            testHarness.stopTest();
                            testButtonText.setString("Run Test");
                            statusText.setString("Test stopped");
                        }
                    }
                }
            }
        }

        // Periodically update the position indicator
        if (debugTimer.getElapsedTime().asSeconds() >= 0.5f) { // Update twice per second
            std::pair<size_t, float> matchResult = { 0, 0.0f };

            if (testHarness.isRunningTest() && testTimer.getElapsedTime().asSeconds() >= 0.1f) {
                // Use test signal instead of live audio
                matchResult = testHarness.processTestFrame();
                testTimer.restart();

                // Update status
                std::stringstream ss;
                ss << "Test running... Confidence: " << std::fixed << std::setprecision(2)
                    << (matchResult.second * 100.0f) << "%";
                statusText.setString(ss.str());
            }
            else {
                // Normal mode - use live audio
                auto current_magnitudes = liveSpectrogram.getCurrentMagnitudes();
                if (!current_magnitudes.empty()) {
                    matchResult = matcher.findMatchWithConfidence(current_magnitudes);

                    // Update status text in normal mode
                    std::stringstream ss;
                    ss << "Current position: " << std::fixed << std::setprecision(1)
                        << matcher.getTimestamp(matchResult.first) << "s  Confidence: "
                        << std::fixed << std::setprecision(0) << (matchResult.second * 100.0f) << "%";
                    statusText.setString(ss.str());
                }
            }

            // Update position indicators on both spectrograms
            double timestamp = matcher.getTimestamp(matchResult.first);
            liveSpectrogram.updatePositionIndicator(timestamp, matchResult.second);
            staticSpectrogram.updatePositionIndicator(timestamp, matchResult.second);

            debugTimer.restart();
        }

        // Draw everything in one window
        window.clear(sf::Color(10, 10, 10));

        // Draw spectrograms
        liveSpectrogram.draw();
        staticSpectrogram.draw();

        // Draw UI elements
        window.draw(testButton);
        window.draw(testButtonText);
        window.draw(statusText);

        window.display();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    // Clean up
    capture.stop();
    fftw_cleanup_threads();

    return 0;
}