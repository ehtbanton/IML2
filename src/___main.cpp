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
#include <atomic>
#include <limits> // Added for std::numeric_limits

// Finally, project-specific headers
#include "AudioFingerprinter.h"
#include "AudioMatcher.h"
#include "SpectrogramVisualizer.h"
#include "AudioCapture.h"
#include "TestHarness.h"
#include "SpectrogramCache.h"

// Global atomic flag for controlling background processing
std::atomic<bool> g_processingActive(true);

// Background processing thread function with improved matching
void backgroundProcessing(AudioMatcher* matcher, Spectrogram* liveSpectrogram,
    StaticSpectrogram* staticSpectrogram,
    std::atomic<double>* currentTimestamp,
    std::atomic<float>* currentConfidence) {

    sf::Clock updateTimer;
    sf::Clock statsTimer;
    int frameCounter = 0;

    // Give the capture system time to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "Background processing started" << std::endl;

    while (g_processingActive) {
        // Update at a higher rate for more responsive matching
        if (updateTimer.getElapsedTime().asMilliseconds() >= 30) {
            auto current_magnitudes = liveSpectrogram->getCurrentMagnitudes();

            if (!current_magnitudes.empty()) {
                frameCounter++;

                // Try multiple matching approaches
                auto matchResult = matcher->findMatchWithConfidence(current_magnitudes);

                // Update atomic values that main thread can read
                *currentTimestamp = matcher->getTimestamp(matchResult.first);
                *currentConfidence = matchResult.second;

                // Print diagnostics every 30 frames
                if (frameCounter % 30 == 0) {
                    std::cout << "Current match position: " << *currentTimestamp
                        << "s, confidence: " << (*currentConfidence * 100.0f)
                        << "%, frame: " << frameCounter << std::endl;
                }
            }

            updateTimer.restart();
        }

        // Print stats every 10 seconds
        if (statsTimer.getElapsedTime().asSeconds() >= 10.0f) {
            matcher->printStats();
            statsTimer.restart();
        }

        // Sleep to prevent CPU thrashing
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::cout << "Background processing stopped" << std::endl;
}

int main() {
    // Set process priority to above normal
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);

    // Initialize FFTW threads for better performance
    fftw_init_threads();
    fftw_plan_with_nthreads(4);  // Use 4 threads for parallel execution

    std::cout << "Audio Position Matcher Initializing..." << std::endl;
    std::cout << "--------------------------------" << std::endl;

    // Get the path to the executable directory and determine the audio file path
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string exePath(path);
    std::string exeDir = exePath.substr(0, exePath.find_last_of("\\/"));
    std::string audioPath = exeDir + "\\beethoven.wav";

    std::cout << "Looking for audio file at: " << audioPath << std::endl;

    // *** FIRST: Initialize the AudioMatcher and generate fingerprints BEFORE creating the window ***
    std::cout << "Analyzing audio file and generating fingerprints..." << std::endl;
    AudioMatcher matcher(audioPath);
    if (!matcher.initialize()) {
        std::cerr << "Failed to initialize audio matcher" << std::endl;
        std::cerr << "Please ensure the file exists at: " << audioPath << std::endl;
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return -1;
    }

    // Run self-test to verify fingerprinting works
    std::cout << "Running self-test to verify fingerprinting..." << std::endl;
    bool testPassed = matcher.runSelfTest();
    std::cout << "Self-test " << (testPassed ? "passed!" : "failed!") << std::endl;

    // Retrieve the static spectrogram data from the matcher
    const auto& staticSpectroData = matcher.getStaticSpectrogram();
    std::cout << "Fingerprint generation complete!" << std::endl;
    std::cout << "Processed " << staticSpectroData.size() << " frames." << std::endl;

    // *** SECOND: Now that initialization is done, create the SFML window ***
    std::cout << "Creating window and initializing UI..." << std::endl;

    // Get desktop mode for fullscreen resolution
    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    // Create a fullscreen window
    sf::RenderWindow window(desktopMode, "Audio Position Matcher",
        sf::Style::Fullscreen);

    // Use VSync for steady frame rate
    window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    // Calculate positions with more spacing for fullscreen
    float windowWidth = static_cast<float>(desktopMode.width);
    float windowHeight = static_cast<float>(desktopMode.height);

    // Left spectrogram at 10% from left edge
    sf::Vector2f livePos(windowWidth * 0.1f, windowHeight * 0.15f);
    // Right spectrogram at 55% from left edge
    sf::Vector2f staticPos(windowWidth * 0.55f, windowHeight * 0.15f);

    // Create an instance of our static spectrogram display using the static data
    StaticSpectrogram staticSpectrogram(window, staticSpectroData, "Static Spectrogram (Reference)", staticPos);

    // Initialize the live spectrogram
    Spectrogram liveSpectrogram(window, "Real-time Spectrogram (Live Capture)", livePos);

    // Make spectrograms a bit smaller relative to screen size
    float spectrogramWidth = windowWidth * 0.35f;  // 35% of screen width
    float spectrogramHeight = windowHeight * 0.6f; // 60% of screen height

    // Set spectrogram positions and sizes
    liveSpectrogram.setSize(sf::Vector2f(spectrogramWidth, spectrogramHeight));
    staticSpectrogram.setSize(sf::Vector2f(spectrogramWidth, spectrogramHeight));

    // Initialize the audio capture with high priority
    std::cout << "Starting audio capture..." << std::endl;
    AudioCapture capture;
    capture.setSpectrogram(&liveSpectrogram);
    if (!capture.start()) {
        std::cerr << "Failed to start audio capture" << std::endl;
        std::cerr << "Make sure audio devices are properly configured" << std::endl;
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return -1;
    }

    // Create a font used for UI
    sf::Font font;
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
        std::cerr << "Failed to load font, using default" << std::endl;
    }

    // Simple status text at the bottom center of the screen
    sf::Text statusText("Audio Position Matcher - Press ESC to exit", font, 18);
    statusText.setPosition(windowWidth * 0.5f - 150.0f, windowHeight * 0.9f);
    statusText.setFillColor(sf::Color::White);

    // Create atomic variables for thread communication
    std::atomic<double> currentTimestamp(0.0);
    std::atomic<float> currentConfidence(0.0f);

    // Start background processing thread
    std::cout << "Starting background processing..." << std::endl;
    std::thread processingThread(backgroundProcessing, &matcher, &liveSpectrogram,
        &staticSpectrogram, &currentTimestamp, &currentConfidence);

    // Set thread priority to time critical
    SetThreadPriority(processingThread.native_handle(), THREAD_PRIORITY_HIGHEST);

    std::cout << "Application initialized successfully. Monitoring audio..." << std::endl;

    // FPS counter variables
    sf::Clock fpsClock;
    int frameCount = 0;
    float lastFpsUpdate = 0.0f;
    float fps = 0.0f;
    sf::Text fpsText("FPS: 0", font, 14);
    fpsText.setPosition(10.0f, 10.0f);
    fpsText.setFillColor(sf::Color::Yellow);

    // Reduced update interval for UI updates to save CPU
    sf::Clock uiUpdateClock;
    bool needsUIUpdate = true;

    // Main loop: poll events and draw both spectrograms in one window
    while (window.isOpen()) {
        // FPS calculation
        frameCount++;
        float currentTime = fpsClock.getElapsedTime().asSeconds();
        if (currentTime - lastFpsUpdate >= 1.0f) {
            fps = static_cast<float>(frameCount) / (currentTime - lastFpsUpdate);
            frameCount = 0;
            lastFpsUpdate = currentTime;
            fpsText.setString("FPS: " + std::to_string(static_cast<int>(fps)));
        }

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
            {
                window.close();
                g_processingActive = false;
            }
            else if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0, 0, static_cast<float>(event.size.width), static_cast<float>(event.size.height));
                window.setView(sf::View(visibleArea));

                // Recalculate positions for spectrograms
                float newWidth = static_cast<float>(event.size.width);
                float newHeight = static_cast<float>(event.size.height);

                livePos = sf::Vector2f(newWidth * 0.1f, newHeight * 0.15f);
                staticPos = sf::Vector2f(newWidth * 0.55f, newHeight * 0.15f);

                float newSpectrogramWidth = newWidth * 0.35f;
                float newSpectrogramHeight = newHeight * 0.6f;

                liveSpectrogram.setPosition(livePos);
                staticSpectrogram.setPosition(staticPos);

                liveSpectrogram.setSize(sf::Vector2f(newSpectrogramWidth, newSpectrogramHeight));
                staticSpectrogram.setSize(sf::Vector2f(newSpectrogramWidth, newSpectrogramHeight));

                // Update status text position
                statusText.setPosition(newWidth * 0.5f - 150.0f, newHeight * 0.9f);

                needsUIUpdate = true;
            }
        }

        // Update UI only every 50ms (20 times per second) to improve performance
        if (uiUpdateClock.getElapsedTime().asMilliseconds() >= 50 || needsUIUpdate) {
            // Use the values updated by background thread
            double timestamp = currentTimestamp.load();
            float confidence = currentConfidence.load();

            // Only update UI if we have values or need a UI update
            if (confidence > 0.0f || needsUIUpdate) {
                // Update position indicators with red vertical line only (no text)
                // The position indicator shows the detected position in the song
                liveSpectrogram.updatePositionIndicator(timestamp, confidence);
                staticSpectrogram.updatePositionIndicator(timestamp, confidence);

                // Update status text
                std::stringstream ss;
                int minutes = static_cast<int>(timestamp) / 60;
                int seconds = static_cast<int>(timestamp) % 60;
                ss << "Position: " << minutes << ":"
                    << std::setfill('0') << std::setw(2) << seconds
                    << " - Confidence: " << std::fixed << std::setprecision(0)
                    << (confidence * 100.0f) << "%";
                statusText.setString(ss.str());
            }

            uiUpdateClock.restart();
            needsUIUpdate = false;
        }

        // Clear the window with a dark background
        window.clear(sf::Color(10, 10, 10));

        // Draw spectrograms
        liveSpectrogram.draw();
        staticSpectrogram.draw();

        // Draw minimal UI elements
        window.draw(statusText);
        window.draw(fpsText);

        // Display the window contents
        window.display();
    }

    // Clean up
    std::cout << "Shutting down..." << std::endl;
    g_processingActive = false;
    if (processingThread.joinable()) {
        processingThread.join();
    }

    capture.stop();
    fftw_cleanup_threads();

    return 0;
}