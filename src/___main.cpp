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

// Finally, project-specific headers
#include "AudioFingerprinter.h"
#include "AudioMatcher.h"
#include "SpectrogramVisualizer.h"
#include "AudioCapture.h"
#include "TestHarness.h"
#include "SpectrogramCache.h"

// Global atomic flag for controlling background processing
std::atomic<bool> g_processingActive(true);

// Background processing thread function
void backgroundProcessing(AudioMatcher* matcher, Spectrogram* liveSpectrogram,
    StaticSpectrogram* staticSpectrogram,
    std::atomic<double>* currentTimestamp,
    std::atomic<float>* currentConfidence) {
    sf::Clock updateTimer;

    while (g_processingActive) {
        // Only update at most every 100ms (10 times per second) to reduce CPU usage
        if (updateTimer.getElapsedTime().asMilliseconds() >= 100) {
            auto current_magnitudes = liveSpectrogram->getCurrentMagnitudes();

            if (!current_magnitudes.empty()) {
                auto matchResult = matcher->findMatchWithConfidence(current_magnitudes);

                // Update atomic values that main thread can read
                *currentTimestamp = matcher->getTimestamp(matchResult.first);
                *currentConfidence = matchResult.second;
            }

            updateTimer.restart();
        }

        // Sleep to prevent CPU thrashing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {
    // Set process priority to above normal
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);

    // Initialize FFTW threads for better performance
    fftw_init_threads();
    fftw_plan_with_nthreads(4);  // Use 4 threads for parallel execution

    // Create a single SFML window for both spectrograms
    sf::RenderWindow window(sf::VideoMode(1280, 720), "Audio Position Matcher",
        sf::Style::Default);

    // Use VSync for steady frame rate
    window.setVerticalSyncEnabled(true);
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

    // Initialize the audio capture with high priority
    AudioCapture capture;
    capture.setSpectrogram(&liveSpectrogram);
    if (!capture.start()) {
        std::cerr << "Failed to start audio capture" << std::endl;
        std::cerr << "Make sure audio devices are properly configured" << std::endl;
        return -1;
    }

    // Initialize test harness
    AudioMatchTestHarness testHarness(&matcher);

    // Create a font used for UI
    sf::Font font;
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
        std::cerr << "Failed to load font, using default" << std::endl;
    }

    // Create UI elements for test controls
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

    // Create atomic variables for thread communication
    std::atomic<double> currentTimestamp(0.0);
    std::atomic<float> currentConfidence(0.0f);

    // Start background processing thread
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

    // Test related variables
    sf::Clock testTimer;
    bool testActive = false;

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
                livePos = sf::Vector2f(50.0f, 50.0f);
                staticPos = sf::Vector2f(static_cast<float>(event.size.width) / 2.0f + 25.0f, 50.0f);

                liveSpectrogram.setPosition(livePos);
                staticSpectrogram.setPosition(staticPos);

                liveSpectrogram.handleResize();
                staticSpectrogram.handleResize();

                needsUIUpdate = true;
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
                            testActive = true;
                            testTimer.restart();
                        }
                        else {
                            testHarness.stopTest();
                            testButtonText.setString("Run Test");
                            statusText.setString("Test stopped");
                            testActive = false;
                        }
                        needsUIUpdate = true;
                    }
                }
            }
        }

        // Update UI only every 50ms (20 times per second) to improve performance
        if (uiUpdateClock.getElapsedTime().asMilliseconds() >= 50 || needsUIUpdate) {
            // Test mode or normal mode UI updates
            if (testActive) {
                if (testTimer.getElapsedTime().asSeconds() >= 0.1f) {
                    auto matchResult = testHarness.processTestFrame();

                    // Update status text
                    std::stringstream ss;
                    ss << "Test running... Confidence: " << std::fixed << std::setprecision(2)
                        << (matchResult.second * 100.0f) << "%";
                    statusText.setString(ss.str());

                    // Update position indicators
                    double timestamp = matcher.getTimestamp(matchResult.first);
                    liveSpectrogram.updatePositionIndicator(timestamp, matchResult.second);
                    staticSpectrogram.updatePositionIndicator(timestamp, matchResult.second);

                    testTimer.restart();
                }
            }
            else {
                // Use the values updated by background thread
                double timestamp = currentTimestamp.load();
                float confidence = currentConfidence.load();

                // Only update UI if we have values
                if (confidence > 0.0f || needsUIUpdate) {
                    // Update position indicators
                    liveSpectrogram.updatePositionIndicator(timestamp, confidence);
                    staticSpectrogram.updatePositionIndicator(timestamp, confidence);

                    // Update status text
                    std::stringstream ss;
                    ss << "Current position: " << std::fixed << std::setprecision(1)
                        << timestamp << "s  Confidence: "
                        << std::fixed << std::setprecision(0) << (confidence * 100.0f) << "%";
                    statusText.setString(ss.str());
                }
            }

            uiUpdateClock.restart();
            needsUIUpdate = false;
        }

        // Clear the window with a dark background
        window.clear(sf::Color(10, 10, 10));

        // Draw spectrograms
        liveSpectrogram.draw();
        staticSpectrogram.draw();

        // Draw UI elements
        window.draw(testButton);
        window.draw(testButtonText);
        window.draw(statusText);
        window.draw(fpsText);

        // Display the window contents
        window.display();
    }

    // Clean up
    g_processingActive = false;
    if (processingThread.joinable()) {
        processingThread.join();
    }

    capture.stop();
    fftw_cleanup_threads();

    return 0;
}