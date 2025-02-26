#pragma once

// First define these to prevent Windows macro conflicts
#define NOMINMAX 
#define WIN32_LEAN_AND_MEAN

// SFML headers MUST come before Windows headers
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

// Now it's safe to include Windows headers
#include <windows.h>

// Standard library headers
#include <vector>
#include <deque>
#include <mutex>
#include <chrono>

// Forward declaration for FFTW types
struct fftw_plan_s;
typedef struct fftw_plan_s* fftw_plan;
typedef double fftw_complex[2];

// Forward declaration
struct SharedMemory;

// Base class for both static and live spectrograms
class SpectrogramBase {
protected:
    static const size_t FFT_SIZE = 1024;

    sf::RenderWindow& window;
    sf::VertexArray vertices;
    sf::Font font;
    sf::RectangleShape spectrogramBackground;
    sf::Vector2f spectrogramPosition;
    sf::Vector2f spectrogramSize;
    std::vector<sf::Text> frequencyLabels;
    std::vector<sf::Text> timeLabels;

    // Position indicator
    sf::RectangleShape positionIndicator;
    sf::Text positionText;
    sf::Text confidenceText;
    sf::Text titleText;
    bool showPositionIndicator;

    // Title for this spectrogram
    std::string title;

    sf::Color getColor(float magnitude);
    sf::Text createText(const sf::Font& font, const std::string& content, unsigned int size, const sf::Vector2f& pos);

    // Add normalization function to ensure consistency
    std::vector<float> normalizeSpectrum(const std::vector<float>& magnitudes);

    // New methods for color matching
    void matchStaticSpectrogramColor(sf::Color& color, float normalizedFreq);
    void applyFrequencyBoosting(std::vector<float>& magnitudes);

public:
    SpectrogramBase(sf::RenderWindow& win, const std::string& title, const sf::Vector2f& position);
    virtual ~SpectrogramBase() {}

    virtual void draw() = 0;
    virtual void handleResize() = 0;
    virtual void updatePositionIndicator(double seconds, float confidence);
    void setPosition(const sf::Vector2f& position);
    void setSize(const sf::Vector2f& size);
};

// Live spectrogram that processes real-time audio
class Spectrogram : public SpectrogramBase {
private:
    static const size_t HOP_SIZE = 512;
    static const size_t HISTORY_SIZE = 2812;
    std::vector<float> overlapBuffer;

    // Shared memory handles
    HANDLE hMapFile;
    SharedMemory* sharedMem;

    // FFT and audio processing
    std::vector<double> window_function;
    std::deque<std::vector<float>> magnitude_history;
    std::mutex history_mutex;
    std::deque<double> column_timestamps;
    std::chrono::steady_clock::time_point start_time;
    bool first_sample;

    // FFTW variables
    double* fft_in;
    fftw_complex* fft_out;
    fftw_plan fft_plan;

    // View parameters
    double time_window;
    float currentConfidence;

    void initializeFFT();
    void initializeUI();
    void updateVertexPositions();

public:
    Spectrogram(sf::RenderWindow& win, const std::string& title, const sf::Vector2f& position);
    ~Spectrogram();

    void processSamples(const std::vector<float>& samples);
    void draw() override;
    void handleResize() override;
    double getTimeWindow() const;
    std::vector<float> getCurrentMagnitudes();
    void resetState() {
        std::unique_lock<std::mutex> lock(history_mutex);
        magnitude_history.clear();
        column_timestamps.clear();
        first_sample = true;
    }
};

// Static spectrogram that displays pre-processed audio
class StaticSpectrogram : public SpectrogramBase {
private:
    const std::vector<std::vector<float>>& spectrogramData;

    void initializeUI();
    void updateVertexPositions();

public:
    StaticSpectrogram(sf::RenderWindow& win, const std::vector<std::vector<float>>& data,
        const std::string& title, const sf::Vector2f& position);

    void draw() override;
    void handleResize() override;
};