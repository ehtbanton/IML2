#pragma once

#include "SpectrogramBase.h"
#include <deque>
#include <mutex>
#include <chrono>

// Forward declaration for FFTW types
struct fftw_plan_s;
typedef struct fftw_plan_s* fftw_plan;
typedef double fftw_complex[2];

// Forward declaration
struct SharedMemory;

// Live spectrogram that processes real-time audio
class LiveSpectrogramImpl : public SpectrogramBase {
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
    LiveSpectrogramImpl(sf::RenderWindow& win, const std::string& title, const sf::Vector2f& position);
    ~LiveSpectrogramImpl();

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