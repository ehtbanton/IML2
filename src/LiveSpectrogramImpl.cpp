#define _USE_MATH_DEFINES
#include "LiveSpectrogramImpl.h"
#include <fftw3.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

#pragma pack(push, 1)
struct SharedMemory {
    float magnitudes[512];  // Half of FFT_SIZE
    double timestamp;
    bool new_data_available;
};
#pragma pack(pop)

LiveSpectrogramImpl::LiveSpectrogramImpl(sf::RenderWindow& win, const std::string& titleText, const sf::Vector2f& position)
    : SpectrogramBase(win, titleText, position),
    window_function(FFT_SIZE),
    first_sample(true),
    hMapFile(NULL),
    sharedMem(NULL),
    currentConfidence(0.0f),
    fft_in(nullptr),
    fft_out(nullptr),
    fft_plan(nullptr)
{
    const char* sharedMemName = "Local\\SpectrogramData";
    hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(SharedMemory),
        sharedMemName
    );

    if (hMapFile) {
        sharedMem = static_cast<SharedMemory*>(MapViewOfFile(
            hMapFile,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            sizeof(SharedMemory)
        ));

        if (sharedMem) {
            sharedMem->new_data_available = false;
            memset(sharedMem->magnitudes, 0, sizeof(sharedMem->magnitudes));
            sharedMem->timestamp = 0.0;
        }
    }

    initializeFFT();

    // Initialize empty vertex array - don't set the size yet
    vertices.setPrimitiveType(sf::Quads);

    // Initialize UI elements
    spectrogramSize = sf::Vector2f(800, 600); // Default size, will be updated later
    initializeUI();

    // Only resize vertex array once we know the size
    size_t totalVertices = HISTORY_SIZE * (FFT_SIZE / 2) * 4;
    vertices.resize(totalVertices);

    // Now update positions
    updateVertexPositions();
}

void LiveSpectrogramImpl::initializeFFT() {
    fft_in = fftw_alloc_real(FFT_SIZE);
    fft_out = fftw_alloc_complex(FFT_SIZE / 2 + 1);
    fft_plan = fftw_plan_dft_r2c_1d(FFT_SIZE, fft_in, fft_out, FFTW_MEASURE);

    for (size_t i = 0; i < FFT_SIZE; ++i) {
        window_function[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (FFT_SIZE - 1)));
    }
}

LiveSpectrogramImpl::~LiveSpectrogramImpl() {
    if (sharedMem) {
        UnmapViewOfFile(sharedMem);
    }
    if (hMapFile) {
        CloseHandle(hMapFile);
    }

    if (fft_plan) {
        fftw_destroy_plan(fft_plan);
    }
    if (fft_in) {
        fftw_free(fft_in);
    }
    if (fft_out) {
        fftw_free(fft_out);
    }
}

void LiveSpectrogramImpl::initializeUI() {
    // Use the spectrogramSize directly (no recalculation needed)
    spectrogramBackground.setSize(spectrogramSize);
    spectrogramBackground.setPosition(spectrogramPosition);
    spectrogramBackground.setFillColor(sf::Color(20, 20, 20));
    spectrogramBackground.setOutlineColor(sf::Color::White);
    spectrogramBackground.setOutlineThickness(1.0f);

    // Initialize position indicator
    positionIndicator.setSize(sf::Vector2f(2.0f, spectrogramSize.y));
    positionIndicator.setFillColor(sf::Color(255, 0, 0, 200));
    positionIndicator.setPosition(spectrogramPosition.x, spectrogramPosition.y);

    positionText = createText(font, "0:00", 14,
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y + spectrogramSize.y + 15.0f));
    positionText.setFillColor(sf::Color::Red);

    confidenceText = createText(font, "Confidence: 0%", 14,
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y + spectrogramSize.y + 15.0f));
    confidenceText.setFillColor(sf::Color::Red);

    // Title text
    titleText = createText(font, title, 16,
        sf::Vector2f(spectrogramPosition.x + spectrogramSize.x / 2 - 80,
            spectrogramPosition.y - 30.0f));
    titleText.setFillColor(sf::Color::White);

    frequencyLabels.clear();
    float freqPerBin = 48000.0f / static_cast<float>(FFT_SIZE);
    int numLabels = 8;
    const float minFreq = 10.0f;
    const float maxFreq = 24000.0f;

    for (int i = 0; i < numLabels; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(numLabels - 1);
        float freq = minFreq * std::pow(maxFreq / minFreq, t);

        std::stringstream ss;
        if (freq >= 1000.0f) {
            ss << std::fixed << std::setprecision(1) << freq / 1000.0f << " kHz";
        }
        else {
            ss << std::fixed << std::setprecision(0) << freq << " Hz";
        }

        float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
        float yPos = spectrogramPosition.y + spectrogramSize.y * (1.0f - yRatio);

        sf::Text label = createText(font, ss.str(), 10,
            sf::Vector2f(spectrogramPosition.x - 55.0f, yPos - 5.0f));
        frequencyLabels.push_back(label);
    }

    timeLabels.clear();
    float secondsPerColumn = static_cast<float>(HOP_SIZE) / 48000.0f;
    float totalTime = secondsPerColumn * static_cast<float>(HISTORY_SIZE);
    time_window = totalTime;

    for (int i = 0; i <= 30; i += 5) {
        float t = static_cast<float>(i) / 30.0f;
        float time = t * totalTime;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << time << "s";

        float xPos = spectrogramPosition.x + spectrogramSize.x * t;
        sf::Text label = createText(font, ss.str(), 10,
            sf::Vector2f(xPos - 15.0f, spectrogramPosition.y + spectrogramSize.y + 10.0f));
        timeLabels.push_back(label);
    }
}

void LiveSpectrogramImpl::updateVertexPositions() {
    // Calculate frequency scale parameters
    const float sampleRate = 48000.0f;
    const float minFreq = 10.0f;
    const float maxFreq = 24000.0f;
    float freqPerBin = sampleRate / static_cast<float>(FFT_SIZE);

    // Calculate the fixed segment width
    float segmentWidth = spectrogramSize.x / static_cast<float>(HISTORY_SIZE);

    // Get current vertex count
    size_t vertexCount = vertices.getVertexCount();

    // Skip if no vertices
    if (vertexCount == 0) {
        return;
    }

    // Calculate how many frames we can place in the vertex array
    size_t framesPerRow = FFT_SIZE / 2;
    size_t maxFrames = vertexCount / (framesPerRow * 4);

    // Ensure we don't exceed vertex array bounds
    maxFrames = std::min(maxFrames, HISTORY_SIZE);

    // Update vertex positions with fixed segment width
    for (size_t x = 0; x < maxFrames; ++x) {
        float xPosition = spectrogramPosition.x + (x * segmentWidth);

        for (size_t y = 0; y < framesPerRow; ++y) {
            // Calculate vertex index
            size_t idx = (x * framesPerRow + y) * 4;

            // Skip if out of bounds
            if (idx + 3 >= vertexCount) {
                continue;
            }

            // Calculate y position values with logarithmic frequency scale
            float freq = static_cast<float>(y) * freqPerBin;
            freq = std::max(minFreq, std::min(maxFreq, freq));

            float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
            yRatio = std::max(0.0f, std::min(1.0f, yRatio));
            float ypos = spectrogramPosition.y + ((1.0f - yRatio) * spectrogramSize.y);

            // Calculate next y position (for bottom of the quad)
            float nextFreq = static_cast<float>(y + 1) * freqPerBin;
            nextFreq = std::max(minFreq, std::min(maxFreq, nextFreq));
            float nextYRatio = std::log10(nextFreq / minFreq) / std::log10(maxFreq / minFreq);
            nextYRatio = std::max(0.0f, std::min(1.0f, nextYRatio));
            float nextYpos = spectrogramPosition.y + ((1.0f - nextYRatio) * spectrogramSize.y);

            // Set vertex positions for this quad with fixed segment width
            vertices[idx].position = sf::Vector2f(xPosition, ypos);
            vertices[idx + 1].position = sf::Vector2f(xPosition + segmentWidth, ypos);
            vertices[idx + 2].position = sf::Vector2f(xPosition + segmentWidth, nextYpos);
            vertices[idx + 3].position = sf::Vector2f(xPosition, nextYpos);
        }
    }
}

void LiveSpectrogramImpl::processSamples(const std::vector<float>& samples) {
    // Ignore empty sample sets
    if (samples.empty()) {
        return;
    }

    // Use the minimum size needed for processing
    size_t sampleSize = std::min(samples.size(), FFT_SIZE);
    bool isSilence = true;

    // Check if this is a silent frame (check every 8th sample for efficiency)
    for (size_t i = 0; i < sampleSize; i += 8) {
        if (i < sampleSize && std::abs(samples[i]) > 1e-6f) {
            isSilence = false;
            break;
        }
    }

    // Initialize time tracking on first sample
    if (first_sample) {
        start_time = std::chrono::steady_clock::now();
        first_sample = false;

        if (vertices.getVertexCount() < HISTORY_SIZE * (FFT_SIZE / 2) * 4) {
            vertices.resize(HISTORY_SIZE * (FFT_SIZE / 2) * 4);
            updateVertexPositions();
        }

        std::lock_guard<std::mutex> lock(history_mutex);
        magnitude_history.clear();
        column_timestamps.clear();
    }

    // Apply window function to samples
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        // Use zeros for any samples beyond what was provided
        fft_in[i] = static_cast<double>(i < sampleSize ? samples[i] * window_function[i] : 0.0);
    }

    // Execute FFT even for silence (to maintain proper timing)
    fftw_execute(fft_plan);

    // Calculate magnitude spectrum
    std::vector<float> magnitudes(FFT_SIZE / 2, 0.0f);
    for (size_t i = 0; i < FFT_SIZE / 2; ++i) {
        float real = static_cast<float>(fft_out[i][0]);
        float imag = static_cast<float>(fft_out[i][1]);
        magnitudes[i] = std::sqrt(real * real + imag * imag) / (static_cast<float>(FFT_SIZE) * 0.25f);
    }

    // Apply frequency boosting (except for silence frames)
    if (!isSilence) {
        SpectrogramBase::applyFrequencyBoosting(magnitudes);
    }

    // Add this frame to history with proper thread safety
    {
        std::lock_guard<std::mutex> lock(history_mutex);

        // Calculate timestamp with scaling factor
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double seconds = elapsed * 0.5; // Scale time by 0.5 to match reference

        // Add the frame to history
        magnitude_history.push_back(magnitudes);
        column_timestamps.push_back(seconds);

        // Maintain fixed history size
        while (magnitude_history.size() > HISTORY_SIZE) {
            magnitude_history.pop_front();
            column_timestamps.pop_front();
        }

        // Update shared memory if available
        if (sharedMem && !magnitudes.empty()) {
            memcpy(sharedMem->magnitudes, magnitudes.data(),
                std::min(sizeof(float) * magnitudes.size(), sizeof(sharedMem->magnitudes)));
            sharedMem->timestamp = seconds;
            sharedMem->new_data_available = true;
        }
    }
}

void LiveSpectrogramImpl::draw() {
    // Create a temporary copy of the data to avoid locking during drawing
    std::deque<std::vector<float>> history_copy;
    {
        std::lock_guard<std::mutex> lock(history_mutex);
        if (!magnitude_history.empty()) {
            // Use a quick reference instead of copying the whole data
            history_copy = magnitude_history;
        }
    }

    // Draw background elements
    window.draw(spectrogramBackground);
    window.draw(titleText);

    // Draw labels
    for (const auto& label : frequencyLabels) {
        window.draw(label);
    }
    for (const auto& label : timeLabels) {
        window.draw(label);
    }

    if (showPositionIndicator) {
        window.draw(positionIndicator);
    }

    // Skip spectrum drawing if no data
    if (history_copy.empty()) {
        return;
    }

    // Ensure vertex array is properly sized (only do this when needed)
    if (vertices.getVertexCount() != HISTORY_SIZE * (FFT_SIZE / 2) * 4) {
        vertices.resize(HISTORY_SIZE * (FFT_SIZE / 2) * 4);
        updateVertexPositions();
    }

    // Use fixed-width segments for each frame
    size_t historySize = history_copy.size();
    size_t binsPerFrame = FFT_SIZE / 2;

    // Calculate how many display columns we'll actually use
    size_t displayColumns = std::min(historySize, HISTORY_SIZE);

    // Calculate fixed segment width
    float segmentWidth = spectrogramSize.x / HISTORY_SIZE;

    // CRITICAL FIX: Find the last non-silence frame index to stop drawing at that point
    size_t lastNonSilenceIndex = 0;
    for (size_t i = history_copy.size(); i > 0; --i) {
        const auto& magnitudes = history_copy[i - 1];
        bool hasSignal = false;

        // Check a few samples to see if there's any signal
        for (size_t j = 0; j < magnitudes.size(); j += 16) {
            if (j < magnitudes.size() && magnitudes[j] > 1e-6f) {
                hasSignal = true;
                break;
            }
        }

        if (hasSignal) {
            lastNonSilenceIndex = i;
            break;
        }
    }

    // Process each frame in our data
    for (size_t i = 0; i < displayColumns; i++) {
        // Skip if we're past the last non-silence frame at the edge
        if (i > lastNonSilenceIndex && i > displayColumns * 0.9f) {
            continue;
        }

        // Get the corresponding frame from history
        if (i >= history_copy.size()) continue;

        const auto& magnitudes = history_copy[i];
        if (magnitudes.empty()) continue;

        // Normalize the frame to match static spectrogram intensity
        std::vector<float> normalizedMagnitudes = normalizeSpectrum(magnitudes);

        // Calculate position for this frame
        float xPosition = spectrogramPosition.x + (i * segmentWidth);

        // Update colors for each frequency bin
        for (size_t y = 0; y < binsPerFrame && y < normalizedMagnitudes.size(); ++y) {
            // Calculate vertex array index
            size_t idx = (i * binsPerFrame + y) * 4;

            // Skip if out of bounds
            if (idx + 3 >= vertices.getVertexCount()) continue;

            // Set vertex positions for this frame segment (fixed width)
            vertices[idx].position = sf::Vector2f(xPosition, vertices[idx].position.y);
            vertices[idx + 1].position = sf::Vector2f(xPosition + segmentWidth, vertices[idx + 1].position.y);
            vertices[idx + 2].position = sf::Vector2f(xPosition + segmentWidth, vertices[idx + 2].position.y);
            vertices[idx + 3].position = sf::Vector2f(xPosition, vertices[idx + 3].position.y);

            // Calculate normalized frequency for color adjustments
            float normalizedFreq = static_cast<float>(y) / binsPerFrame;

            // Set color for this segment with frequency-dependent adjustments
            sf::Color color = getColor(normalizedMagnitudes[y]);
            SpectrogramBase::matchStaticSpectrogramColor(color, normalizedFreq);

            for (int j = 0; j < 4; ++j) {
                vertices[idx + j].color = color;
            }
        }
    }

    // Draw the vertex array
    window.draw(vertices);
}

void LiveSpectrogramImpl::handleResize() {
    initializeUI();
    updateVertexPositions();
}

double LiveSpectrogramImpl::getTimeWindow() const {
    return time_window;
}

std::vector<float> LiveSpectrogramImpl::getCurrentMagnitudes() {
    if (magnitude_history.empty()) {
        return std::vector<float>();
    }
    std::lock_guard<std::mutex> lock(history_mutex);
    return magnitude_history.empty() ? std::vector<float>() : magnitude_history.back();
}