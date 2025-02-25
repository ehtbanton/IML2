#define _USE_MATH_DEFINES
#include "SpectrogramVisualizer.h"
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

// Base class implementation
SpectrogramBase::SpectrogramBase(sf::RenderWindow& win, const std::string& titleText, const sf::Vector2f& position)
    : window(win),
    vertices(sf::Quads),
    title(titleText),
    spectrogramPosition(position),
    showPositionIndicator(false)
{
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
        std::cerr << "Failed to load font" << std::endl;
    }
}

sf::Color SpectrogramBase::getColor(float magnitude) {
    float db = 20 * std::log10(magnitude + 1e-9f);
    float normalized = (db + 50) / 40.0f;
    normalized = std::max(0.0f, std::min(1.0f, normalized));

    if (normalized < 0.4f) {
        return sf::Color(0, 0, static_cast<sf::Uint8>(normalized * 255 / 0.4f));
    }
    else if (normalized < 0.6f) {
        float t = (normalized - 0.4f) * 5.0f;
        return sf::Color(0, static_cast<sf::Uint8>(t * 255), 255);
    }
    else if (normalized < 0.8f) {
        float t = (normalized - 0.6f) * 5.0f;
        return sf::Color(static_cast<sf::Uint8>(t * 255), 255, static_cast<sf::Uint8>(255 * (1.0f - t)));
    }
    else {
        float t = (normalized - 0.8f) * 5.0f;
        return sf::Color(255, static_cast<sf::Uint8>(255 * (1.0f - t)), 0);
    }
}

sf::Text SpectrogramBase::createText(const sf::Font& font, const std::string& content, unsigned int size, const sf::Vector2f& pos) {
    sf::Text text;
    text.setFont(font);
    text.setString(content);
    text.setCharacterSize(size);
    text.setFillColor(sf::Color::White);
    text.setPosition(pos);
    return text;
}

void SpectrogramBase::updatePositionIndicator(double seconds, float confidence) {
    showPositionIndicator = true;

    // Format timestamp as MM:SS
    int minutes = static_cast<int>(seconds) / 60;
    int secs = static_cast<int>(seconds) % 60;

    std::stringstream ss;
    ss << minutes << ":" << std::setfill('0') << std::setw(2) << secs;
    positionText.setString(ss.str());

    std::stringstream cs;
    cs << "Confidence: " << std::fixed << std::setprecision(0) << (confidence * 100.0f) << "%";
    confidenceText.setString(cs.str());

    // Position along X axis based on seconds into the song
    float xRatio = std::min(1.0f, static_cast<float>(seconds / 30.0f)); // Scale to 30 second window
    float xPos = spectrogramPosition.x + xRatio * spectrogramSize.x;

    positionIndicator.setPosition(xPos, spectrogramPosition.y);
    positionText.setPosition(xPos - 15.0f, spectrogramPosition.y - 20.0f);
    confidenceText.setPosition(spectrogramPosition.x, spectrogramPosition.y - 40.0f);

    // Color based on confidence
    sf::Uint8 alpha = static_cast<sf::Uint8>(std::min(1.0f, confidence) * 255);
    positionIndicator.setFillColor(sf::Color(255, 0, 0, alpha));
}

void SpectrogramBase::setPosition(const sf::Vector2f& position) {
    spectrogramPosition = position;
    handleResize();
}

// Spectrogram (Live) implementation
Spectrogram::Spectrogram(sf::RenderWindow& win, const std::string& titleText, const sf::Vector2f& position)
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
    vertices.resize(HISTORY_SIZE * (FFT_SIZE / 2) * 4);
    initializeUI();
    updateVertexPositions();
}

void Spectrogram::initializeFFT() {
    fft_in = fftw_alloc_real(FFT_SIZE);
    fft_out = fftw_alloc_complex(FFT_SIZE / 2 + 1);
    fft_plan = fftw_plan_dft_r2c_1d(FFT_SIZE, fft_in, fft_out, FFTW_MEASURE);

    for (size_t i = 0; i < FFT_SIZE; ++i) {
        window_function[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (FFT_SIZE - 1)));
    }
}

Spectrogram::~Spectrogram() {
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

void Spectrogram::initializeUI() {
    float windowWidth = static_cast<float>(window.getSize().x);
    float windowHeight = static_cast<float>(window.getSize().y);
    spectrogramSize = sf::Vector2f(windowWidth * 0.45f, windowHeight * 0.45f);

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
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y - 20.0f));
    positionText.setFillColor(sf::Color::Red);

    confidenceText = createText(font, "Confidence: 0%", 14,
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y - 40.0f));
    confidenceText.setFillColor(sf::Color::Red);

    // Title text
    titleText = createText(font, title, 16,
        sf::Vector2f(spectrogramPosition.x + spectrogramSize.x / 2 - 80,
            spectrogramPosition.y - 60.0f));
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

void Spectrogram::updateVertexPositions() {
    const float sampleRate = 48000.0f;
    const float minFreq = 10.0f;
    const float maxFreq = 24000.0f;

    float freqPerBin = sampleRate / static_cast<float>(FFT_SIZE);

    for (size_t x = 0; x < HISTORY_SIZE; ++x) {
        for (size_t y = 0; y < FFT_SIZE / 2; ++y) {
            size_t idx = (x * FFT_SIZE / 2 + y) * 4;

            float xRatio = static_cast<float>(x) / static_cast<float>(HISTORY_SIZE);
            float xpos = spectrogramPosition.x + (xRatio * spectrogramSize.x);
            float width = spectrogramSize.x / static_cast<float>(HISTORY_SIZE);

            float freq = static_cast<float>(y) * freqPerBin;
            freq = std::max(minFreq, std::min(maxFreq, freq));

            float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
            yRatio = std::max(0.0f, std::min(1.0f, yRatio));
            float ypos = spectrogramPosition.y + ((1.0f - yRatio) * spectrogramSize.y);

            float nextFreq = static_cast<float>(y + 1) * freqPerBin;
            nextFreq = std::max(minFreq, std::min(maxFreq, nextFreq));
            float nextYRatio = std::log10(nextFreq / minFreq) / std::log10(maxFreq / minFreq);
            nextYRatio = std::max(0.0f, std::min(1.0f, nextYRatio));
            float nextYpos = spectrogramPosition.y + ((1.0f - nextYRatio) * spectrogramSize.y);

            vertices[idx].position = sf::Vector2f(xpos, ypos);
            vertices[idx + 1].position = sf::Vector2f(xpos + width, ypos);
            vertices[idx + 2].position = sf::Vector2f(xpos + width, nextYpos);
            vertices[idx + 3].position = sf::Vector2f(xpos, nextYpos);
        }
    }
}

void Spectrogram::processSamples(const std::vector<float>& samples) {
    static std::vector<float> buffer;
    buffer.insert(buffer.end(), samples.begin(), samples.end());

    if (first_sample) {
        start_time = std::chrono::steady_clock::now();
        first_sample = false;
    }

    while (buffer.size() >= FFT_SIZE) {
        const float scale = 10.0f;
        for (size_t i = 0; i < FFT_SIZE; ++i) {
            fft_in[i] = buffer[i] * window_function[i] * scale;
        }

        fftw_execute(fft_plan);

        std::vector<float> magnitudes(FFT_SIZE / 2);
        for (size_t i = 0; i < FFT_SIZE / 2; ++i) {
            float real = static_cast<float>(fft_out[i][0]);
            float imag = static_cast<float>(fft_out[i][1]);
            magnitudes[i] = std::sqrt(real * real + imag * imag) / (FFT_SIZE * 2.0f);
        }

        {
            std::unique_lock<std::mutex> lock(history_mutex);
            magnitude_history.push_back(magnitudes);

            if (sharedMem) {
                memcpy(sharedMem->magnitudes, magnitudes.data(), sizeof(float) * (FFT_SIZE / 2));
                sharedMem->timestamp = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start_time).count();
                sharedMem->new_data_available = true;
            }

            auto now = std::chrono::steady_clock::now();
            double seconds = std::chrono::duration<double>(now - start_time).count();
            column_timestamps.push_back(seconds);

            while (magnitude_history.size() > HISTORY_SIZE) {
                magnitude_history.pop_front();
                column_timestamps.pop_front();
            }
        }

        buffer.erase(buffer.begin(), buffer.begin() + HOP_SIZE);
    }
}

void Spectrogram::draw() {
    std::unique_lock<std::mutex> lock(history_mutex);

    window.draw(spectrogramBackground);
    window.draw(titleText);

    for (size_t x = 0; x < magnitude_history.size(); ++x) {
        const auto& magnitudes = magnitude_history[x];
        for (size_t y = 0; y < FFT_SIZE / 2; ++y) {
            size_t idx = (x * FFT_SIZE / 2 + y) * 4;
            sf::Color color = getColor(magnitudes[y]);

            for (int i = 0; i < 4; ++i) {
                vertices[idx + i].color = color;
            }
        }
    }
    window.draw(vertices);

    // Draw position indicator if active
    if (showPositionIndicator) {
        window.draw(positionIndicator);
        window.draw(positionText);
        window.draw(confidenceText);
    }

    for (const auto& label : frequencyLabels) {
        window.draw(label);
    }
    for (const auto& label : timeLabels) {
        window.draw(label);
    }
}

void Spectrogram::handleResize() {
    initializeUI();
    updateVertexPositions();
}

double Spectrogram::getTimeWindow() const {
    return time_window;
}

std::vector<float> Spectrogram::getCurrentMagnitudes() {
    if (magnitude_history.empty()) {
        return std::vector<float>();
    }
    return magnitude_history.back();
}

// StaticSpectrogram implementation
StaticSpectrogram::StaticSpectrogram(sf::RenderWindow& win, const std::vector<std::vector<float>>& data,
    const std::string& titleText, const sf::Vector2f& position)
    : SpectrogramBase(win, titleText, position), spectrogramData(data)
{
    initializeUI();
    updateVertexPositions();
}

void StaticSpectrogram::initializeUI() {
    float winW = static_cast<float>(window.getSize().x);
    float winH = static_cast<float>(window.getSize().y);

    spectrogramSize = sf::Vector2f(winW * 0.45f, winH * 0.45f);

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
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y - 20.0f));
    positionText.setFillColor(sf::Color::Red);

    confidenceText = createText(font, "Confidence: 0%", 14,
        sf::Vector2f(spectrogramPosition.x, spectrogramPosition.y - 40.0f));
    confidenceText.setFillColor(sf::Color::Red);

    // Title text
    titleText = createText(font, title, 16,
        sf::Vector2f(spectrogramPosition.x + spectrogramSize.x / 2 - 80,
            spectrogramPosition.y - 60.0f));
    titleText.setFillColor(sf::Color::White);

    frequencyLabels.clear();
    float sampleRate = 48000.0f;
    float freqPerBin = sampleRate / static_cast<float>(FFT_SIZE);
    const float minFreq = 10.0f;
    const float maxFreq = 24000.0f;
    int numFreqLabels = 8;
    for (int i = 0; i < numFreqLabels; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(numFreqLabels - 1);
        float freq = minFreq * std::pow(maxFreq / minFreq, t);
        std::stringstream ss;
        if (freq >= 1000.0f)
            ss << std::fixed << std::setprecision(1) << (freq / 1000.0f) << " kHz";
        else
            ss << std::fixed << std::setprecision(0) << freq << " Hz";

        float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
        float yPos = spectrogramPosition.y + spectrogramSize.y * (1.0f - yRatio);
        frequencyLabels.push_back(createText(font, ss.str(), 10, sf::Vector2f(spectrogramPosition.x - 55.0f, yPos - 5.0f)));
    }

    // Create time labels along the horizontal axis.
    timeLabels.clear();
    size_t numFrames = spectrogramData.size();
    float secondsPerFrame = 512.0f / 48000.0f;
    float totalTime = static_cast<float>(numFrames) * secondsPerFrame;

    // Corrected loop for time labels (fixed to 30 seconds)
    for (int t = 0; t <= 30; t += 5) {
        float tRatio = static_cast<float>(t) / 30.0f; // Use a fixed 30-second window
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << t << " s";
        float xPos = spectrogramPosition.x + spectrogramSize.x * tRatio;
        timeLabels.push_back(createText(font, ss.str(), 10, sf::Vector2f(xPos - 15.0f, spectrogramPosition.y + spectrogramSize.y + 10.0f)));
    }
}

void StaticSpectrogram::updateVertexPositions() {
    size_t numFrames = spectrogramData.size();
    size_t bins = FFT_SIZE / 2;
    vertices.setPrimitiveType(sf::Quads);
    vertices.resize(numFrames * bins * 4);
    float frameWidth = spectrogramSize.x / static_cast<float>(numFrames);
    float sampleRate = 48000.0f;
    float freqPerBin = sampleRate / static_cast<float>(FFT_SIZE);
    const float minFreq = 10.0f;
    const float maxFreq = 24000.0f;

    for (size_t x = 0; x < numFrames; ++x) {
        for (size_t y = 0; y < bins; ++y) {
            size_t idx = (x * bins + y) * 4;
            float xpos = spectrogramPosition.x + static_cast<float>(x) * frameWidth;
            float freq = static_cast<float>(y) * freqPerBin;
            freq = std::max(minFreq, std::min(maxFreq, freq));
            float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
            float ypos = spectrogramPosition.y + spectrogramSize.y * (1.0f - yRatio);

            float nextFreq = static_cast<float>(y + 1) * freqPerBin;
            nextFreq = std::max(minFreq, std::min(maxFreq, nextFreq));
            float nextYRatio = std::log10(nextFreq / minFreq) / std::log10(maxFreq / minFreq);
            float nextYpos = spectrogramPosition.y + spectrogramSize.y * (1.0f - nextYRatio);

            vertices[idx].position = sf::Vector2f(xpos, ypos);
            vertices[idx + 1].position = sf::Vector2f(xpos + frameWidth, ypos);
            vertices[idx + 2].position = sf::Vector2f(xpos + frameWidth, nextYpos);
            vertices[idx + 3].position = sf::Vector2f(xpos, nextYpos);
        }
    }
}

void StaticSpectrogram::draw() {
    window.draw(spectrogramBackground);
    window.draw(titleText);

    size_t numFrames = spectrogramData.size();
    size_t bins = FFT_SIZE / 2;

    for (size_t x = 0; x < numFrames; ++x) {
        const auto& frame = spectrogramData[x];
        for (size_t y = 0; y < bins && y < frame.size(); ++y) {
            size_t idx = (x * bins + y) * 4;
            sf::Color color = getColor(frame[y]);
            vertices[idx].color = color;
            vertices[idx + 1].color = color;
            vertices[idx + 2].color = color;
            vertices[idx + 3].color = color;
        }
    }
    window.draw(vertices);

    // Draw position indicator if active
    if (showPositionIndicator) {
        window.draw(positionIndicator);
        window.draw(positionText);
        window.draw(confidenceText);
    }

    for (const auto& label : frequencyLabels)
        window.draw(label);
    for (const auto& label : timeLabels)
        window.draw(label);
}

void StaticSpectrogram::handleResize() {
    initializeUI();
    updateVertexPositions();
}