#define _USE_MATH_DEFINES
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <fftw3.h>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <ctime>

#pragma comment(lib, "ole32.lib")

// Shared memory structure for IPC
#pragma pack(push, 1)
struct SharedMemory {
    float magnitudes[512];  // Half of FFT_SIZE
    double timestamp;
    bool new_data_available;
};
#pragma pack(pop)

// Helper function to create text elements
sf::Text createText(const sf::Font& font, const std::string& content, unsigned int size, const sf::Vector2f& position) {
    sf::Text text;
    text.setFont(font);
    text.setString(content);
    text.setCharacterSize(size);
    text.setFillColor(sf::Color::White);
    text.setPosition(position);
    return text;
}

class AudioAnalyzer {
public:
    const std::vector<std::vector<float>>& getReferenceSpectogram() const {
        return reference_spectogram;
    }

    void setReferenceSpectogram(const std::vector<std::vector<float>>& data) {
        reference_spectogram = data;
    }

private:
    static const size_t FFT_SIZE = 1024;
    static const size_t HOP_SIZE = 512;
    std::vector<double> window_function;

    // FFTW variables
    double* fft_in;
    fftw_complex* fft_out;
    fftw_plan fft_plan;

    // Store frequency data for the reference audio
    std::vector<std::vector<float>> reference_spectogram;

    void initializeFFT() {
        // Initialize FFTW
        fft_in = fftw_alloc_real(FFT_SIZE);
        fft_out = fftw_alloc_complex(FFT_SIZE / 2 + 1);
        fft_plan = fftw_plan_dft_r2c_1d(FFT_SIZE, fft_in, fft_out, FFTW_MEASURE);

        // Create Hanning window
        window_function.resize(FFT_SIZE);
        for (size_t i = 0; i < FFT_SIZE; ++i) {
            window_function[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (FFT_SIZE - 1)));
        }
    }

public:
    AudioAnalyzer() {
        initializeFFT();
    }

    ~AudioAnalyzer() {
        fftw_destroy_plan(fft_plan);
        fftw_free(fft_in);
        fftw_free(fft_out);
    }

    bool analyzeFile(const std::string& filepath) {
        sf::SoundBuffer buffer;
        if (!buffer.loadFromFile(filepath)) {
            std::cerr << "Failed to load audio file: " << filepath << std::endl;
            return false;
        }

        const sf::Int16* samples = buffer.getSamples();
        size_t sampleCount = buffer.getSampleCount();
        unsigned int channels = buffer.getChannelCount();
        unsigned int sampleRate = buffer.getSampleRate(); // Get sample rate from buffer

        std::vector<float> monoSamples;
        monoSamples.reserve(sampleCount / channels);

        for (size_t i = 0; i < sampleCount; i += channels) {
            float sum = 0.0f;
            for (unsigned int ch = 0; ch < channels; ++ch) {
                sum += samples[i + ch];
            }
            monoSamples.push_back(sum / (channels * 32768.0f));
        }

        reference_spectogram.clear();
        const float scale = 10.0f;

        // Calculate the number of frames to process for approximately 30 seconds
        size_t targetFrameCount = static_cast<size_t>((30.0 * sampleRate) / HOP_SIZE);
        size_t frameCount = 0;

        for (size_t i = 0; i + FFT_SIZE <= monoSamples.size() && frameCount < targetFrameCount; i += HOP_SIZE) { // Limit frames
            for (size_t j = 0; j < FFT_SIZE; ++j) {
                fft_in[j] = monoSamples[i + j] * window_function[j] * scale;
            }

            fftw_execute(fft_plan);

            std::vector<float> magnitudes(FFT_SIZE / 2);
            for (size_t j = 0; j < FFT_SIZE / 2; ++j) {
                float real = fft_out[j][0];
                float imag = fft_out[j][1];
                magnitudes[j] = std::sqrt(real * real + imag * imag) / (FFT_SIZE * 2.0f);
            }

            reference_spectogram.push_back(magnitudes);
            frameCount++; // Increment frame count
        }

        std::cout << "Static Spectrogram: Processed " << reference_spectogram.size() << " frames (approx. "
            << (static_cast<double>(reference_spectogram.size() * HOP_SIZE) / sampleRate) << " seconds)" << std::endl;


        return true;
    }

    size_t findMatch(const std::vector<float>& current_magnitudes,
        size_t search_window = 100,
        float weight_decay = 0.95f) {
        if (reference_spectogram.empty() || current_magnitudes.empty()) {
            return 0;
        }

        static size_t last_match_position = 0;
        size_t start_pos = (last_match_position > search_window) ?
            last_match_position - search_window : 0;
        size_t end_pos = std::min(last_match_position + search_window,
            reference_spectogram.size());

        float best_match_score = -1.0f;
        size_t best_match_position = last_match_position;

        for (size_t i = start_pos; i < end_pos; ++i) {
            const auto& ref_magnitudes = reference_spectogram[i];

            float position_weight = std::pow(weight_decay,
                std::abs(static_cast<int>(i) - static_cast<int>(last_match_position)));

            float correlation = 0.0f;
            float ref_energy = 0.0f;
            float current_energy = 0.0f;

            for (size_t j = 0; j < std::min(ref_magnitudes.size(), current_magnitudes.size()); ++j) {
                correlation += ref_magnitudes[j] * current_magnitudes[j];
                ref_energy += ref_magnitudes[j] * ref_magnitudes[j];
                current_energy += current_magnitudes[j] * current_magnitudes[j];
            }

            float normalized_correlation = correlation /
                (std::sqrt(ref_energy) * std::sqrt(current_energy) + 1e-6f);

            float match_score = normalized_correlation * position_weight;

            if (match_score > best_match_score) {
                best_match_score = match_score;
                best_match_position = i;
            }
        }

        last_match_position = best_match_position;
        return best_match_position;
    }

    double getTimestamp(size_t position) {
        return position * (static_cast<double>(HOP_SIZE) / 48000.0);
    }
};




class Spectrogram {
private:
    static const size_t FFT_SIZE = 1024;
    static const size_t HOP_SIZE = 512;
    static const size_t HISTORY_SIZE = 2812;
    std::vector<float> overlapBuffer;

    // Shared memory handles
    HANDLE hMapFile;
    SharedMemory* sharedMem;

    // SFML window reference
    sf::RenderWindow& window;
    sf::VertexArray vertices;
    sf::Font font;
    sf::RectangleShape spectrogramBackground;

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
    sf::Vector2f spectrogramPosition;
    sf::Vector2f spectrogramSize;
    std::vector<sf::Text> frequencyLabels;
    std::vector<sf::Text> timeLabels;
    double time_window;

    sf::Color getColor(float magnitude) {
        float db = 20 * std::log10(magnitude + 1e-9);
        float normalized = (db + 50) / 40.0f;
        normalized = std::max(0.0f, std::min(1.0f, normalized));

        if (normalized < 0.4f) {
            return sf::Color(0, 0, std::min(255.0f, normalized * 255.0f / 0.4f));
        }
        else if (normalized < 0.6f) {
            float t = (normalized - 0.4f) * 5.0f;
            return sf::Color(0, t * 255, 255);
        }
        else if (normalized < 0.8f) {
            float t = (normalized - 0.6f) * 5.0f;
            return sf::Color(t * 255, 255, 255 * (1.0f - t));
        }
        else {
            float t = (normalized - 0.8f) * 5.0f;
            return sf::Color(255, 255 * (1.0f - t), 0);
        }
    }

    void initializeUI() {
        if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
            std::cerr << "Failed to load font" << std::endl;
        }

        float windowWidth = window.getSize().x;
        float windowHeight = window.getSize().y;
        spectrogramSize = sf::Vector2f(windowWidth * 0.7f, windowHeight * 0.6f);
        spectrogramPosition = sf::Vector2f(windowWidth * 0.15f, windowHeight * 0.2f);

        spectrogramBackground.setSize(spectrogramSize);
        spectrogramBackground.setPosition(spectrogramPosition);
        spectrogramBackground.setFillColor(sf::Color(20, 20, 20));
        spectrogramBackground.setOutlineColor(sf::Color::White);
        spectrogramBackground.setOutlineThickness(1.0f);

        frequencyLabels.clear();
        float freqPerBin = 48000.0f / FFT_SIZE;
        int numLabels = 11;
        const float minFreq = 10.0f;
        const float maxFreq = 24000.0f;

        for (int i = 0; i < numLabels; ++i) {
            float t = i / float(numLabels - 1);
            float freq = minFreq * std::pow(maxFreq / minFreq, t);
            float binIndex = freq / freqPerBin;

            std::stringstream ss;
            if (freq >= 1000.0f) {
                ss << std::fixed << std::setprecision(1) << freq / 1000.0f << " kHz";
            }
            else {
                ss << std::fixed << std::setprecision(0) << freq << " Hz";
            }

            float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
            float yPos = spectrogramPosition.y + spectrogramSize.y * (1.0f - yRatio);

            sf::Text label = createText(font, ss.str(), 12,
                sf::Vector2f(spectrogramPosition.x - 70.0f, yPos - 6.0f));
            frequencyLabels.push_back(label);
        }

        timeLabels.clear();
        float secondsPerColumn = float(HOP_SIZE) / 48000.0f;
        float totalTime = secondsPerColumn * HISTORY_SIZE;
        time_window = totalTime;

        for (int i = 0; i <= 30; i += 5) {
            float t = i / 30.0f;
            float time = t * totalTime;
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << time << "s";

            float xPos = spectrogramPosition.x + spectrogramSize.x * t;
            sf::Text label = createText(font, ss.str(), 12,
                sf::Vector2f(xPos - 15.0f, spectrogramPosition.y + spectrogramSize.y + 10.0f));
            timeLabels.push_back(label);
        }
    }

public:
    Spectrogram(sf::RenderWindow& win)
        : window(win)
        , vertices(sf::Quads)
        , window_function(FFT_SIZE)
        , first_sample(true)
        , hMapFile(NULL)
        , sharedMem(NULL)
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
            sharedMem = (SharedMemory*)MapViewOfFile(
                hMapFile,
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                sizeof(SharedMemory)
            );

            if (sharedMem) {
                sharedMem->new_data_available = false;
                memset(sharedMem->magnitudes, 0, sizeof(sharedMem->magnitudes));
                sharedMem->timestamp = 0.0;
            }
        }

        fft_in = fftw_alloc_real(FFT_SIZE);
        fft_out = fftw_alloc_complex(FFT_SIZE / 2 + 1);
        fft_plan = fftw_plan_dft_r2c_1d(FFT_SIZE, fft_in, fft_out, FFTW_MEASURE);

        for (size_t i = 0; i < FFT_SIZE; ++i) {
            window_function[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (FFT_SIZE - 1)));
        }

        vertices.resize(HISTORY_SIZE * (FFT_SIZE / 2) * 4);
        initializeUI();
        updateVertexPositions();
    }

    ~Spectrogram() {
        if (sharedMem) {
            UnmapViewOfFile(sharedMem);
        }
        if (hMapFile) {
            CloseHandle(hMapFile);
        }

        fftw_destroy_plan(fft_plan);
        fftw_free(fft_in);
        fftw_free(fft_out);
    }

    void updateVertexPositions() {
        const float sampleRate = 48000.0f;
        const float minFreq = 10.0f;
        const float maxFreq = 24000.0f;

        float freqPerBin = sampleRate / float(FFT_SIZE);

        for (size_t x = 0; x < HISTORY_SIZE; ++x) {
            for (size_t y = 0; y < FFT_SIZE / 2; ++y) {
                size_t idx = (x * FFT_SIZE / 2 + y) * 4;

                float xRatio = x / float(HISTORY_SIZE);
                float xpos = spectrogramPosition.x + (xRatio * spectrogramSize.x);
                float width = spectrogramSize.x / float(HISTORY_SIZE);

                float freq = y * freqPerBin;
                freq = std::max(minFreq, std::min(maxFreq, freq));

                float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
                yRatio = std::max(0.0f, std::min(1.0f, yRatio));
                float ypos = spectrogramPosition.y + ((1.0f - yRatio) * spectrogramSize.y);

                float nextFreq = (y + 1) * freqPerBin;
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

    void processSamples(const std::vector<float>& samples) {
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
                float real = fft_out[i][0];
                float imag = fft_out[i][1];
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

    void draw() {
        std::unique_lock<std::mutex> lock(history_mutex);

        window.draw(spectrogramBackground);

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

        for (const auto& label : frequencyLabels) {
            window.draw(label);
        }
        for (const auto& label : timeLabels) {
            window.draw(label);
        }
    }

    void handleResize() {
        initializeUI();
        updateVertexPositions();
    }

    double getTimeWindow() const {
        return time_window;
    }

    std::vector<float> getCurrentMagnitudes() {
        if (magnitude_history.empty()) {
            return std::vector<float>();
        }
        return magnitude_history.back();
    }

};




class StaticSpectrogram {
private:
    static const size_t FFT_SIZE = 1024;
    const std::vector<std::vector<float>>& spectrogramData;
    sf::RenderWindow& window;
    sf::VertexArray vertices;
    sf::Font font;
    sf::RectangleShape spectrogramBackground;
    sf::Vector2f spectrogramPosition;
    sf::Vector2f spectrogramSize;
    std::vector<sf::Text> frequencyLabels;
    std::vector<sf::Text> timeLabels;

    sf::Text createText(const sf::Font& font, const std::string& content, unsigned int size, const sf::Vector2f& pos) {
        sf::Text text;
        text.setFont(font);
        text.setString(content);
        text.setCharacterSize(size);
        text.setFillColor(sf::Color::White);
        text.setPosition(pos);
        return text;
    }

    sf::Color getColor(float magnitude) {
        float db = 20 * std::log10(magnitude + 1e-9f);
        float normalized = (db + 50) / 40.0f;
        normalized = std::max(0.f, std::min(1.f, normalized));

        if (normalized < 0.4f)
            return sf::Color(0, 0, static_cast<sf::Uint8>(normalized * 255 / 0.4f));
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

    void initializeUI() {
        if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
            std::cerr << "Failed to load font" << std::endl;
        }
        float winW = static_cast<float>(window.getSize().x);
        float winH = static_cast<float>(window.getSize().y);

        spectrogramSize = sf::Vector2f(winW * 0.7f, winH * 0.6f);
        spectrogramPosition = sf::Vector2f(winW * 0.15f, winH * 0.2f);

        spectrogramBackground.setSize(spectrogramSize);
        spectrogramBackground.setPosition(spectrogramPosition);
        spectrogramBackground.setFillColor(sf::Color(20, 20, 20));
        spectrogramBackground.setOutlineColor(sf::Color::White);
        spectrogramBackground.setOutlineThickness(1.0f);

        frequencyLabels.clear();
        float sampleRate = 48000.0f;
        float freqPerBin = sampleRate / float(FFT_SIZE);
        const float minFreq = 10.f;
        const float maxFreq = 24000.f;
        int numFreqLabels = 11;
        for (int i = 0; i < numFreqLabels; ++i) {
            float t = i / float(numFreqLabels - 1);
            float freq = minFreq * std::pow(maxFreq / minFreq, t);
            std::stringstream ss;
            if (freq >= 1000.f)
                ss << std::fixed << std::setprecision(1) << (freq / 1000.f) << " kHz";
            else
                ss << std::fixed << std::setprecision(0) << freq << " Hz";

            float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
            float yPos = spectrogramPosition.y + spectrogramSize.y * (1.f - yRatio);
            frequencyLabels.push_back(createText(font, ss.str(), 12, sf::Vector2f(spectrogramPosition.x - 70.f, yPos - 6.f)));
        }

        // Create time labels along the horizontal axis.
        timeLabels.clear();
        size_t numFrames = spectrogramData.size();
        float secondsPerFrame = 512.f / 48000.f;
        float totalTime = numFrames * secondsPerFrame;

        // Corrected loop for time labels (fixed to 30 seconds)
        for (int t = 0; t <= 30; t += 5) {
            float tRatio = t / 30.0f; // Use a fixed 30-second window
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << t << " s";
            float xPos = spectrogramPosition.x + spectrogramSize.x * tRatio;
            timeLabels.push_back(createText(font, ss.str(), 12, sf::Vector2f(xPos - 15.f, spectrogramPosition.y + spectrogramSize.y + 10.f)));
        }
    }


    void updateVertexPositions() {
        size_t numFrames = spectrogramData.size();
        size_t bins = FFT_SIZE / 2;
        vertices.setPrimitiveType(sf::Quads);
        vertices.resize(numFrames * bins * 4);
        float frameWidth = spectrogramSize.x / numFrames;
        float sampleRate = 48000.f;
        float freqPerBin = sampleRate / float(FFT_SIZE);
        const float minFreq = 10.f;
        const float maxFreq = 24000.f;

        for (size_t x = 0; x < numFrames; ++x) {
            for (size_t y = 0; y < bins; ++y) {
                size_t idx = (x * bins + y) * 4;
                float xpos = spectrogramPosition.x + x * frameWidth;
                float freq = y * freqPerBin;
                freq = std::max(minFreq, std::min(maxFreq, freq));
                float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
                float ypos = spectrogramPosition.y + spectrogramSize.y * (1.f - yRatio);

                float nextFreq = (y + 1) * freqPerBin;
                nextFreq = std::max(minFreq, std::min(maxFreq, nextFreq));
                float nextYRatio = std::log10(nextFreq / minFreq) / std::log10(maxFreq / minFreq);
                float nextYpos = spectrogramPosition.y + spectrogramSize.y * (1.f - nextYRatio);

                vertices[idx].position = sf::Vector2f(xpos, ypos);
                vertices[idx + 1].position = sf::Vector2f(xpos + frameWidth, ypos);
                vertices[idx + 2].position = sf::Vector2f(xpos + frameWidth, nextYpos);
                vertices[idx + 3].position = sf::Vector2f(xpos, nextYpos);
            }
        }
    }

public:
    StaticSpectrogram(sf::RenderWindow& win, const std::vector<std::vector<float>>& data)
        : spectrogramData(data), window(win), vertices(sf::Quads)
    {
        initializeUI();
        updateVertexPositions();
    }

    void draw() {
        window.draw(spectrogramBackground);

        size_t numFrames = spectrogramData.size();
        size_t bins = FFT_SIZE / 2;

        for (size_t x = 0; x < numFrames; ++x) {
            const auto& frame = spectrogramData[x];
            for (size_t y = 0; y < bins; ++y) {
                size_t idx = (x * bins + y) * 4;
                sf::Color color = getColor(frame[y]);
                vertices[idx].color = color;
                vertices[idx + 1].color = color;
                vertices[idx + 2].color = color;
                vertices[idx + 3].color = color;
            }
        }
        window.draw(vertices);

        for (const auto& label : frequencyLabels)
            window.draw(label);
        for (const auto& label : timeLabels)
            window.draw(label);
    }

    void handleResize() {
        initializeUI();
        updateVertexPositions();
    }
};




class AudioCapture {
private:
    static const size_t BUFFER_SIZE = 2048;
    static const size_t TARGET_SAMPLES = 48000;
    size_t batchSize;

    IMMDeviceEnumerator* deviceEnumerator;
    IMMDevice* device;
    IAudioClient* audioClient;
    IAudioCaptureClient* captureClient;
    WAVEFORMATEX* pwfx;

    std::atomic<size_t> totalSamplesProcessed = 0;
    std::chrono::steady_clock::time_point lastPrintTime;
    std::chrono::steady_clock::time_point batchStartTime;

    std::thread captureThread;
    std::atomic<bool> isRunning;
    std::vector<float> buffer1;
    std::vector<float> buffer2;
    std::vector<float>* currentBuffer;
    std::mutex bufferMutex;
    unsigned int sampleRate;
    unsigned int numChannels;
    Spectrogram* spectrogram;

    std::vector<float> stretchAudio(const std::vector<float>& input) {
        std::vector<float> output(TARGET_SAMPLES, 0.0f);
        size_t samplesToCopy = std::min(input.size(), TARGET_SAMPLES);
        if (samplesToCopy > 0) {
            std::copy_n(input.begin(), samplesToCopy, output.begin());
        }
        return output;
    }

    bool initializeDevice() {
        HRESULT hr = CoCreateInstance(
            __uuidof(MMDeviceEnumerator),
            nullptr,
            CLSCTX_ALL,
            __uuidof(IMMDeviceEnumerator),
            reinterpret_cast<void**>(&deviceEnumerator)
        );
        if (FAILED(hr)) return false;

        hr = deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
        return SUCCEEDED(hr);
    }

    void cleanupDevice() {
        if (captureClient) { captureClient->Release(); captureClient = nullptr; }
        if (audioClient) { audioClient->Release(); audioClient = nullptr; }
        if (device) { device->Release(); device = nullptr; }
        if (deviceEnumerator) { deviceEnumerator->Release(); deviceEnumerator = nullptr; }
        if (pwfx) { CoTaskMemFree(pwfx); pwfx = nullptr; }
    }

    void processBatch(std::vector<float>& batchBuffer) {
        if (!spectrogram) return;

        std::vector<float> processBuffer;
        if (batchBuffer.size() >= TARGET_SAMPLES) {
            processBuffer.assign(batchBuffer.begin(), batchBuffer.begin() + TARGET_SAMPLES);
        }
        else {
            processBuffer = batchBuffer;
            processBuffer.resize(TARGET_SAMPLES, 0.0f);
        }

        bool isCompletelyEmpty = true;
        for (const float& sample : processBuffer) {
            if (std::abs(sample) > 1e-6) {
                isCompletelyEmpty = false;
                break;
            }
        }

        std::vector<float> processedBatch;
        if (isCompletelyEmpty && processBuffer.empty()) {
            processedBatch = std::vector<float>(TARGET_SAMPLES, 0.0f);
        }
        else {
            processedBatch = stretchAudio(processBuffer);
        }

        spectrogram->processSamples(processedBatch);
    }



    void captureLoop() {
        batchStartTime = std::chrono::steady_clock::now();
        lastPrintTime = batchStartTime;
        currentBuffer = &buffer1;  // Start with buffer1

        while (isRunning) {
            UINT32 packetLength = 0;
            HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                std::cerr << "Failed to get next packet size" << std::endl;
                continue;
            }

            auto now = std::chrono::steady_clock::now();
            double elapsedSeconds = std::chrono::duration<double>(now - batchStartTime).count();

            // If a second has passed, switch buffers and process the filled one
            if (elapsedSeconds >= 1.0) {
                // Switch buffers
                std::vector<float>* bufferToProcess = currentBuffer;
                currentBuffer = (currentBuffer == &buffer1) ? &buffer2 : &buffer1;
                currentBuffer->clear();
                currentBuffer->reserve(batchSize);

                // Process the filled buffer in a separate thread to avoid missing samples
                std::thread processingThread([this, bufferToProcess]() {
                    processBatch(*bufferToProcess);
                    });
                processingThread.detach();  // Let it run independently

                batchStartTime = now;
            }

            if (packetLength == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            while (packetLength > 0) {
                BYTE* data;
                UINT32 numFramesAvailable;
                DWORD flags;

                hr = captureClient->GetBuffer(&data, &numFramesAvailable, &flags, nullptr, nullptr);
                if (FAILED(hr)) {
                    std::cerr << "Failed to get buffer" << std::endl;
                    break;
                }

                if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                    float* floatData = reinterpret_cast<float*>(data);

                    for (size_t i = 0; i < numFramesAvailable; ++i) {
                        float sum = 0.0f;
                        for (unsigned int ch = 0; ch < numChannels; ++ch) {
                            sum += floatData[i * numChannels + ch];
                        }
                        currentBuffer->push_back(sum / static_cast<float>(numChannels));
                    }
                }

                hr = captureClient->ReleaseBuffer(numFramesAvailable);
                if (FAILED(hr)) {
                    std::cerr << "Failed to release buffer" << std::endl;
                    break;
                }

                hr = captureClient->GetNextPacketSize(&packetLength);
                if (FAILED(hr)) {
                    std::cerr << "Failed to get next packet size" << std::endl;
                    break;
                }
            }
        }
    }

public:
    AudioCapture()
        : deviceEnumerator(nullptr)
        , device(nullptr)
        , audioClient(nullptr)
        , captureClient(nullptr)
        , pwfx(nullptr)
        , isRunning(false)
        , buffer1()
        , buffer2()
        , currentBuffer(nullptr)
        , sampleRate(48000)
        , numChannels(2)
        , spectrogram(nullptr)
        , batchSize(48000)
    {
        HRESULT hr = CoInitialize(nullptr);
        if (SUCCEEDED(hr)) {
            initializeDevice();
        }
        buffer1.reserve(batchSize);
        buffer2.reserve(batchSize);
    }

    ~AudioCapture() {
        stop();
        cleanupDevice();
        CoUninitialize();
    }

    void setSpectrogram(Spectrogram* spec) {
        spectrogram = spec;
    }

    bool start() {
        if (!device || isRunning) return false;

        HRESULT hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&audioClient));
        if (FAILED(hr)) return false;

        hr = audioClient->GetMixFormat(&pwfx);
        if (FAILED(hr)) return false;

        pwfx->nSamplesPerSec = 48000;
        pwfx->nAvgBytesPerSec = pwfx->nSamplesPerSec * pwfx->nBlockAlign;

        hr = audioClient->Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            0,
            0,
            pwfx,
            nullptr
        );
        if (FAILED(hr)) return false;

        hr = audioClient->GetService(IID_PPV_ARGS(&captureClient));
        if (FAILED(hr)) return false;

        hr = audioClient->Start();
        if (FAILED(hr)) return false;

        sampleRate = pwfx->nSamplesPerSec;
        numChannels = pwfx->nChannels;
        batchSize = sampleRate;

        isRunning = true;
        captureThread = std::thread(&AudioCapture::captureLoop, this);
        SetThreadPriority(captureThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
        return true;
    }

    void stop() {
        isRunning = false;
        if (captureThread.joinable()) {
            captureThread.join();
        }
        if (audioClient) {
            audioClient->Stop();
        }
    }
};




class TestSignalGenerator : public sf::SoundStream {
private:
    static const size_t SAMPLE_RATE = 48000;
    static const size_t SAMPLES_PER_CHUNK = 2048;

    double currentPhase;
    double currentFrequency;
    double currentTime;

    double startFreq;
    double endFreq;
    double sweepDuration;
    bool isLogarithmic;

    virtual bool onGetData(Chunk& data) override {
        static std::vector<sf::Int16> chunk(SAMPLES_PER_CHUNK);

        for (size_t i = 0; i < SAMPLES_PER_CHUNK; ++i) {
            // Update time
            if (currentTime >= sweepDuration) {
                if (getLoop()) {
                    currentTime = 0;
                    currentPhase = 0;
                    currentFrequency = startFreq;
                }
                else {
                    return false;
                }
            }

            // Calculate current frequency
            if (isLogarithmic) {
                double t = currentTime / sweepDuration;
                currentFrequency = startFreq * std::exp(t * std::log(endFreq / startFreq));
            }
            else {
                double t = currentTime / sweepDuration;
                currentFrequency = startFreq + (endFreq - startFreq) * t;
            }

            // Generate sample
            double amplitude = 0.5; // Reduced amplitude to avoid clipping
            chunk[i] = static_cast<sf::Int16>(32767.0 * amplitude * std::sin(currentPhase));

            // Update phase and time
            currentPhase += 2.0 * M_PI * currentFrequency / SAMPLE_RATE;
            while (currentPhase >= 2.0 * M_PI) {
                currentPhase -= 2.0 * M_PI;
            }

            currentTime += 1.0 / SAMPLE_RATE;
        }

        data.samples = chunk.data();
        data.sampleCount = SAMPLES_PER_CHUNK;
        return true;
    }

    virtual void onSeek(sf::Time timeOffset) override {
        currentTime = timeOffset.asSeconds();

        // Recalculate frequency for the new position
        if (isLogarithmic) {
            double t = currentTime / sweepDuration;
            currentFrequency = startFreq * std::exp(t * std::log(endFreq / startFreq));
        }
        else {
            double t = currentTime / sweepDuration;
            currentFrequency = startFreq + (endFreq - startFreq) * t;
        }

        currentPhase = 0;  // Reset phase to avoid clicks
    }

public:
    TestSignalGenerator()
        : currentPhase(0)
        , currentFrequency(0)
        , currentTime(0)
        , startFreq(0)
        , endFreq(0)
        , sweepDuration(0)
        , isLogarithmic(true)
    {
        initialize(1, SAMPLE_RATE);  // Mono, 48kHz
    }

    void setupSweep(double startFrequency, double endFrequency, double duration, bool logarithmic = true) {
        stop();

        startFreq = startFrequency;
        endFreq = endFrequency;
        sweepDuration = duration;
        isLogarithmic = logarithmic;

        currentTime = 0;
        currentPhase = 0;
        currentFrequency = startFreq;
    }

    void startSweep(bool loop = false) {
        setLoop(loop);
        play();
    }

    // Debug function to check current state
    void printDebugInfo() const {
        std::cout << "Current time: " << currentTime
            << "s, Freq: " << currentFrequency
            << "Hz, Phase: " << currentPhase << std::endl;
    }
};




struct SpectrogramCache {
    static const uint32_t MAGIC_NUMBER = 0x53504543;  // "SPEC" in ASCII
    uint32_t version = 1;
    uint64_t timestamp;
    std::vector<std::vector<float>> spectogram_data;

    bool serialize(const std::string& filepath) const {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) return false;

        // Write header
        file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));

        // Write spectogram dimensions
        uint64_t num_frames = spectogram_data.size();
        uint64_t frame_size = spectogram_data.empty() ? 0 : spectogram_data[0].size();
        file.write(reinterpret_cast<const char*>(&num_frames), sizeof(num_frames));
        file.write(reinterpret_cast<const char*>(&frame_size), sizeof(frame_size));

        // Write data
        for (const auto& frame : spectogram_data) {
            file.write(reinterpret_cast<const char*>(frame.data()),
                frame.size() * sizeof(float));
        }

        return true;
    }

    bool deserialize(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) return false;

        // Read and verify header
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != MAGIC_NUMBER) return false;

        // Read version
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) return false;

        // Read timestamp
        file.read(reinterpret_cast<char*>(&timestamp), sizeof(timestamp));

        // Read dimensions
        uint64_t num_frames, frame_size;
        file.read(reinterpret_cast<char*>(&num_frames), sizeof(num_frames));
        file.read(reinterpret_cast<char*>(&frame_size), sizeof(frame_size));

        // Resize and read data
        spectogram_data.resize(num_frames);
        for (auto& frame : spectogram_data) {
            frame.resize(frame_size);
            file.read(reinterpret_cast<char*>(frame.data()),
                frame_size * sizeof(float));
        }

        return true;
    }
};




class SpectrumMatcher {
private:
    static const size_t WINDOW_LENGTH = 16;  // 16 frames per analysis window
    static const size_t WINDOW_OVERLAP = 8;  // 8 frame overlap
    static const size_t WINDOW_STEP = WINDOW_LENGTH - WINDOW_OVERLAP;  // 8 frames step
    static const size_t FRAMES_PER_SECOND = 48000 / 512;  // Sample rate / hop size
    static const size_t WINDOWS_IN_10_SECONDS = (10 * FRAMES_PER_SECOND) / WINDOW_STEP;

    // Store preprocessed reference data as overlapped windows
    std::vector<std::vector<float>> reference_windows;
    size_t last_match_position = 0;

    // Process frames into a normalized distribution using overlapping windows
    std::vector<float> processWindow(const std::vector<std::vector<float>>& frames, size_t start_idx) {
        if (start_idx + WINDOW_LENGTH > frames.size()) return std::vector<float>();

        // Initialize with size of frequency bins (512/2 = 256 bins)
        std::vector<float> averaged(frames[0].size(), 0.0f);
        float total = 0.0f;

        // Average all frames in the window
        for (size_t i = 0; i < WINDOW_LENGTH && (start_idx + i) < frames.size(); i++) {
            const auto& frame = frames[start_idx + i];
            for (size_t bin = 0; bin < frame.size(); bin++) {
                averaged[bin] += frame[bin];
                total += frame[bin];
            }
        }

        // Normalize so area = 1
        if (total > 1e-6f) {
            for (float& val : averaged) {
                val /= total;
            }
        }

        return averaged;
    }

    // Process a sequence of frames into overlapping windows
    std::vector<std::vector<float>> processFrameSequence(const std::vector<std::vector<float>>& frames) {
        std::vector<std::vector<float>> windows;
        for (size_t i = 0; i + WINDOW_LENGTH <= frames.size(); i += WINDOW_STEP) {
            auto window = processWindow(frames, i);
            if (!window.empty()) {
                windows.push_back(window);
            }
        }
        return windows;
    }

public:
    void setReferenceData(const std::vector<std::vector<float>>& data) {
        std::cout << "Processing " << data.size() << " reference frames into overlapping windows..." << std::endl;
        reference_windows = processFrameSequence(data);
        std::cout << "Created " << reference_windows.size() << " reference windows" << std::endl;
        last_match_position = 0;
    }

    size_t findMatch(const std::vector<float>& current_magnitudes) {
        static std::deque<std::vector<float>> recent_frames;
        static int debug_counter = 0;
        debug_counter++;

        if (reference_windows.empty() || current_magnitudes.empty()) return 0;

        // Add current frame to recent history
        recent_frames.push_back(current_magnitudes);
        if (recent_frames.size() > FRAMES_PER_SECOND * 10) {  // Keep 10 seconds of history
            recent_frames.pop_front();
        }

        // Debug output
        if (debug_counter % 30 == 0) {
            std::cout << "\nDEBUG: Current frame buffer size: " << recent_frames.size()
                << " frames (" << (recent_frames.size() * 512.0 / 48000.0) << " seconds)" << std::endl;
        }

        // Wait until we have enough frames for at least one full window
        if (recent_frames.size() < WINDOW_LENGTH) {
            if (debug_counter % 30 == 0) {
                std::cout << "Waiting for more frames. Have " << recent_frames.size()
                    << "/" << WINDOW_LENGTH << std::endl;
            }
            return last_match_position;
        }

        // Process recent frames into overlapping windows
        std::vector<std::vector<float>> recent_windows = processFrameSequence(
            std::vector<std::vector<float>>(recent_frames.begin(), recent_frames.end())
        );

        if (recent_windows.empty()) {
            if (debug_counter % 30 == 0) {
                std::cout << "No recent windows processed!" << std::endl;
            }
            return last_match_position;
        }

        if (debug_counter % 30 == 0) {
            std::cout << "Processed " << recent_windows.size() << " recent windows" << std::endl;
        }

        // Match the sequence against reference data
        float best_score = -1.0f;
        size_t best_window_idx = last_match_position / WINDOW_STEP;

        // Search through reference windows
        for (size_t i = 0; i + recent_windows.size() <= reference_windows.size(); i++) {
            float sequence_score = 0.0f;
            size_t matches = 0;

            // Compare each window in the sequence
            for (size_t j = 0; j < recent_windows.size(); j++) {
                const auto& current_window = recent_windows[j];
                const auto& ref_window = reference_windows[i + j];

                // Calculate correlation
                float correlation = 0.0f;
                float norm1 = 0.0f;
                float norm2 = 0.0f;

                for (size_t k = 0; k < current_window.size(); k++) {
                    correlation += current_window[k] * ref_window[k];
                    norm1 += current_window[k] * current_window[k];
                    norm2 += ref_window[k] * ref_window[k];
                }

                if (norm1 > 0 && norm2 > 0) {
                    sequence_score += correlation / std::sqrt(norm1 * norm2);
                    matches++;
                }
            }

            float score = matches > 0 ? sequence_score / matches : 0.0f;

            if (score > best_score) {
                best_score = score;
                best_window_idx = i;
            }
        }

        if (debug_counter % 30 == 0) {
            std::cout << "Best score: " << best_score
                << " at " << (best_window_idx * WINDOW_STEP * 512.0 / 48000.0)
                << "s" << std::endl;
        }

        // Use a fixed threshold for now
        const float MATCH_THRESHOLD = 0.7f;  // Lowered from 0.8
        if (best_score > MATCH_THRESHOLD) {
            last_match_position = best_window_idx * WINDOW_STEP;
            if (debug_counter % 30 == 0) {
                std::cout << "Match found! Score: " << best_score << std::endl;
            }
        }
        else if (debug_counter % 30 == 0) {
            std::cout << "No match - score " << best_score << " below threshold " << MATCH_THRESHOLD << std::endl;
        }

        return last_match_position;
    }
};




class AudioMatcher {
private:
    AudioAnalyzer analyzer;
    SpectrumMatcher matcher;
    SpectrogramCache cache;
    std::string cache_path;
    std::string audio_path;
    bool is_initialized = false;
    mutable std::mutex matcher_mutex;  // Add mutex for thread safety

    bool needsUpdate() {
        // Check if cache file exists
        std::ifstream cache_test(cache_path);
        if (!cache_test.good()) {
            return true;
        }
        cache_test.close();

        // Check if audio file exists
        std::ifstream audio_test(audio_path);
        if (!audio_test.good()) {
            return false;  // Audio file doesn't exist, can't update
        }
        audio_test.close();

        // Get file modification times using Windows API
        WIN32_FILE_ATTRIBUTE_DATA audioAttrib, cacheAttrib;
        if (!GetFileAttributesExA(audio_path.c_str(), GetFileExInfoStandard, &audioAttrib) ||
            !GetFileAttributesExA(cache_path.c_str(), GetFileExInfoStandard, &cacheAttrib)) {
            return true;  // If we can't get attributes, assume we need to update
        }

        ULARGE_INTEGER audioTime, cacheTime;
        audioTime.LowPart = audioAttrib.ftLastWriteTime.dwLowDateTime;
        audioTime.HighPart = audioAttrib.ftLastWriteTime.dwHighDateTime;
        cacheTime.LowPart = cacheAttrib.ftLastWriteTime.dwLowDateTime;
        cacheTime.HighPart = cacheAttrib.ftLastWriteTime.dwHighDateTime;

        return audioTime.QuadPart > cacheTime.QuadPart;
    }

public:
    AudioMatcher(const std::string& audio_file)
        : audio_path(audio_file)
        , cache_path(audio_file + ".cache") {}

    const std::vector<std::vector<float>>& getStaticSpectrogram() const {
        return cache.spectogram_data;
    }

    bool initialize() {
        if (is_initialized) return true;

        // Check if audio file exists
        std::ifstream audio_test(audio_path);
        if (!audio_test.good()) {
            std::cerr << "Audio file not found: " << audio_path << std::endl;
            return false;
        }
        audio_test.close();

        bool needs_analysis = needsUpdate();

        if (needs_analysis) {
            std::cout << "Analyzing audio file..." << std::endl;

            if (!analyzer.analyzeFile(audio_path)) {
                std::cerr << "Failed to analyze audio file" << std::endl;
                return false;
            }

            // Store analysis results in cache
            cache.timestamp = std::time(nullptr);
            cache.spectogram_data = analyzer.getReferenceSpectogram();

            if (!cache.serialize(cache_path)) {
                std::cerr << "Failed to save cache file" << std::endl;
                return false;
            }

            std::cout << "Analysis complete and cached" << std::endl;
        }
        else {
            std::cout << "Loading cached analysis..." << std::endl;

            if (!cache.deserialize(cache_path)) {
                std::cerr << "Failed to load cache file, will re-analyze" << std::endl;
                return initialize();
            }
        }

        // Initialize the matcher with the spectral data
        matcher.setReferenceData(cache.spectogram_data);

        is_initialized = true;
        return true;
    }

    size_t findMatch(const std::vector<float>& current_magnitudes) {
        if (!is_initialized) {
            std::cerr << "AudioMatcher not initialized" << std::endl;
            return 0;
        }
        return matcher.findMatch(current_magnitudes);
    }

    double getTimestamp(size_t position) const {
        return position * (static_cast<double>(512) / 48000.0);  // HOP_SIZE / SAMPLE_RATE
    }
};




int main() {
    // Create two SFML windows: one for live audio and one for the static audio
    sf::RenderWindow liveWindow(sf::VideoMode(1280, 720), "Real-time Spectrogram", sf::Style::Default | sf::Style::Resize);
    sf::RenderWindow staticWindow(sf::VideoMode(1280, 720), "Static Spectrogram", sf::Style::Default | sf::Style::Resize);

    liveWindow.setFramerateLimit(60);
    staticWindow.setFramerateLimit(60);

    // Get the path to the executable directory and determine the audio file path
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string exePath(path);
    std::string exeDir = exePath.substr(0, exePath.find_last_of("\\/"));
    std::string audioPath = exeDir + "\\beethoven.wav";

    // Initialize the AudioMatcher (which analyzes the static audio file)
    AudioMatcher matcher(audioPath);
    if (!matcher.initialize()) {
        std::cerr << "Failed to initialize audio matcher" << std::endl;
        std::cerr << "Tried to load from: " << audioPath << std::endl;
        return -1;
    }

    // Retrieve the static spectrogram data from the matcher
    const auto& staticSpectroData = matcher.getStaticSpectrogram();

    // Create an instance of our static spectrogram display using the static data
    StaticSpectrogram staticSpectrogram(staticWindow, staticSpectroData);

    // Initialize the live spectrogram and audio capture as before
    Spectrogram liveSpectrogram(liveWindow);
    AudioCapture capture;
    capture.setSpectrogram(&liveSpectrogram);
    if (!capture.start()) {
        std::cerr << "Failed to start audio capture" << std::endl;
        return -1;
    }

    sf::Clock debugTimer;
    // Main loop: poll events and draw both windows
    while (liveWindow.isOpen() && staticWindow.isOpen()) {
        sf::Event event;
        // Process events for the live window
        while (liveWindow.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
            {
                liveWindow.close();
            }
            else if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
                liveWindow.setView(sf::View(visibleArea));
                liveSpectrogram.handleResize();
            }
        }
        // Process events for the static window
        while (staticWindow.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape))
            {
                staticWindow.close();
            }
            else if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
                staticWindow.setView(sf::View(visibleArea));
                staticSpectrogram.handleResize();
            }
        }

        // For debugging: periodically print out the live matching position
        if (debugTimer.getElapsedTime().asSeconds() >= 1.0f) {
            auto current_magnitudes = liveSpectrogram.getCurrentMagnitudes();
            if (!current_magnitudes.empty()) {
                size_t match_position = matcher.findMatch(current_magnitudes);
                double timestamp = matcher.getTimestamp(match_position);
                std::cout << "Current position in audio: " << timestamp << " seconds" << std::endl;
            }
            debugTimer.restart();
        }

        // Draw live spectrogram
        liveWindow.clear(sf::Color(10, 10, 10));
        liveSpectrogram.draw();
        liveWindow.display();

        // Draw static spectrogram
        staticWindow.clear(sf::Color(10, 10, 10));
        staticSpectrogram.draw();
        staticWindow.display();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    capture.stop();
    return 0;
}