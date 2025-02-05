#define _USE_MATH_DEFINES
#include <SFML/Graphics.hpp>
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

class Spectrogram {
private:
    static const size_t FFT_SIZE = 1024;
    static const size_t HOP_SIZE = 512;
    // Calculate HISTORY_SIZE based on desired time window (10 seconds) at 48kHz
    // HISTORY_SIZE = (sample_rate * desired_seconds) / HOP_SIZE
    // For 48kHz and 10 seconds: (48000 * 10) / 512 ≈ 937
    static const size_t HISTORY_SIZE = 937;

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
    double time_window;  // Store actual time window in seconds

    sf::Color getColor(float magnitude) {
        float db = 20 * std::log10(magnitude + 1e-6);
        float normalized = (db + 50) / 100.0f; // -50 to +50 dB range
        normalized = std::max(0.0f, std::min(1.0f, normalized));

        // Enhanced color mapping for better visualization
        if (normalized < 0.25f) {
            return sf::Color(0, 0, 100 + (normalized * 4 * 155));
        }
        else if (normalized < 0.5f) {
            float t = (normalized - 0.25f) * 4;
            return sf::Color(0, t * 255, 255);
        }
        else if (normalized < 0.75f) {
            float t = (normalized - 0.5f) * 4;
            return sf::Color(t * 255, 255, 255 - (t * 255));
        }
        else {
            float t = (normalized - 0.75f) * 4;
            return sf::Color(255, 255 - (t * 255), 0);
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

        // Setup spectrogram background
        spectrogramBackground.setSize(spectrogramSize);
        spectrogramBackground.setPosition(spectrogramPosition);
        spectrogramBackground.setFillColor(sf::Color(20, 20, 20));
        spectrogramBackground.setOutlineColor(sf::Color::White);
        spectrogramBackground.setOutlineThickness(1.0f);

        // Create frequency labels
        frequencyLabels.clear();
        float minFreq = 10.0f;
        float maxFreq = 24000.0f;  // Adjusted for 48kHz sampling rate (Nyquist frequency)
        int numLabels = 11;

        for (int i = 0; i < numLabels; ++i) {
            float t = i / float(numLabels - 1);
            float freq = minFreq * std::pow(maxFreq / minFreq, t);

            std::stringstream ss;
            if (freq >= 1000.0f) {
                ss << std::fixed << std::setprecision(1) << freq / 1000.0f << " kHz";
            }
            else {
                ss << std::fixed << std::setprecision(0) << freq << " Hz";
            }

            float yPos = spectrogramPosition.y + spectrogramSize.y * (1.0f - t);
            sf::Text label = createText(font, ss.str(), 12,
                sf::Vector2f(spectrogramPosition.x - 70.0f, yPos - 6.0f));
            frequencyLabels.push_back(label);
        }

        // Create time labels for 10 second window
        timeLabels.clear();
        float secondsPerColumn = float(HOP_SIZE) / 48000.0f;  // Adjusted for 48kHz
        float totalTime = secondsPerColumn * HISTORY_SIZE;
        time_window = totalTime;

        for (int i = 0; i <= 10; ++i) {
            float t = i / 10.0f;
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
        // Initialize shared memory
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

        // Initialize FFTW
        fft_in = fftw_alloc_real(FFT_SIZE);
        fft_out = fftw_alloc_complex(FFT_SIZE / 2 + 1);
        fft_plan = fftw_plan_dft_r2c_1d(FFT_SIZE, fft_in, fft_out, FFTW_MEASURE);

        // Create Hanning window
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
        for (size_t x = 0; x < HISTORY_SIZE; ++x) {
            for (size_t y = 0; y < FFT_SIZE / 2; ++y) {
                size_t idx = (x * FFT_SIZE / 2 + y) * 4;

                float xRatio = x / float(HISTORY_SIZE);
                float yRatio = y / float(FFT_SIZE / 2);

                float xpos = spectrogramPosition.x + (xRatio * spectrogramSize.x);
                float ypos = spectrogramPosition.y + ((1.0f - yRatio) * spectrogramSize.y);
                float width = spectrogramSize.x / float(HISTORY_SIZE);
                float height = spectrogramSize.y / float(FFT_SIZE / 2);

                vertices[idx].position = sf::Vector2f(xpos, ypos);
                vertices[idx + 1].position = sf::Vector2f(xpos + width, ypos);
                vertices[idx + 2].position = sf::Vector2f(xpos + width, ypos + height);
                vertices[idx + 3].position = sf::Vector2f(xpos, ypos + height);
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
            // Apply window function and prepare FFT input
            for (size_t i = 0; i < FFT_SIZE; ++i) {
                fft_in[i] = buffer[i] * window_function[i] * 32768.0;
            }

            // Perform FFT
            fftw_execute(fft_plan);

            // Calculate magnitudes
            std::vector<float> magnitudes(FFT_SIZE / 2);
            for (size_t i = 0; i < FFT_SIZE / 2; ++i) {
                float real = fft_out[i][0];
                float imag = fft_out[i][1];
                magnitudes[i] = std::sqrt(real * real + imag * imag) / FFT_SIZE;
            }

            // Update history and shared memory
            {
                std::lock_guard<std::mutex> lock(history_mutex);
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
        std::lock_guard<std::mutex> lock(history_mutex);

        window.draw(spectrogramBackground);

        // Update and draw spectrogram
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

        // Draw labels
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
};

class AudioCapture {
private:
    static const size_t BUFFER_SIZE = 2048;
    static const size_t TARGET_SAMPLES = 48000;  // Adjusted to 48kHz
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
    std::vector<float> captureBuffer;
    std::mutex bufferMutex;
    unsigned int sampleRate;
    unsigned int numChannels;
    Spectrogram* spectrogram;

    std::vector<float> stretchAudio(const std::vector<float>& input) {
        if (input.empty()) return std::vector<float>(TARGET_SAMPLES, 0.0f);

        std::vector<float> output(TARGET_SAMPLES);

        if (input.size() < TARGET_SAMPLES) {
            float ratio = static_cast<float>(input.size() - 1) / (TARGET_SAMPLES - 1);

            for (size_t i = 0; i < TARGET_SAMPLES; ++i) {
                float pos = i * ratio;
                size_t idx1 = static_cast<size_t>(pos);
                size_t idx2 = std::min(idx1 + 1, input.size() - 1);
                float frac = pos - idx1;

                // Linear interpolation between samples
                output[i] = input[idx1] * (1.0f - frac) + input[idx2] * frac;
            }
        }
        else {
            // If we have more samples than needed, use simple downsampling
            float ratio = static_cast<float>(input.size()) / TARGET_SAMPLES;
            for (size_t i = 0; i < TARGET_SAMPLES; ++i) {
                size_t idx = static_cast<size_t>(i * ratio);
                output[i] = input[idx];
            }
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

        auto now = std::chrono::steady_clock::now();
        double batchDuration = std::chrono::duration<double>(now - batchStartTime).count();

        // Only process if we have enough samples or enough time has passed
        if (batchBuffer.size() >= TARGET_SAMPLES * 0.9 || batchDuration >= 1.0) {
            // Check if the batch contains any significant audio
            bool isCompletelyEmpty = true;
            for (const float& sample : batchBuffer) {
                if (std::abs(sample) > 1e-6) {  // Very small threshold
                    isCompletelyEmpty = false;
                    break;
                }
            }

            std::vector<float> processedBatch;
            if (isCompletelyEmpty && batchBuffer.size() == 0) {
                // Only use silent batch if we actually got no samples
                processedBatch = std::vector<float>(TARGET_SAMPLES, 0.0f);
                std::cout << "No samples received - Using silent batch" << std::endl;
            }
            else {
                processedBatch = stretchAudio(batchBuffer);
            }

            spectrogram->processSamples(processedBatch);
            totalSamplesProcessed += batchBuffer.size();

            std::cout << "Batch processed - Samples: " << batchBuffer.size()
                << ", Empty: " << isCompletelyEmpty
                << ", Duration: " << batchDuration << "s" << std::endl;

            // Clear the batch buffer and reset timer
            batchBuffer.clear();
            batchBuffer.reserve(batchSize);
            batchStartTime = now;
        }
    }

    void captureLoop() {
        std::vector<float> batchBuffer;
        batchBuffer.reserve(TARGET_SAMPLES);
        batchStartTime = std::chrono::steady_clock::now();
        lastPrintTime = batchStartTime;

        while (isRunning) {
            UINT32 packetLength = 0;
            HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                std::cerr << "Failed to get next packet size" << std::endl;
                continue;
            }

            // If no packets are available, just wait a bit
            if (packetLength == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsedSeconds = std::chrono::duration<double>(now - batchStartTime).count();

                if (elapsedSeconds >= 1.0) {
                    processBatch(batchBuffer);
                }

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

                // Only skip if explicitly marked as silent
                if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                    float* floatData = reinterpret_cast<float*>(data);

                    // Convert to mono and accumulate samples
                    for (size_t i = 0; i < numFramesAvailable; ++i) {
                        float sum = 0.0f;
                        for (unsigned int ch = 0; ch < numChannels; ++ch) {
                            sum += floatData[i * numChannels + ch];
                        }
                        batchBuffer.push_back(sum / static_cast<float>(numChannels));
                    }
                }

                hr = captureClient->ReleaseBuffer(numFramesAvailable);
                if (FAILED(hr)) {
                    std::cerr << "Failed to release buffer" << std::endl;
                    break;
                }

                // Process batch if we have enough samples or enough time has passed
                auto now = std::chrono::steady_clock::now();
                double elapsedSeconds = std::chrono::duration<double>(now - batchStartTime).count();

                if (batchBuffer.size() >= TARGET_SAMPLES || elapsedSeconds >= 1.0) {
                    processBatch(batchBuffer);
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
        , captureBuffer(BUFFER_SIZE)
        , sampleRate(48000)  // Default to 48kHz
        , numChannels(2)
        , spectrogram(nullptr)
        , batchSize(48000)   // Default to 1 second at 48kHz
    {
        HRESULT hr = CoInitialize(nullptr);
        if (SUCCEEDED(hr)) {
            initializeDevice();
        }
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

        // Force 48kHz sample rate
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
        batchSize = sampleRate;  // One second of audio

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

int main() {
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    sf::RenderWindow window(sf::VideoMode(1280, 720), "Real-time Spectrogram",
        sf::Style::Default | sf::Style::Resize);
    window.setFramerateLimit(60);

    Spectrogram spectrogram(window);
    AudioCapture capture;

    capture.setSpectrogram(&spectrogram);

    if (!capture.start()) {
        std::cerr << "Failed to start audio capture" << std::endl;
        return -1;
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
            else if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
                window.setView(sf::View(visibleArea));
                spectrogram.handleResize();
            }
        }

        window.clear(sf::Color(10, 10, 10));
        spectrogram.draw();
        window.display();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    capture.stop();
    return 0;
}