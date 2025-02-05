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
    // Calculate HISTORY_SIZE based on desired time window (30 seconds) at 48kHz
    // HISTORY_SIZE = (sample_rate * desired_seconds) / HOP_SIZE
    // For 48kHz and 30 seconds: (48000 * 30) / 512 ≈ 2812
    static const size_t HISTORY_SIZE = 2812;

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
        // Convert magnitude to decibels
        float db = 20 * std::log10(magnitude + 1e-9);

        // Wider dynamic range: -50dB to -10dB (40dB range)
        float normalized = (db + 50) / 40.0f;
        normalized = std::max(0.0f, std::min(1.0f, normalized));

        // Modified thresholds to show more gradual color changes
        if (normalized < 0.4f) {  // More space for darker blues
            return sf::Color(0, 0, std::min(255.0f, normalized * 255.0f / 0.4f));
        }
        else if (normalized < 0.6f) {  // Gradual transition to cyan
            float t = (normalized - 0.4f) * 5.0f;
            return sf::Color(0, t * 255, 255);
        }
        else if (normalized < 0.8f) {  // Gradual transition to yellow
            float t = (normalized - 0.6f) * 5.0f;
            return sf::Color(t * 255, 255, 255 * (1.0f - t));
        }
        else {  // Red for the loudest signals
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

        // Setup spectrogram background
        spectrogramBackground.setSize(spectrogramSize);
        spectrogramBackground.setPosition(spectrogramPosition);
        spectrogramBackground.setFillColor(sf::Color(20, 20, 20));
        spectrogramBackground.setOutlineColor(sf::Color::White);
        spectrogramBackground.setOutlineThickness(1.0f);

        // Create frequency labels
        frequencyLabels.clear();
        float freqPerBin = 48000.0f / FFT_SIZE;
        int numLabels = 11;
        const float minFreq = 10.0f;
        const float maxFreq = 24000.0f;

        for (int i = 0; i < numLabels; ++i) {
            float t = i / float(numLabels - 1);
            // Calculate frequency using logarithmic scale
            float freq = minFreq * std::pow(maxFreq / minFreq, t);
            // Find nearest bin
            float binIndex = freq / freqPerBin;

            std::stringstream ss;
            if (freq >= 1000.0f) {
                ss << std::fixed << std::setprecision(1) << freq / 1000.0f << " kHz";
            }
            else {
                ss << std::fixed << std::setprecision(0) << freq << " Hz";
            }

            // Position using logarithmic scale
            float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
            float yPos = spectrogramPosition.y + spectrogramSize.y * (1.0f - yRatio);

            sf::Text label = createText(font, ss.str(), 12,
                sf::Vector2f(spectrogramPosition.x - 70.0f, yPos - 6.0f));
            frequencyLabels.push_back(label);
        }

        // Create time labels for 30 second window
        timeLabels.clear();
        float secondsPerColumn = float(HOP_SIZE) / 48000.0f;  // Adjusted for 48kHz
        float totalTime = secondsPerColumn * HISTORY_SIZE;
        time_window = totalTime;

        for (int i = 0; i <= 30; i += 5) {  // Show label every 5 seconds
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

    // In the Spectrogram class, replace updateVertexPositions with:
    void updateVertexPositions() {
        const float sampleRate = 48000.0f;
        const float minFreq = 10.0f;  // Minimum frequency for log scale
        const float maxFreq = 24000.0f;  // Maximum frequency (Nyquist)

        // Calculate frequency for each bin
        float freqPerBin = sampleRate / float(FFT_SIZE);

        // Calculate vertex positions using logarithmic frequency mapping
        for (size_t x = 0; x < HISTORY_SIZE; ++x) {
            for (size_t y = 0; y < FFT_SIZE / 2; ++y) {
                size_t idx = (x * FFT_SIZE / 2 + y) * 4;

                // Calculate x position (time axis - remains linear)
                float xRatio = x / float(HISTORY_SIZE);
                float xpos = spectrogramPosition.x + (xRatio * spectrogramSize.x);
                float width = spectrogramSize.x / float(HISTORY_SIZE);

                // Calculate frequency for this bin
                float freq = y * freqPerBin;
                freq = std::max(minFreq, std::min(maxFreq, freq));

                // Calculate y position using logarithmic mapping
                float yRatio = std::log10(freq / minFreq) / std::log10(maxFreq / minFreq);
                yRatio = std::max(0.0f, std::min(1.0f, yRatio));
                float ypos = spectrogramPosition.y + ((1.0f - yRatio) * spectrogramSize.y);

                // Calculate height for this bin (difference between this and next frequency)
                float nextFreq = (y + 1) * freqPerBin;
                nextFreq = std::max(minFreq, std::min(maxFreq, nextFreq));
                float nextYRatio = std::log10(nextFreq / minFreq) / std::log10(maxFreq / minFreq);
                nextYRatio = std::max(0.0f, std::min(1.0f, nextYRatio));
                float nextYpos = spectrogramPosition.y + ((1.0f - nextYRatio) * spectrogramSize.y);

                // Assign vertex positions
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
            // Apply window function and prepare FFT input
            // Reduce input amplitude significantly
            const float scale = 10.0f;
            for (size_t i = 0; i < FFT_SIZE; ++i) {
                fft_in[i] = buffer[i] * window_function[i] * scale;
            }

            // Perform FFT
            fftw_execute(fft_plan);

            // Calculate magnitudes with improved scaling
            std::vector<float> magnitudes(FFT_SIZE / 2);
            for (size_t i = 0; i < FFT_SIZE / 2; ++i) {
                float real = fft_out[i][0];
                float imag = fft_out[i][1];
                // Square root for power spectrum, normalized by FFT size
                magnitudes[i] = std::sqrt(real * real + imag * imag) / (FFT_SIZE * 2.0f);
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


int main() {
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    sf::RenderWindow window(sf::VideoMode(1280, 720), "Real-time Spectrogram",
        sf::Style::Default | sf::Style::Resize);
    window.setFramerateLimit(60);

    Spectrogram spectrogram(window);
    AudioCapture capture;
    TestSignalGenerator signalGen;

    // Start with a slower sweep for testing
    signalGen.setupSweep(20.0, 2000.0, 10.0, true);  // 20 Hz to 2 kHz over 10 seconds
    signalGen.startSweep(true);

    capture.setSpectrogram(&spectrogram);

    if (!capture.start()) {
        std::cerr << "Failed to start audio capture" << std::endl;
        return -1;
    }

    sf::Clock debugTimer;
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
            else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) {
                    signalGen.stop();
                    signalGen.setupSweep(20.0, 2000.0, 10.0, true);
                    signalGen.startSweep(true);
                }
                else if (event.key.code == sf::Keyboard::Up) {
                    // Faster sweep
                    signalGen.stop();
                    signalGen.setupSweep(20.0, 2000.0, 5.0, true);
                    signalGen.startSweep(true);
                }
                else if (event.key.code == sf::Keyboard::Down) {
                    // Slower sweep
                    signalGen.stop();
                    signalGen.setupSweep(20.0, 2000.0, 15.0, true);
                    signalGen.startSweep(true);
                }
            }
        }

        // Print debug info every second
        if (debugTimer.getElapsedTime().asSeconds() >= 1.0f) {
            signalGen.printDebugInfo();
            debugTimer.restart();
        }

        window.clear(sf::Color(10, 10, 10));
        spectrogram.draw();
        window.display();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    capture.stop();
    return 0;
}