#include "AudioAnalyzer.hpp"
#include <iostream>
#include <algorithm>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "winmm.lib")

AudioAnalyzer::AudioAnalyzer()
    : deviceEnumerator(nullptr)
    , device(nullptr)
    , audioClient(nullptr)
    , captureClient(nullptr)
    , pwfx(nullptr)
    , isRunning(false)
    , captureBuffer(BUFFER_SIZE, 0.0f)
    , processBuffer(BUFFER_SIZE, 0.0f)
    , bufferReady(false)
    , fftBuffer(FFT_SIZE)
    , windowFunction(FFT_SIZE)
    , currentVolume(0.0f)
    , currentCentroid(0.0f)
    , volumeSmoothing(0.8f)
    , centroidSmoothing(0.7f)
{
    initializeWindowFunction();

    // Initialize COM
    HRESULT hr = CoInitialize(nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM" << std::endl;
        return;
    }

    // Create device enumerator
    hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        nullptr,
        CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator),
        (void**)&deviceEnumerator
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator" << std::endl;
        return;
    }

    // Get default render device
    hr = deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default audio endpoint" << std::endl;
        return;
    }
}

AudioAnalyzer::~AudioAnalyzer() {
    stop();

    if (captureClient) captureClient->Release();
    if (audioClient) audioClient->Release();
    if (device) device->Release();
    if (deviceEnumerator) deviceEnumerator->Release();
    if (pwfx) CoTaskMemFree(pwfx);

    CoUninitialize();
}

void AudioAnalyzer::initializeWindowFunction() {
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        windowFunction[i] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / (FFT_SIZE - 1))));
    }
}

bool AudioAnalyzer::start() {
    if (!device) return false;

    HRESULT hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to activate audio client" << std::endl;
        return false;
    }

    hr = audioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
        std::cerr << "Failed to get mix format" << std::endl;
        return false;
    }

    hr = audioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK,
        0,
        0,
        pwfx,
        nullptr
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client" << std::endl;
        return false;
    }

    hr = audioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&captureClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to get capture client" << std::endl;
        return false;
    }

    hr = audioClient->Start();
    if (FAILED(hr)) {
        std::cerr << "Failed to start audio client" << std::endl;
        return false;
    }

    isRunning = true;
    captureThread = std::thread(&AudioAnalyzer::captureLoop, this);
    return true;
}

void AudioAnalyzer::stop() {
    isRunning = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }
    if (audioClient) {
        audioClient->Stop();
    }
}

void AudioAnalyzer::captureLoop() {
    while (isRunning) {
        UINT32 packetLength = 0;
        HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) continue;

        while (packetLength > 0) {
            BYTE* data;
            UINT32 numFramesAvailable;
            DWORD flags;

            hr = captureClient->GetBuffer(&data, &numFramesAvailable, &flags, nullptr, nullptr);
            if (FAILED(hr)) break;

            if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                std::unique_lock<std::mutex> lock(captureMutex);
                float* floatData = reinterpret_cast<float*>(data);
                size_t channels = pwfx->nChannels;

                // Mix down to mono
                for (UINT32 i = 0; i < numFramesAvailable && i < BUFFER_SIZE; i++) {
                    float sample = 0.0f;
                    for (size_t ch = 0; ch < channels; ch++) {
                        sample += floatData[i * channels + ch];
                    }
                    sample /= static_cast<float>(channels);
                    captureBuffer[i] = sample;
                }
                bufferReady = true;
            }

            hr = captureClient->ReleaseBuffer(numFramesAvailable);
            if (FAILED(hr)) break;

            hr = captureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AudioAnalyzer::swapBuffers() {
    if (bufferReady) {
        std::unique_lock<std::mutex> lock(captureMutex);
        processBuffer.swap(captureBuffer);
        bufferReady = false;
    }
}

void AudioAnalyzer::update() {
    swapBuffers();

    float rmsLevel = calculateRMS();
    calculateFFT();
    float centroid = calculateSpectralCentroid();

    currentVolume = smoothValue(currentVolume, rmsLevel, volumeSmoothing);
    currentCentroid = smoothValue(currentCentroid, centroid, centroidSmoothing);

    normalizeValue(currentVolume);
    normalizeValue(currentCentroid);

    static int frameCount = 0;
    if (++frameCount % 30 == 0) {
        std::cout << "\rVolume: " << (currentVolume * 100.0f) << "% Centroid: "
            << (currentCentroid * 100.0f) << "%" << std::flush;
    }
}

float AudioAnalyzer::calculateRMS() {
    float sum = 0.0f;
    for (const auto& sample : processBuffer) {
        sum += sample * sample;
    }
    return std::sqrt(sum / static_cast<float>(BUFFER_SIZE));
}

void AudioAnalyzer::calculateFFT() {
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        if (i < processBuffer.size()) {
            fftBuffer[i] = std::complex<float>(processBuffer[i] * windowFunction[i], 0.0f);
        }
        else {
            fftBuffer[i] = std::complex<float>(0.0f, 0.0f);
        }
    }

    size_t n = FFT_SIZE;
    for (size_t k = 2; k <= n; k *= 2) {
        for (size_t j = 0; j < n; j += k) {
            for (size_t i = j; i < j + k / 2; ++i) {
                size_t im = i + k / 2;
                float angle = -2.0f * static_cast<float>(M_PI) * static_cast<float>(i - j) / static_cast<float>(k);
                std::complex<float> w(std::cos(angle), std::sin(angle));
                std::complex<float> t = fftBuffer[i];
                std::complex<float> u = fftBuffer[im] * w;
                fftBuffer[i] = t + u;
                fftBuffer[im] = t - u;
            }
        }
    }
}

float AudioAnalyzer::calculateSpectralCentroid() {
    float weightedSum = 0.0f;
    float magnitudeSum = 0.0f;

    for (size_t i = 0; i < FFT_SIZE / 2; ++i) {
        float magnitude = std::abs(fftBuffer[i]);
        float frequency = static_cast<float>(i * SAMPLE_RATE) / static_cast<float>(FFT_SIZE);

        weightedSum += magnitude * frequency;
        magnitudeSum += magnitude;
    }

    if (magnitudeSum > 0.0f) {
        return weightedSum / (magnitudeSum * static_cast<float>(SAMPLE_RATE / 2));
    }
    return 0.0f;
}

void AudioAnalyzer::normalizeValue(float& value, float minValue, float maxValue) {
    value = std::max(minValue, std::min(value, maxValue));
}

float AudioAnalyzer::smoothValue(float current, float target, float smoothing) {
    return current * smoothing + target * (1.0f - smoothing);
}