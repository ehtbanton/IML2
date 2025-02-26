#include "AudioCapture.h"
#include "SpectrogramVisualizer.h"
#include <iostream>

#pragma comment(lib, "ole32.lib")

// Function to normalize audio to a consistent RMS level
void normalizeAudioBatch(std::vector<float>& samples, float targetRMS = 0.1f) {
    if (samples.empty()) return;

    // Calculate current RMS
    float sumSquares = 0.0f;
    for (const float& sample : samples) {
        sumSquares += sample * sample;
    }

    float currentRMS = std::sqrt(sumSquares / static_cast<float>(samples.size()));

    // Avoid division by very small numbers and only normalize if needed
    if (currentRMS > 1e-6f) {
        // Calculate scaling factor
        float scale = targetRMS / currentRMS;

        // Apply scaling
        for (float& sample : samples) {
            sample *= scale;
        }
    }
}

AudioCapture::AudioCapture()
    : deviceEnumerator(nullptr)
    , device(nullptr)
    , audioClient(nullptr)
    , captureClient(nullptr)
    , pwfx(nullptr)
    , isRunning(false)
    , sampleRate(48000)
    , numChannels(2)
    , spectrogram(nullptr)
    , totalSamplesProcessed(0)
    , batchSize(48000)
{
    HRESULT hr = CoInitialize(nullptr);
    if (SUCCEEDED(hr)) {
        initializeDevice();
    }
    captureBuffer.reserve(batchSize);
}

AudioCapture::~AudioCapture() {
    stop();
    cleanupDevice();
    CoUninitialize();
}

void AudioCapture::setSpectrogram(Spectrogram* spec) {
    spectrogram = spec;
}

bool AudioCapture::start() {
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

    // Clear capture buffer
    {
        std::lock_guard<std::mutex> lock(bufferMutex);
        captureBuffer.clear();
        captureBuffer.reserve(batchSize);
    }

    isRunning = true;
    captureThread = std::thread(&AudioCapture::captureLoop, this);

    // Set thread priority to time critical for best audio performance
    SetThreadPriority(captureThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
    return true;
}

void AudioCapture::stop() {
    isRunning = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }
    if (audioClient) {
        audioClient->Stop();
    }
}

bool AudioCapture::initializeDevice() {
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

void AudioCapture::cleanupDevice() {
    if (captureClient) { captureClient->Release(); captureClient = nullptr; }
    if (audioClient) { audioClient->Release(); audioClient = nullptr; }
    if (device) { device->Release(); device = nullptr; }
    if (deviceEnumerator) { deviceEnumerator->Release(); deviceEnumerator = nullptr; }
    if (pwfx) { CoTaskMemFree(pwfx); pwfx = nullptr; }
}

// CRITICAL FIX: Simplified AudioCapture processing to avoid freezing
void AudioCapture::processBatch(std::vector<float>& batchBuffer) {
    if (!spectrogram || batchBuffer.empty()) return;

    // Process in smaller chunks to avoid blocking too long
    static const size_t MAX_FRAMES_PER_BATCH = 10;
    static const size_t STEP_SIZE = FFT_SIZE / 2;  // 50% overlap

    // Limit the number of frames we process to avoid freezing
    size_t framesToProcess = std::min(
        (batchBuffer.size() - FFT_SIZE) / STEP_SIZE + 1,
        MAX_FRAMES_PER_BATCH
    );

    for (size_t i = 0; i < framesToProcess; i++) {
        size_t startPos = i * STEP_SIZE;
        if (startPos + FFT_SIZE > batchBuffer.size()) break;

        // Extract frame
        std::vector<float> frame(FFT_SIZE);
        for (size_t j = 0; j < FFT_SIZE; j++) {
            frame[j] = batchBuffer[startPos + j];
        }

        // Normalize if not empty
        bool isEmpty = true;
        for (float sample : frame) {
            if (std::abs(sample) > 1e-6f) {
                isEmpty = false;
                break;
            }
        }

        if (!isEmpty) {
            normalizeAudioBatch(frame);
        }

        // Process the frame (this will be done in the audio thread)
        spectrogram->processSamples(frame);
    }
}

// CRITICAL FIX: Simplified captureLoop to prevent excessive processing
void AudioCapture::captureLoop() {
    batchStartTime = std::chrono::steady_clock::now();

    // Use simpler capture buffer handling
    std::vector<float> captureBuffer;
    captureBuffer.reserve(batchSize);

    while (isRunning) {
        UINT32 packetLength = 0;
        HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsedSeconds = std::chrono::duration<double>(now - batchStartTime).count();

        // Process data every 0.1 seconds (faster but not too frequent)
        if (elapsedSeconds >= 0.1 && captureBuffer.size() >= FFT_SIZE) {
            processBatch(captureBuffer);

            // Keep a small overlap for continuous processing
            size_t overlapSize = std::min(FFT_SIZE, captureBuffer.size());
            std::vector<float> overlap(captureBuffer.end() - overlapSize, captureBuffer.end());
            captureBuffer = overlap;
            batchStartTime = now;
        }

        if (packetLength == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Process audio packets
        while (packetLength > 0) {
            BYTE* data;
            UINT32 numFramesAvailable;
            DWORD flags;

            hr = captureClient->GetBuffer(&data, &numFramesAvailable, &flags, nullptr, nullptr);
            if (FAILED(hr)) break;

            if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                float* floatData = reinterpret_cast<float*>(data);

                // Add samples to buffer (simplified)
                size_t startIdx = captureBuffer.size();
                captureBuffer.resize(startIdx + numFramesAvailable);

                for (size_t i = 0; i < numFramesAvailable; ++i) {
                    // Mix to mono
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < numChannels && (i * numChannels + ch) < (numFramesAvailable * numChannels); ++ch) {
                        sum += floatData[i * numChannels + ch];
                    }
                    captureBuffer[startIdx + i] = sum / static_cast<float>(numChannels);
                }
            }

            hr = captureClient->ReleaseBuffer(numFramesAvailable);
            if (FAILED(hr)) break;

            hr = captureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) break;
        }
    }
}