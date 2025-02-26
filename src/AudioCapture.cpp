#include "AudioCapture.h"
#include "SpectrogramVisualizer.h"
#include <iostream>

#pragma comment(lib, "ole32.lib")

// Function to normalize audio to a consistent RMS level
void normalizeAudioBatch(std::vector<float>& samples, float targetRMS = 0.15f) {
    if (samples.empty()) return;

    // Calculate current RMS
    float sumSquares = 0.0f;
    for (const float& sample : samples) {
        sumSquares += sample * sample;
    }

    float currentRMS = std::sqrt(sumSquares / static_cast<float>(samples.size()));

    // Avoid division by very small numbers and only normalize if needed
    if (currentRMS > 1e-6f) {
        // Calculate scaling factor and apply slightly stronger scaling to match static
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

void AudioCapture::processBatch(std::vector<float>& batchBuffer) {
    if (!spectrogram || batchBuffer.empty()) return;

    // Use 50% overlap for good spectral resolution with reasonable performance
    static const size_t STEP_SIZE = FFT_SIZE / 2;  // 50% overlap

    // Process in smaller batches to prevent UI blocking
    size_t processedFrames = 0;
    size_t maxFramesToProcess = 10; // Process at most 10 frames per batch

    for (size_t startPos = 0;
        startPos + FFT_SIZE <= batchBuffer.size() && processedFrames < maxFramesToProcess;
        startPos += STEP_SIZE, processedFrames++) {

        // Extract frame with bounds checking
        std::vector<float> frame(FFT_SIZE);
        for (size_t j = 0; j < FFT_SIZE; j++) {
            if (startPos + j < batchBuffer.size()) {
                frame[j] = batchBuffer[startPos + j];
            }
            else {
                frame[j] = 0.0f;
            }
        }

        // Check if frame contains audio or is silent
        bool isEmpty = true;
        for (size_t i = 0; i < frame.size(); i += 16) {
            if (i < frame.size() && std::abs(frame[i]) > 1e-6f) {
                isEmpty = false;
                break;
            }
        }

        // Apply normalization if not silent
        if (!isEmpty) {
            normalizeAudioBatch(frame, 0.15f);
        }

        // Process the frame
        try {
            spectrogram->processSamples(frame);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in processSamples: " << e.what() << std::endl;
        }
    }
}

// Updated AudioCapture captureLoop for improved silence handling and update rate
void AudioCapture::captureLoop() {
    batchStartTime = std::chrono::steady_clock::now();
    lastPrintTime = batchStartTime;

    std::vector<float> localCaptureBuffer;
    localCaptureBuffer.reserve(batchSize);

    // Timer for sending silence frames when no audio is detected
    sf::Clock silenceTimer;

    while (isRunning) {
        UINT32 packetLength = 0;
        HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsedSeconds = std::chrono::duration<double>(now - batchStartTime).count();

        // Process data every 0.1 seconds for faster updates
        if (elapsedSeconds >= 0.1 && localCaptureBuffer.size() >= FFT_SIZE) {
            processBatch(localCaptureBuffer);

            // Keep a small overlap for continuous processing
            size_t overlapSize = std::min(FFT_SIZE, localCaptureBuffer.size());
            std::vector<float> overlap(localCaptureBuffer.end() - overlapSize, localCaptureBuffer.end());
            localCaptureBuffer = overlap;
            batchStartTime = now;

            // Reset silence timer when we process audio
            silenceTimer.restart();
        }

        // If no data is available, check if we need to send a silence frame
        if (packetLength == 0) {
            // If it's been 100ms since last audio, send a silence frame
            if (silenceTimer.getElapsedTime().asMilliseconds() > 100) {
                // Send a silence frame to maintain display updates
                std::vector<float> silenceFrame(FFT_SIZE, 0.0f);
                if (spectrogram) {
                    spectrogram->processSamples(silenceFrame);
                }
                silenceTimer.restart();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
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

                size_t startIdx = localCaptureBuffer.size();
                localCaptureBuffer.resize(startIdx + numFramesAvailable);

                for (size_t i = 0; i < numFramesAvailable; ++i) {
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < numChannels && (i * numChannels + ch) < (numFramesAvailable * numChannels); ++ch) {
                        sum += floatData[i * numChannels + ch];
                    }
                    localCaptureBuffer[startIdx + i] = sum / static_cast<float>(numChannels);
                }
            }
            else {
                // Handle silent audio packets by adding zeros
                size_t startIdx = localCaptureBuffer.size();
                localCaptureBuffer.resize(startIdx + numFramesAvailable, 0.0f);
            }

            hr = captureClient->ReleaseBuffer(numFramesAvailable);
            if (FAILED(hr)) break;

            hr = captureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) break;
        }
    }
}