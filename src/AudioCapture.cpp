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
    , buffer1()
    , buffer2()
    , currentBuffer(nullptr)
    , sampleRate(48000)
    , numChannels(2)
    , spectrogram(nullptr)
    , batchSize(48000)
    , totalSamplesProcessed(0)
{
    HRESULT hr = CoInitialize(nullptr);
    if (SUCCEEDED(hr)) {
        initializeDevice();
    }
    buffer1.reserve(batchSize);
    buffer2.reserve(batchSize);
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

std::vector<float> AudioCapture::stretchAudio(const std::vector<float>& input) {
    std::vector<float> output(TARGET_SAMPLES, 0.0f);
    size_t samplesToCopy = std::min(input.size(), TARGET_SAMPLES);
    if (samplesToCopy > 0) {
        std::copy_n(input.begin(), samplesToCopy, output.begin());
    }
    return output;
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
    if (!spectrogram) return;

    // Avoid unnecessary memory allocations
    static std::vector<float> processBuffer;
    processBuffer.clear();

    if (batchBuffer.size() >= TARGET_SAMPLES) {
        processBuffer.assign(batchBuffer.begin(), batchBuffer.begin() + TARGET_SAMPLES);
    }
    else {
        processBuffer = batchBuffer;
        processBuffer.resize(TARGET_SAMPLES, 0.0f);
    }

    // Quickly determine if buffer is empty (silence)
    bool isCompletelyEmpty = true;
    for (size_t i = 0; i < processBuffer.size(); i += 16) { // Check every 16th sample
        if (std::abs(processBuffer[i]) > 1e-6) {
            isCompletelyEmpty = false;
            break;
        }
    }

    if (!isCompletelyEmpty) {
        // Normalize the audio buffer to match the reference audio's level
        normalizeAudioBatch(processBuffer);
    }

    std::vector<float> processedBatch;
    if (isCompletelyEmpty && processBuffer.empty()) {
        processedBatch.resize(TARGET_SAMPLES, 0.0f);
    }
    else {
        processedBatch = std::move(processBuffer); // Move instead of copy
    }

    spectrogram->processSamples(processedBatch);
}

void AudioCapture::captureLoop() {
    batchStartTime = std::chrono::steady_clock::now();
    lastPrintTime = batchStartTime;
    currentBuffer = &buffer1;  // Start with buffer1

    while (isRunning) {
        UINT32 packetLength = 0;
        HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsedSeconds = std::chrono::duration<double>(now - batchStartTime).count();

        // If a second has passed, switch buffers and process the filled one
        if (elapsedSeconds >= 0.5) { // Process twice per second for better responsiveness
            // Switch buffers
            std::vector<float>* bufferToProcess = currentBuffer;
            currentBuffer = (currentBuffer == &buffer1) ? &buffer2 : &buffer1;
            currentBuffer->clear();
            currentBuffer->reserve(batchSize);

            // Process the filled buffer in a separate thread to avoid missing samples
            if (bufferToProcess->size() > 100) { // Only process non-empty buffers
                std::thread processingThread([this, bufferToProcess]() {
                    processBatch(*bufferToProcess);
                    });
                processingThread.detach();  // Let it run independently
            }

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
                break;
            }

            if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                float* floatData = reinterpret_cast<float*>(data);

                // Ensure buffer has capacity to avoid reallocations
                if (currentBuffer->size() + numFramesAvailable > currentBuffer->capacity()) {
                    currentBuffer->reserve(currentBuffer->capacity() + batchSize);
                }

                // Optimize the inner loop
                currentBuffer->resize(currentBuffer->size() + numFramesAvailable);
                size_t currentSize = currentBuffer->size() - numFramesAvailable;

                // Process in batches to improve cache locality
                for (size_t i = 0; i < numFramesAvailable; ++i) {
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < numChannels; ++ch) {
                        sum += floatData[i * numChannels + ch];
                    }
                    (*currentBuffer)[currentSize + i] = sum / static_cast<float>(numChannels);
                }
            }

            hr = captureClient->ReleaseBuffer(numFramesAvailable);
            if (FAILED(hr)) {
                break;
            }

            hr = captureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                break;
            }
        }
    }
}