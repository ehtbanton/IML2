#include "AudioCapture.h"
#include "SpectrogramVisualizer.h"
#include <iostream>

#pragma comment(lib, "ole32.lib")

AudioCapture::AudioCapture()
    : deviceEnumerator(nullptr)
    , device(nullptr)
    , audioClient(nullptr)
    , captureClient(nullptr)
    , pwfx(nullptr)
    , isRunning(false)
    , buffer1()
    , buffer2()
    , currentBuffer(&buffer1)
    , sampleRate(48000)
    , numChannels(2)
    , spectrogram(nullptr)
    , totalSamplesProcessed(0)
    , mainThread(false)
    , offsetThread(false)
{
    HRESULT hr = CoInitialize(nullptr);
    if (SUCCEEDED(hr)) {
        initializeDevice();
    }

    // Pre-allocate buffers to avoid reallocations
    buffer1.reserve(sampleRate);  // 1 second of audio
    buffer2.reserve(sampleRate);
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

    // Set to 48kHz sample rate
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

    // Clear buffers before starting
    buffer1.clear();
    buffer2.clear();
    currentBuffer = &buffer1;

    // Start main capture and processing threads
    isRunning = true;
    mainThread = true;
    offsetThread = true;

    // Start the main capture thread
    captureThread = std::thread(&AudioCapture::captureLoop, this);
    SetThreadPriority(captureThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);

    // Start the main processing thread (processes every 0.5s)
    mainProcessThread = std::thread(&AudioCapture::mainProcessLoop, this);
    SetThreadPriority(mainProcessThread.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

    // Start the offset processing thread (offset by 0.25s)
    offsetProcessThread = std::thread(&AudioCapture::offsetProcessLoop, this);
    SetThreadPriority(offsetProcessThread.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

    return true;
}

void AudioCapture::stop() {
    isRunning = false;
    mainThread = false;
    offsetThread = false;

    if (captureThread.joinable()) {
        captureThread.join();
    }

    if (mainProcessThread.joinable()) {
        mainProcessThread.join();
    }

    if (offsetProcessThread.joinable()) {
        offsetProcessThread.join();
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

void AudioCapture::processBatch(const std::vector<float>& batchBuffer, bool isOffset) {
    if (!spectrogram) return;

    // Skip if buffer is too small
    if (batchBuffer.size() < 1024) {
        return;
    }

    // For debugging purposes
    std::string threadType = isOffset ? "Offset" : "Main";

    // We want to normalize the frequency spectrum
    // Ensure we take exactly TARGET_SAMPLES / 2 samples (for half-second processing)
    std::vector<float> processBuffer;
    const size_t halfSecondSamples = TARGET_SAMPLES / 2;

    if (batchBuffer.size() >= halfSecondSamples) {
        // Take exactly half a second of audio
        processBuffer.assign(batchBuffer.end() - halfSecondSamples, batchBuffer.end());
    }
    else {
        // In case we don't have enough samples, pad with zeros
        processBuffer = batchBuffer;
        processBuffer.resize(halfSecondSamples, 0.0f);
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
        // Process the samples in the spectrogram
        spectrogram->processSamples(processBuffer);

        // Log processing for debug
        std::cout << threadType << " Thread processed " << processBuffer.size()
            << " samples at " << std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count()
            << "s" << std::endl;
    }
}

void AudioCapture::mainProcessLoop() {
    // Main processing thread - processes every 0.5 seconds
    auto startTime = std::chrono::steady_clock::now();

    while (mainThread) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();

        if (elapsedMs >= 500) { // Every 0.5 seconds
            // Take a snapshot of the current buffer
            std::vector<float> snapshot;
            {
                std::lock_guard<std::mutex> lock(bufferMutex);
                snapshot = *currentBuffer; // Copy the entire buffer
            }

            // Process the snapshot
            processBatch(snapshot, false);

            // Reset timer
            startTime = now;
        }

        // Sleep for a short time to avoid consuming too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AudioCapture::offsetProcessLoop() {
    // Offset processing thread - wait 0.25s to start, then process every 0.5s
    std::this_thread::sleep_for(std::chrono::milliseconds(250)); // Initial offset

    auto startTime = std::chrono::steady_clock::now();

    while (offsetThread) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();

        if (elapsedMs >= 500) { // Every 0.5 seconds
            // Take a snapshot of the current buffer
            std::vector<float> snapshot;
            {
                std::lock_guard<std::mutex> lock(bufferMutex);
                snapshot = *currentBuffer; // Copy the entire buffer
            }

            // Process the snapshot
            processBatch(snapshot, true);

            // Reset timer
            startTime = now;
        }

        // Sleep for a short time to avoid consuming too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AudioCapture::captureLoop() {
    auto lastPrintTime = std::chrono::steady_clock::now();

    while (isRunning) {
        UINT32 packetLength = 0;
        HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
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

                // Lock the buffer while we add samples
                std::lock_guard<std::mutex> lock(bufferMutex);

                // Reserve space for new samples to avoid reallocations
                size_t oldSize = currentBuffer->size();
                currentBuffer->resize(oldSize + numFramesAvailable);

                // Convert multi-channel to mono by averaging channels
                for (UINT32 i = 0; i < numFramesAvailable; ++i) {
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < numChannels; ++ch) {
                        sum += floatData[i * numChannels + ch];
                    }
                    (*currentBuffer)[oldSize + i] = sum / static_cast<float>(numChannels);
                }

                // Keep buffer size reasonable (retain at most 5 seconds of audio)
                const size_t maxBufferSize = 5 * sampleRate;
                if (currentBuffer->size() > maxBufferSize) {
                    currentBuffer->erase(currentBuffer->begin(),
                        currentBuffer->begin() + (currentBuffer->size() - maxBufferSize));
                }

                // Track total samples processed
                totalSamplesProcessed += numFramesAvailable;
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

        // Periodic stats output (every 5 seconds)
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastPrintTime).count();

        if (elapsed >= 5) {
            std::lock_guard<std::mutex> lock(bufferMutex);
            std::cout << "Buffer size: " << currentBuffer->size() << " samples, "
                << "Total processed: " << totalSamplesProcessed << " samples" << std::endl;
            lastPrintTime = now;
        }
    }
}