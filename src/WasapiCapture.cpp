// WasapiCapture.cpp
#include "WasapiCapture.hpp"
#include <iostream>
#include <functiondiscoverykeys_devpkey.h>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "winmm.lib")

WasapiCapture::WasapiCapture(size_t bufferSize)
    : deviceEnumerator(nullptr)
    , device(nullptr)
    , audioClient(nullptr)
    , captureClient(nullptr)
    , pwfx(nullptr)
    , isRunning(false)
    , captureBuffer(bufferSize, 0.0f)
    , swapBuffer(bufferSize, 0.0f)
    , bufferReady(false)
    , sampleRate(44100)
    , numChannels(2)
{
    // Initialize COM
    HRESULT hr = CoInitialize(nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM" << std::endl;
        return;
    }

    initializeDevice();
}

WasapiCapture::~WasapiCapture() {
    stop();
    cleanupDevice();
    CoUninitialize();
}

bool WasapiCapture::initializeDevice() {
    HRESULT hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        nullptr,
        CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator),
        (void**)&deviceEnumerator
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator" << std::endl;
        return false;
    }

    hr = deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default audio endpoint" << std::endl;
        return false;
    }

    return true;
}

void WasapiCapture::cleanupDevice() {
    if (captureClient) captureClient->Release();
    if (audioClient) audioClient->Release();
    if (device) device->Release();
    if (deviceEnumerator) deviceEnumerator->Release();
    if (pwfx) CoTaskMemFree(pwfx);
}

bool WasapiCapture::start() {
    if (!device || isRunning) return false;

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

    sampleRate = pwfx->nSamplesPerSec;
    numChannels = pwfx->nChannels;

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
    captureThread = std::thread(&WasapiCapture::captureLoop, this);
    return true;
}

void WasapiCapture::stop() {
    isRunning = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }
    if (audioClient) {
        audioClient->Stop();
    }
}

bool WasapiCapture::getBuffer(std::vector<float>& outBuffer) {
    if (!bufferReady) return false;

    std::lock_guard<std::mutex> lock(bufferMutex);
    outBuffer.swap(swapBuffer);
    bufferReady = false;
    return true;
}

void WasapiCapture::captureLoop() {
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
                std::lock_guard<std::mutex> lock(bufferMutex);
                float* floatData = reinterpret_cast<float*>(data);

                // Mix down to mono and store in capture buffer
                for (UINT32 i = 0; i < numFramesAvailable && i < captureBuffer.size(); i++) {
                    float sample = 0.0f;
                    for (size_t ch = 0; ch < numChannels; ch++) {
                        sample += floatData[i * numChannels + ch];
                    }
                    captureBuffer[i] = sample / static_cast<float>(numChannels);
                }

                // Swap buffers
                captureBuffer.swap(swapBuffer);
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