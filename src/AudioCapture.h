#pragma once

// First define these to prevent Windows macro conflicts
#define NOMINMAX 
#define WIN32_LEAN_AND_MEAN

// Include SFML before Windows headers
#include <SFML/System.hpp>

// Now safe to include Windows headers
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>

// Standard library headers
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>

// Forward declaration
class Spectrogram;

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

    std::atomic<size_t> totalSamplesProcessed;
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

    std::vector<float> stretchAudio(const std::vector<float>& input);
    bool initializeDevice();
    void cleanupDevice();
    void processBatch(std::vector<float>& batchBuffer);
    void captureLoop();

public:
    AudioCapture();
    ~AudioCapture();

    void setSpectrogram(Spectrogram* spec);
    bool start();
    void stop();
};