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

    IMMDeviceEnumerator* deviceEnumerator;
    IMMDevice* device;
    IAudioClient* audioClient;
    IAudioCaptureClient* captureClient;
    WAVEFORMATEX* pwfx;

    std::atomic<size_t> totalSamplesProcessed;

    std::thread captureThread;       // Thread for audio capture
    std::thread mainProcessThread;   // Thread for main processing (every 0.5s)
    std::thread offsetProcessThread; // Thread for offset processing (0.25s offset, then every 0.5s)

    std::atomic<bool> isRunning;     // Main control flag
    std::atomic<bool> mainThread;    // Control flag for main processing thread
    std::atomic<bool> offsetThread;  // Control flag for offset processing thread

    std::vector<float> buffer1;      // First audio buffer
    std::vector<float> buffer2;      // Second audio buffer
    std::vector<float>* currentBuffer; // Pointer to the active buffer

    std::mutex bufferMutex;          // Mutex for thread-safe buffer access
    unsigned int sampleRate;
    unsigned int numChannels;
    Spectrogram* spectrogram;

    bool initializeDevice();
    void cleanupDevice();
    void processBatch(const std::vector<float>& batchBuffer, bool isOffset);
    void captureLoop();              // Function for the capture thread
    void mainProcessLoop();          // Function for main processing thread
    void offsetProcessLoop();        // Function for offset processing thread

public:
    AudioCapture();
    ~AudioCapture();

    void setSpectrogram(Spectrogram* spec);
    bool start();
    void stop();
};