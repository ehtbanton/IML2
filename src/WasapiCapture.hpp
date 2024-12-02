
// WasapiCapture.hpp
#ifndef WASAPI_CAPTURE_HPP
#define WASAPI_CAPTURE_HPP

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

class WasapiCapture {
public:
    WasapiCapture(size_t bufferSize = 2048);
    ~WasapiCapture();

    bool start();
    void stop();
    bool isStarted() const { return isRunning; }

    // Get the latest audio data
    bool getBuffer(std::vector<float>& outBuffer);

    // Get audio format info
    unsigned int getSampleRate() const { return sampleRate; }
    unsigned int getChannels() const { return numChannels; }

private:
    // WASAPI related members
    IMMDeviceEnumerator* deviceEnumerator;
    IMMDevice* device;
    IAudioClient* audioClient;
    IAudioCaptureClient* captureClient;
    WAVEFORMATEX* pwfx;

    // Thread management
    std::thread captureThread;
    std::atomic<bool> isRunning;

    // Double buffering
    std::vector<float> captureBuffer;
    std::vector<float> swapBuffer;
    std::mutex bufferMutex;
    std::atomic<bool> bufferReady;

    // Audio format info
    unsigned int sampleRate;
    unsigned int numChannels;

    void captureLoop();
    bool initializeDevice();
    void cleanupDevice();
};

#endif // WASAPI_CAPTURE_HPP