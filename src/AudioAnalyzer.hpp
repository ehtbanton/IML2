#ifndef AUDIO_ANALYZER_HPP
#define AUDIO_ANALYZER_HPP

#include <SFML/Audio.hpp>
#include <vector>
#include <complex>
#include <cmath>
#include <memory>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <thread>
#include <mutex>
#include <atomic>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class AudioAnalyzer {
private:
    static const size_t SAMPLE_RATE = 44100;
    static const size_t BUFFER_SIZE = 2048;
    static const size_t FFT_SIZE = 2048;

    // WASAPI related members
    IMMDeviceEnumerator* deviceEnumerator;
    IMMDevice* device;
    IAudioClient* audioClient;
    IAudioCaptureClient* captureClient;
    WAVEFORMATEX* pwfx;

    // Thread management
    std::thread captureThread;
    std::atomic<bool> isRunning;

    // Double buffering to prevent deadlocks
    std::vector<float> captureBuffer;
    std::vector<float> processBuffer;
    std::mutex captureMutex;
    bool bufferReady;

    std::vector<std::complex<float>> fftBuffer;
    std::vector<float> windowFunction;

    float currentVolume;
    float currentCentroid;
    float volumeSmoothing;
    float centroidSmoothing;

    void processAudioData();
    void calculateFFT();
    float calculateRMS();
    float calculateSpectralCentroid();
    void normalizeValue(float& value, float minValue = 0.0f, float maxValue = 1.0f);
    float smoothValue(float current, float target, float smoothing);
    void initializeWindowFunction();
    void captureLoop();
    void swapBuffers();

public:
    AudioAnalyzer();
    ~AudioAnalyzer();

    bool start();
    void stop();
    void update();

    float getVolume() const { return currentVolume; }
    float getSpectralCentroid() const { return currentCentroid; }
};

#endif // AUDIO_ANALYZER_HPP