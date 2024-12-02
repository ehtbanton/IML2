// AudioAnalyzer.hpp
#ifndef AUDIO_ANALYZER_HPP
#define AUDIO_ANALYZER_HPP

#include "WasapiCapture.hpp"
#include <vector>
#include <complex>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class AudioAnalyzer {
public:
    AudioAnalyzer(size_t bufferSize = 2048);
    ~AudioAnalyzer();

    bool start();
    void stop();
    void update();

    float getVolume() const { return currentVolume; }
    float getSpectralCentroid() const { return currentCentroid; }

private:
    static const size_t FFT_SIZE = 2048;

    std::unique_ptr<WasapiCapture> audioCapture;
    std::vector<float> processBuffer;
    std::vector<std::complex<float>> fftBuffer;
    std::vector<float> windowFunction;

    float currentVolume;
    float currentCentroid;
    float volumeSmoothing;
    float centroidSmoothing;

    void calculateFFT();
    float calculateRMS();
    float calculateSpectralCentroid();
    void normalizeValue(float& value, float minValue = 0.0f, float maxValue = 1.0f);
    float smoothValue(float current, float target, float smoothing);
    void initializeWindowFunction();
};

#endif // AUDIO_ANALYZER_HPP