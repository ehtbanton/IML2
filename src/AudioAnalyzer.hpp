// AudioAnalyzer.hpp
#ifndef AUDIO_ANALYZER_HPP
#define WINDOW_DISPLAY_CONTROLLER_HPP

#include <SFML/Audio.hpp>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class AudioAnalyzer {
private:
    static const size_t SAMPLE_RATE = 44100;
    static const size_t BUFFER_SIZE = 2048;

    // Test signal parameters
    float time;
    float frequency;
    float amplitude;

    float currentVolume;
    float currentCentroid;

    // Helper functions
    void generateTestSignal();
    void normalizeValue(float& value, float minValue = 0.0f, float maxValue = 1.0f);
    float smoothValue(float current, float target, float smoothing);

    float volumeSmoothing;
    float centroidSmoothing;

public:
    AudioAnalyzer();

    bool start();
    void stop();
    void update();

    // Getters for normalized values (0.0 to 1.0)
    float getVolume() const { return currentVolume; }
    float getSpectralCentroid() const { return currentCentroid; }
};

#endif // AUDIO_ANALYZER_HPP