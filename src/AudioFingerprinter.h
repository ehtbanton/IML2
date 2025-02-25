#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <utility>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>

// Custom hash function for vectors - needed for exact matching
namespace std {
    template<typename T>
    struct hash<std::vector<T>> {
        size_t operator()(const std::vector<T>& v) const {
            std::hash<T> hasher;
            size_t seed = v.size();
            // Sample every 10th value to keep hash computation fast
            for (size_t i = 0; i < v.size(); i += 10) {
                seed ^= hasher(v[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // Custom equality operator for vectors - needed for exact matching
    template<typename T>
    bool operator==(const std::vector<T>& lhs, const std::vector<T>& rhs) {
        if (lhs.size() != rhs.size()) return false;
        for (size_t i = 0; i < lhs.size(); i++) {
            // Use approximate equality for floating point
            if (std::abs(lhs[i] - rhs[i]) > 1e-6f) {
                return false;
            }
        }
        return true;
    }
}

class AudioFingerprinter {
public:
    enum class DebugLevel {
        None = 0,
        Basic = 1,
        Detailed = 2,
        Verbose = 3
    };

private:
    // Parameters for fingerprinting
    static const size_t FFT_SIZE = 1024;
    static const size_t HOP_SIZE = 512;
    static const size_t NUM_PEAKS = 6;
    static const size_t MAX_TIME_DELTA = 100;
    static const size_t HISTORY_SIZE = 10 * 48000 / HOP_SIZE; // ~10 seconds of audio
    static const size_t MIN_BUFFER_FOR_MATCH = MAX_TIME_DELTA + 10;
    static const size_t MATCH_INTERVAL = 5; // Only perform full matching every N frames
    static const size_t MAX_SILENCE_FRAMES = 10;
    static constexpr float ENERGY_THRESHOLD = 0.0001f;
    static constexpr float MIN_ACCEPT_CONFIDENCE = 0.3f;
    static constexpr float MIN_CONTINUE_CONFIDENCE = 0.1f;

    // Hash table for reference fingerprints
    std::unordered_map<uint64_t, std::vector<size_t>> fingerprint_table;
    std::unordered_map<uint64_t, int> fingerprint_counts;

    // Direct exact frame-to-position lookup table for guaranteed matching
    std::unordered_map<std::vector<float>, size_t> exact_matches;

    // Recent audio frame buffer
    std::deque<std::vector<float>> frame_buffer;

    // Reference spectrogram (for testing)
    std::vector<std::vector<float>> reference_spectrogram;

    // Match state
    size_t last_match_position = 0;
    float last_confidence = 0.0f;
    int last_match_counter = 0;
    int silence_counter = 0;

    // Debug settings
    DebugLevel debug_level;
    bool test_mode;
    std::ofstream debug_log;
    bool log_file_open;

    // Create hash from frequency peaks and time delta
    uint64_t createHash(int freq1, int freq2, int delta);

    // Extract peaks from spectrum
    std::vector<std::pair<int, float>> extractPeaks(const std::vector<float>& spectrum, int frame_index = 0);

    // Generate fingerprints from frame sequence
    void generateFingerprints(const std::vector<std::vector<float>>& frames);

    // Debug logging
    void openDebugLog();
    void closeDebugLog();
    void logDebug(const std::string& message, DebugLevel level);
    std::string getCurrentTimestamp();

public:
    AudioFingerprinter();
    ~AudioFingerprinter();

    // Debug control
    void setDebugLevel(DebugLevel level);
    void enableTestMode(bool enable);

    // Initialize with reference spectrogram
    void setReferenceData(const std::vector<std::vector<float>>& spectrogram);

    // Find match with confidence score
    std::pair<size_t, float> findMatchWithConfidence(const std::vector<float>& current_frame);

    // Backward compatibility method
    size_t findMatch(const std::vector<float>& current_frame);

    // Reset state
    void reset();

    // Testing methods
    void injectTestData(const std::vector<std::vector<float>>& test_frames, size_t expected_position);
    std::vector<std::vector<float>> extractTestSegment(size_t start_position, size_t length);
    bool runSelfTest();
};