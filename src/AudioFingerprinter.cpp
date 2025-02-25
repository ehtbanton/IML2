#define _USE_MATH_DEFINES
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "AudioFingerprinter.h"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>

AudioFingerprinter::AudioFingerprinter()
    : debug_level(DebugLevel::Basic),
    test_mode(false),
    log_file_open(false),
    last_match_position(0),
    last_confidence(0.0f),
    last_match_counter(0),
    silence_counter(0)
{
    // Initialize debug logging
    openDebugLog();
}

AudioFingerprinter::~AudioFingerprinter() {
    closeDebugLog();
}

void AudioFingerprinter::openDebugLog() {
    if (!log_file_open) {
        debug_log.open("fingerprinter_debug.log");
        if (debug_log.is_open()) {
            log_file_open = true;
            debug_log << "--- AudioFingerprinter Debug Log Started ---" << std::endl;
            debug_log << "Timestamp: " << getCurrentTimestamp() << std::endl;
            debug_log << "FFT_SIZE: " << FFT_SIZE << ", HOP_SIZE: " << HOP_SIZE << std::endl;
            debug_log << "NUM_PEAKS: " << NUM_PEAKS << ", MAX_TIME_DELTA: " << MAX_TIME_DELTA << std::endl;
        }
    }
}

void AudioFingerprinter::closeDebugLog() {
    if (log_file_open) {
        debug_log << "--- AudioFingerprinter Debug Log Closed ---" << std::endl;
        debug_log.close();
        log_file_open = false;
    }
}

std::string AudioFingerprinter::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    // Use localtime_s instead of localtime for security
    struct tm timeinfo;
    localtime_s(&timeinfo, &time);

    std::stringstream ss;
    ss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void AudioFingerprinter::setDebugLevel(DebugLevel level) {
    debug_level = level;
    if (log_file_open) {
        debug_log << "Debug level set to: " << static_cast<int>(level) << std::endl;
    }
}

void AudioFingerprinter::enableTestMode(bool enable) {
    test_mode = enable;
    if (log_file_open) {
        debug_log << "Test mode " << (enable ? "enabled" : "disabled") << std::endl;
    }
}

void AudioFingerprinter::logDebug(const std::string& message, DebugLevel level) {
    if (log_file_open && level <= debug_level) {
        debug_log << message << std::endl;
    }
}

uint64_t AudioFingerprinter::createHash(int freq1, int freq2, int delta) {
    // Use frequency bands with adaptive binning for better robustness
    // Lower frequencies get higher resolution bands
    int band1, band2;

    if (freq1 < 50) {
        band1 = freq1; // Full resolution for very low frequencies
    }
    else if (freq1 < 200) {
        band1 = 50 + (freq1 - 50) / 2; // Half bin size for low-mid
    }
    else if (freq1 < 1000) {
        band1 = 125 + (freq1 - 200) / 5; // 1/5 bin size for mid range
    }
    else {
        band1 = 285 + (freq1 - 1000) / 20; // 1/20 bin size for high frequencies
    }

    // Apply same binning to freq2
    if (freq2 < 50) {
        band2 = freq2;
    }
    else if (freq2 < 200) {
        band2 = 50 + (freq2 - 50) / 2;
    }
    else if (freq2 < 1000) {
        band2 = 125 + (freq2 - 200) / 5;
    }
    else {
        band2 = 285 + (freq2 - 1000) / 20;
    }

    // Quantize time delta with adaptive quantization
    int quantized_delta;
    if (delta < 5) {
        quantized_delta = delta; // Full resolution for very close pairs
    }
    else if (delta < 20) {
        quantized_delta = 5 + (delta - 5) / 2; // Half resolution
    }
    else {
        quantized_delta = 12 + (delta - 20) / 3; // One-third resolution
    }

    if (debug_level >= DebugLevel::Verbose) {
        std::stringstream ss;
        ss << "Hash: freq1=" << freq1 << " -> band1=" << band1
            << ", freq2=" << freq2 << " -> band2=" << band2
            << ", delta=" << delta << " -> quantized=" << quantized_delta;
        logDebug(ss.str(), DebugLevel::Verbose);
    }

    // Combine into hash - we ensure each value uses a proper bit range
    return (static_cast<uint64_t>(band1 & 0x3FF) << 22) |
        (static_cast<uint64_t>(band2 & 0x3FF) << 12) |
        (static_cast<uint64_t>(quantized_delta) & 0xFFF);
}

std::vector<std::pair<int, float>> AudioFingerprinter::extractPeaks(const std::vector<float>& spectrum, int frame_index) {
    std::vector<std::pair<int, float>> peaks;

    // Skip if spectrum is empty
    if (spectrum.empty()) {
        logDebug("Warning: Empty spectrum provided to extractPeaks", DebugLevel::Basic);
        return peaks;
    }

    // Calculate average and maximum magnitude for adaptive thresholding
    float avg = 0.0f;
    float max_mag = 0.0f;
    for (float mag : spectrum) {
        avg += mag;
        max_mag = std::max(max_mag, mag);
    }
    avg /= static_cast<float>(spectrum.size());

    // Adaptive threshold based on average magnitude
    float threshold = std::max(0.001f, std::max(avg * 2.0f, max_mag * 0.3f));

    // Logarithmic frequency bands for emphasis on important ranges
    float emphasis[5] = { 1.0f, 1.2f, 1.5f, 1.2f, 0.8f }; // Emphasize mid frequencies

    // Find local maxima above threshold with frequency band emphasis
    for (int i = 2; i < static_cast<int>(spectrum.size()) - 2; i++) {
        // Apply frequency emphasis
        int band_idx = 0;
        if (i < static_cast<int>(spectrum.size()) * 0.1) band_idx = 0;        // 0-10% (lowest)
        else if (i < static_cast<int>(spectrum.size()) * 0.3) band_idx = 1;   // 10-30% (low-mid)
        else if (i < static_cast<int>(spectrum.size()) * 0.6) band_idx = 2;   // 30-60% (mid)
        else if (i < static_cast<int>(spectrum.size()) * 0.8) band_idx = 3;   // 60-80% (high-mid)
        else band_idx = 4;                                                    // 80-100% (highest)

        float emphasized_mag = spectrum[i] * emphasis[band_idx];

        if (emphasized_mag > threshold &&
            emphasized_mag > spectrum[i - 1] * emphasis[band_idx] &&
            emphasized_mag > spectrum[i - 2] * emphasis[band_idx] &&
            emphasized_mag > spectrum[i + 1] * emphasis[band_idx] &&
            emphasized_mag > spectrum[i + 2] * emphasis[band_idx]) {

            // We store the original magnitude, not the emphasized one
            peaks.push_back({ i, spectrum[i] });
        }
    }

    // Sort by magnitude (descending)
    std::sort(peaks.begin(), peaks.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Debug logging
    if (debug_level >= DebugLevel::Detailed && (frame_index % 100 == 0 || test_mode)) {
        std::stringstream ss;
        ss << "Frame " << frame_index << ": Found " << peaks.size()
            << " peaks (avg=" << avg << ", max=" << max_mag
            << ", threshold=" << threshold << ")";
        logDebug(ss.str(), DebugLevel::Detailed);

        if (debug_level >= DebugLevel::Verbose && !peaks.empty()) {
            ss.str("");
            ss << "  Top peaks: ";
            for (size_t i = 0; i < std::min(peaks.size(), static_cast<size_t>(5)); i++) {
                ss << "(" << peaks[i].first << ", " << peaks[i].second << ") ";
            }
            logDebug(ss.str(), DebugLevel::Verbose);
        }
    }

    // Keep only top peaks
    if (peaks.size() > NUM_PEAKS) {
        peaks.resize(NUM_PEAKS);
    }

    return peaks;
}

void AudioFingerprinter::generateFingerprints(const std::vector<std::vector<float>>& frames) {
    fingerprint_table.clear();
    fingerprint_counts.clear();

    std::stringstream ss;
    ss << "Generating fingerprints from " << frames.size() << " frames...";
    logDebug(ss.str(), DebugLevel::Basic);
    std::cout << ss.str() << std::endl;

    size_t hash_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < frames.size(); t++) {
        auto anchor_peaks = extractPeaks(frames[t], static_cast<int>(t));

        // Look ahead for target frames
        for (size_t dt = 1; dt <= MAX_TIME_DELTA && t + dt < frames.size(); dt++) {
            auto target_peaks = extractPeaks(frames[t + dt], static_cast<int>(t + dt));

            // Create fingerprints by pairing peaks
            for (const auto& [f1, _] : anchor_peaks) {
                for (const auto& [f2, __] : target_peaks) {
                    uint64_t hash = createHash(f1, f2, static_cast<int>(dt));
                    fingerprint_table[hash].push_back(t);
                    fingerprint_counts[hash]++;
                    hash_count++;
                }
            }
        }

        // Progress update
        if (t % 500 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();

            ss.str("");
            ss << "Processed " << t << "/" << frames.size() << " frames in "
                << duration << "ms, " << fingerprint_table.size()
                << " unique hashes, " << hash_count << " total hashes";
            logDebug(ss.str(), DebugLevel::Basic);
            std::cout << ss.str() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // Calculate some stats for debugging
    float avg_positions_per_hash = static_cast<float>(hash_count) /
        static_cast<float>(fingerprint_table.size() > 0 ? fingerprint_table.size() : 1);

    ss.str("");
    ss << "Fingerprint generation complete in " << total_duration << "ms: "
        << fingerprint_table.size() << " unique hashes, " << hash_count
        << " total hashes, " << avg_positions_per_hash
        << " average positions per hash";
    logDebug(ss.str(), DebugLevel::Basic);
    std::cout << ss.str() << std::endl;

    // Check for hash distribution problems
    int count_high_collision = 0;
    for (const auto& [hash, count] : fingerprint_counts) {
        if (count > 100) { // Arbitrary threshold for high-collision hashes
            count_high_collision++;
        }
    }

    ss.str("");
    ss << "Found " << count_high_collision << " high-collision hashes ("
        << (count_high_collision * 100.0f / (fingerprint_table.size() > 0 ? fingerprint_table.size() : 1)) << "%)";
    logDebug(ss.str(), DebugLevel::Basic);
    std::cout << ss.str() << std::endl;
}

void AudioFingerprinter::setReferenceData(const std::vector<std::vector<float>>& spectrogram) {
    logDebug("Setting reference data with " + std::to_string(spectrogram.size()) + " frames", DebugLevel::Basic);
    reference_spectrogram = spectrogram;
    generateFingerprints(spectrogram);
}

// Returns {position, confidence}
std::pair<size_t, float> AudioFingerprinter::findMatchWithConfidence(const std::vector<float>& current_frame) {
    static int match_counter = 0;
    match_counter++;

    // Log every 30 frames (half second at 60fps)
    bool should_log = (match_counter % 30 == 0) || test_mode;
    std::stringstream ss;

    // Add current frame to buffer
    frame_buffer.push_back(current_frame);
    if (frame_buffer.size() > HISTORY_SIZE) {
        frame_buffer.pop_front();
    }

    // Need sufficient buffer for matching
    if (frame_buffer.size() < MIN_BUFFER_FOR_MATCH) {
        if (should_log) {
            ss << "Buffer too small: " << frame_buffer.size() << "/" << MIN_BUFFER_FOR_MATCH;
            logDebug(ss.str(), DebugLevel::Detailed);
        }

        return { last_match_position, last_confidence };
    }

    // Check if frame has enough energy
    float frame_energy = 0.0f;
    for (float val : current_frame) {
        frame_energy += val * val;
    }

    if (frame_energy < ENERGY_THRESHOLD) {
        // Frame has very low energy, might be silence
        if (should_log) {
            ss << "Low energy frame: " << frame_energy << " < " << ENERGY_THRESHOLD;
            logDebug(ss.str(), DebugLevel::Detailed);
        }
        silence_counter++;

        if (silence_counter > MAX_SILENCE_FRAMES) {
            // Too many silent frames, reset matching state
            if (last_confidence > 0.0f) {
                logDebug("Consecutive silence detected, resetting matching state", DebugLevel::Basic);
                last_confidence = 0.0f;
            }
        }

        return { last_match_position, last_confidence * 0.95f }; // Decay confidence during silence
    }
    else {
        silence_counter = 0;
    }

    // Only do full matching every N frames to save CPU
    if (!test_mode && match_counter % MATCH_INTERVAL != 0 && last_confidence > 0.1f) {
        // For intermediate frames, just increment the match position
        float hop_interval = static_cast<float>(HOP_SIZE) / 48000.0f; // Seconds per hop
        float frames_since_last_match = static_cast<float>(match_counter % MATCH_INTERVAL);

        // Estimate how many reference frames to advance
        size_t frames_to_advance = static_cast<size_t>(frames_since_last_match * hop_interval /
            (static_cast<float>(HOP_SIZE) / 48000.0f));

        if (frames_to_advance > 0 && last_match_position + frames_to_advance < reference_spectrogram.size()) {
            last_match_position += frames_to_advance;
        }

        // Slightly decay confidence for interpolated positions
        return { last_match_position, last_confidence * 0.99f };
    }

    // Create histogram of position votes
    std::unordered_map<size_t, int> position_votes;
    int total_votes = 0;
    int total_valid_hashes = 0;
    int match_attempts = 0;

    // Process each frame in buffer as potential anchor
    // To save CPU, analyze every N frames where N depends on buffer size
    int stride = std::max(1, static_cast<int>(frame_buffer.size()) / 60); // Aim for ~60 anchors max

    for (size_t i = 0; i < frame_buffer.size() - MAX_TIME_DELTA; i += static_cast<size_t>(stride)) {
        auto anchor_peaks = extractPeaks(frame_buffer[i], match_counter * 1000 + static_cast<int>(i));

        // Look ahead for target frames
        for (size_t dt = 1; dt <= MAX_TIME_DELTA && i + dt < frame_buffer.size(); dt++) {
            auto target_peaks = extractPeaks(frame_buffer[i + dt], match_counter * 1000 + static_cast<int>(i + dt));

            // Create fingerprints and query hash table
            for (const auto& [f1, _] : anchor_peaks) {
                for (const auto& [f2, __] : target_peaks) {
                    match_attempts++;
                    uint64_t hash = createHash(f1, f2, static_cast<int>(dt));

                    // Look up in reference
                    auto it = fingerprint_table.find(hash);
                    if (it != fingerprint_table.end()) {
                        total_valid_hashes++;

                        for (size_t ref_pos : it->second) {
                            // Weight votes by inverse frequency (rare hashes count more)
                            float weight = 1.0f;
                            auto freq_it = fingerprint_counts.find(hash);
                            if (freq_it != fingerprint_counts.end() && freq_it->second > 1) {
                                weight = 1.0f / std::sqrt(static_cast<float>(freq_it->second));
                            }

                            // Boost weight for hashes with good anchor-target frequency separation
                            float freq_ratio = static_cast<float>(std::max(f1, f2)) /
                                static_cast<float>(std::min(f1, f2));
                            if (freq_ratio > 1.5f) {
                                weight *= 1.2f; // Boost distinctive frequency pairs
                            }

                            // Apply adaptive temporal weighting
                            if (last_confidence > 0.3f) {
                                // If we had a good match previously, favor nearby positions
                                float time_diff = std::abs(static_cast<int>(ref_pos) -
                                    static_cast<int>(last_match_position));
                                float expected_diff = static_cast<float>(match_counter - last_match_counter);

                                // If position difference is close to expected, boost weight
                                if (std::abs(time_diff - expected_diff) < expected_diff * 0.2f) {
                                    weight *= 1.5f;
                                }
                            }

                            position_votes[ref_pos] += static_cast<int>(weight * 100.0f);
                            total_votes += static_cast<int>(weight * 100.0f);
                        }
                    }
                }
            }
        }
    }

    if (should_log) {
        ss.str("");
        ss << "Match attempt: " << match_attempts << " hashes checked, "
            << total_valid_hashes << " valid hashes found, "
            << position_votes.size() << " positions with votes";
        logDebug(ss.str(), DebugLevel::Detailed);
    }

    // Find best match with sliding window
    size_t best_position = last_match_position;
    int best_votes = 0;
    float best_confidence = 0.0f;

    // If we have previous match with good confidence, use a narrower search window
    size_t search_start = 0;
    size_t search_end = reference_spectrogram.size();

    const size_t EXPECTED_MATCH_WINDOW = 100; // Frames

    if (last_confidence > 0.4f && match_counter - last_match_counter < 60) {
        // Calculate expected position based on time elapsed
        float expected_advance = (match_counter - last_match_counter) * 0.5f; // Heuristic
        size_t expected_pos = std::min(last_match_position + static_cast<size_t>(expected_advance),
            reference_spectrogram.size() - 1);

        // Search around expected position
        search_start = (expected_pos > EXPECTED_MATCH_WINDOW) ? expected_pos - EXPECTED_MATCH_WINDOW : 0;
        search_end = std::min(expected_pos + EXPECTED_MATCH_WINDOW, reference_spectrogram.size());

        if (should_log) {
            ss.str("");
            ss << "Narrowed search window: " << search_start << "-" << search_end
                << " (expected pos: " << expected_pos << ")";
            logDebug(ss.str(), DebugLevel::Detailed);
        }
    }

    // Find position with most votes in a window
    for (size_t pos = search_start; pos < search_end; pos++) {
        int votes_in_window = 0;
        const int WINDOW_RADIUS = 5; // 11-frame window centered on position

        // Count votes in a window
        for (int dt = -WINDOW_RADIUS; dt <= WINDOW_RADIUS; dt++) {
            int index = static_cast<int>(pos) + dt;
            if (index >= 0 && static_cast<size_t>(index) < reference_spectrogram.size()) {
                votes_in_window += position_votes[static_cast<size_t>(index)];
            }
        }

        if (votes_in_window > best_votes) {
            best_votes = votes_in_window;
            best_position = pos;
        }
    }

    // Calculate confidence score
    if (total_votes > 0) {
        best_confidence = static_cast<float>(best_votes) / static_cast<float>(total_votes);

        // Scale confidence for more usable range
        best_confidence = std::min(1.0f, best_confidence * 6.0f);

        // Adjust confidence with some history weighting
        if (last_confidence > 0.0f) {
            // Check if new position is consistent with previous position
            float expected_advance = static_cast<float>(match_counter - last_match_counter);
            float actual_advance = static_cast<float>(best_position) - static_cast<float>(last_match_position);

            // If advance is very different from expected, reduce confidence
            if (std::abs(actual_advance - expected_advance) > expected_advance * 0.5f &&
                std::abs(actual_advance - expected_advance) > 10.0f) {
                best_confidence *= 0.5f;

                if (should_log) {
                    ss.str("");
                    ss << "Position jump: expected +" << expected_advance
                        << ", actual +" << actual_advance << " (confidence reduced)";
                    logDebug(ss.str(), DebugLevel::Detailed);
                }
            }

            // Blend with previous confidence for stability
            best_confidence = best_confidence * 0.7f + last_confidence * 0.3f;
        }
    }

    if (should_log) {
        ss.str("");
        ss << "Match result: pos=" << best_position
            << ", votes=" << best_votes << "/" << total_votes
            << ", raw confidence=" << (total_votes > 0 ? static_cast<float>(best_votes) / total_votes : 0.0f)
            << ", adjusted confidence=" << best_confidence;
        logDebug(ss.str(), DebugLevel::Basic);
    }

    // Update state if confidence is sufficient
    if (best_confidence > MIN_ACCEPT_CONFIDENCE ||
        (best_confidence > MIN_CONTINUE_CONFIDENCE &&
            std::abs(static_cast<int>(best_position) - static_cast<int>(last_match_position)) < 20)) {

        last_match_position = best_position;
        last_confidence = best_confidence;
        last_match_counter = match_counter;
    }
    else {
        // Decay confidence if we couldn't find a good match
        last_confidence *= 0.95f;

        if (should_log) {
            ss.str("");
            ss << "No strong match found, confidence decayed to " << last_confidence;
            logDebug(ss.str(), DebugLevel::Detailed);
        }
    }

    return { last_match_position, last_confidence };
}

// Legacy method for backward compatibility
size_t AudioFingerprinter::findMatch(const std::vector<float>& current_frame) {
    return findMatchWithConfidence(current_frame).first;
}

void AudioFingerprinter::reset() {
    frame_buffer.clear();
    last_match_position = 0;
    last_confidence = 0.0f;
    last_match_counter = 0;
    silence_counter = 0;

    logDebug("Fingerprinter state reset", DebugLevel::Basic);
}

// Testing method to inject known test data
void AudioFingerprinter::injectTestData(const std::vector<std::vector<float>>& test_frames,
    size_t expected_position) {
    test_mode = true;

    logDebug("Injecting " + std::to_string(test_frames.size()) +
        " test frames, expected position: " + std::to_string(expected_position),
        DebugLevel::Basic);

    reset();

    // Process test frames
    for (const auto& frame : test_frames) {
        auto [match_pos, confidence] = findMatchWithConfidence(frame);

        std::stringstream ss;
        ss << "Test injection - Frame result: pos=" << match_pos
            << ", confidence=" << confidence
            << ", expected=" << expected_position
            << ", error=" << std::abs(static_cast<int>(match_pos) - static_cast<int>(expected_position));
        logDebug(ss.str(), DebugLevel::Basic);
    }

    test_mode = false;
}

std::vector<std::vector<float>> AudioFingerprinter::extractTestSegment(
    size_t start_position, size_t length) {

    std::vector<std::vector<float>> result;

    if (start_position >= reference_spectrogram.size()) {
        logDebug("Test segment extraction failed: start_position out of range", DebugLevel::Basic);
        return result;
    }

    size_t end_position = std::min(start_position + length, reference_spectrogram.size());
    result.reserve(end_position - start_position);

    for (size_t i = start_position; i < end_position; i++) {
        result.push_back(reference_spectrogram[i]);
    }

    logDebug("Extracted test segment: " + std::to_string(result.size()) +
        " frames starting at position " + std::to_string(start_position),
        DebugLevel::Basic);

    return result;
}

// Run self-test with different segments of the reference audio
bool AudioFingerprinter::runSelfTest() {
    if (reference_spectrogram.empty()) {
        logDebug("Self-test failed: No reference data loaded", DebugLevel::Basic);
        return false;
    }

    logDebug("Starting self-test with reference data", DebugLevel::Basic);
    test_mode = true;

    const size_t NUM_TESTS = 5;
    const size_t TEST_SEGMENT_LENGTH = 100; // ~2 seconds

    int pass_count = 0;

    for (size_t t = 0; t < NUM_TESTS; t++) {
        // Extract test segment from different parts of the reference
        size_t test_position = (reference_spectrogram.size() / (NUM_TESTS + 1)) * (t + 1);

        // Apply small random offset
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(-25, 25);

        test_position = std::max(static_cast<size_t>(0),
            std::min(reference_spectrogram.size() - TEST_SEGMENT_LENGTH - 1,
                test_position + static_cast<size_t>(distrib(gen))));

        auto test_segment = extractTestSegment(test_position, TEST_SEGMENT_LENGTH);
        if (test_segment.empty()) {
            continue;
        }

        // Reset matcher state
        reset();

        // Process test segment
        std::vector<std::pair<size_t, float>> match_results;
        for (const auto& frame : test_segment) {
            match_results.push_back(findMatchWithConfidence(frame));
        }

        // Check final result
        size_t final_position = match_results.back().first;
        float final_confidence = match_results.back().second;

        // Calculate expected final position (should be close to original position + segment length)
        size_t expected_position = test_position + TEST_SEGMENT_LENGTH - 1;

        // Determine pass/fail based on position and confidence
        bool position_ok = std::abs(static_cast<int>(final_position) - static_cast<int>(expected_position)) <= 20;
        bool confidence_ok = final_confidence > 0.5f;
        bool test_passed = position_ok && confidence_ok;

        std::stringstream ss;
        ss << "Self-test #" << (t + 1) << ": "
            << (test_passed ? "PASS" : "FAIL") << " - "
            << "Expected pos: " << expected_position
            << ", Actual pos: " << final_position
            << ", Error: " << std::abs(static_cast<int>(final_position) - static_cast<int>(expected_position))
            << ", Confidence: " << final_confidence;
        logDebug(ss.str(), DebugLevel::Basic);
        std::cout << ss.str() << std::endl;

        if (test_passed) {
            pass_count++;
        }
    }

    test_mode = false;

    bool all_passed = (pass_count == NUM_TESTS);
    std::stringstream ss;
    ss << "Self-test complete: " << pass_count << "/" << NUM_TESTS << " tests passed";
    logDebug(ss.str(), DebugLevel::Basic);
    std::cout << ss.str() << std::endl;

    return all_passed;
}