#include "SpectrogramCache.h"
#include <fstream>
#include <iostream>

// Implementation of the SpectrogramCache methods
bool SpectrogramCache::serialize(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));

    // Write spectogram dimensions
    uint64_t num_frames = spectogram_data.size();
    uint64_t frame_size = spectogram_data.empty() ? 0 : spectogram_data[0].size();
    file.write(reinterpret_cast<const char*>(&num_frames), sizeof(num_frames));
    file.write(reinterpret_cast<const char*>(&frame_size), sizeof(frame_size));

    // Write data
    for (const auto& frame : spectogram_data) {
        file.write(reinterpret_cast<const char*>(frame.data()),
            frame.size() * sizeof(float));
    }

    return file.good();
}

bool SpectrogramCache::deserialize(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << filepath << std::endl;
        return false;
    }

    // Read and verify header
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MAGIC_NUMBER) {
        std::cerr << "Invalid file format: Magic number mismatch" << std::endl;
        return false;
    }

    // Read version
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "Unsupported version: " << version << std::endl;
        return false;
    }

    // Read timestamp
    file.read(reinterpret_cast<char*>(&timestamp), sizeof(timestamp));

    // Read dimensions
    uint64_t num_frames, frame_size;
    file.read(reinterpret_cast<char*>(&num_frames), sizeof(num_frames));
    file.read(reinterpret_cast<char*>(&frame_size), sizeof(frame_size));

    // Resize and read data
    spectogram_data.resize(num_frames);
    for (auto& frame : spectogram_data) {
        frame.resize(frame_size);
        file.read(reinterpret_cast<char*>(frame.data()),
            frame_size * sizeof(float));
    }

    return file.good();
}