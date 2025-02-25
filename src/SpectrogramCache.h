#pragma once

#include <vector>
#include <string>
#include <cstdint>

// Cache structure for storing and retrieving spectrograms
struct SpectrogramCache {
    static const uint32_t MAGIC_NUMBER = 0x53504543;  // "SPEC" in ASCII
    uint32_t version = 1;
    uint64_t timestamp;
    std::vector<std::vector<float>> spectogram_data;

    bool serialize(const std::string& filepath) const;
    bool deserialize(const std::string& filepath);
};