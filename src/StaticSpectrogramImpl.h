#pragma once

#include "SpectrogramBase.h"

// Static spectrogram that displays pre-processed audio
class StaticSpectrogramImpl : public SpectrogramBase {
private:
    const std::vector<std::vector<float>>& spectrogramData;

    void initializeUI();
    void updateVertexPositions();

public:
    StaticSpectrogramImpl(sf::RenderWindow& win, const std::vector<std::vector<float>>& data,
        const std::string& title, const sf::Vector2f& position);

    void draw() override;
    void handleResize() override;
};