from audio_similarity import AudioSimilarityAnalyzer

def chooseColour(similarity_analyzer):
    if similarity_analyzer.similarity_threshold > 0.85:
        colour = [0, 255, 0]
    else:
        colour = [255, 0, 0]
    return colour