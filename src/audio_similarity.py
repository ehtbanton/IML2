import numpy as np
from collections import deque
from scipy.spatial.distance import cosine
import time

class AudioSimilarityAnalyzer:
    def __init__(self, window_size=44100, similarity_threshold=0.85):
        """
        Initialize the similarity analyzer with sliding window search.
        
        Args:
            window_size (int): Number of samples in each window (default 1s at 44.1kHz)
            similarity_threshold (float): Threshold for considering segments similar (0-1)
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.reference_buffer = None
        self.history_buffer = deque(maxlen=44100 * 30)  # Store 30 seconds of history
        self.is_recording_reference = False
        self.detected_similarities = []
        
    def compute_spectral_features(self, magnitudes):
        """Compute spectral centroid and rolloff."""
        freqs = np.arange(len(magnitudes))
        # Normalize magnitudes
        magnitudes = np.array(magnitudes)
        magnitudes_sum = np.sum(magnitudes) + 1e-8
        
        # Spectral centroid
        centroid = np.sum(freqs * magnitudes) / magnitudes_sum
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitudes)
        rolloff = np.where(cumsum >= 0.85 * cumsum[-1])[0][0]
        
        return np.array([centroid, rolloff])

    def compute_band_energies(self, magnitudes, num_bands=8):
        """Compute energy in frequency bands."""
        bands = np.array_split(magnitudes, num_bands)
        return np.array([np.sum(band ** 2) for band in bands])

    def compute_fingerprint(self, magnitudes):
        """Compute comprehensive audio fingerprint."""
        spectral = self.compute_spectral_features(magnitudes)
        band_energies = self.compute_band_energies(magnitudes)
        
        # Combine features and normalize
        features = np.concatenate([spectral, band_energies])
        return features / (np.linalg.norm(features) + 1e-8)

    def compute_similarity(self, segment1, segment2):
        """Compute similarity between two segments using cosine similarity."""
        # Convert to fingerprints if not already
        if len(segment1.shape) == 1:
            segment1 = self.compute_fingerprint(segment1)
        if len(segment2.shape) == 1:
            segment2 = self.compute_fingerprint(segment2)
            
        similarity = 1 - cosine(segment1, segment2)
        return similarity

    def find_matches(self, current_window):
        """
        Search through history buffer for matches to current window.
        Returns list of (position, similarity) tuples.
        """
        matches = []
        current_fingerprint = self.compute_fingerprint(current_window)
        
        # Convert history buffer to numpy array for efficient processing
        history = np.array(list(self.history_buffer))
        
        # Search through history with sliding window
        step_size = len(current_window) // 2
        for i in range(0, len(history) - len(current_window), step_size):
            window = history[i:i + len(current_window)]
            if len(window) == len(current_window):
                similarity = self.compute_similarity(current_fingerprint, window)
                if similarity >= self.similarity_threshold:
                    matches.append({
                        'position': i,
                        'similarity': float(similarity),  # Convert to native Python float
                        'timestamp': time.time()
                    })
        
        return matches

    def process_frame(self, magnitudes):
        """
        Process a new frame of frequency magnitudes.
        Implements sliding window search through history.
        """
        # Add new data to history
        for mag in magnitudes:
            self.history_buffer.append(mag)
        
        if self.is_recording_reference:
            if self.reference_buffer is None:
                self.reference_buffer = []
            self.reference_buffer.extend(magnitudes)
            return {
                "status": "recording_reference",
                "frames_recorded": len(self.reference_buffer)
            }

        # Ensure we have enough history
        if len(self.history_buffer) < self.window_size:
            return {
                "status": "collecting_data",
                "frames_collected": len(self.history_buffer)
            }

        # Get current window (last second of data)
        current_window = np.array(list(self.history_buffer)[-self.window_size:])

        # Find matches in history
        matches = self.find_matches(current_window)
        
        # Store detected matches
        if matches:
            self.detected_similarities.extend(matches)
            
            # Keep only recent detections (last 10 seconds)
            current_time = time.time()
            self.detected_similarities = [
                m for m in self.detected_similarities 
                if current_time - m['timestamp'] < 10
            ]

            # Print similarity information for matches
            print(f"\nSimilarity scores for current window:")
            for match in matches:
                print(f"Position: {match['position']:6d} | Similarity: {match['similarity']:.3f}")
        else:
            print("\rNo matches found in current window     ", end="", flush=True)

        # Return analysis results
        best_match = max(matches, key=lambda x: x['similarity']) if matches else None
        return {
            "status": "analyzing",
            "similarity_score": best_match['similarity'] if best_match else 0.0,
            "is_similar": bool(matches),
            "matches": matches,
            "buffer_size": len(self.history_buffer)
        }

    def start_recording_reference(self):
        """Start recording a new reference pattern."""
        self.is_recording_reference = True
        self.reference_buffer = []
        print("\nStarting to record new reference pattern...")
        
    def stop_recording_reference(self):
        """Stop recording reference pattern."""
        self.is_recording_reference = False
        if len(self.reference_buffer) > 0:
            print("\nReference pattern recorded successfully!")
            return True
        print("\nReference pattern too short, please try again.")
        return False