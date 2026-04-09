import numpy as np
from config import settings


def extract_features(audio_data):
    """Normalize raw audio and return it for YAMNet-based classifier."""
    audio_data = audio_data.astype(np.float32)
    max_val = np.max(np.abs(audio_data))
    if max_val < 1e-6:
        return None
    audio_data = audio_data / max_val
    return audio_data
