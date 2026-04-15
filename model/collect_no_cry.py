"""
Collect 'no_cry' samples:
1. Record ambient audio from USB mic
2. Generate synthetic non-cry samples (silence, noise, tones)
"""
import os
import sys
import time
import ctypes
import numpy as np
import soundfile as sf
import soxr

# Suppress ALSA
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    c_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    asound.snd_lib_error_set_handler(c_handler(lambda *a: None))
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

OUTPUT_DIR = os.path.join(settings.BASE_DIR, 'Baby Cry Sence Dataset', 'no_cry')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE_SR = 48000
TARGET_SR = settings.SAMPLE_RATE
DURATION = settings.DURATION
NUM_RECORD = 50      # number of ambient recordings
NUM_SYNTHETIC = 150  # number of synthetic samples


def record_ambient():
    """Record ambient audio from USB mic."""
    import sounddevice as sd

    print(f"Recording {NUM_RECORD} ambient samples ({DURATION}s each)...")
    print("Make normal sounds: talk, clap, walk around, play music, stay silent...")
    print()

    block = int(DEVICE_SR * DURATION)

    for i in range(NUM_RECORD):
        audio = sd.rec(block, samplerate=DEVICE_SR, channels=1, device=settings.AUDIO_DEVICE, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Resample to target
        audio = soxr.resample(audio, DEVICE_SR, TARGET_SR)

        # Normalize
        mx = np.max(np.abs(audio))
        if mx > 1e-6:
            audio = audio / mx * 0.8

        path = os.path.join(OUTPUT_DIR, f'ambient_{i:03d}.wav')
        sf.write(path, audio, TARGET_SR)
        print(f"  [{i+1}/{NUM_RECORD}] Recorded {path}")
        time.sleep(0.5)

    print(f"\nRecorded {NUM_RECORD} ambient samples.")


def generate_synthetic():
    """Generate synthetic non-cry audio samples."""
    print(f"\nGenerating {NUM_SYNTHETIC} synthetic samples...")
    samples_per_type = NUM_SYNTHETIC // 5
    count = 0
    n_samples = int(TARGET_SR * DURATION)

    for i in range(samples_per_type):
        # 1. Silence with minimal noise
        audio = np.random.randn(n_samples).astype(np.float32) * 0.001
        sf.write(os.path.join(OUTPUT_DIR, f'silence_{i:03d}.wav'), audio, TARGET_SR)
        count += 1

        # 2. White noise (various levels)
        level = np.random.uniform(0.01, 0.1)
        audio = np.random.randn(n_samples).astype(np.float32) * level
        sf.write(os.path.join(OUTPUT_DIR, f'whitenoise_{i:03d}.wav'), audio, TARGET_SR)
        count += 1

        # 3. Low frequency hum (AC, fan, etc.)
        freq = np.random.uniform(50, 200)
        t = np.linspace(0, DURATION, n_samples)
        audio = (np.sin(2 * np.pi * freq * t) * np.random.uniform(0.05, 0.2)).astype(np.float32)
        audio += np.random.randn(n_samples).astype(np.float32) * 0.01
        sf.write(os.path.join(OUTPUT_DIR, f'hum_{i:03d}.wav'), audio, TARGET_SR)
        count += 1

        # 4. Random tones (music-like, speech-like fundamentals)
        n_tones = np.random.randint(1, 4)
        audio = np.zeros(n_samples, dtype=np.float32)
        for _ in range(n_tones):
            freq = np.random.uniform(100, 1000)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.02, 0.1)
            audio += amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
        audio += np.random.randn(n_samples).astype(np.float32) * 0.005
        sf.write(os.path.join(OUTPUT_DIR, f'tones_{i:03d}.wav'), audio, TARGET_SR)
        count += 1

        # 5. Filtered noise (band-limited, simulates environment)
        from scipy.signal import butter, lfilter
        low = np.random.uniform(100, 500)
        high = np.random.uniform(1000, 4000)
        b, a = butter(4, [low / (TARGET_SR/2), high / (TARGET_SR/2)], btype='band')
        audio = lfilter(b, a, np.random.randn(n_samples)).astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-6) * np.random.uniform(0.05, 0.15)
        sf.write(os.path.join(OUTPUT_DIR, f'filtered_{i:03d}.wav'), audio, TARGET_SR)
        count += 1

    print(f"Generated {count} synthetic samples.")


if __name__ == "__main__":
    print("=" * 40)
    print("  no_cry Dataset Collector")
    print("=" * 40)

    if "--no-record" not in sys.argv:
        record_ambient()

    generate_synthetic()

    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
    print(f"\nTotal no_cry samples: {total}")
    print(f"Saved to: {OUTPUT_DIR}")
