import time
import os
import ctypes
import numpy as np
from collections import deque

try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    c_error_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    error_handler = c_error_handler(py_error_handler)
    asound.snd_lib_error_set_handler(error_handler)
except Exception:
    pass

from storage.event_store import init_db
from actions.notifier import send_alert
from actions.lullaby_player import play_random_lullaby
from model.classifier import predict
from config import settings

print("Model loaded, starting audio...", flush=True)

from audio.capture import listen_and_predict

COOLDOWN_SECONDS = 5
CONFIDENCE_THRESHOLD = settings.THRESHOLD

last_alert_time = 0

# Rolling buffer: always keeps last 10 seconds (5 chunks x 2s)
audio_buffer = deque(maxlen=5)

# Directory for temporary cry audio clips
CLIP_DIR = os.path.join(settings.BASE_DIR, "storage", "clips")
os.makedirs(CLIP_DIR, exist_ok=True)


def save_audio_clip(audio_data):
    """Save raw audio as a .wav file and return the path."""
    import soundfile as sf
    from datetime import datetime
    filename = datetime.now().strftime("cry_%Y%m%d_%H%M%S.wav")
    filepath = os.path.join(CLIP_DIR, filename)
    sf.write(filepath, audio_data, settings.SAMPLE_RATE)

    # Keep only last 50 clips to save disk space
    clips = sorted(
        [os.path.join(CLIP_DIR, f) for f in os.listdir(CLIP_DIR) if f.endswith(".wav")],
        key=os.path.getmtime,
    )
    for old in clips[:-50]:
        try:
            os.remove(old)
        except Exception:
            pass

    return filepath


def on_cry_detected(features, raw_audio=None):
    global last_alert_time

    # Always store audio, even before cry is confirmed
    if raw_audio is not None:
        audio_buffer.append(raw_audio)

    cry_type, confidence, all_probs = predict(features)
    import numpy as _np
    _rms = float(_np.sqrt(_np.mean(_np.asarray(features, dtype=_np.float32) ** 2)))
    print(f"[pred] {cry_type:<10} conf={confidence:.3f} rms={_rms:.4f}", flush=True)

    if cry_type in ("unknown", "no_cry"):
        return
    if confidence < CONFIDENCE_THRESHOLD:
        return

    now = time.time()
    if now - last_alert_time < COOLDOWN_SECONDS:
        return

    last_alert_time = now
    print(f"Cry detected: {cry_type} ({confidence:.2f}) | {all_probs}", flush=True)

    # Combine full buffer into one 10s clip
    audio_path = None
    if len(audio_buffer) > 0:
        try:
            combined = np.concatenate(list(audio_buffer))
            audio_path = save_audio_clip(combined)
            duration = len(combined) / settings.SAMPLE_RATE
            print(f"Audio clip saved: {audio_path} ({duration:.1f}s)", flush=True)
        except Exception as e:
            print(f"Failed to save audio clip: {e}", flush=True)

    send_alert(cry_type, confidence=confidence, all_probs=all_probs, audio_path=audio_path)
    play_random_lullaby()


if __name__ == "__main__":
    init_db()
    print("=" * 40, flush=True)
    print("  Baby Cry Analyzer - Real-Time Mode", flush=True)
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}", flush=True)
    print(f"  Cooldown: {COOLDOWN_SECONDS}s", flush=True)
    print(f"  Audio buffer: {audio_buffer.maxlen * settings.DURATION}s", flush=True)
    print("=" * 40, flush=True)
    try:
        listen_and_predict(on_cry_detected)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
