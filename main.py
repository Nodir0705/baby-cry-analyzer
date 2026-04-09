import time
import ctypes

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


def on_cry_detected(features):
    global last_alert_time

    cry_type, confidence, all_probs = predict(features)
    # live visibility: log every prediction
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
    send_alert(cry_type, confidence=confidence, all_probs=all_probs)
    play_random_lullaby()


if __name__ == "__main__":
    init_db()
    print("=" * 40, flush=True)
    print("  Baby Cry Analyzer - Real-Time Mode", flush=True)
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}", flush=True)
    print(f"  Cooldown: {COOLDOWN_SECONDS}s", flush=True)
    print("=" * 40, flush=True)
    try:
        listen_and_predict(on_cry_detected)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
