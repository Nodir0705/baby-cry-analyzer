import sounddevice as sd
import numpy as np
import soxr
import threading
import queue
from config import settings
from audio.preprocessor import extract_features

# USB mic only supports 48000 Hz
DEVICE_SAMPLE_RATE = 44100

audio_queue = queue.Queue(maxsize=10)


def listen_and_predict(callback):
    device = getattr(settings, "AUDIO_DEVICE", None)
    block_samples = int(DEVICE_SAMPLE_RATE * settings.DURATION)

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, flush=True)

        audio = indata[:, 0].copy()
        energy = np.sqrt(np.mean(audio ** 2))
        if energy < 0.01:
            return

        try:
            audio_queue.put_nowait(audio)
        except queue.Full:
            pass  # drop frame if processing is behind

    def process_thread():
        while True:
            audio = audio_queue.get()
            try:
                # Fast resample with soxr
                audio_resampled = soxr.resample(
                    audio.astype(np.float32),
                    DEVICE_SAMPLE_RATE,
                    settings.SAMPLE_RATE,
                )
                features = extract_features(audio_resampled)
                if features is not None:
                    callback(features)
            except Exception as e:
                print(f"Processing error: {e}", flush=True)

    # Start processing in background thread
    t = threading.Thread(target=process_thread, daemon=True)
    t.start()

    with sd.InputStream(
        callback=audio_callback,
        device=device,
        channels=settings.CHANNELS,
        samplerate=DEVICE_SAMPLE_RATE,
        blocksize=block_samples,
    ):
        print(f"Listening on device {device} at {DEVICE_SAMPLE_RATE}Hz "
              f"(resampling to {settings.SAMPLE_RATE}Hz)", flush=True)
        print("Press Ctrl+C to stop", flush=True)
        while True:
            sd.sleep(500)
