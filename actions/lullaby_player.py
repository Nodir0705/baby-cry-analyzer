import os
import random
import subprocess
import threading
from config import settings

SPEAKER_DEVICE = "plughw:3,0"

def play_random_lullaby():
    files = [f for f in os.listdir(settings.LULLABIES_DIR) if f.endswith(('.mp3', '.wav'))]
    if not files:
        print("No lullabies found!", flush=True)
        return
    
    file_to_play = os.path.join(settings.LULLABIES_DIR, random.choice(files))
    print(f"Playing lullaby: {file_to_play}", flush=True)
    
    def _play():
        try:
            subprocess.run(
                ["aplay", "-D", SPEAKER_DEVICE, file_to_play],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except Exception as e:
            print(f"Lullaby playback error: {e}", flush=True)
    
    t = threading.Thread(target=_play, daemon=True)
    t.start()
