import os
import random
import subprocess
import threading
from config import settings


def find_usb_speaker():
    """Auto-detect USB speaker card number from aplay -l."""
    try:
        result = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "USB Audio" in line:
                card = line.split(":")[0].replace("card ", "").strip()
                return card
    except Exception as e:
        print(f"Speaker detection error: {e}", flush=True)
    return None


def play_random_lullaby():
    card = find_usb_speaker()
    if not card:
        print("USB speaker not found!", flush=True)
        return

    files = [f for f in os.listdir(settings.LULLABIES_DIR) if f.endswith(('.mp3', '.wav'))]
    if not files:
        print("No lullabies found!", flush=True)
        return

    file_to_play = os.path.join(settings.LULLABIES_DIR, random.choice(files))
    print(f"Playing lullaby: {file_to_play} on card {card}", flush=True)

    def _play():
        try:
            subprocess.run(
                ["aplay", "-D", f"plughw:{card},0", file_to_play],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except Exception as e:
            print(f"Lullaby playback error: {e}", flush=True)

    t = threading.Thread(target=_play, daemon=True)
    t.start()
