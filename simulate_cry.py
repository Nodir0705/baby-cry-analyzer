import os
import sys
import numpy as np
import librosa
from config import settings
from model.classifier import predict
from actions.notifier import send_alert
from actions.lullaby_player import play_random_lullaby

def run_simulation():
    print("--- Baby Cry Simulation Started ---")
    
    # 1. Pick a random audio sample from the 'hungry' dataset
    dataset_path = os.path.join(settings.BASE_DIR, 'Baby Cry Sence Dataset', 'hungry')
    files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    
    if not files:
        print("Error: No test files found in dataset.")
        return
        
    test_file = os.path.join(dataset_path, files[0])
    print(f"Loading test file: {os.path.basename(test_file)}")
    
    # 2. Extract features
    y, sr = librosa.load(test_file, sr=settings.SAMPLE_RATE, duration=settings.DURATION)
    if len(y) < settings.SAMPLE_RATE * settings.DURATION:
        y = np.pad(y, (0, settings.SAMPLE_RATE * settings.DURATION - len(y)))
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Predict
    print("Running AI inference...")
    cry_type = predict(mel_spec_db)
    print(f"AI Prediction: {cry_type}")
    
    # 4. Trigger Actions
    if cry_type != "unknown":
        print("Triggering Telegram alert...")
        success = send_alert(cry_type)
        if success:
            print("Telegram alert sent successfully!")
        else:
            print("Telegram alert failed (check token/chat_id).")
            
        print("Triggering Lullaby playback (simulation)...")
        # In this environment, sounddevice might fail, but we'll try to trigger the logic
        try:
            play_random_lullaby()
            print("Lullaby playback triggered.")
        except Exception as e:
            print(f"Lullaby trigger failed (expected in server env): {e}")

if __name__ == "__main__":
    from storage.event_store import init_db
    init_db()
    run_simulation()
