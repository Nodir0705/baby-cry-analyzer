import os
import librosa
import numpy as np
# import pandas as pd
from config import settings

DATASET_PATH = os.path.join(settings.BASE_DIR, 'Baby Cry Sence Dataset')

def prepare_dataset():
    features = []
    labels = []
    
    classes = ['hungry', 'scared', 'discomfort', 'belly_pain'] # Matching proposal categories
    
    for label in classes:
        class_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_path):
            continue
            
        print(f"Processing class: {label}")
        for file in os.listdir(class_path):
            if file.endswith('.wav'):
                file_path = os.path.join(class_path, file)
                try:
                    # Load audio
                    y, sr = librosa.load(file_path, sr=settings.SAMPLE_RATE, duration=settings.DURATION)
                    # Pad if shorter than duration
                    if len(y) < settings.SAMPLE_RATE * settings.DURATION:
                        y = np.pad(y, (0, settings.SAMPLE_RATE * settings.DURATION - len(y)))
                    
                    # Extract Mel Spectrogram
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    features.append(mel_spec_db)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    X, y = prepare_dataset()
    print(f"Dataset prepared. Features shape: {X.shape}, Labels shape: {y.shape}")
    # Save processed data for training
    np.save(os.path.join(settings.BASE_DIR, 'model', 'X_features.npy'), X)
    np.save(os.path.join(settings.BASE_DIR, 'model', 'y_labels.npy'), y)
    print("Features and labels saved to model/ directory.")
