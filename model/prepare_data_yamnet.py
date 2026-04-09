import os
import random
import sklearn  # noqa: F401 — must import before TF on aarch64
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from config import settings

DATASET_PATH = os.path.join(settings.BASE_DIR, 'Baby Cry Sence Dataset')
DONATE_PATH = os.path.join(settings.BASE_DIR, 'donateacry_corpus')

TARGET_PER_CLASS = 500
BOOST_TARGET = 700
BOOST_CLASSES = {'belly_pain', 'discomfort', 'scared'}

CLASSES = ['hungry', 'scared', 'discomfort', 'belly_pain', 'no_cry']

# Mapping from donateacry_corpus folder names to our class labels
DONATE_CLASS_MAP = {
    'discomfort': 'discomfort',
    'belly_pain': 'belly_pain',
    'tired': 'discomfort',  # acoustically similar low-energy distress
}

print("Loading YAMNet...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet loaded!")


# ── Augmentation primitives ──────────────────────────────────────────────────

def add_noise(data, noise_factor=0.005):
    return data + noise_factor * np.random.randn(len(data))


def time_shift(data):
    shift = int(len(data) * random.uniform(-0.3, 0.3))
    return np.roll(data, shift)


def pitch_shift(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)


def time_stretch(data, rate=1.1):
    stretched = librosa.effects.time_stretch(y=data, rate=rate)
    target_len = len(data)
    if len(stretched) > target_len:
        stretched = stretched[:target_len]
    else:
        stretched = np.pad(stretched, (0, target_len - len(stretched)))
    return stretched


def volume_perturb(data):
    return data * random.uniform(0.5, 1.5)


def random_eq(data, sr):
    n_fft = 2048
    stft = librosa.stft(data, n_fft=n_fft)
    n_bins = stft.shape[0]
    band_start = random.randint(0, n_bins - 20)
    band_width = random.randint(5, 20)
    gain = random.uniform(0.3, 2.0)
    stft[band_start:band_start + band_width, :] *= gain
    return librosa.istft(stft, length=len(data))


def random_augment_chain(y, sr, num_transforms=3):
    transforms = [
        lambda d: add_noise(d, noise_factor=random.uniform(0.002, 0.015)),
        lambda d: time_shift(d),
        lambda d: pitch_shift(d, sr, n_steps=random.choice([-4, -3, -2, -1, 1, 2, 3, 4])),
        lambda d: time_stretch(d, rate=random.choice([0.75, 0.85, 0.9, 1.1, 1.15, 1.25])),
        lambda d: volume_perturb(d),
        lambda d: random_eq(d, sr),
    ]
    chosen = random.sample(transforms, k=min(num_transforms, len(transforms)))
    for fn in chosen:
        y = fn(y)
    return y


def mixup(y1, y2):
    alpha = random.uniform(0.3, 0.7)
    min_len = min(len(y1), len(y2))
    return alpha * y1[:min_len] + (1 - alpha) * y2[:min_len]


# ── YAMNet multi-pool embedding extraction ───────────────────────────────────

def extract_yamnet_embedding(y, sr):
    """Extract 3072-dim multi-pool YAMNet embedding (avg+max+std) from audio."""
    # YAMNet expects 16kHz mono float32
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    # Pad to at least 0.96s (YAMNet minimum window)
    min_len = int(16000 * 0.96)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    waveform = tf.constant(y, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(waveform)
    # Multi-pool: avg + max + std → 3072-dim vector
    avg_emb = tf.reduce_mean(embeddings, axis=0)
    max_emb = tf.reduce_max(embeddings, axis=0)
    std_emb = tf.math.reduce_std(embeddings, axis=0)
    combined = tf.concat([avg_emb, max_emb, std_emb], axis=0).numpy()
    return combined


# ── Load raw audio files from a directory ────────────────────────────────────

def load_class_audio(class_path):
    """Load all .wav files from a directory, return list of (filename, audio) tuples."""
    audio_items = []
    wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    for file in wav_files:
        file_path = os.path.join(class_path, file)
        try:
            y, sr = librosa.load(file_path, sr=settings.SAMPLE_RATE, duration=settings.DURATION)
            audio_items.append((file, y))
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    return audio_items


# ── Dataset preparation with split-before-augment ────────────────────────────

def prepare_dataset():
    # Step 1: Collect all raw originals per class (from both datasets)
    raw_per_class = {cls: [] for cls in CLASSES}

    # Load from main dataset
    for label in CLASSES:
        class_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_path):
            print(f"WARNING: class folder not found: {class_path}")
            continue
        items = load_class_audio(class_path)
        raw_per_class[label].extend(items)
        print(f"[Main] {label}: {len(items)} files")

    # Load from donateacry_corpus (merge additional data)
    if os.path.exists(DONATE_PATH):
        for donate_folder, target_label in DONATE_CLASS_MAP.items():
            donate_class_path = os.path.join(DONATE_PATH, donate_folder)
            if not os.path.exists(donate_class_path):
                print(f"[Donate] folder not found: {donate_class_path}, skipping")
                continue
            items = load_class_audio(donate_class_path)
            # Prefix filenames to avoid collisions
            items = [(f"donate_{donate_folder}_{fn}", audio) for fn, audio in items]
            raw_per_class[target_label].extend(items)
            print(f"[Donate] {donate_folder} → {target_label}: {len(items)} files")
    else:
        print(f"WARNING: donateacry_corpus not found at {DONATE_PATH}, skipping merge")

    print("\n--- Raw originals per class ---")
    for cls in CLASSES:
        print(f"  {cls}: {len(raw_per_class[cls])}")

    # Step 2: Split originals into train/test FIRST (80/20 at original-file level)
    train_audio = {cls: [] for cls in CLASSES}
    test_audio = {cls: [] for cls in CLASSES}

    for label in CLASSES:
        items = raw_per_class[label]
        random.shuffle(items)

        # For no_cry: undersample before splitting
        if label == 'no_cry' and len(items) > TARGET_PER_CLASS:
            items = random.sample(items, TARGET_PER_CLASS)

        n_test = max(1, int(len(items) * 0.2))
        test_items = items[:n_test]
        train_items = items[n_test:]

        train_audio[label] = [audio for _, audio in train_items]
        test_audio[label] = [audio for _, audio in test_items]

        print(f"  {label}: {len(train_items)} train / {len(test_items)} test originals")

    # Step 3: Extract test embeddings (NO augmentation — originals only)
    print("\n--- Extracting test embeddings (originals only) ---")
    test_features = []
    test_labels = []
    for label in CLASSES:
        for y in test_audio[label]:
            emb = extract_yamnet_embedding(y, settings.SAMPLE_RATE)
            test_features.append(emb)
            test_labels.append(label)
        print(f"  {label}: {len(test_audio[label])} test samples")

    # Step 4: Augment training split and extract embeddings
    print("\n--- Augmenting training split and extracting embeddings ---")
    train_features = []
    train_labels = []

    for label in CLASSES:
        raw_audio = train_audio[label]
        if not raw_audio:
            print(f"  No training audio for {label}, skipping")
            continue

        effective_count = len(raw_audio)
        target = BOOST_TARGET if label in BOOST_CLASSES else TARGET_PER_CLASS

        if label == 'no_cry' or effective_count >= target:
            aug_multiplier = 0
        else:
            # Ceiling division to ensure we reach target
            aug_multiplier = max(0, -(-target // effective_count) - 1)

        # Fill remaining gap with mixup for ANY class below target (not just boost)
        after_aug = effective_count * (1 + aug_multiplier)
        mixup_count = max(0, target - after_aug)

        print(f"  {label}: {effective_count} originals, aug={aug_multiplier}x, mixup={mixup_count}")

        # 1. Original training samples
        for y in raw_audio:
            emb = extract_yamnet_embedding(y, settings.SAMPLE_RATE)
            train_features.append(emb)
            train_labels.append(label)

            # 2. Augmented copies
            for _ in range(aug_multiplier):
                y_aug = random_augment_chain(y, settings.SAMPLE_RATE, num_transforms=3)
                emb_aug = extract_yamnet_embedding(y_aug, settings.SAMPLE_RATE)
                train_features.append(emb_aug)
                train_labels.append(label)

        # 3. Mixup samples
        for _ in range(mixup_count):
            y1, y2 = random.sample(raw_audio, 2)
            y_mix = mixup(y1, y2)
            y_mix = random_augment_chain(y_mix, settings.SAMPLE_RATE, num_transforms=2)
            emb_mix = extract_yamnet_embedding(y_mix, settings.SAMPLE_RATE)
            train_features.append(emb_mix)
            train_labels.append(label)

        class_count = train_labels.count(label)
        print(f"  Total training samples for {label}: {class_count}")

    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    X_test = np.array(test_features)
    y_test = np.array(test_labels)

    print("\n" + "=" * 50)
    print("Final dataset (split-before-augment, no leakage):")
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Embedding dim: {X_train.shape[1]}")
    print("\nTrain class distribution:")
    for cls in CLASSES:
        print(f"  {cls}: {np.sum(y_train == cls)}")
    print("\nTest class distribution (originals only):")
    for cls in CLASSES:
        print(f"  {cls}: {np.sum(y_test == cls)}")
    print("=" * 50)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_dataset()
    model_dir = os.path.join(settings.BASE_DIR, 'model')
    np.save(os.path.join(model_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(model_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(model_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(model_dir, 'y_test.npy'), y_test)
    print(f"\nSaved X_train.npy, y_train.npy, X_test.npy, y_test.npy to {model_dir}")
