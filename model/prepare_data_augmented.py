import os
import random
# Import sklearn early to avoid TLS allocation error on aarch64 (Jetson)
import sklearn  # noqa: F401
import librosa
import numpy as np
from config import settings

DATASET_PATH = os.path.join(settings.BASE_DIR, 'Baby Cry Sence Dataset')

TARGET_PER_CLASS = 400
# Higher target for classes with very few originals — more diversity needed
BOOST_TARGET = 600
BOOST_CLASSES = {'belly_pain', 'discomfort'}

CLASSES = ['hungry', 'scared', 'discomfort', 'belly_pain', 'no_cry']


# ── Augmentation primitives ──────────────────────────────────────────────────

def add_white_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def add_pink_noise(data, noise_factor=0.005):
    """1/f noise via numpy FFT."""
    n = len(data)
    white = np.fft.rfft(np.random.randn(n))
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1  # avoid div by zero
    pink = white / np.sqrt(freqs)
    pink_signal = np.fft.irfft(pink, n=n)
    pink_signal = pink_signal / (np.abs(pink_signal).max() + 1e-8)
    return data + noise_factor * pink_signal


def time_shift(data, shift_fraction=None):
    if shift_fraction is None:
        shift_fraction = random.uniform(-0.3, 0.3)
    shift = int(len(data) * shift_fraction)
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


def volume_perturb(data, gain=None):
    if gain is None:
        gain = random.uniform(0.5, 1.5)
    return data * gain


def random_eq(data, sr):
    """Boost or cut a random frequency band."""
    from librosa.filters import mel as mel_filter
    n_fft = 2048
    stft = librosa.stft(data, n_fft=n_fft)
    n_bins = stft.shape[0]
    # Pick a random band to boost/cut
    band_start = random.randint(0, n_bins - 20)
    band_width = random.randint(5, 20)
    gain = random.uniform(0.3, 2.0)
    stft[band_start:band_start + band_width, :] *= gain
    return librosa.istft(stft, length=len(data))


def spec_augment(mel_spec, num_time_masks=2, num_freq_masks=2, max_mask=10):
    """Mask random time/frequency bands on mel spectrogram."""
    aug = mel_spec.copy()
    n_freq, n_time = aug.shape

    for _ in range(num_freq_masks):
        f = random.randint(1, min(max_mask, n_freq - 1))
        f0 = random.randint(0, n_freq - f)
        aug[f0:f0 + f, :] = aug.min()

    for _ in range(num_time_masks):
        t = random.randint(1, min(max_mask, n_time - 1))
        t0 = random.randint(0, n_time - t)
        aug[:, t0:t0 + t] = aug.min()

    return aug


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_mel_spec(y, sr):
    target_len = int(settings.SAMPLE_RATE * settings.DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    y = y[:target_len]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


# ── Augmentation chains for minority classes ─────────────────────────────────

def random_augment_chain(y, sr, num_transforms=3):
    """Apply a random chain of augmentations."""
    transforms = [
        lambda d: add_white_noise(d, noise_factor=random.uniform(0.002, 0.015)),
        lambda d: add_pink_noise(d, noise_factor=random.uniform(0.002, 0.015)),
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


def mixup(y1, y2, alpha=None):
    """Blend two waveforms from the same class."""
    if alpha is None:
        alpha = random.uniform(0.3, 0.7)
    min_len = min(len(y1), len(y2))
    return alpha * y1[:min_len] + (1 - alpha) * y2[:min_len]


# ── Dataset preparation ─────────────────────────────────────────────────────

def prepare_dataset():
    features = []
    labels = []

    for label in CLASSES:
        class_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_path):
            print(f"WARNING: class folder not found: {class_path}")
            continue

        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        raw_count = len(wav_files)
        target = BOOST_TARGET if label in BOOST_CLASSES else TARGET_PER_CLASS
        print(f"\nProcessing class: {label} ({raw_count} raw files, target: {target})")

        # For no_cry: undersample to TARGET_PER_CLASS
        if label == 'no_cry' and raw_count > TARGET_PER_CLASS:
            wav_files = random.sample(wav_files, TARGET_PER_CLASS)
            print(f"  Undersampled to {len(wav_files)} files")

        # Load all raw audio for this class (needed for mixup)
        raw_audio = []
        for file in wav_files:
            file_path = os.path.join(class_path, file)
            try:
                y, sr = librosa.load(file_path, sr=settings.SAMPLE_RATE, duration=settings.DURATION)
                raw_audio.append(y)
            except Exception as e:
                print(f"  Error loading {file}: {e}")

        if not raw_audio:
            print(f"  No audio loaded for {label}, skipping")
            continue

        # Calculate augmentation multiplier
        effective_count = len(raw_audio)
        if label == 'no_cry' or effective_count >= target:
            aug_multiplier = 0
        else:
            aug_multiplier = max(0, (target // effective_count) - 1)

        # How many mixup samples to add (for boosted classes)
        mixup_count = 0
        if label in BOOST_CLASSES:
            remaining = target - effective_count * (1 + aug_multiplier)
            mixup_count = max(0, remaining)

        print(f"  Loaded: {effective_count}, Aug multiplier: {aug_multiplier}x, Mixup: {mixup_count}")

        for y in raw_audio:
            # 1. Original
            mel = extract_mel_spec(y, sr)
            features.append(mel)
            labels.append(label)

            # 2. Augmented copies (always 3 transforms for diversity)
            for i in range(aug_multiplier):
                y_aug = random_augment_chain(y, settings.SAMPLE_RATE, num_transforms=3)
                mel_aug = extract_mel_spec(y_aug, settings.SAMPLE_RATE)
                # Apply SpecAugment (60% chance)
                if random.random() < 0.6:
                    mel_aug = spec_augment(mel_aug)
                features.append(mel_aug)
                labels.append(label)

        # 3. Mixup samples — blend pairs of originals for genuinely new waveforms
        for _ in range(mixup_count):
            y1, y2 = random.sample(raw_audio, 2)
            y_mix = mixup(y1, y2)
            # Also augment the mixed sample
            y_mix = random_augment_chain(y_mix, settings.SAMPLE_RATE, num_transforms=2)
            mel_mix = extract_mel_spec(y_mix, settings.SAMPLE_RATE)
            if random.random() < 0.5:
                mel_mix = spec_augment(mel_mix)
            features.append(mel_mix)
            labels.append(label)

        class_count = labels.count(label)
        print(f"  Total samples for {label}: {class_count}")

    X = np.array(features)
    y = np.array(labels)

    # Print final distribution
    print("\n" + "=" * 40)
    print("Final class distribution:")
    for cls in CLASSES:
        count = np.sum(y == cls)
        print(f"  {cls}: {count}")
    print(f"Total: {len(y)}")
    print("=" * 40)

    return X, y


if __name__ == "__main__":
    X, y = prepare_dataset()
    print(f"\nAugmented Dataset prepared. Features shape: {X.shape}, Labels shape: {y.shape}")
    np.save(os.path.join(settings.BASE_DIR, 'model', 'X_features_augmented.npy'), X)
    np.save(os.path.join(settings.BASE_DIR, 'model', 'y_labels_augmented.npy'), y)
    print("Augmented features and labels saved to model/ directory.")
