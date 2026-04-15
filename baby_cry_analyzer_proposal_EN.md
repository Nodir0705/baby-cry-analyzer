# Baby Cry Analyzer - Smart Lullaby System

## Project Title

**"AI-Powered Baby Cry Analysis and Automatic Soothing System"**

---

## 1. Project Description

This project aims to develop an AI-based smart system that detects baby crying sounds in real-time and classifies their causes. The system continuously listens to audio signals through a microphone and identifies the type of baby crying using deep learning models.

### Detectable States

| Cry Type | Description |
|----------|-------------|
| **Hunger** | Baby is hungry and needs feeding |
| **Fear** | Baby is scared or startled |
| **Pain** | Baby is experiencing discomfort or pain |
| **Discomfort** | Diaper change needed, too hot/cold |

### System Behavior

When the system detects baby crying:

1. **Sends notification** to parent's phone via Telegram Bot
2. **Indicates probable cause** (e.g., "Baby is crying due to hunger")
3. **Automatically plays** soothing music or lullaby
4. **Visual indicator** via LED status lights

### Main Goals

- Help parents understand baby's needs faster
- Reduce parental stress
- Automate baby care assistance
- Work both **online and offline**

---

## 2. System Requirements

| Requirement | Decision |
|-------------|----------|
| **Hardware** | Raspberry Pi 4 + USB Mic + Speaker |
| **Notification** | Telegram Bot |
| **Latency** | Real-time (<500ms) |
| **Connectivity** | Hybrid — works offline, syncs when online |

---

## 3. System Architecture

### Hardware Setup

```
┌─────────────────────────────────────────────────────┐
│                 RASPBERRY PI DEVICE                 │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │   USB    │───▶│  Audio   │───▶│   AI Model   │  │
│  │   Mic    │    │  Buffer  │    │  (Inference) │  │
│  └──────────┘    └──────────┘    └──────────────┘  │
│                                         │          │
│                                         ▼          │
│                                  ┌──────────────┐  │
│                                  │   Cry Type   │  │
│                                  │   Detected?  │  │
│                                  └──────────────┘  │
│                                    │          │    │
│                              YES ──┘          └── NO (loop)
│                                │                   │
│         ┌──────────────────────┼────────────┐     │
│         ▼                      ▼            ▼     │
│  ┌────────────┐    ┌────────────┐    ┌─────────┐ │
│  │  Telegram  │    │   Play     │    │   LED   │ │
│  │   Alert    │    │  Lullaby   │    │ Status  │ │
│  └────────────┘    └────────────┘    └─────────┘ │
│                                                   │
└─────────────────────────────────────────────────────┘
```

### Offline + Online Behavior

```
┌─────────────────────────────────────────────────────────────┐
│                      CRY DETECTED                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      Internet Available?       │
              └───────────────────────────────┘
                     │                 │
                    YES                NO
                     │                 │
                     ▼                 ▼
        ┌─────────────────┐   ┌─────────────────────┐
        │ Send Telegram   │   │ Save to local queue │
        │ Notification    │   │ (SQLite)            │
        └─────────────────┘   └─────────────────────┘
                     │                 │
                     └────────┬────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Play Lullaby +    │
                   │   LED Indicator     │
                   └─────────────────────┘
                              │
                              ▼
              (Background thread checks connectivity)
                              │
                    When online again:
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Sync queued events  │
                   │ to Telegram         │
                   └─────────────────────┘
```

---

## 4. Hardware Requirements

| Component | Example | Approx. Cost |
|-----------|---------|--------------|
| Raspberry Pi 4 (4GB) | RPi 4 Model B | $55-60 |
| USB Microphone | ReSpeaker / any USB mic | $10-30 |
| Speaker | 3.5mm or USB speaker | $10-20 |
| MicroSD Card (32GB+) | SanDisk | $10 |
| Power Supply | Official RPi 5V/3A | $10 |
| (Optional) Case + LED | For demo aesthetics | $10 |

**Total Estimated Cost: ~$100-130**

---

## 5. Software Architecture

```
project/
│
├── main.py                  # Entry point, orchestrates everything
│
├── audio/
│   ├── capture.py           # Mic input, continuous streaming
│   └── preprocessor.py      # MFCC extraction, normalization
│
├── model/
│   ├── classifier.py        # Load & run inference (TFLite/ONNX)
│   └── cry_model.tflite     # Trained model file
│
├── actions/
│   ├── lullaby_player.py    # Play audio files
│   ├── led_controller.py    # GPIO LED status
│   └── notifier.py          # Telegram + offline queue logic
│
├── storage/
│   └── event_store.py       # SQLite for offline events
│
├── config/
│   └── settings.py          # Thresholds, Telegram token, etc.
│
├── lullabies/               # Audio files (.mp3/.wav)
│   ├── lullaby_01.mp3
│   └── lullaby_02.mp3
│
└── requirements.txt
```

---

## 6. Datasets

| Dataset | Description | Use Case |
|---------|-------------|----------|
| **Donate-a-Cry** | Various baby cry types (~450 samples) | Primary training data |
| **Baby Cry Dataset (Kaggle)** | Labeled with hunger, pain, discomfort | Classification training |
| **ESC-50** | Environmental sounds | Negative samples (reduce false positives) |
| **Custom recordings** | Self-recorded samples | Local adaptation |

### Data Augmentation Techniques

- Noise injection
- Pitch shifting
- Time stretching
- Room impulse response simulation

---

## 7. Model Architecture

### Constraints for Real-Time on Raspberry Pi

| Factor | Recommendation |
|--------|----------------|
| **Audio window** | 1-2 seconds of audio per inference |
| **Model size** | <20MB (ideally <10MB) |
| **Model format** | TensorFlow Lite or ONNX Runtime |
| **Features** | Mel-spectrogram or MFCC |

### Recommended Architecture

```
Audio (1.5s) → Mel-Spectrogram → MobileNetV2 (small) → 4 classes
                   │
            (64 mel bands × 94 time frames)
```

**Estimated inference time on RPi 4: 100-200ms** ✅

### Model Pipeline

1. **Feature Extraction**: Mel-spectrogram (64 bands)
2. **Model**: MobileNetV2-small or custom CNN
3. **Output**: 4 classes (hunger, fear, pain, discomfort)
4. **Confidence threshold**: >0.7 to trigger action

---

## 8. Technology Stack

| Category | Technologies |
|----------|--------------|
| **AI/ML** | Python, PyTorch/TensorFlow, Librosa |
| **Inference** | TensorFlow Lite, ONNX Runtime |
| **Hardware** | Raspberry Pi 4, USB Microphone, Speaker |
| **Notification** | Telegram Bot API |
| **Storage** | SQLite (offline events) |
| **Audio** | PyAudio, sounddevice |

---

## 9. Development Timeline (11 weeks)

| Phase | Duration | Tasks |
|-------|----------|-------|
| **1. Dataset & Preprocessing** | 2 weeks | Download data, augmentation pipeline, feature extraction |
| **2. Model Training** | 3 weeks | Train on PC/Colab, optimize, convert to TFLite |
| **3. RPi Core System** | 2 weeks | Audio capture, inference loop, lullaby playback |
| **4. Telegram + Offline Logic** | 1 week | Bot setup, SQLite queue, sync logic |
| **5. Integration & Testing** | 2 weeks | End-to-end testing, latency optimization |
| **6. Documentation & Demo** | 1 week | Thesis writing, demo preparation |

---

## 10. Evaluation Metrics

| Metric | Target |
|--------|--------|
| **Accuracy** | >85% |
| **F1-Score** | >0.80 |
| **Inference Latency** | <500ms |
| **False Alarm Rate** | <5% per day |
| **Offline Functionality** | 100% core features |

---

## 11. Final Deliverables

- ✅ Trained AI model for baby cry classification
- ✅ Raspberry Pi-based standalone device
- ✅ Telegram Bot for notifications
- ✅ Automatic lullaby playback system
- ✅ Offline event storage and sync
- ✅ Technical documentation / Thesis
- ✅ Live demonstration

---

## 12. Future Enhancements

- 📊 Analytics dashboard (web-based)
- 👶 Per-baby personalization
- 🏠 Smart home integration (Home Assistant)
- ⌚ Smartwatch notifications
- 🔊 Voice announcements

---

## 13. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Small dataset | Heavy augmentation + transfer learning |
| False positives (TV, other babies) | Confidence thresholding + negative samples |
| RPi performance | Optimized TFLite model + quantization |
| Network failures | Offline-first design with queue sync |

---

*Document Version: 1.0*  
*Last Updated: January 2025*
