# Baby Cry Analyzer - Project Status

## 🚀 Overview
The **AI-Powered Baby Cry Analysis and Automatic Soothing System** is a sophisticated edge AI project designed to run on NVIDIA Jetson (Xavier) and Raspberry Pi. It detects baby crying in real-time, classifies the reason (Hungry, Scared, Discomfort, Belly Pain), sends a Telegram notification, and automatically plays a soothing lullaby.

---

## 🏗 Overall Structure

### **1. Directory Map**
The project is organized into modular components:

*   **`audio/`**: Core audio handling.
    *   `capture.py`: Manages the microphone stream and real-time inference loop.
    *   `preprocessor.py`: Extracts Mel-spectrogram features from raw audio signals.
*   **`model/`**: AI Brain of the system.
    *   `cry_model.tflite`: Optimized high-performance CNN model for edge inference.
    *   `train_cnn.py`: GPU-accelerated training script for the Convolutional Neural Network.
    *   `prepare_data_augmented.py`: Advanced data augmentation pipeline.
    *   `classifier.py`: Hybrid logic that selects the best available model (TFLite vs. Baseline).
*   **`actions/`**: Response system.
    *   `notifier.py`: Telegram Bot integration for instant mobile alerts.
    *   `lullaby_player.py`: Logic for playing random MP3/WAV files from the lullabies folder.
*   **`storage/`**: Data persistence.
    *   `event_store.py`: Manages the SQLite database for logging every cry event.
*   **`config/`**: Configuration.
    *   `settings.py`: Centralized management of API tokens, thresholds, and paths.

---

## 🏆 Achievements to Date

### **1. Advanced AI Training**
*   **Model:** Implemented a Deep Convolutional Neural Network (CNN) that processes audio spectrograms as images.
*   **GPU Acceleration:** Successfully linked the environment to the Jetson's GPU (TensorFlow 2.12.0) for rapid model training.
*   **Data Augmentation:** Created a pipeline to inject **White Noise** and perform **Pitch Shifting** (High/Low).
*   **Dataset Expansion:** Increased the training data from 448 original samples to **1,792 augmented samples**.

### **2. Performance Metrics**
*   **Training Accuracy:** **96.86%**
*   **Validation Accuracy:** **88.58%** (Excellent performance on unseen data)
*   **Category Strength:** The model is exceptionally reliable for **"Hungry" (91% Precision)** and **"Scared" (88% Precision)** categories.

### **3. System Integration**
*   **Telegram Verified:** A custom bot (@xv_jarvis_bot) is active and successfully sends alerts to the user's mobile device.
*   **Edge Optimization:** The model is converted to **TFLite format (3.6MB)**, making it ultra-fast and lightweight for the Raspberry Pi.
*   **Hybrid Inference:** The system can intelligently switch between the advanced CNN and a baseline RandomForest model if files are missing.

---

## 🛠 How to Use

### **Training & Stats**
*   To evaluate the latest model performance:
    `python model/evaluate_stats.py`
*   To retrain the CNN with GPU acceleration:
    `python model/train_cnn.py`

### **Testing & Simulation**
*   To run a full end-to-end simulation (Detection -> Telegram -> Lullaby):
    `python simulate_cry.py`

---

## ⏭ Next Steps
1.  **PortAudio Installation:** Install system-level audio libraries (`sudo apt-get install portaudio19-dev`) to activate the physical microphone on the RPi.
2.  **Offline Sync:** Implement the background logic to sync events saved in SQLite to Telegram once internet connectivity is restored.
3.  **Data Collection:** Record more samples for "Belly Pain" and "Discomfort" to balance the categories further.

---
**Status:** *Ready for Hardware Deployment*  
**Date:** February 7, 2026
