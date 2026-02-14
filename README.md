# Swasthya Scan 🩺

**AI-Enhanced Telemedicine Diagnostic Support for Rural Healthcare**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Swasthya Scan addresses healthcare disparities in rural communities through lightweight, offline-capable AI models. The platform provides diagnostic support for common conditions where specialist access is limited.

## Models

All models are optimized for mobile deployment using TensorFlow Lite:

| Domain | Model File | Task | Accuracy | Size |
|--------|-----------|------|----------|------|
| 🫀 Cardiology | `swasthya_scan_ecg.tflite` | ECG Arrhythmia (5-class) | 82.76% | 2.3 MB |
| 🫁 Pulmonology | `swasthya_scan_xray.tflite` | Chest X-ray Pneumonia | 92%+ | 5 MB |
| 🩺 Dermatology | `swasthya_scan_derm.tflite` | Melanoma Detection | 85-90% | 4 MB |

**Total:** ~11 MB for complete diagnostic suite

## Features

- ✅ **Offline-capable**: Works in low/no connectivity areas
- ✅ **Lightweight**: All models < 5 MB each
- ✅ **Mobile-optimized**: TensorFlow Lite format
- ✅ **Clinically relevant**: Trained on standard medical datasets
- ✅ **Open source**: MIT License

## Model Details

### ECG Arrhythmia Detection
- **Dataset**: MIT-BIH Arrhythmia (109,446 samples)
- **Classes**: Normal, Supraventricular, Ventricular, Fusion, Unknown
- **Input**: 187-point ECG signal
- **Architecture**: 1D CNN with 3 conv blocks

### Chest X-ray Pneumonia Detection
- **Dataset**: Chest X-ray Images (5,863 samples)
- **Classes**: Normal, Pneumonia (binary)
- **Input**: 224×224 grayscale image
- **Architecture**: EfficientNetB0 with custom head

### Melanoma Detection
- **Dataset**: Melanoma Skin Cancer (10,000 images)
- **Classes**: Benign, Malignant (binary)
- **Input**: 128×128 RGB image
- **Architecture**: MobileNetV2 (alpha=0.5)

## Performance Metrics

### ECG Model
![ECG Confusion Matrix](docs/ecg_confusion_matrix.png)

### Chest X-ray Model
![X-ray Confusion Matrix](docs/xray_confusion_matrix.png)

### Dermatology Model
![Derm Confusion Matrix](docs/derm_confusion_matrix.png)

## Quick Start

### Python Inference Example

```python
import numpy as np
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(
    model_path="models/swasthya_scan_ecg.tflite"
)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input (example ECG signal)
ecg_signal = np.random.rand(1, 187, 1).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details['index'], ecg_signal)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details['index'])

print(f"Prediction: {prediction}")
