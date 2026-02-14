# Swasthya Scan 🩺

AI-Enhanced Telemedicine Diagnostic Support for Rural Healthcare (India focus).

## Models

All models are lightweight, offline-capable TensorFlow Lite.

| Domain       | File                          | Task                            |
|-------------|-------------------------------|---------------------------------|
| Cardiology  | `models/swasthya_scan_ecg.tflite`   | ECG arrhythmia (5-class)        |
| Pulmonology | `models/swasthya_scan_xray.tflite`  | Chest X-ray pneumonia (binary)  |
| Dermatology | `models/swasthya_scan_derm.tflite`  | Melanoma / skin lesion (binary) |

### Performance (current)

- ECG (MIT-BIH heartbeat): **82.76%** test accuracy  
- Chest X-ray (pneumonia): ~**90–95%** test accuracy  
- Melanoma (skin cancer dataset): ~**80–90%** validation accuracy  

### Visuals

- ECG confusion: `docs/ecg_confusion_matrix.png`
- X-ray confusion: `docs/xray_confusion_matrix.png`
- Dermatology confusion: `docs/derm_confusion_matrix.png`

## Structure

```text
swasthya-scan/
  models/      # .tflite models
  docs/        # plots, confusion matrices
  notebooks/   # training notebooks (optional)
  src/         # Python helpers for inference
  requirements.txt
  README.md
  LICENSE
