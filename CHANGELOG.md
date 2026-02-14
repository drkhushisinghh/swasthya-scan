# Changelog

All notable changes to Swasthya Scan will be documented in this file.

## [1.0.0] - 2026-02-14

### Added
- ECG arrhythmia detection model (5 classes)
  - Input: 187-point ECG signal
  - Accuracy: 82.76%
  - Size: 2.3 MB

- Chest X-ray pneumonia detection model (binary)
  - Input: 224×224 grayscale image
  - Accuracy: 92%+
  - Size: 5 MB

- Skin condition classification model (7 classes)
  - Input: 224×224 RGB image
  - Accuracy: 85-90%
  - Size: 4 MB

### Validated
- All models tested on Kaggle GPU environment
- Input/output shapes verified
- Inference performance benchmarked
- Confusion matrices generated

### Documentation
- Complete README with usage examples
- Model specifications documented
- Mobile integration examples (Android/iOS)
- Datasets and citations included

---

## Future Releases

### [1.1.0] - Planned
- [ ] Ophthalmology: Diabetic retinopathy detection
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Mobile app (Android/iOS)
- [ ] Edge device optimization

### [2.0.0] - Planned
- [ ] Federated learning support
- [ ] Real-time video analysis
- [ ] Multi-modal fusion (ECG + X-ray)
- [ ] Clinical validation results
