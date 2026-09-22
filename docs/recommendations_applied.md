# Professor recommendations applied

| Recommendation | Previous state | Implemented change | Evidence/file |
|---|---|---|---|
| Normalization | Not specified | Baseline uses `Rescaling(1/255)`; MobileNetV2 path uses `mobilenet_v2.preprocess_input` (never both together) | `src/neurodriver_cnn/models/baseline.py`, `src/neurodriver_cnn/models/mobilenetv2.py` |
| Strategic augmentation | Not specified | Train-only horizontal flip, modest brightness/contrast/zoom; vertical flip and 90/180deg rotation explicitly excluded | `notebooks/01_cnn_baseline_mobilenetv2.ipynb` (Section 6) |
| Simple CNN baseline | Not specified | 3x Conv/Pool -> Flatten -> Dense -> Dropout -> logits -> Softmax as an explicit weak reference point | `src/neurodriver_cnn/models/baseline.py` |
| Flatten in baseline | Not specified | Flatten used only in the baseline (never after MobileNetV2, which uses GlobalAveragePooling2D) | `src/neurodriver_cnn/models/baseline.py` vs `src/neurodriver_cnn/models/mobilenetv2.py` |
| Dropout | Not specified | Dropout(0.4) in baseline head, configurable Dropout in MobileNetV2 head | `configs/training_config.json`, both model files |
| EarlyStopping | Not specified | `EarlyStopping(monitor="val_loss", restore_best_weights=True)` | `src/neurodriver_cnn/models/baseline.py::build_baseline_callbacks` |
| ReduceLROnPlateau | Not specified | `ReduceLROnPlateau(monitor="val_loss", factor=0.5, ...)` | `src/neurodriver_cnn/models/baseline.py::build_baseline_callbacks` |
| Leakage prevention | Not specified | Split-before-augmentation, augmentation TRAIN-only, group-aware split by real video ID, TEST never tuned on | `src/neurodriver_cnn/data/manifest.py`, `scripts/05_validate_dataset.py` |
| Input shape | Not specified | Fixed 224x224x3 RGB across baseline and MobileNetV2 | `configs/dataset_config.json`, both model files |
| Class imbalance | Not specified | `reports/sampling_plan.md` reports minority/majority ratio and a class_weight recommendation | `scripts/04_build_experiment_subset.py` |
| F1 | Not specified | F1 macro + F1 weighted + per-class report, never Accuracy alone | `src/neurodriver_cnn/evaluation/metrics.py::classification_metrics` |
| Confusion matrix | Not specified | `confusion_matrix()` over the 4 classes, plus a Full-Frame-vs-ROI label transition matrix | `src/neurodriver_cnn/evaluation/metrics.py`, `scripts/03_analyze_manifest.py` |

Update this table as further recommendations are received and addressed.
