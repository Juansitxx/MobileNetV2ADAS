# NeuroDriver CNN ADAS Colombia

## Core purpose
Academic research prototype derived from NeuroDriver. Current scope: compact CNN-based visual ADAS perception, not the complete autonomous-driving stack.

Roadmap:
`ImageNet MobileNetV2 → BDD100K source-domain learning → NeuroDriver Colombian dashcam data → Colombian fine-tuning → Knowledge Distillation from NeuroDriver Teacher → compact Student → web perception/alert service`

The final web service performs perception/alerts only. It does not directly control steering, throttle, braking, CAN, or other vehicle actuators.

## Regional scope
Final focus: Colombian driving.

BDD100K is only the initial SOURCE DOMAIN. Never describe it as Colombian data.
The future NeuroDriver dashcam dataset is the TARGET DOMAIN — Colombia.

## Main task
Frame-level multiclass image classification with exactly:
- `CLEAR = 0`
- `VEHICLE = 1`
- `PEDESTRIAN = 2`
- `MIXED = 3`

BDD100K bounding boxes are used only to generate frame-level ground truth. The CNN receives RGB images, not bounding boxes.

Do not convert the project into YOLO, SSD, Faster R-CNN, RetinaNet, Detectron, or another object-detection training task unless explicitly requested.

## Category policy
Initial mapping:
- Vehicle: `car`, `truck`, `bus`, `motorcycle`
- Pedestrian: `pedestrian`
- Auxiliary: `rider`, `bicycle`

Inspect the real annotation schema before assuming exact names.

Motorcycle belongs to `VEHICLE` for the four-class output but must remain explicit through fields such as `has_motorcycle` and `num_motorcycles`. Future evaluation must support a motorcycle-containing-frame slice. Do not silently merge bicycle/rider into motorcycle, pedestrian, or vehicle.

## Ground-truth strategies
Implement and compare:
1. Full-frame labeling.
2. ADAS ROI labeling.

Initial configurable ROI:
- x_min = 0.20
- x_max = 0.80
- y_min = 0.35
- y_max = 1.00
- bbox intersection ratio threshold = 0.35

Keep ROI parameters in configuration. Do not permanently choose Full Frame vs ROI until real statistics and visual examples are reviewed.

## Dataset architecture
Training must depend on a Common Manifest rather than BDD100K-specific folders:

`BDD100K adapter → Common Manifest ← NeuroDriver adapter`

Prefer manifest image paths instead of copying images into class folders. Treat `data/raw/` as read-only.

## Data leakage
- Split before augmentation.
- Augment TRAIN only.
- Reserve TEST for final evaluation.
- Preserve real sequence/video/run/group IDs when available.
- Never invent group IDs.
- Future NeuroDriver data must be split by run/session/video, not adjacent random frames.

Preferred BDD strategy:
`official TRAIN → internal TRAIN + VALIDATION`
`official VAL → academic TEST`

Default seed: `42`.

## Baseline CNN
Keep intentionally simple:
`Input 224×224×3 → Rescaling(1/255) → Conv/Pool ×3 → Flatten → Dense → Dropout → logits → Softmax`

Use EarlyStopping and ReduceLROnPlateau. Baseline exists only as an experimental reference point.

## MobileNetV2 Student
Primary architecture:
`MobileNetV2(ImageNet, include_top=False, 224×224×3) → GlobalAveragePooling2D → Dropout → Dense(4, name="logits") → Softmax(name="predictions")`

Use `tf.keras.applications.mobilenet_v2.preprocess_input`.
Do not also apply `Rescaling(1./255)` in the MobileNetV2 path.
Do not use Flatten after MobileNetV2.
Keep pre-Softmax logits accessible for Knowledge Distillation.

## Knowledge Distillation
Knowledge Distillation is a central future technique alongside Colombian fine-tuning.

Teacher: NeuroDriver or a NeuroDriver-derived high-capacity perception model.
Student: MobileNetV2.

The same Student must support both normal supervised training and future Teacher-Student distillation.

Conceptual future loss:
`L_total = alpha * L_supervised + (1 - alpha) * L_distillation`

Distillation will use temperature-scaled Teacher/Student outputs, typically with KL divergence.

Do not create fake Teacher logits, pseudo-results, or fake KD training. Teacher is used only during training; final deployment runs the Student alone.

Future data contracts should be extendable with:
- `teacher_predictions_available`
- `teacher_logits_path`
- `teacher_model_version`
- `teacher_adapter_version`
- `label_source`

Do not claim KD itself shrinks the architecture; MobileNetV2 is the compact Student and KD transfers knowledge.

## Fine-tuning + KD progression
`ImageNet MobileNetV2 → BDD100K supervised training → source-model evaluation → Colombian NeuroDriver integration → Colombian fine-tuning → NeuroDriver→MobileNetV2 KD → comparative evaluation → Student-only deployment`

## Metrics
Never rely on Accuracy alone. Prepare/measure:
- Accuracy, Precision, Recall
- F1 macro, F1 weighted, per-class metrics
- confusion matrix
- parameter count, model size, inference latency
- motorcycle subset performance

Never invent metrics.

## Academic requirements
Explicitly address normalization, strategic augmentation, simple CNN baseline, Flatten in baseline, Dropout, EarlyStopping, ReduceLROnPlateau, input shape, leakage prevention, class imbalance, F1, and confusion matrix.

Maintain documentation mapping:
`professor recommendation → implemented change → evidence/file`

## Augmentation
TRAIN only. Reasonable: horizontal flip for current non-left/right labels, modest brightness/contrast, modest zoom, very small rotation only if justified.

Never use vertical flips or 90°/180° road rotations.

## Execution environment
Initial training: Google Colab.
Later local training: modern Intel i7 + NVIDIA RTX 4060.

Keep code portable to Windows/Linux/Colab. Use `pathlib`. Never hardcode personal paths.

## Current scope
Focus on dataset preparation/audit, Common Manifest, notebooks, baseline CNN, MobileNetV2 readiness, evaluation utilities, Colombian-domain migration design, KD-ready architecture/contracts, and academic documentation.

Do not drift into FastAPI implementation, frontend, mobile apps, Raspberry Pi, TFLite/ONNX, CAN control, or long production training unless explicitly requested.

## Engineering rules
- Inspect relevant files before making claims.
- Preserve existing user work.
- Keep raw data read-only.
- Use modular code and tests.
- Keep notebooks readable; move reusable logic into `src/`.
- Handle missing data with clear human-readable messages.
- Never fabricate data, labels, metrics, training results, or KD results.
- Do not start long GPU training automatically.
- Prefer incremental, reproducible progress.

## Current task instructions
For the current implementation milestone, read `PHASE_1.md` first.

`PROJECT_SPEC.md` is a full reference document. Read it only when `PHASE_1.md` leaves a material requirement unresolved or the user explicitly asks for the full specification.

If the user's latest explicit instruction conflicts with these files, follow the user's latest instruction and document the decision.
