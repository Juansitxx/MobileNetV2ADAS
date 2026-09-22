# NeuroDriver CNN ADAS Colombia — Phase Specification

## 1. Objective

Build a significant academic prototype for lightweight CNN-based ADAS visual perception.

Research progression:

```text
ImageNet MobileNetV2
→ BDD100K source-domain learning
→ NeuroDriver Colombian dashcam dataset
→ Colombian fine-tuning
→ Knowledge Distillation: NeuroDriver Teacher → MobileNetV2 Student
→ comparative evaluation
→ Student-only web perception service
```

Current milestone: prepare the dataset pipeline, academic notebooks, a simple CNN baseline, a KD-ready MobileNetV2 architecture, evaluation utilities, and documentation. Full MobileNetV2 training, Colombian adaptation, real KD, and web deployment are later phases.

---

## 2. Research scope

The original NeuroDriver ecosystem is broader than this project. This work isolates visual perception.

In scope:
- forward-facing dashcam imagery;
- CNN image classification;
- BDD100K as initial source domain;
- future Colombian NeuroDriver data as target domain;
- MobileNetV2 transfer learning/fine-tuning;
- future Teacher-Student Knowledge Distillation;
- eventual web-based perception/alerts.

Out of scope now:
- steering/braking/throttle control;
- CAN control;
- object-detector training;
- FastAPI/frontend implementation;
- Raspberry Pi/mobile deployment;
- TFLite/ONNX;
- long production training;
- fake KD experiments.

---

## 3. Regional methodology

Final focus: Colombian driving.

BDD100K is provisional SOURCE DOMAIN data, not Colombian data.

Future NeuroDriver dashcam data captured in Colombia is the TARGET DOMAIN.

Do not use fixed/elevated Colombian CCTV as the primary training domain for this dashcam model unless a later experiment explicitly studies that domain shift.

Future evaluation should compare:
1. source-trained model on source test;
2. source-trained model on Colombian held-out test;
3. Colombian fine-tuned model on the same Colombian test;
4. Colombian fine-tuned + KD Student on the same Colombian test.

---

## 4. Primary classification problem

Exactly four output classes:

```text
CLEAR      = 0
VEHICLE    = 1
PEDESTRIAN = 2
MIXED      = 3
```

Definitions:
- CLEAR: no relevant vehicle and no relevant pedestrian.
- VEHICLE: ≥1 relevant vehicle, no relevant pedestrian.
- PEDESTRIAN: ≥1 relevant pedestrian, no relevant vehicle.
- MIXED: ≥1 relevant vehicle and ≥1 relevant pedestrian.

Initial object mapping:
- vehicle: car, truck, bus, motorcycle;
- pedestrian: pedestrian;
- auxiliary: rider, bicycle.

Inspect real BDD100K annotation names before finalizing parser assumptions.

Motorcycle remains inside VEHICLE for the main output but must stay explicit through:
- `has_motorcycle`
- `num_motorcycles`

Do not silently merge rider/bicycle into other categories.

---

## 5. How BDD100K annotations are used

This is NOT an object-detection training project.

BDD100K bounding boxes are used to generate frame-level classification ground truth.

Pipeline:

```text
BDD100K image + official object annotations
→ labeling rules
→ frame class
→ Common Manifest
→ CNN
```

The CNN input is the RGB image only.

---

## 6. Two labeling strategies

Implement both before choosing the experiment strategy.

### 6.1 Full Frame

Any valid mapped object in the full image contributes.

Store:
- `label_fullframe`
- `has_vehicle_fullframe`
- `has_pedestrian_fullframe`
- `has_motorcycle_fullframe`
- relevant counts.

### 6.2 ADAS ROI

Initial normalized configuration:

```json
{
  "x_min": 0.20,
  "x_max": 0.80,
  "y_min": 0.35,
  "y_max": 1.00,
  "bbox_intersection_threshold": 0.35,
  "min_bbox_area_ratio": 0.0
}
```

A bbox is initially relevant if:
1. its center is inside ROI; OR
2. `intersection_area / bbox_area >= threshold`.

Also compute:
`bbox_area_ratio = bbox_area / image_area`.

Do not discard tiny/distant objects during the first audit unless later evidence justifies a threshold.

### 6.3 Selection

Compare Full Frame vs ROI using:
- class counts/percentages;
- label-transition matrix;
- visual examples;
- PEDESTRIAN availability;
- MIXED availability;
- motorcycle representation;
- likely semantic noise;
- relevance to forward ADAS perception.

Record the final chosen label strategy in `docs/decisions_log.md`.

---

## 7. Target repository structure

Use approximately:

```text
PROJECT_ROOT/
├── CLAUDE.md
├── PROJECT_SPEC.md
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── dataset_config.json
│   └── training_config.json
├── data/
│   ├── README.md
│   ├── raw/bdd100k/
│   ├── interim/
│   └── processed/manifests/
├── docs/
│   ├── project_scope.md
│   ├── dataset_methodology.md
│   ├── bdd100k_setup.md
│   ├── colombian_domain_strategy.md
│   ├── neurodriver_dataset_contract.md
│   ├── teacher_student_contract.md
│   ├── recommendations_applied.md
│   ├── decisions_log.md
│   ├── future_knowledge_distillation.md
│   ├── future_web_service.md
│   └── presentation_outline.md
├── notebooks/
│   ├── 00_dataset_exploration.ipynb
│   └── 01_cnn_baseline_mobilenetv2.ipynb
├── reports/
│   ├── schema_analysis.md
│   ├── dataset_audit.md
│   ├── dataset_audit.json
│   ├── sampling_plan.md
│   ├── experiment_status.md
│   └── figures/
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_inspect_bdd100k.py
│   ├── 02_build_manifest.py
│   ├── 03_analyze_manifest.py
│   ├── 04_build_experiment_subset.py
│   └── 05_validate_dataset.py
├── src/neurodriver_cnn/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   ├── labeling/
│   ├── models/
│   ├── evaluation/
│   ├── distillation/
│   └── utils/
└── tests/
    ├── test_roi.py
    ├── test_frame_labels.py
    └── test_manifest.py
```

Adjust only when technically justified.

Keep raw/interim data and model checkpoints out of Git.

---

## 8. Configuration

### `configs/dataset_config.json`

Include:
- seed = 42;
- source dataset = BDD100K;
- target domain = NeuroDriver Colombia;
- class mappings;
- ROI;
- bbox threshold;
- min bbox area ratio;
- target experimental size ≈ 8000;
- image size = 224×224.

### `configs/training_config.json`

Prepare:
- image size;
- batch size;
- baseline epochs;
- MobileNet head epochs;
- fine-tuning epochs;
- learning rates;
- EarlyStopping patience;
- ReduceLROnPlateau patience.

Also include future KD configuration:

```json
"distillation": {
  "enabled": false,
  "alpha": null,
  "temperature": null
}
```

Do not select scientifically “final” alpha/T values yet.

---

## 9. BDD100K setup

Inspect existing files first.

If BDD100K is absent:
- do not use unofficial mirrors;
- do not download huge archives without control;
- document official/manual acquisition;
- specify only required image and object-detection annotation packages;
- complete the codebase and notebooks as far as possible.

Create `docs/bdd100k_setup.md` with exact expected placement and verification commands.

Do not require segmentation, tracking, lane masks, etc. for this phase.

---

## 10. Environment checker

`scripts/00_check_environment.py` should report clearly:
- Python/OS;
- project root;
- free disk;
- TensorFlow availability;
- GPU visibility;
- BDD image availability;
- BDD annotation availability.

Use readable `[OK]`, `[INFO]`, `[WARNING]`, `[MISSING]` messages rather than expected-condition tracebacks.

---

## 11. Annotation inspection and parser

`scripts/01_inspect_bdd100k.py`:
- inspect actual JSON schema;
- list keys;
- count records;
- list/count categories;
- inspect bbox structure;
- inspect weather/scene/timeofday;
- preserve real sequence/video identifiers if present.

Write `reports/schema_analysis.md`.

Implement reusable parsing in:
`src/neurodriver_cnn/data/bdd100k.py`.

Use modular functions, `pathlib`, type hints, and useful docstrings.

---

## 12. Bounding-box/ROI logic

For each bbox compute:
- x1, y1, x2, y2;
- width, height, area;
- center;
- normalized center;
- bbox area ratio;
- category.

Create pure/testable ROI helpers in:
`src/neurodriver_cnn/labeling/roi.py`.

Create frame classification logic in:
`src/neurodriver_cnn/labeling/frame_labels.py`.

Expected return structure may include:

```python
{
    "label": "MIXED",
    "has_vehicle": True,
    "has_pedestrian": True,
    "has_motorcycle": True,
    "num_vehicles": 3,
    "num_pedestrians": 1,
    "num_motorcycles": 2,
    "num_riders": 1,
    "num_bicycles": 0
}
```

---

## 13. Full manifest

`scripts/02_build_manifest.py` should create:

`data/processed/manifests/bdd100k_manifest.csv`

Recommended fields:
- image_id/path;
- source dataset/split;
- real group/sequence ID if available;
- width/height;
- weather/scene/timeofday;
- object count;
- Full Frame counts/flags/label;
- ROI counts/flags/label;
- ROI config values;
- image/annotation availability;
- valid sample flag.

Never invent group IDs.

Validate class invariants before saving.

---

## 14. Common Manifest contract

Create `docs/neurodriver_dataset_contract.md`.

Required future training fields:
- `image_path`
- `label`
- `split`
- `source_dataset`
- `has_vehicle`
- `has_pedestrian`
- `has_motorcycle`

Optional:
- image_id;
- group/run/session/video ID;
- timestamp;
- counts;
- environment attributes;
- label provenance;
- Teacher prediction metadata.

The future NeuroDriver adapter must output this same contract so notebooks do not require rewriting.

---

## 15. Leakage and split rules

Rules:
- split before augmentation;
- augmentation only on TRAIN;
- TEST never used for hyperparameter tuning;
- use real group-aware splitting when possible;
- future NeuroDriver adjacent frames must not cross splits.

Preferred BDD split:
- official TRAIN → internal TRAIN + VALIDATION;
- official VAL → academic TEST.

Use seed 42.

---

## 16. Dataset audit and figures

`scripts/03_analyze_manifest.py` should create:
- `reports/dataset_audit.md`
- `reports/dataset_audit.json`

Report:
- total/valid/missing samples;
- resolutions;
- category counts;
- Full Frame distribution;
- ROI distribution;
- label-transition matrix;
- motorcycle representation;
- weather/time/scene if available;
- evidence-based limitations and potential source→Colombia domain shift.

Create figures when real data exists:
- `class_distribution_fullframe.png`
- `class_distribution_roi.png`
- `fullframe_vs_roi.png`
- `motorcycle_distribution.png`
- `roi_examples.png`
- `class_examples.png`

ROI visualizations should show ROI, bbox/category, relevance, and both labels.

---

## 17. Experimental subset

Target approximately 4,000–8,000 images; prefer near 8,000 if class availability permits.

Do not fabricate balance, duplicate files, or generate synthetic samples at dataset-preparation time.

`reports/sampling_plan.md` should state:
- class availability;
- minority/majority ratio;
- possible balanced subset;
- recommended practical subset;
- whether class weights may be appropriate.

`scripts/04_build_experiment_subset.py` creates:
`data/processed/manifests/experiment_manifest.csv`.

Include label provenance and split.

Save manifest-generation metadata/config for reproducibility.

---

## 18. Dataset validation/tests

`scripts/05_validate_dataset.py` should check:
- file existence;
- expected labels;
- duplicate IDs/paths;
- split overlap;
- group overlap when real group IDs exist.

Optional flags may perform image verification and exact file-hash duplicate checks.

Tests must cover:
- CLEAR / VEHICLE / PEDESTRIAN / MIXED logic;
- car/truck/bus/motorcycle behavior;
- motorcycle + pedestrian → MIXED;
- rider/bicycle isolation;
- ROI inside/outside/intersection;
- manifest invariants.

Run tests and fix failures.

---

## 19. Notebook 00 — dataset exploration

`notebooks/00_dataset_exploration.ipynb`

Academic sections:
1. NeuroDriver context.
2. Colombian problem focus.
3. Scope reduction.
4. BDD100K as provisional source domain.
5. Colombia as future target domain.
6. Four-class task.
7. Motorcycle handling.
8. Full Frame method.
9. ADAS ROI method.
10. Dataset audit.
11. Distribution comparison.
12. Visual examples.
13. Splits/leakage.
14. Domain-shift limitations.
15. Experimental subset.
16. Conclusions.

No CNN training here.

---

## 20. Notebook 01 — CNN baseline + MobileNetV2

`notebooks/01_cnn_baseline_mobilenetv2.ipynb`

Designed primarily for Google Colab, but portable to local Windows/Linux.

Include:
- environment/GPU detection;
- Python/NumPy/TensorFlow seed = 42;
- configurable project/data root;
- Common Manifest loading;
- label validation;
- examples/distributions;
- TensorFlow data pipeline;
- augmentation;
- simple baseline CNN;
- callbacks;
- evaluation functions;
- MobileNetV2 construction;
- forward-pass smoke test;
- future fine-tuning cells;
- KD-readiness explanation;
- clear DONE/PENDING markers.

Do not hardcode personal Google Drive paths.

---

## 21. Input pipeline and augmentation

Input: RGB 224×224×3.

Use efficient `tf.data` with:
- TRAIN shuffle;
- batching;
- prefetch AUTOTUNE;
- sensible caching only when appropriate.

Augmentation only on TRAIN.

Allowed:
- horizontal flip;
- modest brightness/contrast;
- modest zoom;
- very small rotation if justified.

Never use vertical flip or 90°/180° road rotations.

Document horizontal flip as acceptable for the current labels because they do not encode left/right semantics.

---

## 22. Baseline CNN

Implement reusable model in:
`src/neurodriver_cnn/models/baseline.py`.

Suggested architecture:

```text
Input 224×224×3
→ Rescaling(1/255)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + MaxPool
→ Flatten
→ Dense(128)
→ Dropout(0.4)
→ Dense(4, logits)
→ Softmax
```

Use:
- Adam;
- EarlyStopping on val_loss with restore_best_weights;
- ReduceLROnPlateau.

If real data is available, a short initial training run is allowed to generate honest preliminary metrics. Do not run long training automatically.

---

## 23. MobileNetV2 Student

Implement in:
`src/neurodriver_cnn/models/mobilenetv2.py`.

Use:
- `tf.keras.applications.MobileNetV2`
- ImageNet weights
- `include_top=False`
- `input_shape=(224,224,3)`
- `mobilenet_v2.preprocess_input`
- frozen backbone initially
- `base_model(x, training=False)`
- GlobalAveragePooling2D
- Dropout
- `Dense(4, name="logits")`
- `Softmax(name="predictions")`

Do not:
- use `Rescaling(1./255)` in the same MobileNet pipeline;
- use Flatten after MobileNetV2.

Show total/trainable/non-trainable params.

If data exists, perform a real batch forward pass and validate output `(batch, 4)` without NaN.

Full MobileNet training is not required for the immediate milestone.

---

## 24. Fine-tuning design

Prepare but do not necessarily run:
1. train head with frozen backbone;
2. unfreeze selected late layers;
3. recompile;
4. use low LR (order ~1e-5);
5. fine-tune carefully;
6. manage BatchNormalization deliberately.

Distinguish:
- source-domain MobileNet training/fine-tuning on BDD;
- later target-domain fine-tuning on Colombian NeuroDriver data.

---

## 25. Evaluation

Never use Accuracy alone.

Prepare:
- Accuracy;
- Precision;
- Recall;
- F1 macro;
- F1 weighted;
- per-class report;
- confusion matrix.

Prepare model comparison fields:
- performance metrics;
- total/trainable parameters;
- saved model size;
- average inference latency.

Prepare `evaluate_motorcycle_subset(...)` using `has_motorcycle == True`.

Do not fabricate missing values.

---

## 26. Academic metric interpretation

Notebook Markdown should explain, using real values only:
- train/validation divergence and possible overfitting;
- class imbalance;
- Precision/Recall differences;
- PEDESTRIAN F1/Recall;
- confusion among VEHICLE, PEDESTRIAN, MIXED.

Do not claim causality from a confusion matrix.

---

## 27. Professor recommendations

Create `docs/recommendations_applied.md` with:

```text
Recommendation | Previous state | Implemented change | Evidence/file
```

Explicitly cover:
- normalization;
- strategic augmentation;
- baseline CNN;
- Flatten;
- Dropout;
- EarlyStopping;
- leakage prevention;
- input shape;
- class imbalance;
- F1;
- confusion matrix.

---

## 28. Colombian adaptation strategy

Create `docs/colombian_domain_strategy.md`.

Future stages:
1. Integrate NeuroDriver Colombia frames and metadata.
2. Convert to Common Manifest.
3. Split Colombia by run/session/video into TRAIN/VAL/TEST.
4. Keep Colombia TEST untouched.
5. Evaluate BDD-trained Student on Colombia TEST before adaptation.
6. Fine-tune on Colombia TRAIN/VAL.
7. Re-evaluate same held-out Colombia TEST.
8. Add NeuroDriver Teacher KD.
9. Compare gains/losses, including motorcycle subset.

Potential differences such as motorcycle density, road infrastructure, signage, traffic behavior, vehicle mix, road condition, and pedestrian context must be treated as hypotheses until measured.

---

## 29. Knowledge Distillation contract

KD is a core future technique, not an unrelated optional feature.

Teacher:
NeuroDriver or a NeuroDriver-derived high-capacity perception system.

Student:
MobileNetV2.

The Teacher may not naturally output the four Student classes, so plan a:
`NeuroDriverTeacherAdapter`.

Create `docs/teacher_student_contract.md`.

Conceptual Teacher output:

```python
{
    "logits": ...,
    "probabilities": ...,
    "teacher_version": ...,
    "adapter_version": ...
}
```

Student logits must remain directly accessible before Softmax.

---

## 30. Future KD data/loss

Extend the Common Manifest contract with optional:
- `teacher_predictions_available`
- `teacher_logits_path`
- `teacher_model_version`
- `teacher_adapter_version`
- `label_source`

Possible provenance:
- `BDD100K_GROUND_TRUTH`
- `NEURODRIVER_GROUND_TRUTH`
- `TEACHER_PSEUDO_LABEL`
- `MANUAL_REVIEW`

Never equate Teacher pseudo-labels with verified human ground truth.

Document future loss:

```text
teacher_soft = softmax(teacher_logits / T)
student_soft = softmax(student_logits / T)

L_total =
alpha * L_supervised
+
(1 - alpha) * KL(teacher_soft || student_soft)
```

Treat `alpha` and `T` as experimental hyperparameters.

Do not generate fake Teacher predictions now.

---

## 31. KD experimental design

Keep later experiments separable:

- A: simple CNN baseline.
- B: MobileNetV2 + BDD supervised learning.
- C: MobileNetV2 + Colombian fine-tuning.
- D: MobileNetV2 + Colombian fine-tuning + NeuroDriver KD.

For KD ablation, keep Student architecture, data split, and evaluation conditions comparable.

Evaluate KD with:
- classification metrics;
- motorcycle subset;
- parameters/model size;
- latency.

Do not claim KD itself reduces architecture size; the Student architecture is compact and KD transfers knowledge.

---

## 32. KD scaffolding

Create:

```text
src/neurodriver_cnn/distillation/
├── __init__.py
└── README.md
```

Document future:
- TeacherAdapter;
- cached Teacher predictions;
- distillation loss;
- custom Keras Distiller/training loop.

Do not implement fake KD.

---

## 33. Future web service

Create `docs/future_web_service.md`.

Concept:

```text
Dashcam/image
→ POST /predict
→ FastAPI
→ MobileNetV2 Student
→ class + confidence
→ visual ADAS alert
```

Teacher is absent during deployment inference.

Do not implement the service now.

---

## 34. Documentation

Create/update:
- `README.md`
- `docs/project_scope.md`
- `docs/dataset_methodology.md`
- `docs/bdd100k_setup.md`
- `docs/colombian_domain_strategy.md`
- `docs/neurodriver_dataset_contract.md`
- `docs/teacher_student_contract.md`
- `docs/recommendations_applied.md`
- `docs/decisions_log.md`
- `docs/future_knowledge_distillation.md`
- `docs/future_web_service.md`
- `docs/presentation_outline.md`
- `reports/experiment_status.md`

README should describe the system as an academic/research prototype, never as production-ready autonomous driving.

---

## 35. Presentation outline

Approximately 8 slides:
1. Colombian driving problem/context.
2. Original NeuroDriver ecosystem.
3. Scope reduction to CNN visual perception.
4. Dataset strategy: BDD source → Colombia target.
5. Four classes + explicit motorcycle tracking.
6. CNN baseline.
7. MobileNetV2 transfer learning + future NeuroDriver KD.
8. Current progress and next steps.

Do not claim KD is already implemented.

---

## 36. Immediate milestone definition

A strong near-term result is:

- repository structure complete;
- BDD setup documented;
- parser/labeling/ROI tested;
- Common Manifest architecture;
- audit/sampling/split code;
- Notebook 00;
- Notebook 01;
- baseline CNN implemented;
- MobileNetV2 Student implemented with exposed logits;
- callbacks/evaluation prepared;
- motorcycle slice prepared;
- Colombian migration documented;
- KD contracts/scaffolding prepared;
- honest status report.

If real BDD data is available:
- generate real statistics;
- create experimental manifest;
- optionally run a short baseline training for preliminary metrics.

Not required yet:
- full MobileNet training;
- actual Colombian fine-tuning;
- real KD;
- web deployment.

---

## 37. Execution order

1. Inspect repository and preserve existing work.
2. Read `CLAUDE.md` and this file.
3. Create/update structure and configs.
4. Prepare official BDD setup instructions.
5. Implement environment checker.
6. Inspect real BDD schema when available.
7. Implement parser.
8. Implement/test ROI and frame labeling.
9. Build manifest when data permits.
10. Audit, visualize, and select/recommend label strategy.
11. Build reproducible subset and validate splits.
12. Build Notebook 00.
13. Implement baseline.
14. Implement KD-ready MobileNetV2 Student.
15. Implement evaluation utilities.
16. Build Notebook 01.
17. Add KD contracts/scaffolding.
18. Complete academic docs/presentation outline.
19. Run tests/validation.
20. Produce executive status report.

Do not start long GPU training automatically.

---

## 38. If BDD100K is unavailable

Continue all work that does not require real samples.

Use minimal synthetic fixtures only for unit tests; never present them as experiment data.

At completion, state exactly:
- what official BDD100K packages are required;
- where to obtain them;
- where to place them;
- which command to run next.

---

## 39. Non-fabrication rule

Never fabricate:
- dataset counts;
- class distributions;
- metrics;
- loss curves;
- confusion matrices;
- BDD→Colombia comparisons;
- KD results.

Use `Pending` whenever a real experiment has not been executed.

---

## 40. Final status report

Report concisely:

1. Overall status.
2. Files created/modified.
3. BDD100K availability.
4. Real dataset findings, if available.
5. Full Frame vs ROI findings/status.
6. Class distribution.
7. Motorcycle representation/handling.
8. Split/leakage validation.
9. Notebook 00 status.
10. Notebook 01 status.
11. Baseline status.
12. MobileNetV2 status.
13. KD-readiness status.
14. Tests.
15. Real metrics.
16. Blockers.
17. Manual user action required.
18. Recommended next step.

Do not automatically execute the next major phase or long training.
