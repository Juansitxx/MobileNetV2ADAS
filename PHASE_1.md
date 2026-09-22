# Phase 1 — NeuroDriver CNN ADAS Colombia

## Goal
Create a strong academic milestone without long training.

By the end of this phase the repository should contain:
- a reproducible BDD100K preparation/audit pipeline;
- Full Frame vs ADAS ROI frame labels;
- a Common Manifest compatible with future NeuroDriver Colombia data;
- explicit motorcycle metadata;
- dataset exploration notebook;
- simple CNN baseline;
- MobileNetV2 Student with accessible logits;
- evaluation utilities;
- Colombian fine-tuning roadmap;
- Knowledge Distillation contracts/scaffolding;
- academic documentation.

Full MobileNetV2 training, Colombian fine-tuning, real KD, and the web service are NOT required yet.

## Research path
`ImageNet → MobileNetV2 → BDD100K source domain → NeuroDriver Colombia target domain → fine-tuning + NeuroDriver Teacher KD → compact Student → web perception service`

BDD100K is not Colombian. Never present it as such.

## Classification
Exactly four classes:
- CLEAR = 0
- VEHICLE = 1
- PEDESTRIAN = 2
- MIXED = 3

Initial mapping:
- vehicle: car, truck, bus, motorcycle
- pedestrian: pedestrian
- auxiliary: rider, bicycle

Motorcycle remains inside VEHICLE but MUST also produce:
- `has_motorcycle`
- `num_motorcycles`

Do not add MOTORCYCLE as a fifth output class.

## BDD100K use
This is frame-level classification, not detector training.

Use official BDD100K object annotations only to build frame ground truth.
CNN input = RGB frame.

Do not implement/train YOLO, SSD, Faster R-CNN, RetinaNet, Detectron, etc.

## Labeling methods
Implement both:

### Full Frame
All mapped objects in frame count.

### ADAS ROI
Initial normalized ROI:
- x_min 0.20
- x_max 0.80
- y_min 0.35
- y_max 1.00
- bbox intersection ratio threshold 0.35
- min bbox area ratio 0.0 initially

A bbox is ROI-relevant when:
- bbox center is inside ROI; OR
- intersection_area / bbox_area >= threshold.

Keep all values configurable.

Do not permanently choose Full Frame vs ROI until real distributions and examples are reviewed.

## Repository structure
Create/update approximately:

```text
configs/
  dataset_config.json
  training_config.json
data/
  raw/bdd100k/
  interim/
  processed/manifests/
docs/
notebooks/
  00_dataset_exploration.ipynb
  01_cnn_baseline_mobilenetv2.ipynb
reports/
  figures/
scripts/
  00_check_environment.py
  01_inspect_bdd100k.py
  02_build_manifest.py
  03_analyze_manifest.py
  04_build_experiment_subset.py
  05_validate_dataset.py
src/neurodriver_cnn/
  data/
  labeling/
  models/
  evaluation/
  distillation/
  utils/
tests/
```

Preserve existing work. Treat `data/raw/` as read-only.

Configure `.gitignore` so raw datasets, checkpoints, models, environments, caches, and notebook checkpoints are not accidentally committed.

## Dataset setup
First inspect the current repository.

If BDD100K is absent:
- do not use unofficial mirrors;
- do not download huge archives automatically;
- document the official/manual download;
- specify only the image and object-detection packages needed;
- continue implementing everything that can be prepared without real data.

Create `docs/bdd100k_setup.md`.

## Environment check
`scripts/00_check_environment.py` should clearly report:
- Python/OS;
- project root;
- free disk;
- TensorFlow installed?;
- GPU visible?;
- BDD images found?;
- BDD annotations found?.

Use readable `[OK]`, `[INFO]`, `[WARNING]`, `[MISSING]` messages.

## Inspect real schema
`scripts/01_inspect_bdd100k.py` must inspect the actual annotations before parser assumptions become permanent.

Report:
- record count;
- keys;
- categories/frequencies;
- bbox schema;
- weather/scene/timeofday;
- real sequence/video/group identifier if available.

Write `reports/schema_analysis.md`.

Never invent sequence IDs.

## Parser
Implement reusable BDD parsing in:
`src/neurodriver_cnn/data/bdd100k.py`

Use modular functions, pathlib, type hints, useful docstrings.

For bboxes compute:
- coordinates;
- width/height/area;
- center;
- normalized center;
- bbox_area_ratio;
- category.

## ROI + frame labeling
Implement pure/testable ROI functions in:
`src/neurodriver_cnn/labeling/roi.py`

Implement:
- `classify_fullframe(...)`
- `classify_roi(...)`

in:
`src/neurodriver_cnn/labeling/frame_labels.py`

Return label, vehicle/pedestrian/motorcycle flags and counts, plus rider/bicycle counts when available.

## Manifest
`scripts/02_build_manifest.py` should generate:
`data/processed/manifests/bdd100k_manifest.csv`

Include at least:
- image_id/path;
- source dataset/split;
- real group ID if available;
- width/height;
- weather/scene/timeofday if available;
- full-frame label/flags/counts;
- ROI label/flags/counts;
- motorcycle fields;
- image/annotation availability;
- valid_sample.

Validate class invariants before saving.

## Common Manifest
Create `docs/neurodriver_dataset_contract.md`.

Training notebooks must consume a Common Manifest rather than BDD-specific folders.

Minimum future fields:
- image_path
- label
- split
- source_dataset
- has_vehicle
- has_pedestrian
- has_motorcycle

Optional:
- image/group/run/session/video IDs;
- timestamp;
- object counts;
- environment attributes;
- label_source;
- Teacher prediction metadata.

Future NeuroDriverAdapter must emit this same schema.

## Leakage / splits
Rules:
- split before augmentation;
- augmentation only TRAIN;
- never tune on TEST;
- use real group-aware splitting when possible;
- future NeuroDriver adjacent frames must not cross splits.

Preferred BDD scheme:
- official TRAIN → internal TRAIN + VALIDATION;
- official VAL → academic TEST.

Default seed = 42.

## Audit
`scripts/03_analyze_manifest.py` should generate:
- `reports/dataset_audit.md`
- `reports/dataset_audit.json`

Report real values only:
- sample counts;
- missing/invalid files;
- resolutions;
- object/category counts;
- Full Frame class distribution;
- ROI class distribution;
- label transition matrix;
- motorcycle representation;
- weather/time/scene distributions if available;
- dataset limitations/domain-shift notes.

Generate figures when data exists:
- class_distribution_fullframe.png
- class_distribution_roi.png
- fullframe_vs_roi.png
- motorcycle_distribution.png
- roi_examples.png
- class_examples.png

## Experimental subset
Target roughly 4,000–8,000 samples; prefer near 8,000 if class availability permits.

Do not fabricate balance, duplicate files, or generate synthetic training images here.

`reports/sampling_plan.md` should analyze:
- minority/majority class;
- ratios;
- possible balanced subset;
- recommended practical subset;
- future class_weight need.

`scripts/04_build_experiment_subset.py` creates:
`data/processed/manifests/experiment_manifest.csv`

Make sampling/splits reproducible and save generation metadata/config.

## Validation/tests
`scripts/05_validate_dataset.py` should check:
- paths;
- labels;
- duplicate IDs/paths;
- split overlap;
- group overlap when applicable.

Unit tests must cover:
- no objects → CLEAR;
- car/truck/bus/motorcycle → VEHICLE;
- motorcycle sets has_motorcycle;
- pedestrian → PEDESTRIAN;
- vehicle + pedestrian → MIXED;
- rider/bicycle are not silently remapped;
- ROI inside/outside/intersection cases;
- manifest invariants.

Run tests and fix failures.

## Notebook 00
Create `notebooks/00_dataset_exploration.ipynb`.

Academic sections:
- NeuroDriver context;
- Colombian focus;
- scope reduction;
- BDD source vs Colombia target;
- four classes;
- motorcycle policy;
- Full Frame method;
- ADAS ROI method;
- audit/distributions;
- examples;
- splits/leakage;
- domain shift;
- experimental subset;
- conclusions.

No CNN training in Notebook 00.

## Notebook 01
Create `notebooks/01_cnn_baseline_mobilenetv2.ipynb`.

Designed for Colab but portable locally.

Include:
- environment/GPU check;
- deterministic seeds;
- configurable project/data root;
- Common Manifest loading;
- label validation;
- tf.data pipeline;
- train-only augmentation;
- baseline CNN;
- callbacks/evaluation;
- MobileNetV2 construction;
- forward-pass smoke test;
- future fine-tuning cells;
- KD-readiness explanation;
- clear DONE/PENDING status.

No personal hardcoded paths.

## Preprocessing / augmentation
Image = RGB 224×224×3.

Baseline:
- `Rescaling(1./255)`.

MobileNetV2:
- `tf.keras.applications.mobilenet_v2.preprocess_input`.
- do NOT also use `Rescaling(1./255)`.

Train-only augmentation:
- horizontal flip;
- modest brightness/contrast;
- modest zoom;
- very small rotation only if justified.

Never vertical flip or 90°/180° rotations.

## Baseline CNN
Implement reusable baseline in:
`src/neurodriver_cnn/models/baseline.py`

Concept:
`Input → Rescaling → Conv/Pool ×3 → Flatten → Dense(128) → Dropout(0.4) → Dense logits → Softmax`

Use:
- Adam;
- EarlyStopping;
- ReduceLROnPlateau.

If real data is available, a short initial training run is allowed for honest preliminary metrics. Do not start long training automatically.

## MobileNetV2 Student
Implement:
`src/neurodriver_cnn/models/mobilenetv2.py`

Use:
- ImageNet weights;
- include_top=False;
- input 224×224×3;
- frozen backbone initially;
- preprocess_input;
- `base_model(x, training=False)`;
- GlobalAveragePooling2D;
- Dropout;
- `Dense(4, name="logits")`;
- `Softmax(name="predictions")`.

Do not use Flatten after MobileNetV2.

Keep logits directly accessible for future KD.

Show total/trainable/non-trainable params.
If real data exists, run a real batch forward-pass smoke test.

Full MobileNetV2 training is not required now.

## Future fine-tuning
Prepare code/design for:
1. train head with frozen backbone;
2. unfreeze selected late layers;
3. recompile with low LR (~1e-5 order);
4. carefully handle BatchNormalization.

Keep BDD source training and later Colombian domain fine-tuning conceptually distinct.

## Evaluation
Prepare:
- Accuracy;
- Precision;
- Recall;
- F1 macro/weighted;
- per-class report;
- confusion matrix;
- parameter count;
- model size;
- inference latency.

Also prepare:
`evaluate_motorcycle_subset(...)`

Never invent unavailable results.

## Colombian target-domain strategy
Create `docs/colombian_domain_strategy.md`.

Future plan:
1. recover NeuroDriver Colombia dashcam data;
2. convert to Common Manifest;
3. split by run/session/video into Colombia TRAIN/VAL/TEST;
4. keep Colombia TEST untouched;
5. evaluate BDD-trained Student before adaptation;
6. fine-tune on Colombia TRAIN/VAL;
7. evaluate again on same Colombia TEST;
8. add Teacher KD;
9. compare, including motorcycle slice.

Treat possible Colombia-vs-BDD differences as hypotheses until measured.

## Knowledge Distillation
KD is a core future technique, not an optional side idea.

Teacher:
NeuroDriver or a NeuroDriver-derived high-capacity perception model.

Student:
the same MobileNetV2 defined above.

Teacher is used during training only. Final web inference runs Student alone.

Create:
- `docs/teacher_student_contract.md`
- `docs/future_knowledge_distillation.md`
- `src/neurodriver_cnn/distillation/__init__.py`
- `src/neurodriver_cnn/distillation/README.md`

Do NOT perform fake KD.

## Teacher-Student contract
The Student output space remains:
CLEAR / VEHICLE / PEDESTRIAN / MIXED.

Future `NeuroDriverTeacherAdapter` will map real NeuroDriver outputs to compatible Teacher logits/probabilities.

Conceptual Teacher batch output may include:
- logits;
- probabilities;
- teacher_version;
- adapter_version.

Future manifest fields may include:
- teacher_predictions_available;
- teacher_logits_path;
- teacher_model_version;
- teacher_adapter_version;
- label_source.

Possible provenance:
- BDD100K_GROUND_TRUTH
- NEURODRIVER_GROUND_TRUTH
- TEACHER_PSEUDO_LABEL
- MANUAL_REVIEW

Never treat pseudo-labels as verified ground truth.

## KD loss/design
Document, but do not execute:

`L_total = alpha * L_supervised + (1-alpha) * L_distillation`

Use temperature-scaled Teacher/Student logits; KL divergence is the expected starting point.

`alpha` and temperature `T` are experimental hyperparameters, not fixed truths.

Training config should contain:
```json
"distillation": {
  "enabled": false,
  "alpha": null,
  "temperature": null
}
```

Do not generate fake Teacher logits.

## Future experiment ladder
Keep experiments separable:
- A: simple CNN baseline;
- B: MobileNetV2 + BDD supervised;
- C: MobileNetV2 + Colombian fine-tuning;
- D: MobileNetV2 + Colombian fine-tuning + NeuroDriver KD.

Use comparable splits/Student architecture for ablation.

Do not claim KD itself shrinks model size; MobileNetV2 is the compact Student.

## Academic documentation
Create/update:
- README.md
- docs/project_scope.md
- docs/dataset_methodology.md
- docs/bdd100k_setup.md
- docs/colombian_domain_strategy.md
- docs/neurodriver_dataset_contract.md
- docs/teacher_student_contract.md
- docs/recommendations_applied.md
- docs/decisions_log.md
- docs/future_knowledge_distillation.md
- docs/future_web_service.md
- docs/presentation_outline.md
- reports/experiment_status.md

`recommendations_applied.md` must map:
`professor recommendation → previous state → implemented change → evidence/file`

Explicitly cover normalization, augmentation, baseline, Flatten, Dropout, EarlyStopping, leakage, input shape, imbalance, F1, confusion matrix.

## Presentation outline
Prepare ~8 slides:
1. Colombian driving problem/context.
2. Original NeuroDriver.
3. Scope reduction.
4. BDD source → Colombia target strategy.
5. Four classes + motorcycle tracking.
6. Baseline CNN.
7. MobileNetV2 + transfer learning + future KD.
8. Current status + next steps.

Do not claim KD is implemented.

## If BDD100K is missing
Continue all code/docs/notebook scaffolding possible.

Use synthetic values only inside unit-test fixtures; never present them as real experiment results.

At the end state exactly:
- what official BDD package(s) are required;
- where to obtain them;
- where to place them;
- what command the user should run next.

## Non-fabrication
Never fabricate:
- dataset counts;
- class distributions;
- metrics;
- curves;
- confusion matrices;
- Colombia comparisons;
- KD results.

Use `Pending` if not executed.

## Work order
1. Inspect repo/preserve work.
2. Read CLAUDE.md and this file.
3. Build structure/config.
4. BDD setup/environment checker.
5. Schema inspection/parser.
6. ROI/frame labels/tests.
7. Manifest/audit/subset/validation.
8. Notebook 00.
9. Baseline.
10. KD-ready MobileNetV2.
11. Evaluation utilities.
12. Notebook 01.
13. Colombian/KD docs and scaffolding.
14. Run tests/validation.
15. Produce concise status report.

Do not start long GPU training automatically.

## Final report
Return:
1. overall status;
2. files created/modified;
3. BDD availability;
4. real dataset findings;
5. Full Frame vs ROI status/results;
6. class/motorcycle distribution;
7. split/leakage status;
8. Notebook 00 status;
9. Notebook 01 status;
10. baseline status;
11. MobileNetV2 status;
12. KD-readiness status;
13. tests;
14. real metrics;
15. blockers;
16. manual user action required;
17. recommended next step.

Do not automatically execute the next major phase.
