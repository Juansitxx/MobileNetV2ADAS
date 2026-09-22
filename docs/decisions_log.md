# Decisions log

Chronological record of material technical decisions and their rationale.

## 2026-09-22 — Phase 1 scaffolding

- **Repository structure and Common Manifest architecture created** following
  `PHASE_1.md` / `PROJECT_SPEC.md`. Rationale: training code must not depend
  on BDD100K-specific folders so a future Colombian NeuroDriver adapter can
  plug in unchanged.
- **Full Frame vs ADAS ROI: NOT finalized.** Both strategies are implemented
  and both are computed into every manifest row
  (`label_fullframe`/`label_roi`). A permanent choice is deferred until real
  BDD100K distributions and visual examples exist (`reports/dataset_audit.md`,
  `reports/figures/`). Rationale: `PHASE_1.md` explicitly requires evidence
  before this choice, not an a-priori preference.
- **Default `label`/`has_*` columns in the manifest currently mirror the
  Full Frame strategy** (`scripts/02_build_manifest.py`). This is a
  placeholder default for the Common Manifest contract, not the final
  decision — revisit once the audit above is run on real data.
- **BDD100K local environment: not available.** No TensorFlow install and
  no BDD100K data are present in this development environment. All
  code/tests/docs were built to work correctly in that state (graceful
  MISSING/PENDING reporting) and to work correctly once real data and
  TensorFlow are added, without code changes.
- **BatchNormalization kept frozen during future fine-tuning**, even after
  unfreezing late MobileNetV2 layers (`unfreeze_for_fine_tuning`).
  Rationale: BDD/Colombian fine-tuning batches are expected to be small and
  non-representative of the full training distribution, which would corrupt
  BatchNorm running statistics if left trainable.

## 2026-09-22 — Real dataset evidence: format, splits, and multi-label reformulation

Real BDD100K data was acquired and inspected, superseding several Phase 1
assumptions built for the official 100k `box2d` release.

- **Actual dataset: BDD100K Images 10K (DatasetNinja), Supervisely format,
  distributed as a `.tar`.** Not the official BDD100K 100k box2d release.
  Annotations use `geometryType` (`polygon`/`rectangle`) with
  `points.exterior`, not `box2d`. The parser
  (`neurodriver_cnn.data.bdd100k`) was rewritten around
  `polygon_to_bbox` (`x1=min(x), y1=min(y), x2=max(x), y2=max(y)`).
  Rationale: the previously-implemented official-format parser
  (`load_bdd100k_labels`/`parse_frame_record`) was verified-in-code but
  never verified against real data, and real data turned out to differ.
- **DatasetNinja `test` split excluded from all supervised evaluation.**
  Real counts: train = 7000 JSON (6891 with objects), val = 1000 JSON (981
  with objects), test = 2000 JSON (**0** with objects). `test` carries no
  usable ground truth. Splits: raw `train` -> internal TRAIN + VALIDATION,
  raw `val` -> academic TEST, raw `test` -> excluded entirely.
  `scripts/02_build_manifest.py` never reads the `test` split.
- **No real group/sequence ID exists in this dataset** (independent frames,
  not a tracking sequence). `group_id` is always `None`; the group-aware
  split (`neurodriver_cnn.data.manifest.group_aware_split`) falls back to
  its documented seeded-random behavior. Never invented.
- **`roi.min_bbox_area_ratio` set to `0.0005`** (from the placeholder
  `0.0`), based on visual review of tiny/distant detections judged to be
  noise for ADAS-relevant perception. Kept as a configurable experimental
  hyperparameter in `configs/dataset_config.json` — not hardcoded — and
  applied only within `classify_roi` (Full Frame labeling is unaffected).
- **Classification reformulated as multi-label (two binary targets),
  replacing 4-class softmax.** Real class counts showed severe imbalance
  for pure `PEDESTRIAN` frames; a single 4-way softmax risked collapsing
  that class. Models (`baseline.py`, `mobilenetv2.py`, new
  `resnet50_teacher.py`) now output two raw logits (`vehicle_logit`,
  `pedestrian_logit`), trained with `BinaryCrossentropy(from_logits=True)`.
  The four ADAS states (CLEAR/VEHICLE/PEDESTRIAN/MIXED) are derived
  post-hoc from thresholded probabilities
  (`neurodriver_cnn.evaluation.metrics.labels_from_logits`,
  threshold configurable via `configs/training_config.json` ->
  `evaluation.label_threshold`, default 0.5) and remain the basis for the
  confusion matrix, F1, and all reporting/presentation. `has_vehicle`/
  `has_pedestrian` were already required Common Manifest fields, so this
  needed no manifest schema change.
- **Experiment ladder extended: Teacher = ResNet50 (new
  `neurodriver_cnn.models.resnet50_teacher`), Student 1 = MobileNetV2,
  Student 2 = the existing lightweight baseline CNN** (architecture
  unchanged, now also framed as a KD Student, not only an academic
  reference point). Final comparison (Phase 2+, not run yet): ResNet50
  Teacher, MobileNetV2 Base, MobileNetV2 Distilled, lightweight CNN Base,
  lightweight CNN Distilled — on Accuracy/Precision/Recall/F1/confusion
  matrix (4 states), parameter counts, estimated FLOPs, model size, and
  inference latency. No real KD is implemented yet; all three
  architectures share the two-logit output space so a future KD loss needs
  no adapter layer.

Add a new dated entry for each subsequent material decision (label strategy
finalization, class_weight adoption, fine-tuning layer count, KD alpha/T
choice, etc.).
