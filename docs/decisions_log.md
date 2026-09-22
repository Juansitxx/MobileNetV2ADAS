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

## 2026-09-22 — Labeling strategy finalized: ADAS ROI with per-category area thresholds

The Full Frame vs ADAS ROI comparison is closed for this experiment: **ADAS
ROI is the final, fixed frame-labeling strategy.** `scripts/02_build_manifest.py`
now writes the Common Manifest's `label`/`has_vehicle`/`has_pedestrian`/
`has_motorcycle` fields from the ROI classification (previously Full
Frame); `label_fullframe` and its counts/flags remain in the manifest for
reference/audit only, not for training.

- **ROI geometry unchanged:** `x_min=0.20, x_max=0.80, y_min=0.35,
  y_max=1.00`, `bbox_intersection_threshold=0.35`.
- **`min_bbox_area_ratio` is now per-category, not a single global value:**
  - `car`, `truck`, `bus`: `0.01`
  - `motorcycle`: `0.0005`
  - `pedestrian`: `0.0005`
  - `rider`, `bicycle` (not explicitly specified): fall back to `0.0005`
    (`ROIConfig.default_min_bbox_area_ratio`) as a conservative default,
    since these are similarly small objects to motorcycle/pedestrian — this
    is an engineering default, not a measured decision; revisit if evidence
    suggests otherwise.
- **Rationale for `motorcycle=0.0005`:** real analysis showed this
  threshold retains 145 motorcycle-containing frames in TRAIN and 27 in
  VAL, with minimal effect on the overall vehicle-class balance. This
  matters specifically because the eventual target domain is Colombian
  driving, where motorcycle density is expected to be materially higher
  than in BDD100K/US driving — under-representing motorcycles now would
  work against that future adaptation. `car`/`truck`/`bus` keep the larger
  `0.01` threshold because their real-world footprint is much bigger, so a
  detection that small is more likely to be noise/occlusion than a genuine
  distant vehicle.
- **Multi-label training target unchanged:** two independent binary
  targets, `has_vehicle` and `has_pedestrian`, still derive the four ADAS
  states as `(0,0)=CLEAR, (1,0)=VEHICLE, (0,1)=PEDESTRIAN, (1,1)=MIXED` via
  `neurodriver_cnn.labeling.frame_labels.derive_label`. Motorcycle stays
  folded into `has_vehicle`/VEHICLE, with `has_motorcycle`/
  `num_motorcycles` preserved as separate metadata for the dedicated
  motorcycle-subset evaluation slice.

Add a new dated entry for each subsequent material decision (class_weight
adoption, fine-tuning layer count, KD alpha/T choice, etc.).
