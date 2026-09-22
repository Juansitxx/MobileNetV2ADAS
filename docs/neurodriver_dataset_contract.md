# Common Manifest contract

Training/evaluation code depends on this contract, not on BDD100K-specific
folders. Any future data source — most importantly a `NeuroDriverAdapter`
for Colombian dashcam data — must emit a manifest satisfying this same
schema so notebooks/training code require no changes.

## Required fields

| Field | Type | Notes |
|---|---|---|
| `image_path` | str | Path to the RGB frame. |
| `label` | str | One of `CLEAR`, `VEHICLE`, `PEDESTRIAN`, `MIXED`. |
| `split` | str | One of `TRAIN`, `VALIDATION`, `TEST`. |
| `source_dataset` | str | e.g. `BDD100K`, `NEURODRIVER_COLOMBIA`. |
| `has_vehicle` | bool | Derived from ADAS ROI (the final labeling strategy, fixed 2026-09-22). **Also the first of the two training targets** (see "Multi-label training" below). |
| `has_pedestrian` | bool | Derived from ADAS ROI. **Also the second training target.** |
| `has_motorcycle` | bool | Explicit even though motorcycle folds into VEHICLE. |

Validated by `neurodriver_cnn.data.manifest.validate_manifest_invariants`.

## Optional fields (present when available)

- `image_id`, `group_id` / `run_id` / `session_id` / `video_id` (never
  invented — absent when the source dataset doesn't provide one);
- `timestamp`;
- object counts (`num_vehicles`, `num_pedestrians`, `num_motorcycles`,
  `num_riders`, `num_bicycles`);
- environment attributes (`weather`, `scene`, `timeofday`);
- `label_source` — provenance of the label (see below);
- Teacher/KD metadata (see below).

## Multi-label training (updated 2026-09-22)

Models train directly on the two required boolean fields `has_vehicle` /
`has_pedestrian` as independent binary targets (`vehicle_logit`,
`pedestrian_logit`, `BinaryCrossentropy(from_logits=True)`), not on `label`
as a 4-way softmax target — real class counts showed `PEDESTRIAN` is too
imbalanced for a single softmax to represent well. `label` remains in the
manifest and is still the field used for the derived 4-state
CLEAR/VEHICLE/PEDESTRIAN/MIXED reporting (confusion matrix, F1,
presentation), computed the same way at both manifest-build time
(`neurodriver_cnn.labeling.frame_labels.derive_label`) and at model
evaluation time from thresholded predictions
(`neurodriver_cnn.evaluation.metrics.labels_from_logits`). See
`docs/decisions_log.md`.

## BDD100K-specific fields (adapter output, not part of the common contract)

`scripts/02_build_manifest.py` also writes BDD100K-specific columns
(`label_fullframe`, `label_roi`, per-strategy counts/flags, `raw_split`
(the DatasetNinja `train`/`val` split name — `test` is never loaded),
`image_available`, `annotation_available`, `valid_sample`) so Full Frame
remains available for reference/audit even though ADAS ROI is the fixed
training strategy (`label_roi` values are what `label`/`has_vehicle`/
`has_pedestrian`/`has_motorcycle` above are copied from). Common-contract
training code should only read the required/optional fields above, not
these adapter-specific ones.

## Future Knowledge Distillation fields (optional, not populated yet)

- `teacher_predictions_available` (bool)
- `teacher_logits_path` (str)
- `teacher_model_version` (str)
- `teacher_adapter_version` (str)

## `label_source` provenance values

- `BDD100K_GROUND_TRUTH`
- `NEURODRIVER_GROUND_TRUTH`
- `TEACHER_PSEUDO_LABEL`
- `MANUAL_REVIEW`

`TEACHER_PSEUDO_LABEL` rows are never treated as verified ground truth.

## Future `NeuroDriverAdapter`

Not implemented yet. When Colombian NeuroDriver data becomes available, it
must be converted to this same schema (a `NeuroDriverAdapter`, mirroring
`neurodriver_cnn.data.bdd100k`), including group-aware split by
run/session/video — never by adjacent random frames.
