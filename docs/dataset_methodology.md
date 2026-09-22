# Dataset methodology

## Source data

**BDD100K Images 10K**, redistributed by DatasetNinja in **Supervisely
format** (`.tar` of per-image polygon annotations) — not the official
BDD100K 100k `box2d` release originally assumed (see
`docs/decisions_log.md`, 2026-09-22). This is a **provisional SOURCE
DOMAIN**, chosen for its scale and diversity, not because it resembles
Colombian driving. See `docs/bdd100k_setup.md` for acquisition and
`docs/colombian_domain_strategy.md` for the future target domain.

## From polygons to frame labels

This is a frame-level classification project, not an object-detection
project. Supervisely polygon/rectangle annotations are converted to
axis-aligned bounding boxes (`x1=min(x), y1=min(y), x2=max(x), y2=max(y)`)
only to derive a single label per frame:

```text
BDD100K Images 10K frame + Supervisely polygon annotations
-> polygon_to_bbox (derived axis-aligned box)
-> category mapping (vehicle / pedestrian / auxiliary)
-> labeling strategy (Full Frame or ADAS ROI)
-> frame label (CLEAR / VEHICLE / PEDESTRIAN / MIXED) as has_vehicle/has_pedestrian
-> Common Manifest row
-> CNN (RGB image in, two binary targets out — see "Multi-label reformulation" below)
```

Parsing lives in `src/neurodriver_cnn/data/bdd100k.py`; category mapping and
label derivation live in `src/neurodriver_cnn/labeling/`.

## Excluded split: DatasetNinja `test`

Real inspection found `test` has zero annotated objects in all 2000
records (train: 6891/7000 with objects, val: 981/1000). It carries no
usable ground truth and is never loaded by
`scripts/02_build_manifest.py`. See `docs/bdd100k_setup.md`.

## Multi-label reformulation (not 4-class softmax)

Real class counts showed severe imbalance for pure `PEDESTRIAN` frames.
Models are trained on two independent binary targets — `has_vehicle`,
`has_pedestrian` — via `BinaryCrossentropy(from_logits=True)` on two raw
logits (`vehicle_logit`, `pedestrian_logit`), not a 4-way softmax. The four
ADAS states are derived post-hoc from thresholded sigmoid probabilities
(`neurodriver_cnn.evaluation.metrics.labels_from_logits`; threshold
configurable, default 0.5) for confusion-matrix/F1/presentation purposes.
See `docs/decisions_log.md` (2026-09-22).

## Category mapping

- `vehicle`: `car`, `truck`, `bus`, `motorcycle`
- `pedestrian`: `pedestrian`
- `auxiliary` (excluded from vehicle/pedestrian): `rider`, `bicycle`

These are the BDD100K `classTitle` category names as re-exported by
DatasetNinja; `scripts/01_inspect_bdd100k.py` re-confirms them against the
real annotation archive before manifest generation, and
`reports/schema_analysis.md` records
what was actually found.

## Labeling strategy: ADAS ROI (final, fixed 2026-09-22)

**Full Frame** — any mapped object anywhere in the image counts. Still
computed and stored (`label_fullframe` and its counts/flags) for
reference/audit, but no longer used to build the Common Manifest's training
fields.

**ADAS ROI — the strategy actually used for training** (see
`docs/decisions_log.md`, 2026-09-22): only objects relevant to the forward
driving corridor count, additionally excluding objects smaller than a
**per-category** minimum bbox-area ratio (`configs/dataset_config.json` →
`roi.min_bbox_area_ratio_by_category`):

| Category | min_bbox_area_ratio |
|---|---|
| car, truck, bus | 0.01 |
| motorcycle | 0.0005 |
| pedestrian | 0.0005 |
| rider, bicycle (fallback default) | 0.0005 |

`motorcycle=0.0005` was chosen because it retains 145 TRAIN / 27 VAL
motorcycle-containing frames with minimal effect on overall vehicle
balance — important for the future Colombian domain, where motorcycle
density is expected to be materially higher than in BDD100K.
`car`/`truck`/`bus` use a larger 0.01 threshold since their real-world
footprint is much bigger, making a smaller detection more likely to be
noise. Both are implemented in
`neurodriver_cnn.labeling.frame_labels.classify_fullframe`/`classify_roi`
and computed for every frame; only the ROI columns feed
`label`/`has_vehicle`/`has_pedestrian`/`has_motorcycle` in the Common
Manifest.

## Common Manifest

Training code depends on a Common Manifest (see
`docs/neurodriver_dataset_contract.md`), not on BDD100K-specific folder
structure. `data/raw/` is treated as read-only; nothing is copied into
per-class folders.

## Splits and leakage prevention

- Split happens before augmentation; augmentation is applied to TRAIN only.
- TEST is never used for tuning.
- Splitting is group-aware: when a real BDD100K video/run identifier is
  available, all frames from that group stay on one side of the split.
  Group IDs are never invented.
- Preferred BDD scheme: official TRAIN → internal TRAIN + VALIDATION;
  official VAL → academic TEST.
- Seed = 42 throughout.

## Experimental subset

A reproducible subset of roughly 4,000-8,000 samples (preferring ~8,000) is
built from the full manifest for faster iteration
(`scripts/04_build_experiment_subset.py`, `reports/sampling_plan.md`). Class
balance is reported honestly, not fabricated; the sampling seed and
configuration are saved alongside the subset for reproducibility.
