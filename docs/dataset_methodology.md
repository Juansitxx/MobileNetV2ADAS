# Dataset methodology

## Source data

BDD100K (Berkeley DeepDrive) object-detection annotations + images. This is
a **provisional SOURCE DOMAIN**, chosen for its scale and diversity, not
because it resembles Colombian driving. See `docs/bdd100k_setup.md` for
acquisition and `docs/colombian_domain_strategy.md` for the future target
domain.

## From bounding boxes to frame labels

This is a frame-level classification project, not an object-detection
project. BDD100K's official object bounding boxes are used only to derive a
single label per frame:

```text
BDD100K image + official object annotations
-> category mapping (vehicle / pedestrian / auxiliary)
-> labeling strategy (Full Frame or ADAS ROI)
-> frame label (CLEAR / VEHICLE / PEDESTRIAN / MIXED)
-> Common Manifest row
-> CNN (RGB image in, label out)
```

Parsing lives in `src/neurodriver_cnn/data/bdd100k.py`; category mapping and
label derivation live in `src/neurodriver_cnn/labeling/`.

## Category mapping

- `vehicle`: `car`, `truck`, `bus`, `motorcycle`
- `pedestrian`: `pedestrian`
- `auxiliary` (excluded from vehicle/pedestrian): `rider`, `bicycle`

These are the official BDD100K category names as documented publicly;
`scripts/01_inspect_bdd100k.py` re-confirms them against the real annotation
file before manifest generation, and `reports/schema_analysis.md` records
what was actually found.

## Two labeling strategies

**Full Frame** — any mapped object anywhere in the image counts.
**ADAS ROI** — only objects relevant to a configurable forward-driving
corridor count (see `configs/dataset_config.json` → `roi`).

Both are implemented and both are computed for every frame in the manifest
so they can be compared with real statistics (`reports/dataset_audit.md`)
before a permanent strategy is chosen (`docs/decisions_log.md`).

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
