# Experiment status

Generated as part of the Phase 1 milestone, updated 2026-09-22 after real
dataset evidence arrived. This file reports real, observed status only —
see `docs/decisions_log.md` for dated decisions and `PHASE_1.md` for the
milestone definition.

## Environment

- Python 3.11.9, Windows.
- TensorFlow: **installed locally as of 2026-09-22** (specifically to
  verify the baseline CNN's parameter budget with real numbers instead of
  hand math — see `docs/decisions_log.md`).
- GPU: none locally (CPU-only TensorFlow); local target hardware per
  `CLAUDE.md` is an Intel i7 + RTX 4060, with initial training intended on
  Google Colab.
- Run `python scripts/00_check_environment.py` for a live check.

## Dataset availability

**Not present in this repository.** The real, confirmed source is
**BDD100K Images 10K** (DatasetNinja, Supervisely `.tar` format) — not the
official BDD100K 100k `box2d` release originally assumed in the first
Phase 1 pass. No unofficial mirror was used and no large archive was
downloaded automatically. See `docs/bdd100k_setup.md` for acquisition and
`docs/decisions_log.md` (2026-09-22) for the format-change rationale.

## Pipeline implementation status

| Component | Status |
|---|---|
| Repository structure / configs | DONE |
| BDD100K setup documentation | DONE (rewritten 2026-09-22 for the real Supervisely `.tar` source) |
| Environment checker | DONE (verified: reports MISSING correctly for TensorFlow/dataset archive) |
| Supervisely parser (`data/bdd100k.py`) | DONE — rewritten 2026-09-22 for polygon/rectangle geometry (`polygon_to_bbox`); verified end-to-end against a synthetic Supervisely `.tar` fixture |
| Schema inspection script | DONE — rewritten to read directly from the `.tar` (no extraction needed); verified against the synthetic fixture (confirmed empty `test` split, category/tag counts) |
| ROI geometry (`labeling/roi.py`) | DONE, unit-tested |
| Frame labeling (`labeling/frame_labels.py`) | DONE, unit-tested, including the final per-category `min_bbox_area_ratio` filter (car/truck/bus 0.01, motorcycle/pedestrian 0.0005) |
| Common Manifest schema/validation (`data/manifest.py`) | DONE, unit-tested; now reuses `derive_label` (deduplicated from a second copy found during this update) |
| Manifest builder script | DONE — rewritten to extract the `.tar` (idempotent) and build the manifest in one command; excludes the DatasetNinja `test` split; Common Manifest fields now sourced from ADAS ROI (not Full Frame); verified end-to-end against synthetic fixtures |
| Dataset audit script + figures | DONE, unchanged logic, re-verified against the synthetic fixture's manifest |
| Experimental subset builder + sampling plan | DONE, unchanged logic, re-verified against the synthetic fixture's manifest |
| Dataset validation script | DONE, re-verified (0 issues) against the synthetic fixture's manifests |
| Unit tests (ROI, frame labels, manifest, Supervisely parser, baseline model) | DONE — **40/40 passing** (`python -m pytest -q`); includes 2 TensorFlow-backed tests (`test_baseline_model.py`, skipped automatically without TF) |
| Notebook 00 (dataset exploration) | DONE — regenerated 2026-09-22 for the Supervisely source, multi-label framing, and the finalized ADAS ROI per-category thresholds; not executed (no real dataset present) |
| Notebook 01 (baseline + MobileNetV2 + ResNet50 Teacher) | DONE — regenerated 2026-09-22 for multi-label targets, the ResNet50 Teacher smoke test, and the baseline parameter-budget assertion; not executed end-to-end (no real dataset present), but every model-construction cell has now been verified standalone with real TensorFlow |
| Baseline CNN (`models/baseline.py`, Student 2) | DONE — outputs 2 logits with `BinaryCrossentropy(from_logits=True)`; **fixed 2026-09-22**: was ~12.94M params (~13x over the <1M budget) due to `Flatten` on an under-pooled 28x28x128 feature map — added an extra 4x4 pool + reduced Dense to 64 units; verified with real TensorFlow: **494,850 total params** |
| MobileNetV2 Student (`models/mobilenetv2.py`, Student 1) | DONE — 2-logit output, `(model, base_model)` return; verified with real TensorFlow: 2,260,546 total params (2,562 trainable, frozen backbone), `(batch,2)` logits, no NaNs |
| ResNet50 Teacher (`models/resnet50_teacher.py`) | DONE (new) — architecture only, matching 2-logit output space; verified with real TensorFlow: 23,591,810 total params (4,098 trainable, frozen backbone), `(batch,2)` logits, no NaNs; not trained |
| Evaluation utilities (`evaluation/metrics.py`) | DONE — added `sigmoid`, `labels_from_logits`, `multilabel_binary_metrics`, `estimate_flops`; implementation only, not run against real predictions |
| KD contracts/scaffolding | DONE — updated for the 2-logit Teacher/Student contract and the 5-way final comparison ladder; explicitly not implemented |
| Academic documentation | DONE — updated across all affected docs for the real dataset format, split exclusion, multi-label reformulation, and new Teacher/Student roster |

## Full Frame vs ROI

**Decided (2026-09-22): ADAS ROI is the final, fixed labeling strategy.**
Both strategies remain implemented and computed per-frame (Full Frame kept
for reference/audit only), with ROI now using **per-category**
`min_bbox_area_ratio` — car/truck/bus: 0.01, motorcycle/pedestrian: 0.0005
— chosen specifically because 0.0005 retains 145 TRAIN / 27 VAL motorcycle
frames with minimal effect on overall vehicle balance (real evidence),
which matters for the future Colombian domain. See `docs/decisions_log.md`.

## Class / motorcycle distribution

**Pending real data.** No fabricated numbers are reported here. Known real
counts so far are limited to record-level JSON stats from
`docs/bdd100k_setup.md` (train 7000/6891 with objects, val 1000/981, test
2000/0) — not yet a class distribution.

## Split / leakage validation

Logic implemented and unit-tested. The DatasetNinja `test` split is
excluded from manifest building entirely (no usable ground truth). This
dataset has no real group/sequence ID (independent frames), so group-aware
splitting falls back to its documented seeded-random behavior — verified
end-to-end against the synthetic Supervisely fixture, not yet against real
data.

## Real metrics

**Pending.** No model has been trained on real data (only architecture
construction, parameter counts, and forward-pass smoke tests on
zero-tensors have been verified — see the pipeline table above). No
training run yet, per instruction ("no hagas entrenamiento todavía").

## Blockers

1. The real BDD100K Images 10K `.tar` is not present locally (see
   `docs/bdd100k_setup.md`) — the only remaining blocker to a real training
   run.

## Manual user action required

1. (Done locally 2026-09-22, still needed in Colab) Install TensorFlow
   (`pip install tensorflow`, or run in Google Colab which has it
   preinstalled).
2. Download BDD100K Images 10K (DatasetNinja, Supervisely format) and place
   it at `data/raw/bdd100k/bdd100k_10k_supervisely.tar`, per
   `docs/bdd100k_setup.md`.
3. Re-run, in order: `scripts/00` through `scripts/05` (script 02 extracts
   the `.tar` automatically).

## Recommended next step

Once the `.tar` is placed: run `scripts/00_check_environment.py` through
`scripts/05_validate_dataset.py` in order, review
`reports/dataset_audit.md` + `reports/figures/`, record the Full Frame vs
ROI decision in `docs/decisions_log.md`, then open `notebooks/00` and `01`
to inspect the real data and run a short baseline training for honest
preliminary metrics (per-target and derived 4-state).
