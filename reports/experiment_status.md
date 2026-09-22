# Experiment status

Generated as part of the Phase 1 milestone. This file reports real,
observed status only — see `docs/decisions_log.md` for dated decisions and
`PHASE_1.md` for the milestone definition.

## Environment

- Python 3.11.9, Windows.
- TensorFlow: **not installed locally** (`pip install tensorflow` per
  `requirements.txt` to enable notebooks/model code).
- GPU: not checked (requires TensorFlow); local target hardware per
  `CLAUDE.md` is an Intel i7 + RTX 4060, with initial training intended on
  Google Colab.
- Run `python scripts/00_check_environment.py` for a live check.

## BDD100K data

**Not present in this repository.** No unofficial mirror was used and no
large archive was downloaded automatically, per `CLAUDE.md`/`PHASE_1.md`.
See `docs/bdd100k_setup.md` for the official download and expected
placement.

## Pipeline implementation status

| Component | Status |
|---|---|
| Repository structure / configs | DONE |
| BDD100K setup documentation | DONE |
| Environment checker | DONE (verified: reports MISSING correctly for TensorFlow/BDD100K) |
| BDD100K parser (`data/bdd100k.py`) | DONE (schema assumptions documented; to be reconfirmed by `scripts/01_inspect_bdd100k.py` against real data) |
| Schema inspection script | DONE (verified: reports MISSING and writes a pending `reports/schema_analysis.md` correctly) |
| ROI geometry (`labeling/roi.py`) | DONE, unit-tested |
| Frame labeling (`labeling/frame_labels.py`) | DONE, unit-tested |
| Common Manifest schema/validation (`data/manifest.py`) | DONE, unit-tested |
| Manifest builder script | DONE (verified: reports MISSING correctly; logic exercised with a synthetic non-committed fixture) |
| Dataset audit script + figures | DONE (verified: pending-report path and full audit/figures path both exercised with a synthetic non-committed fixture) |
| Experimental subset builder + sampling plan | DONE (verified: pending path and sampling logic both exercised with a synthetic non-committed fixture; pandas 3.0 `groupby.apply` column-drop issue found and fixed) |
| Dataset validation script | DONE (verified against real file-existence and manifest-invariant checks) |
| Unit tests (ROI, frame labels, manifest) | DONE — **28/28 passing** (`python -m pytest -q`) |
| Notebook 00 (dataset exploration) | DONE (scaffolded with all required academic sections; not executed, since no real BDD100K data is present) |
| Notebook 01 (baseline + MobileNetV2) | DONE (scaffolded with all required sections, DONE/PENDING status cells; not executed, since TensorFlow/data are unavailable locally) |
| Baseline CNN (`models/baseline.py`) | DONE (implementation only — no training run, no TensorFlow locally) |
| MobileNetV2 Student (`models/mobilenetv2.py`) | DONE (implementation only, logits exposed for future KD — no forward pass run, no TensorFlow locally) |
| Evaluation utilities (`evaluation/metrics.py`) | DONE (implementation only — not run against real predictions yet) |
| KD contracts/scaffolding | DONE (documentation + `distillation/` package placeholder; explicitly not implemented) |
| Academic documentation | DONE (this file plus all files listed in `PHASE_1.md` Section "Academic documentation") |

## Full Frame vs ROI

**Pending.** Both strategies are implemented and computed per-frame, but no
real distribution/example comparison has been run (requires BDD100K data).
See `docs/decisions_log.md`.

## Class / motorcycle distribution

**Pending real data.** No fabricated numbers are reported here.

## Split / leakage validation

Logic implemented and unit-tested (`tests/test_manifest.py`:
`test_group_aware_split_keeps_group_together`,
`test_check_split_overlap_detects_leaked_group`, etc.). Not yet run against
a real BDD100K manifest.

## Real metrics

**Pending.** No model has been trained on real data in this environment
(no TensorFlow installed, no BDD100K present).

## Blockers

1. TensorFlow is not installed locally (`pip install tensorflow`).
2. BDD100K images + annotations are not present locally (see
   `docs/bdd100k_setup.md`).

## Manual user action required

1. Install TensorFlow (`pip install tensorflow`, or run in Google Colab
   which has it preinstalled).
2. Download BDD100K (images 100k + object-detection labels) from the
   official source and place under `data/raw/bdd100k/` as documented in
   `docs/bdd100k_setup.md`.
3. Re-run, in order: `scripts/00` through `scripts/05`.

## Recommended next step

Once BDD100K is placed: run `scripts/00_check_environment.py` through
`scripts/05_validate_dataset.py` in order, review
`reports/dataset_audit.md` + `reports/figures/`, record the Full Frame vs
ROI decision in `docs/decisions_log.md`, then open `notebooks/00` and `01`
to inspect the real data and run a short baseline training for honest
preliminary metrics.
