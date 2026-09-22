# NeuroDriver CNN ADAS Colombia

**Academic research prototype.** This is not a production autonomous-driving
system and does not control steering, throttle, braking, CAN, or any other
vehicle actuator. It performs visual perception/alert classification only.

Scoped down from the broader NeuroDriver ecosystem to a single problem:
frame-level, 4-class ADAS visual perception, built as a bridge toward
Colombian driving conditions.

## Research progression

```text
ImageNet MobileNetV2
-> BDD100K source-domain learning
-> NeuroDriver Colombian dashcam data
-> Colombian fine-tuning
-> Knowledge Distillation: NeuroDriver Teacher -> MobileNetV2 Student
-> comparative evaluation
-> Student-only web perception service
```

**Current milestone (Phase 1):** dataset pipeline, Common Manifest,
notebooks, CNN baseline, MobileNetV2 Student (with KD-ready logits),
evaluation utilities, and academic documentation. Full MobileNetV2
training, Colombian fine-tuning, real KD, and web deployment are later
phases — see `reports/experiment_status.md` for exact current status.

## Classification task

Four ADAS states, frame-level (not object detection):

- `CLEAR = 0` — no relevant vehicle and no relevant pedestrian
- `VEHICLE = 1` — >=1 relevant vehicle, no relevant pedestrian
- `PEDESTRIAN = 2` — >=1 relevant pedestrian, no relevant vehicle
- `MIXED = 3` — >=1 relevant vehicle and >=1 relevant pedestrian

**Trained as multi-label, not 4-class softmax:** real data showed severe
imbalance for pure `PEDESTRIAN` frames, so models predict two independent
binary targets (`has_vehicle`, `has_pedestrian`, i.e. `vehicle_logit` /
`pedestrian_logit` with `BinaryCrossentropy(from_logits=True)`), and the
four states above are derived post-hoc from thresholded probabilities for
metrics/reporting. See `docs/decisions_log.md` (2026-09-22).

Motorcycles count as `VEHICLE` but remain explicitly tracked via
`has_motorcycle`/`num_motorcycles` for a dedicated evaluation slice.

## Getting started

```bash
pip install -r requirements.txt
python scripts/00_check_environment.py
```

BDD100K Images 10K (DatasetNinja, Supervisely format `.tar`) is **not
bundled** with this repository. See `docs/bdd100k_setup.md` for acquisition
and expected placement at
`data/raw/bdd100k/bdd100k_10k_supervisely.tar`. Once placed:

```bash
python scripts/01_inspect_bdd100k.py       # reads the .tar directly, no extraction needed
python scripts/02_build_manifest.py        # extracts (once) + builds the manifest, one command
python scripts/03_analyze_manifest.py
python scripts/04_build_experiment_subset.py
python scripts/05_validate_dataset.py
```

Then open `notebooks/00_dataset_exploration.ipynb` and
`notebooks/01_cnn_baseline_mobilenetv2.ipynb` (designed for Google Colab,
portable locally).

## Running tests

```bash
python -m pytest -q
```

## Repository layout

```text
configs/        dataset/training configuration (seed, ROI, class mapping, ...)
data/           raw/ (read-only, gitignored) + interim/ + processed/manifests/
docs/           methodology, contracts, strategy, and decision documentation
notebooks/      00 dataset exploration, 01 baseline + MobileNetV2
reports/        audit/schema/sampling reports, figures, experiment status
scripts/        numbered pipeline steps (00 environment check -> 05 validation)
src/neurodriver_cnn/  data parsing, labeling, models, evaluation, distillation scaffolding
tests/          unit tests for ROI, frame labeling, and manifest logic
```

## Key documents

- `docs/project_scope.md` — in/out of scope.
- `docs/dataset_methodology.md` — labeling strategies, splits, leakage rules.
- `docs/neurodriver_dataset_contract.md` — the Common Manifest schema.
- `docs/colombian_domain_strategy.md` — planned source→target adaptation.
- `docs/teacher_student_contract.md` / `docs/future_knowledge_distillation.md` — KD design (not implemented).
- `docs/decisions_log.md` — dated technical decisions and rationale.
- `reports/experiment_status.md` — current honest status, blockers, next step.
