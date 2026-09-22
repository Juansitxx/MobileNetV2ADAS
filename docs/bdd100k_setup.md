# BDD100K setup

**Updated 2026-09-22** — the dataset actually available for this project is
**BDD100K Images 10K**, redistributed by DatasetNinja in **Supervisely
format**, not the official BDD100K 100k `box2d` label release originally
assumed. See `docs/decisions_log.md` for why (real inspection of the
acquired data).

## What to download

From DatasetNinja's BDD100K Images 10K page, download the dataset export as
a single `.tar` archive in Supervisely format.

## Where to place it

```text
data/raw/bdd100k/
└── bdd100k_10k_supervisely.tar
```

`data/raw/` is treated as **read-only** — nothing writes into
`bdd100k_10k_supervisely.tar` itself. `scripts/02_build_manifest.py`
extracts it once (idempotent, marked with a `.extracted` file) into:

```text
data/raw/bdd100k/extracted/
├── train/
│   ├── ann/*.jpg.json
│   └── img/*.jpg
├── val/
│   ├── ann/*.jpg.json
│   └── img/*.jpg
└── test/
    ├── ann/*.jpg.json   # present but empty of objects — never loaded
    └── img/*.jpg
```

(An extra wrapping folder inside the tar is tolerated —
`neurodriver_cnn.data.bdd100k.find_split_dir` searches recursively for
`train`/`val`/`test` directories rather than assuming an exact nesting
depth.)

## Annotation format (Supervisely, not official BDD100K box2d)

Real inspection (`scripts/01_inspect_bdd100k.py`, `reports/schema_analysis.md`)
showed per-image JSON files with **polygon** (occasionally rectangle)
geometry, not `box2d`:

```json
{
  "size": {"width": 1280, "height": 720},
  "tags": [{"name": "weather", "value": "clear"}],
  "objects": [
    {"classTitle": "car", "geometryType": "polygon",
     "points": {"exterior": [[x, y], ...], "interior": []}}
  ]
}
```

`neurodriver_cnn.data.bdd100k.polygon_to_bbox` derives an axis-aligned
bounding box from `points.exterior` via `x1=min(x), y1=min(y), x2=max(x),
y2=max(y)`. This project still does not train an object detector — the
derived box is only used to compute frame-level labels.

## The DatasetNinja `test` split has no usable ground truth

Real counts (see `docs/decisions_log.md`, 2026-09-22):

| Split | JSON records | Records with >=1 object |
|---|---|---|
| train | 7000 | 6891 |
| val | 1000 | 981 |
| test | 2000 | **0** |

`test` is therefore **never loaded** for manifest building or evaluation.
The Common Manifest is built from `train`/`val` only:
raw `train` -> internal TRAIN + VALIDATION, raw `val` -> academic TEST.

## No real sequence/group identifier

BDD100K Images 10K consists of independent frames, not tracking sequences —
there is no real video/run ID to preserve. `group_id` is always `None` in
this dataset (never invented); the group-aware split falls back to a seeded
random per-row split (see `neurodriver_cnn.data.manifest.group_aware_split`).

## Verification / build commands

```bash
python scripts/00_check_environment.py     # expect [OK] for the .tar or its extracted form
python scripts/01_inspect_bdd100k.py       # reads the .tar directly, writes reports/schema_analysis.md
python scripts/02_build_manifest.py        # extracts (once) + builds data/processed/manifests/bdd100k_manifest.csv
python scripts/03_analyze_manifest.py
python scripts/04_build_experiment_subset.py
python scripts/05_validate_dataset.py
```

`scripts/02_build_manifest.py` accepts `--tar PATH` / `--extract-dir PATH`
if the archive is not at the default location above.
