# BDD100K setup

Status as of this Phase 1 scaffolding: **BDD100K is not present in this
repository.** No unofficial mirror is used or recommended; no large archive
is downloaded automatically.

## What to download

From the official BDD100K site (https://bdd-data.berkeley.edu/, login
required) or the official BDD100K download portal, obtain:

1. **Images (100K set)** — the `bdd100k_images_100k.zip` package (train +
   val splits). This project does not need the `test` split for Phase 1
   (no labels are published for it) and does not need the 10K/tracking/
   segmentation/lane packages.
2. **Object detection labels** — the detection-2020 label JSON package
   (commonly `bdd100k_labels_release.zip`, containing
   `bdd100k_labels_images_train.json` and `bdd100k_labels_images_val.json`).
   This project does not need segmentation masks, lane masks, drivable-area
   masks, or tracking annotations for Phase 1.

## Where to place it

Extract so the layout matches:

```text
data/raw/bdd100k/
├── images/
│   └── 100k/
│       ├── train/   # *.jpg
│       └── val/     # *.jpg
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

`data/raw/` is treated as **read-only** by every script and notebook in
this repository — nothing here writes into it.

## Verification

After placing the files, run:

```bash
python scripts/00_check_environment.py
```

Expect `[OK] BDD100K image directories found` and
`[OK] BDD100K annotation file(s) found`. Then run:

```bash
python scripts/01_inspect_bdd100k.py
```

to confirm the real annotation schema (record count, category names, bbox
format, available metadata) before manifest generation, and check
`reports/schema_analysis.md`.

## If you cannot download BDD100K yet

Everything else in this repository (code, tests, docs, notebook
scaffolding) has been prepared without requiring real image data. Once the
files above are placed, re-run, in order:

```bash
python scripts/00_check_environment.py
python scripts/01_inspect_bdd100k.py
python scripts/02_build_manifest.py
python scripts/03_analyze_manifest.py
python scripts/04_build_experiment_subset.py
python scripts/05_validate_dataset.py
```
