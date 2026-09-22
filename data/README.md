# data/

- `raw/` — the original Supervisely `.tar`
  (`bdd100k/bdd100k_10k_supervisely.tar`), read-only, and its extracted
  form (`bdd100k/extracted/`, auto-generated/idempotent by
  `scripts/02_build_manifest.py`). Never written to by hand. See
  `docs/bdd100k_setup.md`.
- `interim/` — intermediate artifacts that are not final manifests. Safe to
  delete and regenerate.
- `processed/manifests/` — the Common Manifest CSVs consumed by training
  code (`bdd100k_manifest.csv`, `experiment_manifest.csv`). See
  `docs/neurodriver_dataset_contract.md`.

Nothing under `raw/` or `interim/` is committed to Git (see `.gitignore`).
