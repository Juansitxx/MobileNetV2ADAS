# data/

- `raw/` — official, unmodified source data (BDD100K). **Read-only.** Never
  written to by any script. See `docs/bdd100k_setup.md`.
- `interim/` — intermediate artifacts that are not final manifests (e.g.
  schema-inspection dumps). Safe to delete and regenerate.
- `processed/manifests/` — the Common Manifest CSVs consumed by training
  code (`bdd100k_manifest.csv`, `experiment_manifest.csv`). See
  `docs/neurodriver_dataset_contract.md`.

Nothing under `raw/` or `interim/` is committed to Git (see `.gitignore`).
