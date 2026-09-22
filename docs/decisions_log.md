# Decisions log

Chronological record of material technical decisions and their rationale.

## 2026-09-22 — Phase 1 scaffolding

- **Repository structure and Common Manifest architecture created** following
  `PHASE_1.md` / `PROJECT_SPEC.md`. Rationale: training code must not depend
  on BDD100K-specific folders so a future Colombian NeuroDriver adapter can
  plug in unchanged.
- **Full Frame vs ADAS ROI: NOT finalized.** Both strategies are implemented
  and both are computed into every manifest row
  (`label_fullframe`/`label_roi`). A permanent choice is deferred until real
  BDD100K distributions and visual examples exist (`reports/dataset_audit.md`,
  `reports/figures/`). Rationale: `PHASE_1.md` explicitly requires evidence
  before this choice, not an a-priori preference.
- **Default `label`/`has_*` columns in the manifest currently mirror the
  Full Frame strategy** (`scripts/02_build_manifest.py`). This is a
  placeholder default for the Common Manifest contract, not the final
  decision — revisit once the audit above is run on real data.
- **BDD100K local environment: not available.** No TensorFlow install and
  no BDD100K data are present in this development environment. All
  code/tests/docs were built to work correctly in that state (graceful
  MISSING/PENDING reporting) and to work correctly once real data and
  TensorFlow are added, without code changes.
- **BatchNormalization kept frozen during future fine-tuning**, even after
  unfreezing late MobileNetV2 layers (`unfreeze_for_fine_tuning`).
  Rationale: BDD/Colombian fine-tuning batches are expected to be small and
  non-representative of the full training distribution, which would corrupt
  BatchNorm running statistics if left trainable.

Add a new dated entry for each subsequent material decision (label strategy
finalization, class_weight adoption, fine-tuning layer count, KD alpha/T
choice, etc.).
