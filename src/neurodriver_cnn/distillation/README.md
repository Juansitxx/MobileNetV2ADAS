# Distillation scaffolding

Status: **not implemented**. This package is a placeholder for the future
Teacher (NeuroDriver) → Student (MobileNetV2) Knowledge Distillation work
described in `docs/teacher_student_contract.md` and
`docs/future_knowledge_distillation.md`.

Planned components (none exist yet):

- `teacher_adapter.py` — `NeuroDriverTeacherAdapter`, mapping real
  NeuroDriver Teacher outputs to the Student's 4-class logit/probability
  space.
- `cache.py` — loading/caching precomputed Teacher predictions
  (`teacher_logits_path` in the Common Manifest) so the Teacher does not
  need to run during every Student training step.
- `losses.py` — the temperature-scaled KL-divergence distillation loss:

  ```text
  teacher_soft = softmax(teacher_logits / T)
  student_soft = softmax(student_logits / T)
  L_total = alpha * L_supervised + (1 - alpha) * KL(teacher_soft || student_soft)
  ```

- `distiller.py` — a custom Keras `Model`/training loop combining
  `L_supervised` and `L_distillation`.

`alpha` and temperature `T` are experimental hyperparameters (see
`configs/training_config.json` → `distillation`, currently `enabled: false`).

Do not add fake Teacher logits or a fake training loop to make this package
"look done" — KD is only implemented once a real NeuroDriver Teacher is
available.
