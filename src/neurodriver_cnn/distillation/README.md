# Distillation scaffolding

Status: **not implemented**. This package is a placeholder for the future
Teacher (ResNet50) -> Student Knowledge Distillation work described in
`docs/teacher_student_contract.md` and `docs/future_knowledge_distillation.md`.

## Roles (updated 2026-09-22 — multi-label reformulation)

- **Teacher**: ResNet50, ImageNet-pretrained (`neurodriver_cnn.models.resnet50_teacher`).
- **Student 1**: MobileNetV2 (`neurodriver_cnn.models.mobilenetv2`).
- **Student 2**: the lightweight custom CNN (`neurodriver_cnn.models.baseline`).

All three output **two raw logits** — `vehicle_logit`, `pedestrian_logit` —
not a 4-way softmax, so Teacher and Student logits are directly comparable
without an adapter layer. The 4-state ADAS label is derived post-hoc from
thresholded sigmoid probabilities
(`neurodriver_cnn.evaluation.metrics.labels_from_logits`).

Final experiment ladder to compare (see `docs/future_knowledge_distillation.md`):
ResNet50 Teacher, MobileNetV2 Base, MobileNetV2 Distilled, lightweight CNN
Base, lightweight CNN Distilled.

## Planned components (none exist yet)

- `teacher_adapter.py` — only needed if a *different* NeuroDriver Teacher
  (beyond the in-repo ResNet50) requires output-space mapping.
- `cache.py` — loading/caching precomputed Teacher predictions
  (`teacher_logits_path` in the Common Manifest) so the Teacher does not
  need to run during every Student training step.
- `losses.py` — the temperature-scaled KL-divergence distillation loss,
  computed per binary target (vehicle, pedestrian):

  ```text
  teacher_soft = sigmoid(teacher_logit / T)
  student_soft = sigmoid(student_logit / T)
  L_total = alpha * L_supervised + (1 - alpha) * KL(teacher_soft || student_soft)
  ```

- `distiller.py` — a custom Keras `Model`/training loop combining
  `L_supervised` (BinaryCrossentropy per target) and `L_distillation`.

`alpha` and temperature `T` are experimental hyperparameters (see
`configs/training_config.json` -> `distillation`, currently `enabled: false`).

Do not add fake Teacher logits or a fake training loop to make this package
"look done" — KD is only implemented once ResNet50 has been trained as a
real Teacher on real data.
