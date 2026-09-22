# Future Knowledge Distillation

Status: **documented only — not implemented.** See
`src/neurodriver_cnn/distillation/README.md` for the code-level scaffolding
and `docs/teacher_student_contract.md` for the Teacher/Student roles.

## Why KD, and why now is too early

KD is a core future technique (alongside Colombian fine-tuning), not an
optional add-on: it is how knowledge from a high-capacity NeuroDriver
Teacher gets transferred into the compact MobileNetV2 Student without
growing the Student. It requires a real Teacher model producing real
predictions on real Colombian data — none of which exists yet in this
repository. Building a fake KD loop now would produce meaningless numbers
that could later be mistaken for evidence.

## Planned loss

**Updated 2026-09-22:** computed per binary target (vehicle, pedestrian),
not over a 4-way softmax — see `docs/decisions_log.md` (real data showed
severe PEDESTRIAN class imbalance, which motivated the multi-label
reformulation of both Teacher and Students).

```text
teacher_soft = sigmoid(teacher_logit / T)   # per target: vehicle, pedestrian
student_soft = sigmoid(student_logit / T)

L_total = alpha * L_supervised + (1 - alpha) * KL(teacher_soft || student_soft)
```

`alpha` and temperature `T` are experimental hyperparameters, not fixed
truths — `configs/training_config.json` keeps them `null` while
`distillation.enabled = false`.

## Planned experiment ladder (for later ablation)

Teacher/Students updated 2026-09-22 (`docs/decisions_log.md`):

- **Teacher**: ResNet50 (`neurodriver_cnn.models.resnet50_teacher`).
- **Student 1**: MobileNetV2.
- **Student 2**: the lightweight custom CNN (existing baseline architecture).

Final comparison:

- ResNet50 Teacher
- MobileNetV2 Base (supervised only)
- MobileNetV2 Distilled (Colombian fine-tuning + ResNet50 Teacher KD)
- Lightweight CNN Base (supervised only)
- Lightweight CNN Distilled (Colombian fine-tuning + ResNet50 Teacher KD)

Data splits and evaluation conditions are kept comparable across all five
so any measured gain is attributable to the technique being added, not to a
confound. KD does not itself shrink a model; each Student architecture is
already compact and KD only transfers knowledge into it.

## Evaluation once KD exists

Same utilities as the rest of this project
(`src/neurodriver_cnn/evaluation/metrics.py`): Accuracy, Precision, Recall,
F1, the 4-state confusion matrix (derived from the two binary targets —
see `docs/teacher_student_contract.md`), motorcycle subset, parameter
count, estimated FLOPs, model size, inference latency — compared across all
five architectures.
