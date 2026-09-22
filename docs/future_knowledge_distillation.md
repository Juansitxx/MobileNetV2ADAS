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

```text
teacher_soft = softmax(teacher_logits / T)
student_soft = softmax(student_logits / T)

L_total = alpha * L_supervised + (1 - alpha) * KL(teacher_soft || student_soft)
```

`alpha` and temperature `T` are experimental hyperparameters, not fixed
truths — `configs/training_config.json` keeps them `null` while
`distillation.enabled = false`.

## Planned experiment ladder (for later ablation)

- **A**: simple CNN baseline.
- **B**: MobileNetV2 + BDD100K supervised training.
- **C**: MobileNetV2 + Colombian fine-tuning.
- **D**: MobileNetV2 + Colombian fine-tuning + NeuroDriver Teacher KD.

Student architecture, data splits, and evaluation conditions are kept
comparable across A-D so any measured gain is attributable to the technique
being added, not to a confound. KD does not itself shrink the model; the
Student architecture is already compact and KD only transfers knowledge
into it.

## Evaluation once KD exists

Same utilities as the rest of this project
(`src/neurodriver_cnn/evaluation/metrics.py`): Accuracy, Precision, Recall,
F1 (macro/weighted), confusion matrix, motorcycle subset, parameter count,
model size, inference latency — compared across A-D.
