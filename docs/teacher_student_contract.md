# Teacher-Student contract

Status: **contract only — no Teacher model has been trained and no real KD exists yet.**

**Updated 2026-09-22:** Teacher/Student output space changed from 4-class
softmax to two independent sigmoid logits — see
`docs/decisions_log.md` for why (severe PEDESTRIAN class imbalance found in
real data).

## Roles

- **Teacher**: ResNet50, ImageNet-pretrained
  (`neurodriver_cnn.models.resnet50_teacher.build_resnet50_teacher`). Used
  only during training.
- **Student 1**: MobileNetV2
  (`neurodriver_cnn.models.mobilenetv2.build_mobilenetv2_student`).
- **Student 2**: the lightweight custom CNN
  (`neurodriver_cnn.models.baseline.build_baseline_cnn`) — same
  architecture used as the academic baseline, now also framed as a KD
  Student.

Used alone at deployment — the Teacher is absent from final inference.

## Shared output space

All three architectures output the **same two raw logits**:
`vehicle_logit`, `pedestrian_logit` (a `Dense(2, name="logits")` layer, no
activation). Since Teacher and both Students already share this space, no
`NeuroDriverTeacherAdapter`-style remapping layer is needed for this
in-repo ResNet50 Teacher — an adapter would only become necessary if a
different, external Teacher with an incompatible output space were
introduced later.

The 4-state ADAS label (CLEAR/VEHICLE/PEDESTRIAN/MIXED) is derived, not
trained directly: `sigmoid(logit) >= threshold` per target, then
`neurodriver_cnn.labeling.frame_labels.derive_label` — see
`neurodriver_cnn.evaluation.metrics.labels_from_logits`.

## Conceptual Teacher batch output (not implemented)

```python
{
    "vehicle_logit": ...,      # shape (batch,), Teacher's own scale
    "pedestrian_logit": ...,   # shape (batch,)
    "teacher_version": ...,
    "adapter_version": ...,
}
```

## Final comparison ladder (Phase 2+, not run yet)

ResNet50 Teacher, MobileNetV2 Base, MobileNetV2 Distilled, lightweight CNN
Base, lightweight CNN Distilled — see
`docs/future_knowledge_distillation.md`.

## Non-fabrication

No fake Teacher logits, no fake Teacher model, and no fake KD training are
implemented anywhere in this repository. Every KD-related field/module here
is scaffolding for Phase 2+, gated behind `configs/training_config.json` →
`distillation.enabled = false`.
