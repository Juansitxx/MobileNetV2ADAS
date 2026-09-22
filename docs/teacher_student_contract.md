# Teacher-Student contract

Status: **contract only — no Teacher model or real KD exists yet.**

## Roles

- **Teacher**: NeuroDriver, or a NeuroDriver-derived high-capacity
  perception model. Used only during training.
- **Student**: the MobileNetV2 architecture defined in
  `src/neurodriver_cnn/models/mobilenetv2.py`. Used alone at deployment —
  the Teacher is absent from final inference.

The Student's output space is fixed: `CLEAR`, `VEHICLE`, `PEDESTRIAN`,
`MIXED`. The Teacher may not naturally output this same space, so a future
`NeuroDriverTeacherAdapter` will map real NeuroDriver Teacher outputs into
compatible Teacher logits/probabilities before any distillation loss is
computed.

## Conceptual Teacher batch output (not implemented)

```python
{
    "logits": ...,           # shape (batch, 4), Teacher's own scale
    "probabilities": ...,    # softmax(logits)
    "teacher_version": ...,
    "adapter_version": ...,
}
```

## Student logits access

`neurodriver_cnn.models.mobilenetv2.build_mobilenetv2_student` returns
`(full_model, logits_model, base_model)`. `logits_model` exposes pre-Softmax
logits directly — required so a future KL-divergence distillation loss
does not need to rebuild or re-wrap the network.

## Non-fabrication

No fake Teacher logits, no fake Teacher model, and no fake KD training are
implemented anywhere in this repository. Every KD-related field/module here
is scaffolding for Phase 2+, gated behind `configs/training_config.json` →
`distillation.enabled = false`.
