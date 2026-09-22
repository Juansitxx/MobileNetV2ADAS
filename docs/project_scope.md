# Project scope

## What this is

An academic research prototype: a compact CNN-based visual perception
component for an ADAS (Advanced Driver Assistance System), scoped down from
the broader NeuroDriver ecosystem, targeting eventual Colombian driving
conditions.

## Research progression

```text
ImageNet MobileNetV2
-> BDD100K source-domain learning
-> NeuroDriver Colombian dashcam data
-> Colombian fine-tuning
-> Knowledge Distillation: NeuroDriver Teacher -> MobileNetV2 Student
-> comparative evaluation
-> Student-only web perception service
```

## In scope (this and near-future phases)

- Forward-facing dashcam frame-level 4-class classification (CLEAR, VEHICLE,
  PEDESTRIAN, MIXED).
- BDD100K as an initial SOURCE DOMAIN (never presented as Colombian data).
- A Common Manifest architecture, independent of BDD100K folder layout, so a
  future NeuroDriver Colombia adapter plugs in without rewriting training
  code.
- Two ground-truth labeling strategies (Full Frame, ADAS ROI), compared with
  real evidence before a permanent choice.
- A simple CNN baseline (also Student 2 in the KD ladder), a MobileNetV2
  transfer-learning Student, and a future ResNet50 Teacher architecture —
  all three sharing a two-logit (vehicle, pedestrian) output space so a
  future KD loss needs no adapter layer.
- Multi-label training (independent `has_vehicle`/`has_pedestrian` binary
  targets) instead of 4-class softmax, adopted after real data showed
  severe PEDESTRIAN class imbalance; the four ADAS states are still derived
  for reporting (see `docs/decisions_log.md`, 2026-09-22).
- Evaluation utilities beyond Accuracy (Precision/Recall/F1, confusion
  matrix, parameter count, estimated FLOPs, model size, latency,
  motorcycle subset).
- Design/documentation for Colombian fine-tuning and Teacher-Student KD
  (not executed yet).

## Explicitly out of scope for now

- Object-detector training (YOLO, SSD, Faster R-CNN, RetinaNet, Detectron).
- Steering/throttle/braking control, CAN bus integration.
- FastAPI/frontend implementation or web deployment.
- Raspberry Pi / mobile deployment, TFLite/ONNX export.
- Long production training runs.
- Real Knowledge Distillation (no Teacher model exists yet) or fabricated
  Teacher/KD results.

## What "done" means for Phase 1

See `PHASE_1.md` for the authoritative milestone definition and
`reports/experiment_status.md` for the current honest status.
