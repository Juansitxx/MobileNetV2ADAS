# Future web perception service

Status: **not implemented in this phase.**

## Concept

```text
Dashcam frame/image
-> POST /predict
-> FastAPI
-> MobileNetV2 Student (only — Teacher absent at inference)
-> class + confidence
-> visual ADAS alert (perception/alert only)
```

The service performs perception and alerting only. It does not send any
command to steering, throttle, braking, CAN, or any other vehicle actuator.

## Why not now

Phase 1's scope is dataset preparation, labeling, the Common Manifest, the
baseline/MobileNetV2 architectures, evaluation utilities, and KD/Colombian
documentation — not deployment. Building a web service before the Student
has been trained and evaluated on real data (BDD100K, then Colombia) would
have nothing real to serve.

## Prerequisites before this is built

1. A MobileNetV2 Student trained (and ideally fine-tuned + distilled) with
   honestly reported metrics.
2. A decided serialization format for the Student (kept out of scope for
   Phase 1 per `CLAUDE.md` — no TFLite/ONNX work yet).
3. An explicit user request to start the FastAPI implementation.
