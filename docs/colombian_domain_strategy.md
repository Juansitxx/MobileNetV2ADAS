# Colombian target-domain strategy

Status: **planning document — no Colombian data integrated yet.**

BDD100K is the provisional SOURCE DOMAIN used to bootstrap the CNN. The
NeuroDriver Colombian dashcam dataset is the TARGET DOMAIN this project is
ultimately aimed at. Fixed/elevated Colombian CCTV footage is not used as
the primary training domain for this dashcam model unless a later, explicit
experiment studies that specific domain shift.

## Planned stages

1. Recover/organize the NeuroDriver Colombia dashcam frames and metadata.
2. Convert them to the Common Manifest schema via a `NeuroDriverAdapter`
   (mirrors the BDD100K adapter's output contract — see
   `docs/neurodriver_dataset_contract.md`).
3. Split Colombian data by run/session/video into TRAIN/VAL/TEST — never by
   adjacent random frames, to avoid leakage.
4. Keep the Colombian TEST split untouched until final evaluation.
5. Evaluate the BDD-trained Student on Colombian TEST **before** any
   adaptation, to measure the raw source→target domain gap.
6. Fine-tune the Student on Colombian TRAIN/VAL (frozen-backbone head
   training, then selective late-layer unfreezing at a low learning rate;
   see `neurodriver_cnn.models.mobilenetv2.unfreeze_for_fine_tuning`).
7. Re-evaluate on the same held-out Colombian TEST split.
8. Add NeuroDriver Teacher → Student Knowledge Distillation (see
   `docs/teacher_student_contract.md`).
9. Compare all four stages, including the motorcycle-containing-frame
   subset, using the same evaluation utilities
   (`src/neurodriver_cnn/evaluation/metrics.py`).

## Comparison matrix (to be filled in Phase 2+ with real numbers)

| Model | Evaluated on | Accuracy | F1 macro | Motorcycle-subset F1 |
|---|---|---|---|---|
| BDD-trained | BDD TEST | Pending | Pending | Pending |
| BDD-trained | Colombia TEST | Pending | Pending | Pending |
| Colombia fine-tuned | Colombia TEST | Pending | Pending | Pending |
| Colombia fine-tuned + KD | Colombia TEST | Pending | Pending | Pending |

## Hypotheses, not assumptions

Until measured, the following are treated as hypotheses only:

- higher motorcycle density in Colombian traffic than in BDD100K;
- different road infrastructure/signage conventions;
- different pedestrian behavior/context (informal crossings, street
  vending, etc.);
- different vehicle mix and road condition;
- different camera mounting position/field of view across NeuroDriver
  devices.

None of these are asserted as fact in this repository's code, docs, or
notebooks before real Colombian data is analyzed.
