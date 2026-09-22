# Presentation outline (~8 slides)

1. **Colombian driving problem/context** — why an ADAS perception model
   tailored to Colombian traffic conditions matters (vehicle mix, motorcycle
   density, informal traffic patterns).
2. **Original NeuroDriver ecosystem** — the broader project this prototype
   is scoped down from; this work isolates visual perception only.
3. **Scope reduction** — what is explicitly in/out of scope for this
   academic prototype (see `docs/project_scope.md`); no vehicle control, no
   object-detector training.
4. **Dataset strategy: BDD source -> Colombia target** — BDD100K as a
   provisional source domain, NeuroDriver Colombia dashcam data as the
   eventual target domain, and the planned adaptation path
   (`docs/colombian_domain_strategy.md`).
5. **Four classes + explicit motorcycle tracking** — CLEAR/VEHICLE/
   PEDESTRIAN/MIXED definitions, category mapping, and why motorcycles stay
   inside VEHICLE while remaining separately trackable.
6. **CNN baseline** — the intentionally simple reference architecture and
   why it exists (normalization, Flatten, Dropout, EarlyStopping,
   ReduceLROnPlateau — see `docs/recommendations_applied.md`).
7. **MobileNetV2 transfer learning + future NeuroDriver KD** — the Student
   architecture, exposed logits, and the planned (not yet implemented)
   Teacher-Student distillation.
8. **Current progress and next steps** — honest status
   (`reports/experiment_status.md`), what is Pending, and the recommended
   next action.

Do not claim Knowledge Distillation is already implemented on any slide.
