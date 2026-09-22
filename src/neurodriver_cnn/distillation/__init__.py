"""Knowledge Distillation scaffolding — NOT implemented yet.

Teacher (ResNet50) and both Students (MobileNetV2, lightweight baseline CNN)
share the same two-logit output space (vehicle_logit, pedestrian_logit) —
see docs/teacher_student_contract.md and
docs/future_knowledge_distillation.md for the design. No fake Teacher
logits or fake KD training are produced by this package.
"""
