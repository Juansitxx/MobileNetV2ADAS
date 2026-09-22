"""ResNet50 Teacher architecture — future Knowledge Distillation Teacher.

Not trained in this phase (see docs/future_knowledge_distillation.md). Kept
architecturally compatible with both Students (MobileNetV2, baseline CNN):
same two-logit output space (``vehicle_logit``, ``pedestrian_logit``), so a
future distillation loss can compare Teacher and Student logits directly
without an adapter layer. See docs/teacher_student_contract.md.
"""

from __future__ import annotations

NUM_TARGETS = 2  # [vehicle_logit, pedestrian_logit]
INPUT_SHAPE = (224, 224, 3)


def build_resnet50_teacher(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
):
    """ImageNet ResNet50(include_top=False) -> GAP -> Dropout -> Dense(2, logits).

    Returns ``(model, base_model)``, mirroring
    ``neurodriver_cnn.models.mobilenetv2.build_mobilenetv2_student``.
    """
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import resnet50

    base_model = resnet50.ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape, name="image")
    x = resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    logits = layers.Dense(NUM_TARGETS, name="logits")(x)

    model = models.Model(inputs=inputs, outputs=logits, name="resnet50_teacher")
    return model, base_model


def compile_teacher(model, learning_rate: float = 1e-3):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.0)],
    )
    return model
