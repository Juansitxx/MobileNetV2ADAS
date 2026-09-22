"""MobileNetV2 Student — the primary architecture and future KD Student.

Pre-Softmax logits are kept directly accessible (``logits_model``) so a
future Knowledge Distillation loss can consume them without rebuilding the
network. See docs/teacher_student_contract.md.
"""

from __future__ import annotations

NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)


def build_mobilenetv2_student(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
):
    """ImageNet MobileNetV2(include_top=False) -> GAP -> Dropout -> Dense(4, logits) -> Softmax.

    Returns ``(full_model, logits_model, base_model)``:
    - ``full_model``: image -> softmax probabilities (deployment/training target).
    - ``logits_model``: image -> pre-softmax logits (for future KD).
    - ``base_model``: the MobileNetV2 backbone (for later selective unfreezing).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import mobilenet_v2

    base_model = mobilenet_v2.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape, name="image")
    # preprocess_input expects raw 0-255 RGB; do NOT also Rescaling(1./255) here.
    x = mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    logits = layers.Dense(NUM_CLASSES, name="logits")(x)
    outputs = layers.Softmax(name="predictions")(logits)

    full_model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_student")
    logits_model = models.Model(inputs=inputs, outputs=logits, name="mobilenetv2_student_logits")

    return full_model, logits_model, base_model


def unfreeze_for_fine_tuning(base_model, unfreeze_from_layer: int | None = None) -> None:
    """Unfreeze late backbone layers for fine-tuning.

    Recompile with a low learning rate (~1e-5) after calling this. BatchNorm
    layers are kept frozen (inference mode) even when unfrozen, since BDD100K
    fine-tuning batches are typically too small/non-representative to safely
    update BatchNorm running statistics.
    """
    from tensorflow.keras import layers as keras_layers

    base_model.trainable = True
    if unfreeze_from_layer is None:
        unfreeze_from_layer = max(0, len(base_model.layers) - 30)

    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, keras_layers.BatchNormalization):
            layer.trainable = False


def compile_student(model, learning_rate: float = 1e-3):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_parameters(model) -> dict:
    """Return total/trainable/non-trainable parameter counts."""
    import numpy as np

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }
