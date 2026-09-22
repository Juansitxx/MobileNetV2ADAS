"""Simple CNN baseline — an intentionally weak academic reference point,
and, unchanged in role, the "CNN ligera propia" Student 2 in the future
Knowledge Distillation ladder (see docs/future_knowledge_distillation.md).

Outputs **two independent logits** (vehicle, pedestrian), not a 4-way
softmax: real class-count inspection showed a severe imbalance for the
pure PEDESTRIAN class, which a single softmax over 4 classes would tend to
collapse. Trained with per-target BinaryCrossentropy(from_logits=True); the
4-state ADAS label (CLEAR/VEHICLE/PEDESTRIAN/MIXED) is derived post-hoc from
thresholded probabilities for metrics/reporting
(``neurodriver_cnn.evaluation.metrics``), never trained directly. See
docs/decisions_log.md (2026-09-22).

Parameter budget (final architecture fixed 2026-09-22, see
docs/decisions_log.md): four Conv/Pool blocks (16/32/64/64 filters) shrink
224x224 down to 14x14 before ``Flatten``, keeping the flattened vector
(12,544) small enough that ``Dense(64)`` on top of it stays cheap.
Verified with real TensorFlow: **863,522 total params**, under the <1M
"lightweight CNN" budget. (An earlier 3-block version needed an extra
pooling hack to hit the same budget — this 4-block architecture hits it
directly, with no hack layer, which is cleaner and is now the fixed
design.)
"""

from __future__ import annotations

NUM_TARGETS = 2  # [vehicle_logit, pedestrian_logit]


def build_baseline_cnn(input_shape: tuple[int, int, int] = (224, 224, 3), dropout: float = 0.4):
    """Input -> Rescaling(1/255) -> Conv/Pool x4 -> Flatten -> Dense -> Dropout -> logits.

    The returned model's output is the raw ``logits`` layer (no activation):
    compile with ``BinaryCrossentropy(from_logits=True)``. Convert to
    probabilities at inference/evaluation time with
    ``neurodriver_cnn.evaluation.metrics.sigmoid``.
    """
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(NUM_TARGETS, name="logits")(x)

    return models.Model(inputs=inputs, outputs=logits, name="baseline_cnn")


def build_baseline_callbacks(patience_es: int = 5, patience_lr: int = 3):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=patience_lr, min_lr=1e-7
        ),
    ]


def compile_baseline(model, learning_rate: float = 1e-3):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # Model output is raw logits, so threshold=0.0 in logit-space is
        # equivalent to probability threshold 0.5 (sigmoid(0) == 0.5) —
        # matches configs/training_config.json -> evaluation.label_threshold.
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.0)],
    )
    return model
