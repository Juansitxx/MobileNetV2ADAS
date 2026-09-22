"""Simple CNN baseline — an intentionally weak academic reference point.

Not meant to compete with MobileNetV2; it exists to demonstrate that
transfer learning materially helps, and to exercise the full pipeline
(normalization, Flatten, Dropout, EarlyStopping, ReduceLROnPlateau) before
those choices are applied to the Student model.
"""

from __future__ import annotations

NUM_CLASSES = 4


def build_baseline_cnn(input_shape: tuple[int, int, int] = (224, 224, 3), dropout: float = 0.4):
    """Input -> Rescaling(1/255) -> Conv/Pool x3 -> Flatten -> Dense -> Dropout -> logits -> Softmax."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(NUM_CLASSES, name="logits")(x)
    outputs = layers.Softmax(name="predictions")(logits)

    return models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")


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
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
