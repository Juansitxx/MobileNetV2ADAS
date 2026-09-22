"""Evaluation utilities shared by the baseline CNN, MobileNetV2 Student, and
the future ResNet50 Teacher.

Never rely on Accuracy alone (see CLAUDE.md). These functions take plain
numpy arrays / lists so they have no TensorFlow import requirement (except
``estimate_flops``/``measure_inference_latency`` which need a real model)
and stay unit-testable; the notebooks pass in values computed from a
trained model.

Models output two raw logits (vehicle, pedestrian) — see
``neurodriver_cnn.models.baseline``/``mobilenetv2``/``resnet50_teacher`` —
not a 4-way softmax. ``labels_from_logits`` converts those two logits into
the 4-state ADAS label (CLEAR/VEHICLE/PEDESTRIAN/MIXED) for reporting;
``classification_metrics``/``confusion_matrix`` then operate on those
derived class indices exactly as before.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from neurodriver_cnn.labeling.frame_labels import derive_label

CLASS_NAMES = ["CLEAR", "VEHICLE", "PEDESTRIAN", "MIXED"]
CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def sigmoid(logits) -> np.ndarray:
    """Numerically stable sigmoid for converting raw model logits to probabilities."""
    logits = np.asarray(logits, dtype=float)
    return np.where(logits >= 0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))


def labels_from_logits(vehicle_logits, pedestrian_logits, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Threshold two-logit model output into has_vehicle/has_pedestrian booleans
    and the derived 4-state ADAS class index, using
    ``neurodriver_cnn.labeling.frame_labels.derive_label`` as the single
    source of truth for the mapping (see configs/training_config.json ->
    evaluation.label_threshold for the default).
    """
    has_vehicle = sigmoid(vehicle_logits) >= threshold
    has_pedestrian = sigmoid(pedestrian_logits) >= threshold
    labels = np.array(
        [CLASS_NAME_TO_INDEX[derive_label(bool(v), bool(p))] for v, p in zip(has_vehicle, has_pedestrian)]
    )
    return {"has_vehicle": has_vehicle, "has_pedestrian": has_pedestrian, "label": labels}


def multilabel_binary_metrics(y_true_vehicle, y_true_pedestrian, y_pred_vehicle, y_pred_pedestrian) -> dict[str, Any]:
    """Per-target (vehicle, pedestrian) binary Precision/Recall/F1/Accuracy.

    Reported separately from the derived 4-state metrics because the two
    binary targets are what the models are actually trained on.
    """
    from sklearn.metrics import precision_recall_fscore_support

    result = {}
    for name, y_true, y_pred in [
        ("vehicle", y_true_vehicle, y_pred_vehicle),
        ("pedestrian", y_true_pedestrian, y_pred_pedestrian),
    ]:
        y_true = np.asarray(y_true, dtype=bool)
        y_pred = np.asarray(y_pred, dtype=bool)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        result[name] = {
            "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    """Accuracy, macro/weighted Precision-Recall-F1, and a per-class report."""
    from sklearn.metrics import classification_report, precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class_report": report,
    }


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    return sk_confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))


def evaluate_motorcycle_subset(y_true, y_pred, has_motorcycle) -> dict[str, Any]:
    """Classification metrics restricted to rows where has_motorcycle == True.

    Returns ``{"n_samples": 0, ...}`` with NaN metrics if no motorcycle
    samples are present, rather than fabricating a result.
    """
    has_motorcycle = np.asarray(has_motorcycle, dtype=bool)
    n = int(has_motorcycle.sum())
    if n == 0:
        return {"n_samples": 0, "note": "No motorcycle-containing samples available."}

    y_true = np.asarray(y_true)[has_motorcycle]
    y_pred = np.asarray(y_pred)[has_motorcycle]
    metrics = classification_metrics(y_true, y_pred)
    metrics["n_samples"] = n
    return metrics


def model_size_bytes(saved_model_path) -> int:
    """Total size in bytes of a saved model file/directory."""
    from pathlib import Path

    path = Path(saved_model_path)
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    raise FileNotFoundError(f"Saved model path not found: {path}")


def estimate_flops(model, input_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Approximate multiply-accumulate FLOPs for one forward pass, via
    TensorFlow's graph profiler. Best-effort: TF's profiler API has shifted
    across versions, so this returns a clear "unavailable" note instead of
    raising if profiling fails on the installed TF version.
    """
    import tensorflow as tf

    try:
        forward_pass = tf.function(
            lambda x: model(x), input_signature=[tf.TensorSpec([1, *input_shape])]
        )
        graph_info = tf.compat.v1.profiler.profile(
            forward_pass.get_concrete_function().graph,
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),
        )
        return {"flops": int(graph_info.total_float_ops), "note": "approximate, per single-image forward pass"}
    except Exception as exc:  # pragma: no cover - depends on TF build/version
        return {"flops": None, "note": f"FLOPs estimation unavailable on this TensorFlow build: {exc}"}


def measure_inference_latency(model, sample_batch, n_repeats: int = 20) -> dict[str, float]:
    """Average/median wall-clock inference latency (ms) over n_repeats forward passes."""
    latencies_ms = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict(sample_batch, verbose=0)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "n_repeats": n_repeats,
    }
