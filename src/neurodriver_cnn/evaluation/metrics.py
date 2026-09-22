"""Evaluation utilities shared by the baseline CNN and MobileNetV2 Student.

Never rely on Accuracy alone (see CLAUDE.md). These functions take plain
numpy arrays / lists so they have no TensorFlow import requirement and stay
unit-testable; the notebooks pass in ``y_true``/``y_pred`` computed from a
trained model.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

CLASS_NAMES = ["CLEAR", "VEHICLE", "PEDESTRIAN", "MIXED"]


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
