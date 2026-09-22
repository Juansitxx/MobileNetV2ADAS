"""Deterministic seeding across random/numpy/TensorFlow."""

from __future__ import annotations


def set_global_seed(seed: int = 42) -> None:
    import os
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed locally; notebooks/Colab will have it.
