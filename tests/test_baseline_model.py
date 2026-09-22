import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed in this environment")

from neurodriver_cnn.evaluation.metrics import count_parameters
from neurodriver_cnn.models.baseline import build_baseline_cnn


def test_baseline_cnn_is_under_one_million_params():
    model = build_baseline_cnn(input_shape=(224, 224, 3), dropout=0.4)
    params = count_parameters(model)
    assert params["total_params"] < 1_000_000, params


def test_baseline_cnn_outputs_two_raw_logits():
    model = build_baseline_cnn(input_shape=(224, 224, 3))
    batch = tf.zeros((2, 224, 224, 3))
    logits = model(batch, training=False)
    assert logits.shape == (2, 2)
