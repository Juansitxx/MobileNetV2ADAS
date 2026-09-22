"""Report the local environment status in human-readable form.

Never raises on an expected "missing" condition (no TensorFlow, no GPU, no
BDD100K data) — those are reported as [WARNING]/[MISSING] lines so the rest
of the pipeline can be inspected without a working ML environment.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TAR_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k" / "bdd100k_10k_supervisely.tar"
EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "bdd100k" / "extracted"


def _line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def check_python_os() -> None:
    _line("OK", f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")


def check_project_root() -> None:
    _line("INFO", f"Project root: {PROJECT_ROOT}")


def check_free_disk() -> None:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    status = "OK" if free_gb > 20 else "WARNING"
    _line(status, f"Free disk on project drive: {free_gb:.1f} GB")


def check_tensorflow() -> bool:
    try:
        import tensorflow as tf

        _line("OK", f"TensorFlow installed: {tf.__version__}")
        return True
    except ImportError:
        _line(
            "MISSING",
            "TensorFlow not installed. Install with `pip install tensorflow` "
            "(see requirements.txt) before running notebooks/models.",
        )
        return False


def check_gpu(tf_available: bool) -> None:
    if not tf_available:
        _line("INFO", "GPU check skipped (TensorFlow not installed).")
        return
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        _line("OK", f"GPU(s) visible to TensorFlow: {[g.name for g in gpus]}")
    else:
        _line("WARNING", "No GPU visible to TensorFlow. CPU-only training will be slow.")


def check_bdd_data() -> bool:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("ann")):
        _line("OK", f"BDD100K (extracted, Supervisely format) found under {EXTRACT_DIR}")
        return True
    if RAW_TAR_PATH.exists():
        _line(
            "OK",
            f"BDD100K Supervisely archive found: {RAW_TAR_PATH} "
            "(not extracted yet — scripts/02_build_manifest.py extracts it automatically).",
        )
        return True
    _line(
        "MISSING",
        f"BDD100K Supervisely archive not found at {RAW_TAR_PATH}. "
        "See docs/bdd100k_setup.md for the official download.",
    )
    return False


def main() -> None:
    print("=== NeuroDriver CNN - Environment Check ===")
    check_python_os()
    check_project_root()
    check_free_disk()
    tf_available = check_tensorflow()
    check_gpu(tf_available)
    check_bdd_data()
    print("=== Done ===")


if __name__ == "__main__":
    main()
