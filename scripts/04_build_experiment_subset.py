"""Build a reproducible ~4,000-8,000 sample experiment subset.

Writes data/processed/manifests/experiment_manifest.csv plus generation
metadata for reproducibility. Never fabricates balance or duplicates files
to force a target size — if class availability cannot reach the target,
the actual achievable size is reported honestly (see
reports/sampling_plan.md, produced by this script).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurodriver_cnn.config import load_dataset_config  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifests" / "bdd100k_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "manifests" / "experiment_manifest.csv"
SAMPLING_PLAN_PATH = PROJECT_ROOT / "reports" / "sampling_plan.md"


def _pending(reason: str) -> None:
    SAMPLING_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLING_PLAN_PATH.write_text(
        f"# Sampling plan\n\n**Pending.** {reason}\n", encoding="utf-8"
    )
    print(f"[MISSING] {reason}")


def main() -> None:
    if not MANIFEST_PATH.exists():
        _pending(
            "No BDD100K manifest found. Run scripts/02_build_manifest.py first "
            "(requires BDD100K data, see docs/bdd100k_setup.md)."
        )
        return

    df = pd.read_csv(MANIFEST_PATH)
    valid = df[df["valid_sample"] == True]  # noqa: E712
    if len(valid) == 0:
        _pending("Manifest has zero valid_sample rows; nothing to sample from.")
        return

    config = load_dataset_config()["experiment_subset"]
    seed = load_dataset_config()["seed"]

    class_counts = valid["label"].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_n = int(class_counts.min())
    majority_n = int(class_counts.max())

    # A fully balanced subset is capped by 4x the minority class count.
    balanced_target_per_class = minority_n
    balanced_subset_size = balanced_target_per_class * len(class_counts)

    preferred = config["target_size_preferred"]
    achievable = min(preferred, len(valid))

    plan_lines = [
        "# Sampling plan",
        "",
        f"- Valid samples available: {len(valid)}",
        f"- Class counts: {class_counts.to_dict()}",
        f"- Minority class: {minority_class} ({minority_n})",
        f"- Majority class: {majority_class} ({majority_n})",
        f"- Majority/minority ratio: {majority_n / minority_n:.2f}" if minority_n > 0 else "- Majority/minority ratio: undefined (0 minority samples)",
        f"- Fully balanced subset size (capped by minority class): {balanced_subset_size}",
        f"- Recommended practical subset size: {achievable} (target range "
        f"{config['target_size_min']}-{config['target_size_max']}, preferred {preferred})",
        "",
        "## class_weight recommendation",
        "",
        (
            f"Majority/minority ratio is {majority_n / minority_n:.2f}; consider "
            "`class_weight` (inverse frequency) during training if this subset "
            "keeps natural class proportions rather than the fully balanced size."
            if minority_n > 0
            else "Cannot compute a ratio: at least one class has zero samples."
        ),
    ]
    SAMPLING_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLING_PLAN_PATH.write_text("\n".join(plan_lines), encoding="utf-8")

    # Reproducible sampling: proportional-to-availability, capped at `achievable`,
    # stratified by label so all four classes remain represented.
    frac = achievable / len(valid)
    subset = valid.groupby("label", group_keys=False).sample(
        frac=min(1.0, frac), random_state=seed
    ).reset_index(drop=True)
    subset["label_source"] = "BDD100K_GROUND_TRUTH"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUTPUT_PATH, index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "target_size_preferred": preferred,
        "achieved_size": len(subset),
        "sampling_fraction": frac,
        "class_counts_in_subset": subset["label"].value_counts().to_dict(),
    }
    (OUTPUT_PATH.parent / "experiment_manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"[OK] Wrote {len(subset)} rows to {OUTPUT_PATH}")
    print(f"[OK] Wrote {SAMPLING_PLAN_PATH}")


if __name__ == "__main__":
    main()
