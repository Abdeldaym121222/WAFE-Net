"""
Ablation Study Runner  —  Section 6 of the paper
==================================================
Trains all four configurations on every dataset and prints Table 7.

Expected Table 7 results
─────────────────────────────────────────────────────────────
Configuration                  FER2013   KDEF    CK+
─────────────────────────────────────────────────────────────
Baseline (EfficientNet-B3+GAP)  71.79    95.89   98.47
+ DRAB                          72.93    97.31   99.50
+ DRAB + WTM (no CDF)           73.85    98.50   99.81
+ DRAB + WTM + CDF (WAFE-Net)   74.21    98.81  100.00

Usage:
    python ablation.py
    # or limit datasets:
    python ablation.py --datasets kdef ckplus
"""

import argparse
import json
from pathlib import Path

import torch

from train import train, DATASET_CONFIG

ABLATION_MODELS = [
    "baseline",
    "baseline_drab",
    "baseline_drab_wtm",
    "wafenet",
]

ABLATION_LABELS = {
    "baseline":          "Baseline (EfficientNet-B3 + GAP)",
    "baseline_drab":     "+ DRAB",
    "baseline_drab_wtm": "+ DRAB + WTM (no CDF)",
    "wafenet":           "+ DRAB + WTM + CDF  (WAFE-Net)",
}

# Default data roots — adjust to your paths
DATA_ROOTS = {
    "fer2013": "D:/New_Model/fer2013",
    "kdef":    "D:/New_Model/KDEF",
    "ckplus":  "D:/New_Model/CKplus",
}


def run_ablation(datasets=("fer2013", "kdef", "ckplus"), save_dir="checkpoints/ablation"):
    results = {ds: {} for ds in datasets}

    for dataset in datasets:
        data_root = DATA_ROOTS[dataset]
        for model_key in ABLATION_MODELS:
            print(f"\n{'#'*70}")
            print(f"  ABLATION: {ABLATION_LABELS[model_key]}  |  {dataset.upper()}")
            print(f"{'#'*70}")

            import types
            args = types.SimpleNamespace(
                dataset   = dataset,
                data_root = data_root,
                model     = model_key,
                epochs    = None,   # use dataset default
                patience  = None,
                save_dir  = save_dir,
            )
            acc = train(args)
            results[dataset][model_key] = round(acc, 2)

    # ── Print Table 7 ─────────────────────────────────────────────
    header_ds = [ds.upper() for ds in datasets]
    col_w     = 14
    name_w    = 42

    print(f"\n\n{'='*80}")
    print("  ABLATION STUDY — Table 7")
    print(f"{'='*80}")
    print(f"{'Configuration':<{name_w}}", end="")
    for ds in header_ds:
        print(f"{ds:>{col_w}}", end="")
    print()
    print("─" * (name_w + col_w * len(datasets)))
    for model_key in ABLATION_MODELS:
        label = ABLATION_LABELS[model_key]
        print(f"{label:<{name_w}}", end="")
        for dataset in datasets:
            val = results[dataset].get(model_key, "—")
            print(f"{str(val):>{col_w}}", end="")
        print()
    print(f"{'='*80}\n")

    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["fer2013", "kdef", "ckplus"],
                        choices=["fer2013", "kdef", "ckplus"])
    parser.add_argument("--save_dir", default="checkpoints/ablation")
    args = parser.parse_args()
    run_ablation(datasets=args.datasets, save_dir=args.save_dir)
