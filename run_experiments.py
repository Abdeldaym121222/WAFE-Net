"""
run_experiments.py — Reproduce ALL paper results
==================================================
Runs WAFE-Net on FER2013, KDEF, and CK+, then prints
Tables 4, 5, 6, 7, and 8 equivalents.

Usage:
    python run_experiments.py
    python run_experiments.py --datasets kdef ckplus   # subset
    python run_experiments.py --skip_ablation          # skip Table 7
"""

import argparse
import types
import json
import time
from pathlib import Path

import torch

from model    import WAFENet
from train    import train, DATASET_CONFIG
from evaluate import full_report, measure_fps, count_flops
from datasets import get_fer2013_loaders, get_kdef_loaders, get_ckplus_loaders

# ── SOTA reference numbers from the paper (Tables 4, 5, 6) ──────
SOTA = {
    "fer2013": [
        ("VGGNet [12]",                    2021, 70.32),
        ("ResNet-50 [13]",                 2021, 71.83),
        ("CBAM-ResNet [17]",               2022, 72.11),
        ("EfficientNet-B4 [15]",           2022, 72.54),
        ("DAN [18]",                       2023, 73.27),
        ("TransFER [31]",                  2023, 73.90),
        ("Hybrid LocalAttn+ViT [21]",      2024, 73.16),
        ("EfficientNet-B4+Transformer[32]",2024, 73.58),
        ("POSTER [22]",                    2023, 74.24),
    ],
    "kdef": [
        ("CNN + DBN [33]",                 2023, 95.29),
        ("CNN + Residual Network [34]",    2023, 93.38),
        ("AFER [35]",                      2023, 93.70),
        ("New CNN [36]",                   2024, 95.00),
        ("ResNet50+CBAM+TCN [37]",         2024, 97.08),
        ("Fine-tuned VGG-19+Histogram[38]",2024, 95.92),
        ("DCRNet [39]",                    2025, 96.25),
        ("FER-MOTION [40]",                2025, 97.81),
        ("MsC-wCA ResNet18 [23]",          2025, 97.48),
    ],
    "ckplus": [
        ("TPAN [41]",                      2021, 92.84),
        ("DeepCNN [42]",                   2021, 98.00),
        ("Fusion-CNN [20]",                2023, 98.22),
        ("ZFER [43]",                      2023, 98.74),
        ("CNN [44]",                       2024, 98.00),
        ("VGG-19 [45]",                    2025, 98.00),
        ("ResNet-50 [46]",                 2025, 98.48),
        ("DCRNet [39]",                    2025, 98.98),
    ],
}

TABLE_TITLES = {
    "fer2013": "Table 4. Comparison on FER2013 (7 classes)",
    "kdef":    "Table 5. Comparison on KDEF",
    "ckplus":  "Table 6. Comparison on CK+",
}

DATA_ROOTS = {
    "fer2013": "D:/New_Model/fer2013",
    "kdef":    "D:/New_Model/KDEF",
    "ckplus":  "D:/New_Model/CKplus",
}

LOADER_MAP = {
    "fer2013": get_fer2013_loaders,
    "kdef":    get_kdef_loaders,
    "ckplus":  get_ckplus_loaders,
}


def print_comparison_table(dataset, our_acc):
    title = TABLE_TITLES[dataset]
    rows  = SOTA[dataset]
    w_name, w_year, w_acc = 46, 6, 14

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Method':<{w_name}} {'Year':>{w_year}} {'Accuracy (%)':>{w_acc}}")
    print(f"{'─'*70}")
    for name, year, acc in rows:
        print(f"{name:<{w_name}} {year:>{w_year}} {acc:>{w_acc}.2f}")
    print(f"{'─'*70}")
    our_label = "WAFE-Net (Proposed)"
    print(f"{our_label:<{w_name}} {'2026':>{w_year}} {our_acc:>{w_acc}.2f}  ★")
    print(f"{'='*70}")


def run_main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_accs  = {}

    # ── 1. Train & evaluate WAFE-Net on each dataset ─────────────
    for dataset in args.datasets:
        a = types.SimpleNamespace(
            dataset   = dataset,
            data_root = DATA_ROOTS[dataset],
            model     = "wafenet",
            epochs    = None,
            patience  = None,
            save_dir  = args.save_dir,
        )
        acc = train(a)
        all_accs[dataset] = acc

    # ── 2. Full per-class reports ─────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  PER-CLASS REPORTS")
    print("=" * 70)
    for dataset in args.datasets:
        cfg          = DATASET_CONFIG[dataset]
        loader_fn    = LOADER_MAP[dataset]
        _, _, test_loader = loader_fn(DATA_ROOTS[dataset], batch_size=cfg["batch_size"])

        best_ckpt = Path(args.save_dir) / f"wafenet_{dataset}_best.pth"
        model     = WAFENet(num_classes=7).to(device)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        full_report(model, test_loader, device, dataset_name=dataset.upper())

    # ── 3. Print comparison tables (Tables 4, 5, 6) ──────────────
    for dataset in args.datasets:
        print_comparison_table(dataset, all_accs[dataset])

    # ── 4. Computational efficiency (Table 8) ────────────────────
    print(f"\n\n{'='*70}")
    print("  Table 8. Computational Efficiency")
    print(f"{'='*70}")
    first_ds  = args.datasets[0]
    best_ckpt = Path(args.save_dir) / f"wafenet_{first_ds}_best.pth"
    model     = WAFENet(num_classes=7).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    avg_ms, fps = measure_fps(model, device)
    count_flops(model)

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nWAFE-Net  |  {total_p:.1f} M params  |  ~{fps:.0f} FPS  ({avg_ms:.1f} ms/img)")
    print("Paper reports: 14.3 M  |  4.1 GFLOPs  |  ~38 FPS")

    # ── 5. Ablation study (Table 7) ──────────────────────────────
    if not args.skip_ablation:
        from ablation import run_ablation
        run_ablation(datasets=args.datasets, save_dir=str(Path(args.save_dir) / "ablation"))

    # ── 6. Summary ───────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  SUMMARY — Final Test Accuracies")
    print(f"{'='*70}")
    targets = {"fer2013": 74.21, "kdef": 98.81, "ckplus": 100.00}
    for ds, acc in all_accs.items():
        tgt = targets.get(ds, "—")
        print(f"  {ds.upper():<10}  Got: {acc:.2f}%   Target (paper): {tgt}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce WAFE-Net paper results")
    parser.add_argument("--datasets",        nargs="+", default=["fer2013","kdef","ckplus"],
                        choices=["fer2013","kdef","ckplus"])
    parser.add_argument("--save_dir",        default="checkpoints")
    parser.add_argument("--skip_ablation",   action="store_true")
    args = parser.parse_args()
    run_main(args)
