"""
Training script for WAFE-Net
=============================
Reproduces the training setup from Section 4.2 of the paper.

Usage:
    python train.py --dataset fer2013 --data_root data/fer2013 --epochs 50
    python train.py --dataset kdef    --data_root data/KDEF    --epochs 100
    python train.py --dataset ckplus  --data_root data/CK+     --epochs 50
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model    import WAFENet, Baseline, BaselinePlusDRAB, BaselinePlusDRABWTM
from datasets import get_fer2013_loaders, get_kdef_loaders, get_ckplus_loaders
from evaluate import evaluate


# ─────────────────────────────────────────────
# Hyperparameters (Table 3)
# ─────────────────────────────────────────────
DATASET_CONFIG = {
    "fer2013": dict(batch_size=64, epochs=50, patience=15,  num_classes=7),
    "kdef":    dict(batch_size=32, epochs=100, patience=50, num_classes=7),
    "ckplus":  dict(batch_size=32, epochs=50,  patience=15, num_classes=7),
}

MODEL_MAP = {
    "wafenet":          WAFENet,
    "baseline":         Baseline,
    "baseline_drab":    BaselinePlusDRAB,
    "baseline_drab_wtm": BaselinePlusDRABWTM,
}


def get_loaders(dataset, data_root, batch_size):
    if dataset == "fer2013":
        return get_fer2013_loaders(data_root, batch_size=batch_size)
    elif dataset == "kdef":
        return get_kdef_loaders(data_root, batch_size=batch_size)
    elif dataset == "ckplus":
        return get_ckplus_loaders(data_root, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total * 100.0


def train(args):
    cfg    = DATASET_CONFIG[args.dataset]
    epochs = args.epochs or cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset.upper()}")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader = get_loaders(
        args.dataset, args.data_root, cfg["batch_size"]
    )

    # Model
    ModelClass = MODEL_MAP[args.model]
    model = ModelClass(num_classes=cfg["num_classes"], pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f} M\n")

    # Optimizer & scheduler (Section 4.2)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    patience     = args.patience or cfg["patience"]
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_ckpt    = Path(args.save_dir) / f"{args.model}_{args.dataset}_best.pth"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device, verbose=False)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc,  2))
        history["val_loss"].append(round(val_loss,   4))
        history["val_acc"].append(round(val_acc,    2))

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.1f}s]")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best val_acc={best_val_acc:.2f}% — checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping triggered after {patience} epochs without improvement.")
                break

    # ── Final evaluation on test set ──────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Loading best checkpoint: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, verbose=True,
                                    dataset_name=args.dataset.upper())

    print(f"\n  ★ Final test accuracy: {test_acc:.2f}%")
    print(f"{'─'*60}\n")

    # Save training history
    hist_path = Path(args.save_dir) / f"{args.model}_{args.dataset}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {hist_path}")

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WAFE-Net")
    parser.add_argument("--dataset",   default="fer2013", choices=["fer2013","kdef","ckplus"])
    parser.add_argument("--data_root", default="data/fer2013")
    parser.add_argument("--model",     default="wafenet",
                        choices=["wafenet","baseline","baseline_drab","baseline_drab_wtm"])
    parser.add_argument("--epochs",    type=int, default=None)
    parser.add_argument("--patience",  type=int, default=None)
    parser.add_argument("--save_dir",  default="checkpoints")
    args = parser.parse_args()
    train(args)
