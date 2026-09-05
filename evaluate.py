"""
Evaluation utilities
=====================
• evaluate()         — loss + accuracy on a DataLoader
• full_report()      — per-class precision/recall/F1 + confusion matrix
• measure_fps()      — FPS / latency measurement (Table 8)
• count_flops()      — GFLOPs estimation (Table 8)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# ─────────────────────────────────────────────
# Basic evaluation
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device, verbose=False, dataset_name=""):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct   / total * 100.0

    if verbose:
        print(f"\n[{dataset_name}]  Loss={avg_loss:.4f}  Accuracy={accuracy:.2f}%  ({correct}/{total})")
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# Full per-class report + confusion matrix
# ─────────────────────────────────────────────
def full_report(model, loader, device, class_names=None, dataset_name=""):
    if class_names is None:
        class_names = EMOTION_LABELS

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100.0

    print(f"\n{'='*60}")
    print(f"  {dataset_name}  —  Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    return accuracy, cm


# ─────────────────────────────────────────────
# FPS / inference latency (Table 8)
# ─────────────────────────────────────────────
def measure_fps(model, device, input_size=(1, 3, 224, 224), n_runs=200, warmup=20):
    """
    Measures average inference time and FPS on `device`.
    Matches the RTX 3060 measurement in Section 7 (~38 FPS).
    """
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    avg_ms = (t_end - t_start) / n_runs * 1000.0
    fps    = 1000.0 / avg_ms

    print(f"\nInference latency: {avg_ms:.1f} ms/image  →  {fps:.1f} FPS")
    return avg_ms, fps


# ─────────────────────────────────────────────
# GFLOPs estimation (Table 8)
# ─────────────────────────────────────────────
def count_flops(model, input_size=(1, 3, 224, 224)):
    """
    Uses thop (ptflops) if available, otherwise prints a note.
    The paper reports 4.1 GFLOPs for WAFE-Net.
    """
    try:
        from thop import profile, clever_format
        dummy = torch.randn(*input_size)
        macs, params = profile(model.cpu(), inputs=(dummy,), verbose=False)
        flops = macs * 2
        flops_str, params_str = clever_format([flops, params], "%.2f")
        print(f"\nGFLOPs : {flops_str}   Params : {params_str}")
        return flops, params
    except ImportError:
        print("\n[Note] Install 'thop' for GFLOPs counting:  pip install thop")
        total_p = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Total params (manual): {total_p:.1f} M")
        return None, total_p
