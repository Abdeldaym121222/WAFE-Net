"""
Dataset loaders for FER2013, KDEF, and CK+
============================================
Each loader returns (train_loader, val_loader, test_loader).

Directory structure expected
─────────────────────────────
FER2013  (official split already in sub-folders):
  data/fer2013/train/<class_name>/...
  data/fer2013/val/<class_name>/...
  data/fer2013/test/<class_name>/...

KDEF  (flat folder, 80/20 split performed here):
  data/KDEF/<class_name>/<image>.jpg

CK+  (peak frames only, subject-independent 80/20 split):
  data/CK+/<class_name>/<image>.png
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Transforms (Section 4.2 & 4.3)
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─────────────────────────────────────────────
# Helper: convert grayscale → 3-channel RGB
# ─────────────────────────────────────────────
class GrayToRGB:
    """If image is grayscale, replicate to 3 channels (for FER2013)."""
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img


fer_train_transform = transforms.Compose([
    GrayToRGB(),
    *train_transform.transforms,
])

fer_val_transform = transforms.Compose([
    GrayToRGB(),
    *val_transform.transforms,
])


# ─────────────────────────────────────────────
# Balanced oversampling helper
# ─────────────────────────────────────────────
def make_balanced_sampler(dataset):
    """Returns a WeightedRandomSampler to balance class distribution."""
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts[targets]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


# ─────────────────────────────────────────────
# FER2013
# ─────────────────────────────────────────────
def get_fer2013_loaders(root="data/fer2013", batch_size=64, num_workers=4):
    train_ds = datasets.ImageFolder(
        os.path.join(root, "train"), transform=fer_train_transform
    )
    val_ds   = datasets.ImageFolder(
        os.path.join(root, "val"),   transform=fer_val_transform
    )
    test_ds  = datasets.ImageFolder(
        os.path.join(root, "test"),  transform=fer_val_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    print(f"FER2013  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    print(f"Classes : {train_ds.classes}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Generic small dataset (KDEF / CK+)
# with stratified 80/20 split and oversampling
# ─────────────────────────────────────────────
def get_small_dataset_loaders(root, batch_size=32, num_workers=4,
                               seed=42, dataset_name="KDEF"):
    full_ds = datasets.ImageFolder(root, transform=val_transform)

    targets = [s[1] for s in full_ds.samples]
    indices = list(range(len(full_ds)))

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=targets,
        random_state=seed
    )

    # Apply training augmentations to the training split
    train_ds_aug = datasets.ImageFolder(root, transform=train_transform)

    train_subset = Subset(train_ds_aug, train_idx)
    test_subset  = Subset(full_ds,       test_idx)

    # Balanced oversampling for training
    train_targets = [targets[i] for i in train_idx]
    class_counts  = np.bincount(train_targets)
    sample_weights = [1.0 / class_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(test_subset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"{dataset_name}  train={len(train_idx):,}  test={len(test_idx):,}")
    print(f"Classes : {full_ds.classes}")
    return train_loader, val_loader, val_loader   # val == test for these datasets


def get_kdef_loaders(root="data/KDEF", batch_size=32, num_workers=4, seed=42):
    return get_small_dataset_loaders(root, batch_size, num_workers, seed, "KDEF")


def get_ckplus_loaders(root="data/CK+", batch_size=32, num_workers=4, seed=42):
    return get_small_dataset_loaders(root, batch_size, num_workers, seed, "CK+")
