"""
data_loader.py
Data loading, domain-specific SSL augmentations, and strict stratified data splitting
for the TCL-CXR empirical comparison study.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit


class TwoCropTransform:
    """
    Generates two independently augmented views of the same input image
    for contrastive self-supervised pretraining (SimCLR and TCL).
    """
    def __init__(self, base_transform):
        # Store the underlying torchvision transformation pipeline
        self.base_transform = base_transform

    def __call__(self, x):
        # Apply the stochastic transformation twice to create a positive pair (x_i, x_j)
        return [self.base_transform(x), self.base_transform(x)]


def get_ssl_transforms(img_size=224):
    """
    Defines domain-specific data augmentations for chest X-ray SSL pretraining.
    Bounded carefully to prevent cutting out peripheral lung lesions or distorting anatomy.
    """
    ssl_transform = transforms.Compose([
        # Simulate slight patient positioning shifts without extreme rotation
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        # Preserve at least 80% of the field of view to retain peripheral nodules
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
        # Simulate X-ray exposure and contrast variations across scanners
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.5),
        # Simulate sensor noise and resolution variations using a 3x3 Gaussian kernel
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.5),
        # Convert PIL image to PyTorch tensor scaled in [0, 1]
        transforms.ToTensor(),
        # Normalize using standard ImageNet statistics expected by timm backbones
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Wrap in TwoCropTransform to return two augmented views per sample
    return TwoCropTransform(ssl_transform)


def get_downstream_transforms(img_size=224, is_train=True):
    """
    Defines transformations for downstream supervised fine-tuning and evaluation.
    """
    if is_train:
        return transforms.Compose([
            # Resize input radiograph to standard backbone resolution
            transforms.Resize((img_size, img_size)),
            # Mild affine perturbation to regularize low-label fine-tuning
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Deterministic preprocessing for validation, held-out testing, and Grad-CAM
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class CXRDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Chest X-Ray radiographs.
    For Shenzhen/Montgomery, binary labels are parsed directly from standard filenames
    (e.g., *_0.png for Normal, *_1.png for Tuberculosis).
    """
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return total number of radiographs in the subset
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load image and convert to 3-channel RGB for compatibility with Xception
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply image transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        # Return image only for unlabeled SSL pretraining
        if self.labels is None:
            return image

        # Return (image, float_label, filename) for downstream classification
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, os.path.basename(img_path)


def load_tb_filepaths_and_labels(data_dir):
    """
    Scans a directory of Shenzhen or Montgomery radiographs and extracts binary TB labels.
    """
    all_files = sorted(glob(os.path.join(data_dir, "*.png")))
    labels = []
    for f in all_files:
        # Standard naming convention: CHNCXR_XXXX_0.png (Normal) vs CHNCXR_XXXX_1.png (TB)
        stem = os.path.splitext(os.path.basename(f))[0]
        label = 1 if stem.endswith("_1") else 0
        labels.append(label)
    return np.array(all_files), np.array(labels)


def get_strict_splits(file_paths, labels, label_fraction=1.0, seed=42):
    """
    Implements the rigorous two-stage evaluation split protocol:
      1. Fixed outer split (seed=42): 70% Training Pool, 10% Validation, 20% Held-out Test.
         (Shenzhen contains 1 image per patient, ensuring patient-level isolation).
      2. Seeded inner split (seed=42, 77, or 123): Stratified sampling of 10%, 25%, 50%,
         or 100% labeled subsets from the 70% Training Pool.
    """
    # Stage 1a: Separate 20% Held-out Test Set using a fixed reference seed (42)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(file_paths, labels))

    files_train_val, labels_train_val = file_paths[train_val_idx], labels[train_val_idx]
    files_test, labels_test = file_paths[test_idx], labels[test_idx]

    # Stage 1b: Separate 10% Validation Set (0.125 of the remaining 80% equals 10% total)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_pool_idx, val_idx = next(sss_val.split(files_train_val, labels_train_val))

    files_train_pool = files_train_val[train_pool_idx]
    labels_train_pool = labels_train_val[train_pool_idx]
    files_val = files_train_val[val_idx]
    labels_val = labels_train_val[val_idx]

    # Stage 2: Sample the target label budget from the 70% Training Pool using the run seed
    if label_fraction < 1.0:
        sss_budget = StratifiedShuffleSplit(
            n_splits=1, train_size=label_fraction, random_state=seed
        )
        sub_idx, _ = next(sss_budget.split(files_train_pool, labels_train_pool))
        files_train = files_train_pool[sub_idx]
        labels_train = labels_train_pool[sub_idx]
    else:
        # Use 100% of the 70% training pool
        files_train, labels_train = files_train_pool, labels_train_pool

    return (files_train, labels_train), (files_val, labels_val), (files_test, labels_test)
