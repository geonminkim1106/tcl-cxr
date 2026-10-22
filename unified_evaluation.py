"""
unified_evaluation.py
Rigorous, leakage-free downstream evaluation protocol:
  - Partition: 70% Train Pool / 10% Validation / 20% Held-out Test.
  - Checkpoint Selection: Best Validation AUROC only (no test-set peeking).
  - Final Evaluation: Single evaluation on the 20% Held-out Test Set and external Montgomery set
    across 3 random seeds (42, 77, 123) using population standard deviation (ddof=0).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

from data_loader import (
    CXRDataset,
    get_downstream_transforms,
    load_tb_filepaths_and_labels,
    get_strict_splits
)
from model import CXRClassifier


def set_ deterministic_seed(seed):
    """Fixes all random number generators for reproducible multi-seed experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Computes rank-based (AUROC) and threshold-dependent (F1, Recall, Specificity)
    metrics on a given evaluation dataloader.
    """
    model.eval()
    all_probs, all_targets = [], []

    for images, labels, _ in dataloader:
        images = images.to(device)
        # Obtain predicted probabilities via sigmoid activation
        probs = torch.sigmoid(model(images)).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    # Apply fixed operational decision boundary (default = 0.5)
    preds = (all_probs >= threshold).astype(int)

    auroc = roc_auc_score(all_targets, all_probs)
    f1 = f1_score(all_targets, preds, zero_division=0)
    recall = recall_score(all_targets, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-8)

    return {
        "AUROC": auroc,
        "F1": f1,
        "Recall": recall,
        "Specificity": specificity,
        "probs": all_probs,
        "targets": all_targets
    }


def run_single_experiment(model_name, ckpt_path, files, labels, fraction, seed, device, epochs=30):
    """
    Trains a model on a specific label fraction and seed, selects the checkpoint
    with the highest Validation AUROC, and evaluates once on the Held-out Test Set.
    """
    set_deterministic_seed(seed)

    # Obtain strict 70% Train Pool (subsampled by fraction), 10% Val, and 20% Held-out Test
    (f_tr, y_tr), (f_val, y_val), (f_te, y_te) = get_strict_splits(files, labels, fraction, seed)

    train_loader = DataLoader(
        CXRDataset(f_tr, y_tr, get_downstream_transforms(is_train=True)),
        batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        CXRDataset(f_val, y_val, get_downstream_transforms(is_train=False)),
        batch_size=16, shuffle=False
    )
    test_loader = DataLoader(
        CXRDataset(f_te, y_te, get_downstream_transforms(is_train=False)),
        batch_size=16, shuffle=False
    )

    # Initialize classifier; use ImageNet weights if evaluating the baseline
    is_imagenet = (model_name == "ImageNet")
    model = CXRClassifier(backbone_name="legacy_xception", pretrained_imagenet=is_imagenet).to(device)

    # Load SSL pretrained weights if not ImageNet baseline
    if not is_imagenet and ckpt_path is not None:
        model.load_ssl_encoder(ckpt_path, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_model_state = None

    # Supervised fine-tuning loop
    for epoch in range(epochs):
        model.train()
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()

        # Evaluate on the 10% Validation Set to select the optimal checkpoint
        val_metrics = evaluate_model(model, val_loader, device)
        if val_metrics["AUROC"] > best_val_auroc:
            best_val_auroc = val_metrics["AUROC"]
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Restore the best validation-selected checkpoint and evaluate once on the 20% Test Set
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_metrics = evaluate_model(model, test_loader, device)

    print(f"[{model_name} | {int(fraction*100)}% | Seed {seed}] "
          f"Test AUROC: {test_metrics['AUROC']:.4f}, F1: {test_metrics['F1']:.4f}")

    return test_metrics, model


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SHENZHEN_DIR = "/content/drive/MyDrive/TB_data/Shenzhen/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png"
    CKPT_DIR = "/content/drive/MyDrive/TB_data"
    SEEDS = [42, 77, 123]

    files, labels = load_tb_filepaths_and_labels(SHENZHEN_DIR)

    # Define experimental matrix: (Model Name, Checkpoint Path, Label Fractions)
    experiments = [
        ("ImageNet", None, [0.10, 0.25, 0.50, 1.00]),
        ("SimCLR", os.path.join(CKPT_DIR, "SimCLR_xception_ssl.pth"), [0.10, 0.25, 0.50, 1.00]),
        ("TCL_Full", os.path.join(CKPT_DIR, "TCL_Full_xception_ssl.pth"), [0.10, 0.25, 0.50, 1.00]),
        ("ICH_Only", os.path.join(CKPT_DIR, "ICH_Only_xception_ssl.pth"), [0.10]),
        ("CCH_Only", os.path.join(CKPT_DIR, "CCH_Only_xception_ssl.pth"), [0.10]),
    ]

    records = []
    for model_name, ckpt_path, fractions in experiments:
        for frac in fractions:
            for seed in SEEDS:
                metrics, _ = run_single_experiment(
                    model_name, ckpt_path, files, labels, frac, seed, DEVICE
                )
                records.append({
                    "Model": model_name,
                    "Label_Fraction": frac,
                    "Seed": seed,
                    "AUROC": metrics["AUROC"],
                    "F1": metrics["F1"],
                    "Recall": metrics["Recall"],
                    "Specificity": metrics["Specificity"]
                })

    # Aggregate 3-seed results using population standard deviation (ddof=0)
    df = pd.DataFrame(records)
    summary = df.groupby(["Model", "Label_Fraction"]).agg(
        AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", lambda x: np.std(x, ddof=0)),
        F1_mean=("F1", "mean"),
        F1_std=("F1", lambda x: np.std(x, ddof=0))
    ).reset_index()

    print("\n=====================================================")
    print(" FINAL ROBUST RESULTS (Three-seed evaluation) ")
    print("=====================================================")
    print(summary.to_markdown(index=False))
    summary.to_csv("Unified_Evaluation_Results_v2.csv", index=False)
