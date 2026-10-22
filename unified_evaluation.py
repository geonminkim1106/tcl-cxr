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
from glob import glob
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


def set_deterministic_seed(seed):
    """Fixes all random number generators for reproducible multi-seed experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_path(base_dir, candidate_names):
    """
    Searches for a pretrained checkpoint (.pth) across the base_dir and its subdirectories
    so that minor filename or folder differences do not trigger a FileNotFoundError.
    """
    for name in candidate_names:
        direct_path = os.path.join(base_dir, name)
        if os.path.exists(direct_path):
            return direct_path
        # Recursive search inside TB_data and current working directory
        matches = glob(os.path.join(base_dir, "**", name), recursive=True)
        if matches:
            return matches[0]
        local_matches = glob(os.path.join(".", "**", name), recursive=True)
        if local_matches:
            return local_matches[0]
    return os.path.join(base_dir, candidate_names[0])


@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Computes rank-based (AUROC) and threshold-dependent (F1, Recall, Specificity)
    metrics on a given evaluation dataloader.
    """
    model.eval()
    all_probs, all_targets = [], []

    for batch in dataloader:
        # Handle both (images, labels, filenames) and (images, labels) formats safely
        images, labels = batch[0], batch[1]
        images = images.to(device)
        # Obtain predicted probabilities via sigmoid activation
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
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
        if os.path.exists(ckpt_path):
            model.load_ssl_encoder(ckpt_path, device=device)
        else:
            raise FileNotFoundError(
                f"[{model_name}] Pretrained checkpoint not found at: {ckpt_path}\n"
                f"Please verify your .pth file location inside /content/drive/MyDrive/TB_data"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_model_state = None

    # Supervised fine-tuning loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, targets = batch[0].to(device), batch[1].to(device).float()
            optimizer.zero_grad()
            logits = model(images).view(-1)
            loss = criterion(logits, targets)
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

    # =========================================================================
    # Google Drive File Paths (Configured to match your exact directory layout)
    # =========================================================================
    TBDATA_DIR = "/content/drive/MyDrive/TB_data"
    SHENZHEN_DIR = "/content/drive/MyDrive/TB_data/Shenzhen/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png"
    MONTGOMERY_DIR = "/content/drive/MyDrive/TB_data/Shenzhen/Montgomery/MontgomerySet/CXR_png"
    NIH_DIR = "/content/drive/MyDrive/TB_data/images_001/images"

    SEEDS = [42, 77, 123]

    # Load Shenzhen dataset for internal fine-tuning and evaluation
    files, labels = load_tb_filepaths_and_labels(SHENZHEN_DIR)
    print(f"Loaded Shenzhen dataset: {len(files)} images (Positive TB: {int(labels.sum())})")

    # Load Montgomery dataset for zero-shot external validation
    mont_files, mont_labels = load_tb_filepaths_and_labels(MONTGOMERY_DIR)
    mont_loader = DataLoader(
        CXRDataset(mont_files, mont_labels, get_downstream_transforms(is_train=False)),
        batch_size=16, shuffle=False
    )
    print(f"Loaded Montgomery dataset: {len(mont_files)} images (Positive TB: {int(mont_labels.sum())})")

    # Automatically locate pretrained SSL checkpoints inside TB_data
    simclr_ckpt = resolve_checkpoint_path(TBDATA_DIR, ["SimCLR_xception_ssl.pth", "simclr_xception.pth", "pretrain_SimCLR.pth"])
    tcl_ckpt    = resolve_checkpoint_path(TBDATA_DIR, ["TCL_Full_xception_ssl.pth", "BOTH_xception_ssl.pth", "tcl_xception.pth", "pretrain_BOTH.pth"])
    ich_ckpt    = resolve_checkpoint_path(TBDATA_DIR, ["ICH_Only_xception_ssl.pth", "ICH_xception_ssl.pth", "pretrain_ICH.pth"])
    cch_ckpt    = resolve_checkpoint_path(TBDATA_DIR, ["CCH_Only_xception_ssl.pth", "CCH_xception_ssl.pth", "pretrain_CCH.pth"])

    # Define experimental matrix: (Model Name, Checkpoint Path, Label Fractions)
    experiments = [
        ("ImageNet", None,        [0.10, 0.25, 0.50, 1.00]),
        ("SimCLR",   simclr_ckpt, [0.10, 0.25, 0.50, 1.00]),
        ("TCL_Full", tcl_ckpt,    [0.10, 0.25, 0.50, 1.00]),
        ("ICH_Only", ich_ckpt,    [0.10]),
        ("CCH_Only", cch_ckpt,    [0.10]),
    ]

    records = []
    ext_records = []

    for model_name, ckpt_path, fractions in experiments:
        for frac in fractions:
            for seed in SEEDS:
                metrics, trained_model = run_single_experiment(
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

                # Run zero-shot external evaluation on Montgomery for the 10% label regime
                if frac == 0.10 and model_name in ["ImageNet", "SimCLR", "TCL_Full"]:
                    ext_metrics = evaluate_model(trained_model, mont_loader, DEVICE)
                    ext_records.append({
                        "Model": model_name,
                        "Seed": seed,
                        "AUROC": ext_metrics["AUROC"],
                        "F1": ext_metrics["F1"],
                        "Recall": ext_metrics["Recall"],
                        "Specificity": ext_metrics["Specificity"]
                    })

    # Aggregate 3-seed Shenzhen results using population standard deviation (ddof=0)
    df = pd.DataFrame(records)
    summary = df.groupby(["Model", "Label_Fraction"]).agg(
        AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", lambda x: np.std(x, ddof=0)),
        F1_mean=("F1", "mean"),
        F1_std=("F1", lambda x: np.std(x, ddof=0))
    ).reset_index()

    print("\n=====================================================")
    print(" FINAL ROBUST RESULTS - SHENZHEN (Three-seed eval) ")
    print("=====================================================")
    print(summary.to_string(index=False))

    out_csv = os.path.join(TBDATA_DIR, "Unified_Evaluation_Results_v2.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved Shenzhen summary to: {out_csv}")

    # Aggregate 3-seed Montgomery external validation results
    if ext_records:
        df_ext = pd.DataFrame(ext_records)
        ext_summary = df_ext.groupby(["Model"]).agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_std=("AUROC", lambda x: np.std(x, ddof=0)),
            F1_mean=("F1", "mean"),
            F1_std=("F1", lambda x: np.std(x, ddof=0)),
            Recall_mean=("Recall", "mean"),
            Recall_std=("Recall", lambda x: np.std(x, ddof=0)),
            Spec_mean=("Specificity", "mean"),
            Spec_std=("Specificity", lambda x: np.std(x, ddof=0))
        ).reset_index()

        print("\n=====================================================")
        print(" EXTERNAL VALIDATION - MONTGOMERY (10% Models) ")
        print("=====================================================")
        print(ext_summary.to_string(index=False))

        ext_csv = os.path.join(TBDATA_DIR, "Montgomery_External_Results_v2.csv")
        ext_summary.to_csv(ext_csv, index=False)
        print(f"Saved Montgomery summary to: {ext_csv}")
