"""
pretrain_ablation.py
Self-Supervised Pretraining script on the NIH ChestX-ray14 dataset (4,005 patient-split subset).
Trains SimCLR, ICH-Only, CCH-Only, and TCL-Full sequentially under identical conditions.
"""

import os
from glob import glob
import torch
from torch.utils.data import DataLoader
from data_loader import CXRDataset, get_ssl_transforms
from model import XceptionSSLModel, NTXentLoss, ClusterContrastiveLoss


def train_ssl_objective(mode, dataloader, device, epochs=50, lr=3e-4, save_dir="/content/drive/MyDrive/TB_data"):
    """
    Executes SSL pretraining for a specified objective ('SimCLR', 'ICH_Only', 'CCH_Only', 'TCL_Full').
    """
    os.makedirs(save_dir, exist_ok=True)
    # Initialize common Xception SSL architecture
    model = XceptionSSLModel(backbone_name="legacy_xception").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Instantiate instance-level (row-space) and cluster-level (column-space) losses
    criterion_ich = NTXentLoss(temperature=0.5).to(device)
    criterion_cch = ClusterContrastiveLoss(temperature=1.0).to(device)

    print(f"\n Starting SSL Pretraining for [{mode}] ({epochs} Epochs)...")
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for views in dataloader:
            # Unpack positive pair of augmented radiographs
            x_i, x_j = views[0].to(device), views[1].to(device)
            optimizer.zero_grad()

            # Forward pass through Xception encoder and projection heads
            _, z_i, c_i = model(x_i)
            _, z_j, c_j = model(x_j)

            # Select objective function based on experimental variant
            if mode in ["SimCLR", "ICH_Only"]:
                # Pure instance-level contrastive learning without cluster constraints
                loss = criterion_ich(z_i, z_j)
            elif mode == "CCH_Only":
                # Pure cluster-level contrastive learning (observed to plateau near ln(1024) ≈ 6.93)
                loss = criterion_cch(c_i, c_j)
            elif mode == "TCL_Full":
                # Twin Contrastive Learning combining both row-space and column-space objectives
                loss = criterion_ich(z_i, z_j) + criterion_cch(c_i, c_j)
            else:
                raise ValueError(f"Unknown SSL mode: {mode}")

            # Backpropagation and parameter update
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{mode}] Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f}")

    # Save purely the encoder state_dict for downstream fine-tuning transfer
    ckpt_path = os.path.join(save_dir, f"{mode}_xception_ssl.pth")
    torch.save({"encoder_state_dict": model.encoder.state_dict()}, ckpt_path)
    print(f"Saved pretrained encoder weights to {ckpt_path}")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NIH_DATA_DIR = "/content/drive/MyDrive/TB_data/images_001/images"  # Path to patient-split 4,005 NIH radiographs

    # Load unlabeled file paths and build SSL DataLoader
    nih_files = sorted(glob(os.path.join(NIH_DATA_DIR, "*.png")))
    ssl_dataset = CXRDataset(file_paths=nih_files, labels=None, transform=get_ssl_transforms(224))
    ssl_loader = DataLoader(ssl_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    # Sequentially pretrain all compared objectives
    for objective_mode in ["SimCLR", "TCL_Full", "ICH_Only", "CCH_Only"]:
        train_ssl_objective(objective_mode, ssl_loader, DEVICE, epochs=50)    
