"""
model.py
Neural network architectures and contrastive loss functions using a common
Xception backbone for fair empirical comparison across SSL objectives.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class XceptionSSLModel(nn.Module):
    """
    Unified Self-Supervised Learning architecture supporting:
      - SimCLR (Unconstrained instance-level contrastive learning)
      - TCL (Twin Contrastive Learning combining ICH and CCH)
      - ICH-Only / CCH-Only (Ablation variants)
    """
    def __init__(self, backbone_name="legacy_xception", proj_dim=128, num_clusters=10, pretrained_imagenet=False):
        super().__init__()
        # Load the common Xception encoder without the classification head (num_classes=0)
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained_imagenet, num_classes=0
        )
        # Retrieve the output feature dimension of Xception (2048)
        feat_dim = self.encoder.num_features

        # Instance-level Contrastive Head (ICH) / SimCLR 2-layer MLP Projection Head
        self.instance_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim)
        )

        # Cluster-level Contrastive Head (CCH) mapping features to cluster assignment probabilities
        self.cluster_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_clusters),
            nn.Softmax(dim=1)  # Enforce soft cluster probability distribution across rows
        )

    def forward(self, x):
        # Extract 2048-dimensional global pooled representation h from Xception
        h = self.encoder(x)
        # Compute L2-normalized instance projection z for row-space NT-Xent loss
        z = F.normalize(self.instance_projector(h), dim=1)
        # Compute soft cluster assignment probabilities c for column-space contrastive loss
        c = self.cluster_projector(h)
        return h, z, c


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy (NT-Xent) Loss.
    Used for SimCLR and the Instance-level Contrastive Head (ICH) in row-space.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        # Concatenate representations from both augmented views: shape (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute pairwise cosine similarity matrix scaled by temperature: shape (2N, 2N)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarity along the main diagonal
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Construct ground-truth indices matching each view i with its positive pair i + N
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        # Compute cross-entropy loss across all in-batch negative pairs
        loss = F.cross_entropy(sim_matrix, pos_indices)
        return loss


class ClusterContrastiveLoss(nn.Module):
    """
    Cluster-level Contrastive Head (CCH) Loss operating in column-space,
    coupled with an entropy regularization term to prevent trivial single-cluster collapse.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, c_i, c_j):
        # Transpose cluster assignment matrices (N, K) -> (K, N) to contrast cluster columns
        c_i_t = F.normalize(c_i.T, dim=1)
        c_j_t = F.normalize(c_j.T, dim=1)
        num_clusters = c_i_t.size(0)

        # Concatenate normalized cluster vectors: shape (2K, N)
        c = torch.cat([c_i_t, c_j_t], dim=0)
        sim_matrix = torch.matmul(c, c.T) / self.temperature

        # Mask self-similarity along the diagonal
        mask = torch.eye(2 * num_clusters, device=c.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Target indices matching cluster k in view 1 with cluster k in view 2
        pos_indices = torch.cat([
            torch.arange(num_clusters, 2 * num_clusters, device=c.device),
            torch.arange(0, num_clusters, device=c.device)
        ], dim=0)

        cluster_loss = F.cross_entropy(sim_matrix, pos_indices)

        # Compute negative entropy penalty to encourage uniform cluster utilization
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()

        ne_loss = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-8)).sum() \
                  + math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-8)).sum()

        return cluster_loss + ne_loss


class CXRClassifier(nn.Module):
    """
    Downstream Tuberculosis binary classification model.
    Discards SSL projection heads and attaches a linear classification layer to Xception.
    """
    def __init__(self, backbone_name="legacy_xception", pretrained_imagenet=False):
        super().__init__()
        # Initialize Xception feature extractor
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained_imagenet, num_classes=0
        )
        feat_dim = self.encoder.num_features
        # Single linear classification head outputting raw binary logit
        self.classifier = nn.Linear(feat_dim, 1)

    def load_ssl_encoder(self, checkpoint_path, device="cpu"):
        """
        Loads pretrained SSL encoder weights and performs a strict integrity check
        to ensure zero missing or unexpected keys.
        """
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Extract encoder weights if wrapped in a full SSL state dictionary
        if "encoder_state_dict" in state_dict:
            state_dict = state_dict["encoder_state_dict"]
        else:
            # Filter out projection head keys (instance_projector / cluster_projector)
            state_dict = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("encoder.") or not ("projector" in k or "classifier" in k)
            }

        # Load into self.encoder and print verification diagnostics
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"  [Weight Check] Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features).squeeze(-1)
        return logits
