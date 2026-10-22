"""
gradcam_vis.py
Fixed for PyTorch 2.x inplace ReLU backward hook conflict and automatic .pth discovery.
"""

import os
from glob import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from data_loader import get_downstream_transforms
from model import CXRClassifier


def disable_inplace_relu(model):
    """
    Disables inplace=True across all ReLU layers in timm's Xception backbone.
    Required in PyTorch 2.x to prevent RuntimeError when using backward hooks for Grad-CAM.
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


class GradCAMExtractor:
    """
    Hooks into the final convolutional block (bn4) of the Xception encoder
    to compute class-discriminative Grad-CAM spatial heatmaps safely.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward hook and tensor-level gradient hook (avoids BackwardHookView conflict)
        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        # Clone output to detach from any downstream inplace operations
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove_hooks(self):
        self.fwd_hook.remove()

    def generate(self, input_tensor):
        """Computes the normalized Grad-CAM heatmap and positive prediction probability."""
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        logit = self.model(input_tensor).view(-1)
        prob = torch.sigmoid(logit).item()

        # Backward pass
        logit.backward()

        # Global average pooling over spatial gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations.detach(), dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize heatmap to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, prob


def plot_gradcam_grid(image_paths, models_dict, device, save_path="/content/drive/MyDrive/TB_data/figure 61.png"):
    """
    Renders the 4x4 qualitative comparison grid and saves it to Google Drive.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    transform = get_downstream_transforms(img_size=224, is_train=False)
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(12, 12))

    col_titles = ["Original CXR", "ImageNet Base", "SimCLR", "TCL"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight="bold")

    for row_idx, img_path in enumerate(image_paths):
        raw_img = Image.open(img_path).convert("RGB").resize((224, 224))
        raw_np = np.array(raw_img) / 255.0

        axes[row_idx, 0].imshow(raw_np)
        axes[row_idx, 0].set_ylabel(os.path.basename(img_path), fontsize=9)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        input_tensor = transform(raw_img).unsqueeze(0).to(device)

        for col_idx, model_key in enumerate(["ImageNet", "SimCLR", "TCL"], start=1):
            model = models_dict[model_key]
            # Hook into bn4 (before act4 ReLU) to completely avoid inplace view errors
            target_layer = model.encoder.bn4 if hasattr(model.encoder, "bn4") else model.encoder.conv4
            extractor = GradCAMExtractor(model, target_layer)
            cam, prob = extractor.generate(input_tensor)
            extractor.remove_hooks()

            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            overlay = 0.55 * raw_np + 0.45 * heatmap
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].axis("off")

            pred_label = "Positive" if prob >= 0.5 else "Negative"
            color = "green" if prob >= 0.5 else "red"
            axes[row_idx, col_idx].text(
                0.5, -0.08, f"Prob: {prob:.3f} ({pred_label})",
                color=color, fontweight="bold", fontsize=10,
                ha="center", transform=axes[row_idx, col_idx].transAxes
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved 4x4 Grad-CAM comparison grid to: {save_path}")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TBDATA_DIR = "/content/drive/MyDrive/TB_data"
    SHENZHEN_DIR = "/content/drive/MyDrive/TB_data/Shenzhen/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png"

    # Print all available .pth files in TB_data so you can verify what is inside your Drive
    all_pth = glob(os.path.join(TBDATA_DIR, "**", "*.pth"), recursive=True)
    print("Found .pth files in TB_data:", [os.path.basename(p) for p in all_pth])

    sample_images = [
        os.path.join(SHENZHEN_DIR, "CHNCXR_0588_1.png"),
        os.path.join(SHENZHEN_DIR, "CHNCXR_0530_1.png"),
        os.path.join(SHENZHEN_DIR, "CHNCXR_0659_1.png"),
        os.path.join(SHENZHEN_DIR, "CHNCXR_0532_1.png"),
    ]

    models_dict = {}
    search_keywords = {
        "ImageNet": ["finetuned_ImageNet", "imagenet"],
        "SimCLR":   ["finetuned_SimCLR", "SimCLR", "simclr"],
        "TCL":      ["finetuned_TCL", "TCL_Full", "BOTH", "tcl"]
    }

    for display_name in ["ImageNet", "SimCLR", "TCL"]:
        is_imagenet = (display_name == "ImageNet")
        model = CXRClassifier(backbone_name="legacy_xception", pretrained_imagenet=is_imagenet).to(DEVICE)

        # Search for matching .pth file in TB_data
        matched_ckpt = None
        for kw in search_keywords[display_name]:
            matches = [p for p in all_pth if kw.lower() in os.path.basename(p).lower()]
            if matches:
                matched_ckpt = matches[0]
                break

        if matched_ckpt is not None:
            state = torch.load(matched_ckpt, map_location=DEVICE)
            if "encoder_state_dict" in state:
                model.encoder.load_state_dict(state["encoder_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print(f"[{display_name}] Loaded weights from: {os.path.basename(matched_ckpt)}")
        else:
            print(f"[{display_name}] No custom .pth found; using default initialization.")

        # CRITICAL FIX: Disable inplace ReLU to prevent BackwardHook RuntimeError
        disable_inplace_relu(model)
        models_dict[display_name] = model

    output_path = "/content/drive/MyDrive/TB_data/figure 61.png"
    plot_gradcam_grid(sample_images, models_dict, DEVICE, save_path=output_path)
