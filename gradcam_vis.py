"""
gradcam_vis.py
Generates the 4x4 Grad-CAM qualitative comparison grid across identical test radiographs
for ImageNet, SimCLR, and TCL, displaying predicted probabilities and spatial attention biases.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from data_loader import get_downstream_transforms


class GradCAMExtractor:
    """
    Hooks into the final depthwise separable convolutional layer of the Xception encoder
    to compute class-discriminative Grad-CAM spatial heatmaps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks on the target convolutional block
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Computes the normalized Grad-CAM heatmap and positive prediction probability."""
        self.model.eval()
        self.model.zero_grad()

        # Forward pass to get binary logit and probability
        logit = self.model(input_tensor)
        prob = torch.sigmoid(logit).item()

        # Backward pass with respect to the positive TB logit
        logit.backward()

        # Global average pooling over spatial gradients: alpha_k = mean_{u,v}(dY / dA^k_{u,v})
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps followed by ReLU
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize heatmap to [0, 1] range
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, prob


def plot_gradcam_grid(image_paths, models_dict, device, save_path="/content/drive/MyDrive/TB_data/figure 61.png"):
    """
    Renders the 4x4 qualitative comparison grid across 4 test patients and 3 models.
    """
    transform = get_downstream_transforms(img_size=224, is_train=False)
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(12, 12))

    col_titles = ["Original CXR", "ImageNet Base", "SimCLR", "TCL"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight="bold")

    for row_idx, img_path in enumerate(image_paths):
        # Load raw grayscale/RGB image for visual overlay
        raw_img = Image.open(img_path).convert("RGB").resize((224, 224))
        raw_np = np.array(raw_img) / 255.0

        # Plot column 0: Original Radiograph
        axes[row_idx, 0].imshow(raw_np)
        axes[row_idx, 0].set_ylabel(img_path.split("/")[-1], fontsize=9)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Prepare normalized input tensor
        input_tensor = transform(raw_img).unsqueeze(0).to(device)

        # Plot columns 1–3: ImageNet, SimCLR, and TCL Grad-CAM overlays
        for col_idx, model_key in enumerate(["ImageNet", "SimCLR", "TCL"], start=1):
            model = models_dict[model_key]
            # Hook into the final convolutional layer of timm's legacy_xception (act4 / conv4)
            extractor = GradCAMExtractor(model, model.encoder.act4)
            cam, prob = extractor.generate(input_tensor)

            # Resize CAM to 224x224 and apply JET colormap
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            # Blend original CXR with heatmap
            overlay = 0.55 * raw_np + 0.45 * heatmap
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].axis("off")

            # Annotate predicted probability and threshold decision below each subplot
            pred_label = "Positive" if prob >= 0.5 else "Negative"
            color = "green" if prob >= 0.5 else "red"
            axes[row_idx, col_idx].text(
                0.5, -0.08, f"Prob: {prob:.3f} ({pred_label})",
                color=color, fontweight="bold", fontsize=10,
                ha="center", transform=axes[row_idx, col_idx].transAxes
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Saved 4x4 Grad-CAM comparison grid to {save_path}")
