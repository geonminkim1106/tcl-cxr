import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from PIL import Image
import timm

def set_seed(seed=42):
    """Locks all random seeds for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_stratified_subset(df, fraction, random_seed):
    """Extracts a stratified subset of the data (e.g., 10%) while preserving class distribution."""
    if fraction == 1.0: return df
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=random_seed)
    subset_idx, _ = next(sss.split(df, df['label']))
    return df.iloc[subset_idx].reset_index(drop=True)

class TBClassificationDataset(Dataset):
    """Downstream dataset for binary TB classification."""
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'filename']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'label']

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class TBClassifier(nn.Module):
    """Downstream Classifier utilizing the pretrained Xception backbone."""
    def __init__(self, pretrained_model_path=None):
        super(TBClassifier, self).__init__()
        
        # Load standard ImageNet weights if no pretrained_model_path is provided
        use_imagenet = pretrained_model_path is None
        self.backbone = timm.create_model('xception', pretrained=use_imagenet, num_classes=0)
        
        # Inject self-supervised weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            state_dict = torch.load(pretrained_model_path, map_location='cpu')
            # Extract only the backbone weights, ignoring the projection heads
            backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
        # Binary classification head replacing the original ImageNet head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1) # Return shape [Batch] for BCEWithLogitsLoss

def evaluate_model(model, val_loader, criterion, device):
    """Evaluates the model on the validation set and returns AUROC and F1 metrics."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float() # Threshold at 0.5

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return {'AUROC': auroc, 'F1': f1}

def analyze_confusion_matrix(model, val_loader, device, threshold=0.5, model_name="Model"):
    """
    Evaluates the model on validation data and prints TP, TN, FP, FN, Precision, and Recall.
    """
    model.eval()
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Ensure output shape is compatible
            outputs = model(images)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
                
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate components of the confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0) # Sensitivity
    
    predicted_positive_ratio = (tp + fp) / len(all_labels) if len(all_labels) > 0 else 0
    
    print(f"=== {model_name} (Threshold: {threshold}) ===")
    print(f"Total Validation Samples: {len(all_labels)}")
    print(f"TP (True Positive - Hit)        : {tp}")
    print(f"FN (False Negative - Missed TB) : {fn}")
    print(f"TN (True Negative - Correct)    : {tn}")
    print(f"FP (False Positive - False Alarm): {fp}")
    print(f"--------------------------------------")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall (Sensitivity) : {recall:.4f}")
    print(f"Predicted Pos. Ratio : {predicted_positive_ratio:.4f}\n")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dir = "/content/drive/MyDrive/TB_data/Shenzhen/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png"
    
    # Paths to the pre-trained models from Phase 1 (Added ImageNet Baseline)
    weight_paths = {
        "ImageNet_Baseline": None,
        "ICH_Only": "/content/drive/MyDrive/TB_data/ich_model.pth",
        "CCH_Only": "/content/drive/MyDrive/TB_data/cch_model.pth",
        "TCL_Full (ICH+CCH)": "/content/drive/MyDrive/TB_data/both_model.pth"
    }

    # Extract labels from filenames (0 for Normal, 1 for TB)
    file_list = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    df_list = [{'filename': f, 'label': int(f.split('_')[-1][0])} for f in file_list]
    real_df = pd.DataFrame(df_list)

    # 80/20 Stratified Split for Train/Validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(real_df, real_df['label']))
    full_train_df, val_df = real_df.iloc[train_idx].reset_index(drop=True), real_df.iloc[val_idx].reset_index(drop=True)

    # Downstream Augmentations (includes standard normalization)
    train_transform = T.Compose([
        T.Resize((224, 224)), T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = TBClassificationDataset(val_df, image_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Set label efficiency to 10%
    frac = 0.1 
    seeds = [42, 77, 123]
    ablation_summary = []

    # Loop through each ablation architecture
    for model_name, w_path in weight_paths.items():
        if w_path is not None and not os.path.exists(w_path): 
            continue
            
        auroc_results, f1_results = [], []
        
        # Evaluate across 3 different random seeds for statistical robustness
        for seed in seeds:
            set_seed(seed)
            train_df = get_stratified_subset(full_train_df, fraction=frac, random_seed=seed)
            train_dataset = TBClassificationDataset(train_df, image_dir, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)

            model = TBClassifier(pretrained_model_path=w_path).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            best_auroc, best_f1 = 0.0, 0.0
            
            # Short fine-tuning cycle (5 epochs) to evaluate representation quality
            for epoch in range(5):
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                val_metrics = evaluate_model(model, val_loader, criterion, device)
                best_auroc = max(best_auroc, val_metrics['AUROC'])
                best_f1 = max(best_f1, val_metrics['F1'])
            
            # Print Confusion Matrix right after the 5-epoch cycle for the 10% data regime
            if frac == 0.1:
                analyze_confusion_matrix(model, val_loader, device, threshold=0.5, model_name=f"{model_name} (Seed {seed})")

            auroc_results.append(best_auroc)
            f1_results.append(best_f1)
 
        # Aggregate results
        ablation_summary.append({
            'Model': model_name,
            'AUROC (Mean ± Std)': f"{np.mean(auroc_results):.4f} ± {np.std(auroc_results):.4f}",
            'F1-Score (Mean ± Std)': f"{np.mean(f1_results):.4f} ± {np.std(f1_results):.4f}"
        })

    # Output markdown table for easy GitHub copy-pasting
    print(pd.DataFrame(ablation_summary).to_markdown(index=False))
