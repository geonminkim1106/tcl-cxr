# TCL-CXR: An Empirical Comparison of Contrastive Learning Objectives for Tuberculosis Classification in Low-Label Regimes

**Independent Research Project**

An empirical investigation comparing instance-level (**SimCLR**) and cluster-level (**Twin Contrastive Learning, TCL**) self-supervised pretraining objectives for label-efficient chest X-ray (CXR) tuberculosis classification.

<p align="center">
  <img src="assets/images/pipeline.png" width="750">
</p>
<p align="center"><i>Figure 1. Overall SSL pretraining (NIH ChestX-ray14) and downstream evaluation pipeline (Shenzhen & Montgomery TB datasets).</i></p>

---

## 1. Experimental Intent & Motivation
In medical imaging, acquiring large-scale expert-annotated data is costly and labor-intensive. While Self-Supervised Learning (SSL) mitigates this by learning visual representations from unlabeled radiographs, it remains unclear whether adding structural complexity—such as explicit cluster-level constraints—to the contrastive objective actually improves downstream diagnostic performance under severe label scarcity.

Originally inspired by the CGRclust Twin Contrastive Learning (TCL) framework developed for unlabeled DNA sequence clustering (Alipour et al., 2024), this project initially aimed to adapt TCL to chest X-ray tuberculosis (TB) classification. However, rigorously controlled multi-seed evaluations revealed a counter-intuitive finding: unconstrained instance-level contrastive learning (SimCLR) consistently outperforms structured cluster-level contrastive learning (TCL) across all label budgets and external domain evaluations.

Consequently, the primary objectives of this study evolved to:
1. **Controlled Empirical Comparison:** Evaluate how structural differences in pretraining objectives (ImageNet transfer learning vs. SimCLR vs. TCL) affect downstream TB classification across varying label budgets (10%, 25%, 50%, and 100%).
2. **Dissecting Objective Components & Failure Modes:** Conduct an ablation study on TCL’s core components—the Instance-level Contrastive Head (ICH) and Cluster-level Contrastive Head (CCH)—and analyze threshold sensitivity to understand why rank-based discrimination (AUROC) and fixed-threshold operational performance (F1-score) diverge.
3. **Visualizing Spatial Bias & Collapse:** Utilize Grad-CAM to investigate whether cluster-level constraints induce spatial representation collapse or shortcut learning on non-pathological image artifacts.

### Research Questions
* **Main Question:** How do different pretraining objectives—ImageNet initialization, SimCLR, and Twin Contrastive Learning (TCL)—affect tuberculosis classification performance under varying label budgets?
  1. *Label Efficiency:* How does downstream classification performance scale across 10%, 25%, 50%, and 100% labeled training subsets?
  2. *External Generalization:* Do the relative performance differences observed on the internal dataset persist on an unseen external hospital cohort (Montgomery dataset)?
  3. *Objective Structure & Visual Patterns:* What do component ablations, threshold sweeps, and Grad-CAM visualizations reveal about optimization stability and spatial attention bias?

---

## 2. Experimental Design

### Datasets & Split Protocol
| Dataset | Purpose | Total Images | Protocol / Split | Status |
| :--- | :--- | :---: | :--- | :---: |
| **NIH ChestX-ray14** | SSL Pretraining | 4,005 used (112,120 total) | Patient-level split, AP/PA views only | Completed |
| **Shenzhen TB** | Downstream Fine-tuning & Test | 662 | 70% Train Pool / 10% Val / 20% Held-out Test | Completed |
| **Montgomery TB** | Zero-shot External Validation | 138 | Full cohort evaluated (Cross-domain) | Completed |

* **Leakage Prevention & Label Budgets:** The NIH dataset was strictly split at the patient level prior to sampling 4,005 unlabeled radiographs. The Shenzhen dataset (inherently one radiograph per patient) was partitioned into a 70% training pool, 10% validation set, and 20% held-out test set. Label budgets (10%, 25%, 50%, and 100%) represent stratified subsets sampled from the 70% training pool (e.g., the 10% budget utilizes ~7% of the total dataset).
* **Unified Evaluation Protocol:** All models share the exact same train/validation/test splits. Model checkpoints are selected exclusively based on validation AUROC, and final metrics are reported on the independent 20% test set across 3 random seeds (42, 77, 123) using population standard deviation ($\text{ddof}=0$).

### Methodology & Architecture
* **Common Backbone:** `Xception` (via `timm`), chosen for its parameter efficiency via depthwise separable convolutions and kept identical across all methods to isolate the effect of the pretraining objective.
* **Domain-Specific SSL Augmentation:** Designed to preserve subtle peripheral lesions while simulating realistic clinical acquisition variations:
  * *Rotation & Translation:* `RandomAffine(degrees=5, translate=(0.02, 0.02))`
  * *Lesion-Preserving Crop:* `RandomResizedCrop(scale=(0.8, 1.0))`
  * *Intensity & Sensor Noise:* `ColorJitter(brightness=0.2, contrast=0.2, p=0.5)` and `GaussianBlur(kernel_size=3, p=0.5)`
* **Compared Pretraining Objectives:**
  1. `ImageNet Baseline`: Standard supervised initialization from natural images.
  2. `SimCLR`: Pure instance-level contrastive learning maximizing agreement between augmented views via a nonlinear projection head (NT-Xent loss).
  3. `TCL (ICH + CCH)`: Dual-head architecture simultaneously enforcing instance-level discrimination (ICH, row-space) and cluster-level assignment constraints (CCH, column-space).
  4. `ICH-Only` & `CCH-Only`: Ablation variants isolating each head of the TCL framework.

---

## 3. Results

### A. Label-Budget Scaling on Shenzhen Dataset (Internal Test)
| Label Budget | SimCLR AUROC | SimCLR F1 | TCL AUROC | TCL F1 | ImageNet AUROC | ImageNet F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | **0.8702 ± 0.0093** | **0.7928 ± 0.0095** | 0.7306 ± 0.0484 | 0.4135 ± 0.2976 | 0.8173 ± 0.0285 | 0.5911 ± 0.0642 |
| **25%** | **0.9053 ± 0.0040** | **0.8460 ± 0.0264** | 0.8324 ± 0.0264 | 0.7591 ± 0.0365 | 0.8664 ± 0.0093 | 0.7795 ± 0.0099 |
| **50%** | **0.9239 ± 0.0069** | **0.8699 ± 0.0094** | 0.9031 ± 0.0075 | 0.8377 ± 0.0184 | 0.9003 ± 0.0086 | 0.8408 ± 0.0062 |
| **100%** | **0.9369 ± 0.0086** | **0.8900 ± 0.0056** | 0.9316 ± 0.0109 | 0.8620 ± 0.0043 | 0.9195 ± 0.0147 | 0.8634 ± 0.0205 |

<p align="center">
  <img src="assets/images/figure 2.jpg" width="750">
</p>
<p align="center"><i>Figure 2. Downstream AUROC and F1-Score scaling curves across labeled data fractions (10%–100%).</i></p>

* **Key Observation:** **SimCLR** consistently achieved the highest AUROC and F1-score with the lowest seed-to-seed variance across all label budgets. Under extreme label scarcity (10%), SimCLR maintained a robust F1-score of **0.7928**, whereas **TCL** suffered from severe optimization instability (F1 **0.4135 ± 0.2976**, including complete prediction collapse at threshold 0.5 in one seed), underperforming even the ImageNet baseline (**0.5911**).

---

### B. External Cross-Domain Generalization (Montgomery Dataset)
| Method | AUROC (Mean ± Std) | F1-Score (Th=0.5) | Recall (Sensitivity) | Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **ImageNet Baseline** | 0.677 ± 0.041 | 0.518 ± 0.160 | 0.575 ± 0.304 | 0.629 ± 0.247 |
| **SimCLR** | **0.774 ± 0.027** | **0.677 ± 0.042** | **0.851 ± 0.028** | 0.513 ± 0.152 |
| **TCL (Full)** | 0.682 ± 0.018 | 0.590 ± 0.014 | 0.598 ± 0.043 | **0.688 ± 0.107** |

<p align="center">
  <img src="assets/images/figure 3.png" width="750">
</p>
<p align="center"><i>Figure 3. Internal (Shenzhen) vs. Cross-Domain (Montgomery) generalization performance.</i></p>

* **Key Observation:** When evaluated directly on the unseen Montgomery cohort without re-tuning, **SimCLR** preserved the highest discriminative power (AUROC **0.774**) and operational F1-score (**0.677**), driven by strong sensitivity (Recall **0.851**). While TCL exhibited higher specificity (**0.688**), its overall diagnostic sensitivity dropped sharply, indicating distinct operating trade-offs under domain shift.

---

### C. Threshold Sensitivity & TCL Component Ablation (10% Label Budget)
| Model Variant | AUROC (Mean ± Std) | F1-Score (Th=0.5, Mean ± Std) |
| :--- | :---: | :---: |
| **ImageNet Baseline** | 0.8173 ± 0.0285 | 0.5911 ± 0.0642 |
| **ICH-Only (Instance Head)** | 0.6992 ± 0.0527 | **0.6339 ± 0.0606** |
| **CCH-Only (Cluster Head)** | **0.8171 ± 0.0322** | 0.5812 ± 0.0844 |
| **TCL-Full (ICH + CCH)** | 0.7306 ± 0.0484 | 0.4135 ± 0.2976 |

<p align="center">
  <img src="assets/images/figure 4.jpg" width="750">
</p>
<p align="center"><i>Figure 4. Threshold sensitivity analysis showing F1-Score and Recall across decision thresholds (0.1–0.9).</i></p>

<p align="center">
  <img src="assets/images/figure 5.png" width="480">
</p>
<p align="center"><i>Figure 5. Ablation study comparing individual TCL heads (ICH-Only, CCH-Only, and TCL-Full) at the 10% label budget.</i></p>

* **Rank vs. Threshold Divergence:** In the ablation study, `CCH-Only` achieved a relatively high AUROC (**0.8171**) but a low F1-score (**0.5812**), whereas `ICH-Only` showed lower AUROC (**0.6992**) but superior F1-score (**0.6339**). During pretraining, the `CCH-Only` loss plateaued near $\ln(1024) \approx 6.93$, indicating stagnant cluster assignment. Combining both heads in `TCL-Full` resulted in negative synergy under the 10% regime, severely degrading probability calibration around the default `0.5` decision boundary.

---

## 4. Qualitative Analysis & Discussion (Grad-CAM)

<p align="center">
  <img src="assets/images/figure 61.jpg" width="750">
</p>
<p align="center"><i>Figure 6. Grad-CAM spatial activation maps and predicted probabilities across identical test radiographs for ImageNet, SimCLR, and TCL (10% label budget).</i></p>

### Spatial Activation Patterns & Potential Bias
1. **TCL's Horizontal Band Artifact & Probabilistic Collapse:** Across multiple test radiographs, `TCL` repeatedly generated a horizontal band-like activation along the lower thorax while outputting nearly invariant probabilities hovering at **0.508–0.509**. This suggests that forcing rigid cluster assignments on heterogeneous CXRs may induce optimization difficulties or spatial shortcut learning rather than anatomically grounded representations.
2. **ImageNet's Boundary & Mediastinal Bias:** `ImageNet` heavily focused on high-contrast edges such as the central cardiac shadow and diaphragm, with output probabilities stagnating at **0.497**, reflecting severe domain shift under minimal fine-tuning labels.
3. **SimCLR's Broader Thoracic Coverage (and Remaining Shortcuts):** Without rigid clustering constraints, `SimCLR` maintained a continuous representation space that activated more broadly across the lung fields and produced better-separated prediction probabilities. However, its heatmaps also partially spilled into peripheral background corners, demonstrating that unconstrained SSL still requires spatial guidance (e.g., lung segmentation masks) to fully eliminate background shortcuts.

---

## 5. Limitations & Future Work

### Limitations
* **Dataset Scale:** Downstream fine-tuning and external validation were conducted on relatively compact public cohorts (Shenzhen and Montgomery), warranting validation on larger multi-center clinical repositories.
* **Single Backbone & Uncalibrated Thresholds:** Experiments were controlled using an Xception backbone and a fixed default threshold (`0.5`) for primary comparisons.
* **Exploratory Nature of Grad-CAM:** Saliency maps provide qualitative insights into spatial attention but do not alone establish causal mechanisms for representation collapse.

### Future Work
* **Lung-Field Constrained SSL:** Integrate anatomical lung segmentation masks during contrastive pretraining to suppress background shortcut learning.
* **Embedding Diagnostics & Threshold Calibration:** Quantify representation collapse via embedding variance/rank metrics and evaluate validation-tuned operating thresholds under cross-domain shift.
* **Architecture Generalization:** Extend the empirical comparison to ResNet/DenseNet/ViT backbones and additional SSL paradigms (e.g., MoCo v2, BYOL).

---

## 6. Repository Structure & Reproducibility

```text
TCL-CXR/
├── assets/
│   └── images/
│       ├── pipeline.png
│       ├── figure 2.jpg          # Label-scaling curves (AUROC & F1)
│       ├── figure 3.png          # Cross-domain generalization (Shenzhen vs. Montgomery)
│       ├── figure 4.jpg          # Threshold sensitivity sweep
│       ├── figure 5.png          # TCL component ablation bar chart
│       └── figure 61.jpg         # 4x4 Grad-CAM qualitative comparison grid
├── data_loader.py
├── model.py
├── pretrain_ablation.py
├── unified_evaluation.py         # Unified 3-seed Val-selected / Held-out Test evaluation
├── gradcam_vis.py
├── requirements.txt
└── README.md
