# TCL-CXR: An Empirical Comparison of Contrastive Learning Objectives for Tuberculosis Classification in Low-Label Regimes

**Independent Research Project**

An empirical comparison of supervised ImageNet pretraining and self-supervised contrastive learning approaches—SimCLR and Twin Contrastive Learning (TCL)—for tuberculosis (TB) classification from chest X-ray (CXR) images under limited-label conditions.

<p align="center">
  <img src="assets/images/pipeline.png" width="750" alt="TCL-CXR experimental pipeline">
</p>
<p align="center"><em>Figure 1. Overview of self-supervised pretraining on NIH ChestX-ray14 and downstream evaluation on the Shenzhen and Montgomery TB datasets.</em></p>

---

## 1. Motivation and Research Questions

In medical imaging, obtaining large, expert-annotated datasets can be costly and labor-intensive. Self-supervised learning (SSL) offers a way to learn image representations from unlabeled radiographs, but it remains an open empirical question whether adding cluster-level constraints to an instance-level contrastive objective improves downstream performance when labeled data are scarce.

This project was initially motivated by the Twin Contrastive Learning (TCL) framework used for unlabeled DNA-sequence clustering (Alipour et al., 2024). The original goal was to investigate whether TCL could be adapted to TB classification from chest X-rays. In the experiments conducted so far, SimCLR achieved higher mean AUROC and F1-score than TCL across the evaluated label budgets. The project therefore focuses on an empirical comparison of pretraining approaches and an examination of the conditions under which their performance differs.

### Research Questions

- **Label Efficiency:** How does performance change when downstream training uses 10%, 25%, 50%, or 100% of the labeled training pool?
- **External Evaluation:** Do the relative performance patterns observed on Shenzhen persist when the models are evaluated on the Montgomery dataset without retraining or threshold tuning?
- **Objective Components and Prediction Behavior:** What do TCL component ablations, threshold-sensitivity analyses, and Grad-CAM visualizations reveal about performance variability and spatial activation patterns?

### Study Objectives

1. Compare supervised ImageNet initialization, SimCLR, and TCL across different labeled-data budgets.
2. Evaluate the individual TCL heads—the Instance-level Contrastive Head (ICH) and Cluster-level Contrastive Head (CCH)—in an ablation study.
3. Examine threshold sensitivity and Grad-CAM visualizations as exploratory analyses of prediction behavior and spatial activation patterns.

---

## 2. Experimental Design

### Datasets and Split Protocol

| Dataset | Role | Dataset Size | Protocol |
| :--- | :--- | :---: | :--- |
| **NIH ChestX-ray14** | SSL pretraining | 4,005 images used from 112,120 available images | AP/PA views only; patient-level split as configured in the experiment |
| **Shenzhen TB** | Downstream training, validation, and internal testing | 662 images | 70% training pool / 10% validation / 20% held-out test |
| **Montgomery TB** | External evaluation | 138 images | Full cohort evaluated for cross-dataset generalization |

The 10%, 25%, 50%, and 100% label budgets are sampled from the 70% Shenzhen training pool. Thus, the label-budget percentages refer to the training pool rather than the full dataset. The Shenzhen dataset is described in this project as containing one radiograph per patient.

All methods are intended to use the same downstream train/validation/test partitions. Checkpoints are selected using validation AUROC, and internal test results are reported across three random seeds (42, 77, and 123). Mean and population standard deviation (`ddof=0`) are reported for the Shenzhen results.

> **Reproducibility note:** Verify the split strategy, patient-level separation, sample counts, and evaluation settings against the released code and logs before treating them as confirmed guarantees. Claims about leakage prevention should only be made if the implementation supports them.

### Models and Pretraining Objectives

All experiments use an Xception backbone through `timm`.

1. **ImageNet Baseline:** Xception initialized with supervised weights learned from natural images.
2. **SimCLR:** Instance-level contrastive learning that brings augmented views of the same image together in representation space while contrasting them with other images, using an NT-Xent-style objective.
3. **TCL (ICH + CCH):** A dual-head approach combining instance-level contrastive learning (ICH) with cluster-level constraints (CCH).
4. **ICH-Only and CCH-Only:** Ablation variants intended to examine the two TCL components separately.

The ImageNet baseline differs from the SSL methods in both pretraining data and supervision. Therefore, comparisons between ImageNet and the SSL methods should be interpreted as comparisons between the tested pretraining approaches, rather than as isolating the effect of a single objective component.

### SSL Augmentations

The pretraining pipeline uses the following augmentations, as specified in the current experiment configuration:

- **Geometric:** `RandomAffine(degrees=5, translate=(0.02, 0.02))`
- **Crop:** `RandomResizedCrop(scale=(0.8, 1.0))`
- **Intensity and Blur:** `ColorJitter(brightness=0.2, contrast=0.2, p=0.5)` and `GaussianBlur(kernel_size=3, p=0.5)`

These augmentations introduce variation in the training views. The crop operation is not guaranteed to preserve every lesion.

---

## 3. Results

### A. Label-Budget Scaling on Shenzhen

The table reports mean ± population standard deviation across three seeds. F1-scores use the fixed decision threshold of 0.5.

| Label Budget | SimCLR AUROC | SimCLR F1 | TCL AUROC | TCL F1 | ImageNet AUROC | ImageNet F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | **0.8702 ± 0.0093** | **0.7928 ± 0.0095** | 0.7306 ± 0.0484 | 0.4135 ± 0.2976 | 0.8173 ± 0.0285 | 0.5911 ± 0.0642 |
| **25%** | **0.9053 ± 0.0040** | **0.8460 ± 0.0264** | 0.8324 ± 0.0264 | 0.7591 ± 0.0365 | 0.8664 ± 0.0093 | 0.7795 ± 0.0099 |
| **50%** | **0.9239 ± 0.0069** | **0.8699 ± 0.0094** | 0.9031 ± 0.0075 | 0.8377 ± 0.0184 | 0.9003 ± 0.0086 | 0.8408 ± 0.0062 |
| **100%** | **0.9369 ± 0.0086** | **0.8900 ± 0.0056** | 0.9316 ± 0.0109 | 0.8620 ± 0.0043 | 0.9195 ± 0.0147 | 0.8634 ± 0.0205 |

<p align="center">
  <img src="assets/images/figure 2.png" width="750" alt="AUROC and F1-score across label budgets">
</p>
<p align="center"><em>Figure 2. AUROC and F1-score across labeled-data budgets on the Shenzhen test set.</em></p>

**Key Observation:** SimCLR achieved the highest mean AUROC and F1-score at all four evaluated label budgets. At the 10% label budget, TCL showed substantial seed-to-seed variability in F1-score (0.4135 ± 0.2976), including one run with an F1-score of 0 at the fixed 0.5 threshold. This result warrants further investigation into prediction behavior and training stability; it does not by itself establish the cause of the low score.

### B. External Evaluation on Montgomery

| Method | AUROC (Mean ± SD) | F1-Score (Threshold = 0.5) | Recall / Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **ImageNet Baseline** | 0.677 ± 0.041 | 0.518 ± 0.160 | 0.575 ± 0.304 | 0.629 ± 0.247 |
| **SimCLR** | **0.774 ± 0.027** | **0.677 ± 0.042** | **0.851 ± 0.028** | 0.513 ± 0.152 |
| **TCL (Full)** | 0.682 ± 0.018 | 0.590 ± 0.014 | 0.598 ± 0.043 | **0.688 ± 0.107** |

<p align="center">
  <img src="assets/images/figure 3.png" width="750" alt="Internal and external evaluation comparison">
</p>
<p align="center"><em>Figure 3. Performance comparison on the Shenzhen internal test set and the Montgomery external cohort.</em></p>

**Key Observation:** In the reported Montgomery evaluation, SimCLR achieved the highest AUROC, F1-score, and recall, while TCL had the highest specificity. This indicates different operating trade-offs at the fixed threshold. These results should be interpreted in light of the small size of the external cohort.

### C. Threshold Sensitivity and TCL Component Ablation

The following ablation results use the 10% label budget. F1-scores are measured at the fixed threshold of 0.5.

| Model Variant | AUROC (Mean ± SD) | F1-Score (Mean ± SD) |
| :--- | :---: | :---: |
| **ImageNet Baseline** | 0.8173 ± 0.0285 | 0.5911 ± 0.0642 |
| **ICH-Only** | 0.6992 ± 0.0527 | **0.6339 ± 0.0606** |
| **CCH-Only** | **0.8171 ± 0.0322** | 0.5812 ± 0.0844 |
| **TCL-Full (ICH + CCH)** | 0.7306 ± 0.0484 | 0.4135 ± 0.2976 |

<p align="center">
  <img src="assets/images/figure 4.png" width="750" alt="F1-score and recall across decision thresholds">
</p>
<p align="center"><em>Figure 4. F1-score and recall across decision thresholds from 0.1 to 0.9.</em></p>

<p align="center">
  <img src="assets/images/figure 5.png" width="480" alt="TCL component ablation comparison">
</p>
<p align="center"><em>Figure 5. Comparison of ICH-Only, CCH-Only, and TCL-Full at the 10% label budget.</em></p>

**Key Observation:** CCH-Only achieved an AUROC similar to the ImageNet baseline but a lower mean F1-score at the fixed threshold. ICH-Only achieved the highest mean F1-score among the listed variants, while TCL-Full had the lowest mean F1-score and substantial variability. Under this experimental setup, combining the two heads did not improve the reported metrics over the better-performing ablation. Further diagnostics are needed to determine why.

The CCH-Only pretraining loss was observed to plateau near `ln(1024) ≈ 6.93`. This is reported as an optimization observation; the loss value alone is insufficient to establish cluster-assignment collapse without considering the loss formulation and additional training diagnostics.

AUROC measures ranking discrimination across thresholds, whereas F1-score depends on a selected decision threshold. A difference between these metrics does not, by itself, demonstrate poor probability calibration. Threshold-sweep results should be interpreted as sensitivity analysis, and any threshold selected for deployment should be chosen using validation data rather than the held-out test set.

---

## 4. Exploratory Grad-CAM Analysis

<p align="center">
  <img src="assets/images/figure 6.png" width="750" alt="Grad-CAM comparison across ImageNet, SimCLR, and TCL">
</p>
<p align="center"><em>Figure 6. Grad-CAM maps and predicted probabilities for the same test radiographs, comparing ImageNet, SimCLR, and TCL at the 10% label budget.</em></p>

The Grad-CAM visualizations provide qualitative examples of where the models' predictions are associated with activation in the selected feature maps. They can motivate further investigation, but they do not establish causal explanations for model behavior.

1. **TCL:** In the examined examples, TCL showed a recurring horizontal band-like activation near the lower thorax and predicted probabilities around 0.508–0.509. These observations raise questions about prediction stability and spatial activation patterns, but do not establish their underlying cause.
2. **ImageNet Baseline:** Some examined maps emphasized high-contrast regions, including boundaries around the cardiac shadow and diaphragm. This may reflect sensitivity to prominent image structures, but the maps alone cannot establish that these features are spurious.
3. **SimCLR:** The examined maps showed activation across broader thoracic regions, with some activation extending into peripheral background areas. This pattern motivates further testing of spatial guidance, such as lung segmentation, but does not prove that segmentation would improve performance.

---

## 5. Limitations and Future Work

### Limitations

- **Dataset Size:** The downstream and external evaluations use relatively small public datasets. Larger and more diverse multi-center cohorts are needed to assess generalizability.
- **Single Backbone:** The current comparison uses Xception and does not establish whether the observed ranking generalizes to other architectures.
- **Fixed Decision Threshold:** The primary F1-score comparisons use a threshold of 0.5. Alternative operating thresholds may produce different precision–recall trade-offs.
- **Exploratory Visualization:** Grad-CAM is qualitative and does not, on its own, establish shortcut learning, causal mechanisms, or representation collapse.
- **Scope of Comparison:** ImageNet pretraining differs from SimCLR and TCL in both data domain and supervision. The observed differences cannot be attributed solely to the contrastive objective.

### Future Work

- **Lung-Field Constrained Learning:** Test whether lung segmentation masks or other spatial constraints affect performance and reduce activation outside the lung fields.
- **Representation Diagnostics:** Evaluate embedding variance, effective rank, and other diagnostics to investigate potential representation collapse.
- **Threshold Analysis:** Compare the fixed-threshold results with operating thresholds selected on validation data, including evaluation under domain shift.
- **Architecture and Method Generalization:** Extend the comparison to other backbones (e.g., ResNet, DenseNet, or ViT) and SSL approaches (e.g., MoCo v2 or BYOL).

---

## 6. Repository Structure and Reproducibility

```text
TCL-CXR/
├── assets/
│   └── images/
│       ├── pipeline.png
│       ├── figure 2.png
│       ├── figure 3.png
│       ├── figure 4.png
│       ├── figure 5.png
│       └── figure 6.png
├── data_loader.py
├── model.py
├── pretrain_ablation.py
├── unified_evaluation.py
├── gradcam_vis.py
├── requirements.txt
└── README.md
```

### Installation and Usage

**Requirements:** Python 3.11+, PyTorch 2.x, `timm`, and `scikit-learn`.

```bash
# 0. Install dependencies (Python 3.11+, PyTorch 2.x, timm, scikit-learn)
pip install -r requirements.txt

# 1. SSL Pretraining on NIH ChestX-ray14
python pretrain_ablation.py

# 2. Unified Downstream Evaluation (10%–100% Label Budgets + Ablation across 3 Seeds)
python unified_evaluation.py

# 3. Grad-CAM & Threshold Sweep Visualization
python gradcam_vis.py
```

Before running the scripts, configure the dataset paths and verify the expected directory structure in `data_loader.py`. Exact dependencies, data paths, checkpoint locations, and output locations should match the current implementation.

## 7. References and Data Sources

- Alipour, F., Hill, K. A., & Kari, L. (2024). CGRclust: Chaos Game Representation for twin contrastive clustering of unlabelled DNA sequences. BMC Genomics, 25, Article 1214. https://doi.org/10.1186/s12864-024-11135-y

## Disclaimer

This repository is for research and educational purposes only. The models have not been established as clinically validated diagnostic systems and should not be used to make medical decisions.
