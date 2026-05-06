# Multi-Modal Deep Learning for Chest Disease Prediction

**Fusing X-ray Imaging and Clinical Vitals for 14-Class Diagnosis**

> Research project · B.Tech CSE, 2nd Year · Paper submission in progress

---

## The Problem

Most chest X-ray AI systems are unimodal — they diagnose from images alone. Real radiologists don't work this way. They combine visual findings with clinical context: the patient's age, heart rate, oxygen saturation.

Beyond that, a widely-ignored flaw in prior work on the NIH ChestX-ray14 dataset is **patient-level data leakage** — random image splits let the same patient appear in both train and test sets, causing models to memorize patient anatomy instead of learning disease features.

This project addresses both problems.

---

## What We Built

A dual-branch fusion neural network that:
- Processes chest X-ray images through a **DenseNet121 (CheXNet)** backbone
- Simultaneously processes clinical vitals through a lightweight **MLP branch**
- Fuses both feature vectors and classifies **14 concurrent chest diseases**
- Uses a **strict patient-level 70/15/15 split** to prevent data leakage
- Generates **Grad-CAM heatmaps** on the fused architecture for explainability

---

## Results

| Model | Mean AUC-ROC | Notes |
|-------|-------------|-------|
| MLP (clinical only) | 0.700 | Baseline tabular |
| ResNet50 | 0.803 | Image only |
| EfficientNetV2 | 0.822 | Image only |
| CheXNet (DenseNet121) | 0.842 | Image only |
| **Ours (CheXNet + MLP)** | **0.8865** | **Multimodal fusion** |

Top per-class AUC on test set:
- Infiltration: **0.985**
- Emphysema: **0.980**
- Atelectasis: **0.971**
- Cardiomegaly: **0.953**

---

## Architecture

```
X-Ray Image ──► DenseNet121 (pretrained) ──► 1024-dim features ──┐
                                                                   ├──► Fusion Head ──► 14-class output
Clinical Vitals ──► MLP (3-layer) ──────────► 16-dim features ───┘
(Age, Gender, Temp, HR, SpO2)
```

**Training details:**
- Loss: Multi-Label Focal Loss (α=0.25, γ=2.0) to handle class imbalance
- Optimizer: AdamW with weight decay
- Early stopping on validation loss (patience=3)
- Image resolution: 320×320
- Evaluation: Test-Time Augmentation (5-crop) on final metrics

---

## Dataset

[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — 112,120 frontal X-rays from 30,805 unique patients, labeled for 14 thoracic diseases.

> **Note on clinical data:** The NIH dataset does not include EHR vitals. We synthesized physiologically plausible vitals (temperature, heart rate, SpO2) with disease-correlated noise to validate the multimodal architecture. Results with real EHR data (e.g., MIMIC-IV) are expected to improve further — this is the primary direction for future work.

---

## Notebooks

| Notebook | Description | Run |
|----------|-------------|-----|
| [`01_training.ipynb`](notebooks/01_training.ipynb) | Full pipeline: EDA → data split → model training | [Kaggle ↗](https://www.kaggle.com/code/crimsonshadow5/modal-chest-disease-prediction-v2) |
| [`02_evaluation.ipynb`](notebooks/02_evaluation.ipynb) | TTA evaluation + Grad-CAM visualization | [Kaggle ↗](https://www.kaggle.com/code/crimsonshadow5/multi-modal-chest-disease-prediction-eval) |

Pretrained weights (`.pth`) available as a Kaggle dataset output from the training notebook.

---

## Repository Structure

```
├── notebooks/
│   ├── 01_training.ipynb       # Training pipeline
│   └── 02_evaluation.ipynb     # Evaluation + Grad-CAM
├── src/
│   ├── model.py                # MultimodalFusionNet architecture
│   └── dataset.py              # MultimodalXRayDataset + transforms
├── assets/
│   ├── roc_curves.png          # Per-class ROC curves
│   └── gradcam_sample.png      # Grad-CAM visualization
├── paper.pdf                   # Research paper (submission in progress)
└── requirements.txt
```

---

## Key Design Decisions

**Why patient-level split?** With ~3.7 scans per patient on average (some patients have 100+), a naive image-level random split guarantees test leakage. Splitting by Patient ID means the model never sees any scan from a test patient during training.

**Why Focal Loss?** The NIH dataset is heavily imbalanced — "No Finding" dominates, and rare classes like Hernia appear in <0.2% of images. Standard BCE gives equal weight to all examples; Focal Loss re-weights toward hard, under-represented cases.

**Why fuse at feature level (late fusion)?** Early fusion (concatenating raw inputs) destroys the spatial structure of the image. Feature-level concatenation lets each branch develop its own learned representation before combining.

---

## Limitations & Future Work

- Clinical vitals are **synthetic**, not real EHR data — the architecture is validated but the clinical branch's contribution is bounded by simulation fidelity
- Future work: train on a naturally multimodal dataset such as [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) paired with MIMIC-IV EHR records
- Transformer-based cross-modal attention (instead of simple concatenation) is a natural architectural upgrade

---

## Requirements

```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
opencv-python>=4.8
Pillow>=10.0
```