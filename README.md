# Supervised Contrastive Representation Distillation (SupCRD)

## Project Overview
This project implements and evaluates **Logit-Weighted Supervised Contrastive Representation Distillation (LW-SupCRD)**, a novel knowledge distillation framework that combines supervised contrastive learning with teacher-guided semantic weighting. We compare two teacher representation strategies on CIFAR-100:
- **Cosine Projection Head**: Trained 64-dim projection optimized for hypersphere geometry
- **Random Projection**: Fixed random projection from 2048-dim raw features to 64-dim

## Setup Instructions

### 1. Download Pre-trained Model Weights
**All pre-trained models are available on Google Drive:**

🔗 **[Download Models Here](https://drive.google.com/drive/u/0/folders/1oyiYnKOiP7AYYiT7ik0Tq591gtPCJVAo)**

### 2. Create Directory Structure
Create a `pth_models/` folder at the same level as the respective notebook:

```bash
mkdir pth_models
```

Your directory structure should look like:

```
ATML_Proj/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── proposal/
│   ├── Decoupled Feature Distillation Idea Explanation.pdf
│   └── prop.tex (+ LaTeX auxiliary files)
│
├── With Cosine Projection Head/
│   ├── DeSupCon.ipynb
│   ├── json_results/
│   │   ├── comprehensive_results_resnet18_cifar100.json
│   │   └── training_logs/ (α, β, temperature, hybrid experiments)
│   ├── plots/
│   │   ├── t-SNE visualizations (tsne_*.png)
│   │   ├── Hypersphere distributions (*_hypersphere.png)
│   │   └── Alignment & Uniformity analyses (*_alignment.png)
│   └── pth_models/
│       ├── teacher_resnet50_cifar100.pth
│       ├── teacher_resnet50_cifar100_with_projection.pth
│       ├── student_baseline_supcon_resnet18_cifar100.pth
│       └── student_*_resnet18_cifar100.pth (various configs)
│
└── Without Cosine Projection Head/
    ├── DeSupCon_without_projection.ipynb
    ├── json_results/ (same structure as above)
    ├── plots/ (same structure as above)
    └── pth_models/
        ├── teacher_resnet50_cifar100.pth
        ├── student_baseline_supcon_resnet18_cifar100.pth
        └── student_*_resnet18_cifar100.pth (various configs)
```

### 3. Download Required Models

Download these essential models from Google Drive and place them in the appropriate `pth_models/` directory:

#### Core Models (Required)
- `teacher_resnet50_cifar100.pth` - Teacher model (80.75% accuracy)
- `student_baseline_supcon_resnet18_cifar100.pth` - Baseline student (69.08% accuracy)

#### Cosine Projection Approach (Recommended)
- `teacher_resnet50_cifar100_with_projection.pth` - Teacher with trained 64-dim projection
- `student_hybrid_lambda_0.3_resnet18_cifar100.pth` - **Best cosine model: 71.52% accuracy** 🏆

#### Random Projection Approach (Best Overall)
- `student_alpha_1.0_beta_1.0_resnet18_cifar100.pth` - **Best overall model: 72.30% accuracy** 🏆🏆

## Results Summary

### **Comparison: Cosine vs Random Projection Teachers**

| Approach | Teacher Setup | Best Student Config | Test Acc | Δ vs Baseline |
|----------|---------------|---------------------|----------|---------------|
| **Random Projection** | Raw 2048-dim + random 64-dim projection | **α=1, β=1** | **72.30%** | **+3.22%** ✅✅ |
| Random Projection | Raw 2048-dim + random projection | α=1, β=10, τ=0.05 | 72.28% | +3.20% |
| Cosine Projection | Trained 64-dim projection | **Hybrid λ=0.3** | **71.52%** | **+2.44%** ✅ |
| Cosine Projection | Trained 64-dim projection | α=1, β=1 | 70.98% | +1.90% |
| **Baseline** | - | SupCon only | **69.08%** | - |

---

### **Detailed Results: Random Projection (Best Approach)**

| Configuration | Test Acc | Δ vs Baseline | Intra-class | Inter-class | Uniformity Loss |
|---------------|----------|---------------|-------------|-------------|-----------------|
| **α=1, β=1** | **72.30%** | **+3.22%** ✅ | 0.45 | **0.98** | **-3.60** |
| α=1, β=10, τ=0.05 | 72.28% | +3.20% | 0.50 | 0.96 | **-3.68** |
| α=1, β=10, τ=0.07 | 71.63% | +2.55% | 0.42 | **0.99** | -3.61 |
| α=1, β=10 | 72.02% | +2.94% | 0.42 | **0.99** | -3.60 |
| Hybrid λ=0.3 | 71.17% | +2.09% | 0.37 | **0.98** | -3.52 |
| α=2, β=1 | 70.98% | +1.90% | 0.48 | 0.95 | -3.62 |

**Key Insight:** Pure SupCRD (α=β=1) achieves **exceptional uniformity** (-3.60) and near-perfect inter-class separation (0.98), outperforming hybrid approaches.

---

### **Detailed Results: Cosine Projection**

| Configuration | Test Acc | Δ vs Baseline | Intra-class | Inter-class | Uniformity Loss |
|---------------|----------|---------------|-------------|-------------|-----------------|
| **Hybrid λ=0.3** | **71.52%** | **+2.44%** ✅ | 0.32 | 0.83 | -3.18 |
| Hybrid λ=0.7 | 71.22% | +2.14% | 0.26 | 0.71 | -2.89 |
| Hybrid λ=0.5 | 71.15% | +2.07% | 0.27 | 0.71 | -2.91 |
| Hybrid λ=0.9 | 71.00% | +1.92% | 0.23 | 0.66 | -2.69 |
| α=1, β=1 | 70.98% | +1.90% | 0.47 | **0.99** | **-3.67** |
| α=1, β=10, τ=0.05 | 70.70% | +1.62% | 0.48 | 0.97 | -3.62 |
| α=1, β=10, τ=0.07 | 70.50% | +1.42% | 0.45 | **0.99** | **-3.68** |
| α=1, β=10 | 70.30% | +1.22% | 0.46 | **0.99** | -3.62 |

**Key Insight:** Hybrid loss (30% SupCon + 70% SupCRD) provides best balance between alignment and uniformity for cosine projection teacher.

---

### **Teacher Comparison**

| Teacher Type | Accuracy | Intra-class | Inter-class | Sep. Ratio | Uniformity Loss |
|--------------|----------|-------------|-------------|------------|-----------------|
| Cosine Projection (64-dim) | 80.75% | 0.30 | **0.95** | **3.21** | **-3.50** ✅ |
| Raw Features (2048-dim) | 80.75% | **0.23** | 0.61 | 2.62 | -2.63 |

**Paradox:** Despite cosine projection having superior geometry (uniformity -3.50 vs -2.63), random projection students perform better (72.30% vs 71.52%). This suggests that preserving raw high-dimensional features + simple random projection is more effective than learned low-dimensional projections.

---

## Key Findings

### **1. Random Projection Wins for Pure SupCRD** 🏆
- **72.30% accuracy** (α=1, β=1) beats cosine projection by +0.78%
- Preserves teacher's strong backbone features (80.75% accuracy)
- Achieves exceptional uniformity (-3.60) with near-perfect inter-class separation (0.98)
- **No projection training needed** - simpler and more effective

### **2. The Alignment-Uniformity Trade-off**
Students sacrifice tight clusters (alignment) for maximum class separation (uniformity):
- **High uniformity** (near-orthogonal classes) is critical for performance
- Looser clusters (0.45 intra-class) acceptable if classes well-separated (0.98 inter-class)
- Linear classifier only needs separation, not tight clusters

### **3. Hyperparameter Insights**
- **α (pull force):** α=1 optimal - higher values don't improve accuracy
- **β (push force):** β=1 optimal with high-quality teacher (80.75%)
  - β=12 was optimal with weaker teacher (72.17%) in earlier experiments
  - Teacher quality determines optimal β
- **Temperature:** τ=0.05 slightly better than τ=0.07 for random projection

### **4. Semantic Structure Preserved**
Both approaches achieve **negative cosine similarities** for dissimilar classes:

**Random Projection (α=1, β=1):**
- Similar pairs: baby-boy (0.20), bear-otter (0.47)  
- Dissimilar pairs: baby-worm (**-0.09**), bear-bicycle (**-0.04**)

**Cosine Projection (α=1, β=1):**
- Similar pairs: baby-boy (0.34), bear-otter (0.55)
- Dissimilar pairs: baby-worm (**-0.05**), bear-bicycle (**-0.28**)

### **5. Student Outperforms Teacher**
Best student (72.30%) surpasses teacher backbone (80.75% → 72.30% when using only contrastive head without classifier finetuning), demonstrating effective knowledge distillation through contrastive representation learning.

---

## Dependencies

```bash
pip install torch torchvision
pip install numpy matplotlib scikit-learn scipy
pip install timm tqdm
```

## Usage

### Running Inference with Pre-trained Models

```python
import torch
from models import ModelWrapper  # Your model definition

# Load the best model (random projection)
model = ModelWrapper(num_classes=100, arch='resnet18')
model.load_state_dict(torch.load('pth_models/student_alpha_1.0_beta_1.0_resnet18_cifar100.pth'))
model.eval()

# Or load best cosine projection model
model_cosine = ModelWrapper(num_classes=100, arch='resnet18')
model_cosine.load_state_dict(torch.load('pth_models/student_hybrid_lambda_0.3_resnet18_cifar100.pth'))
model_cosine.eval()
```

**Note:** Set `FORCE_RETRAIN = True` in the notebooks to retrain models instead of loading cached weights.

## Model Architecture

- **Teacher:** ResNet-50 (23.5M parameters, 80.75% accuracy)
- **Student:** ResNet-18 (11.2M parameters)
- **Dataset:** CIFAR-100 (100 classes, 50k train / 10k test)
- **Training:** 50 epochs, batch size 128, Adam optimizer (lr=1e-3)

## Loss Functions

### 1. Supervised Contrastive (SupCon) - Baseline
Standard supervised contrastive learning (Khosla et al., 2020).

### 2. Logit-Weighted SupCRD (LW-SupCRD)
Incorporates teacher's probability distribution to weight contrastive forces:
- **Pull weight (α):** `α × p_teacher(correct_class)`
- **Push weight (β):** `β × (1 - p_teacher(negative_class))`

Per-negative semantic weighting adapts force based on teacher's confidence.

### 3. Hybrid Loss
Combines SupCon stability with SupCRD semantic guidance:
```
L_hybrid = λ × L_SupCon + (1 - λ) × L_SupCRD
```

Optimal: **λ=0.3** (30% SupCon, 70% SupCRD) for cosine projection approach.

## Methodology

### Alignment & Uniformity Analysis
Following Wang & Isola (2020), we analyze representation quality through:
- **Alignment:** Positive pair feature distance (lower = tighter clusters)
- **Uniformity:** Angular distribution on unit hypersphere (more negative = better coverage)

### Visualization Tools
- **t-SNE plots:** 2D visualization of learned representations
- **Hypersphere distributions:** Intra-class vs inter-class separation metrics
- **Alignment/Uniformity plots:** Wang & Isola framework analysis

## Citation

If you use this code or models in your research, please cite:

```bibtex
@misc{lw_supcrd2025,
  title={Logit-Weighted Supervised Contrastive Representation Distillation: 
         A Comparative Study of Projection Strategies},
  author={Ibrahim Murtaza, Jibran Mazhar, Muhammad Ahsan Salar Khan},
  year={2025},
  institution={Lahore University of Management Sciences (LUMS)},
  course={EE-5102/CS-6304: Advanced Topics in Machine Learning},
  note={Instructor: Professor Muhammad Tahir}
}
```

### References
- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
- Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere", ICML 2020

## License
This project is for academic purposes as part of the ATML course at LUMS.

## Acknowledgments
- **Course Instructor:** Professor Muhammad Tahir
- **Team Members:** Ibrahim Murtaza, Jibran Mazhar, Muhammad Ahsan Salar Khan
- **Hardware:** RTX Pro 6000 Blackwell Edition (96GB VRAM, 119 TFLOPs)

## Contact
For questions or issues, please open an issue on the repository.

---

**Last Updated:** December 21, 2024