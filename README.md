# Logit-Weighted Supervised Contrastive Representation Distillation (LW-SupCRD)

## Project Overview
This project implements and evaluates **Logit-Weighted Supervised Contrastive Representation Distillation (LW-SupCRD)**, a novel knowledge distillation framework that combines supervised contrastive learning with teacher-guided semantic weighting on CIFAR-100.

**Key Innovation:** Uses teacher logits to semantically weight contrastive forces, achieving superior representation learning compared to standard supervised contrastive methods.

## Setup Instructions

### 1. Download Pre-trained Model Weights
**All pre-trained models are available on Google Drive:**

🔗 **[Download Models Here](https://drive.google.com/drive/u/0/folders/1oyiYnKOiP7AYYiT7ik0Tq591gtPCJVAo)**

### 2. Create Directory Structure
Create a `pth_models/` folder at the project root:

```bash
mkdir pth_models
```

Your directory structure should look like:

```
ATML_Proj/
│
├── .gitignore
├── DeSupCon.ipynb
├── desupcon.py
├── LICENSE
├── README.md
├── requirements.txt
│
├── proposal/
│   ├── Decoupled Feature Distillation Idea Explanation.pdf
│   └── prop.tex (+ LaTeX auxiliary files)
│
├── json_results/
│   ├── comprehensive_results_resnet18_cifar100.json
│   └── training_logs/
│       ├── teacher_resnet50_cifar100.json
│       ├── student_baseline_supcon_resnet18_cifar100.json
│       ├── student_baseline_crd_resnet18_cifar100.json
│       ├── student_undistilled_resnet18_cifar100.json
│       ├── student_alpha_*.json (α sweep experiments)
│       ├── student_*_beta_*.json (β sweep experiments)
│       ├── student_*_temp_*.json (temperature sweep)
│       ├── baseline_crd_nobank_training_log.json
│       ├── baseline_crd_bank4096_training_log.json
│       ├── lwsupcrd_nobank_training_log.json
│       └── lwsupcrd_bank4096_training_log.json
│
├── plots/
│   ├── t-SNE visualizations (tsne_*.png)
│   ├── 3D hypersphere distributions (*_hypersphere.html)
│   └── Alignment & Uniformity analyses (*_alignment.png)
│
└── pth_models/
    ├── teacher_resnet50_cifar100.pth (80.75% acc)
    ├── teacher_resnet50_cifar100_with_projection.pth
    ├── student_baseline_supcon_resnet18_cifar100.pth (69.08%)
    ├── student_baseline_crd_resnet18_cifar100.pth (68.05%)
    ├── student_undistilled_resnet18_cifar100.pth (67.93%)
    ├── student_alpha_1.0_beta_10.0_temp_0.07_resnet18_cifar100.pth ⭐ (73.35% - BEST)
    ├── student_baseline_crd_nobank_resnet18_cifar100.pth (68.15%)
    ├── student_baseline_crd_bank4096_resnet18_cifar100.pth (69.56%)
    ├── student_lwsupcrd_nobank_resnet18_cifar100.pth (74.76%)
    ├── student_lwsupcrd_bank4096_resnet18_cifar100.pth (75.63%)
    └── student_*_resnet18_cifar100.pth (various configurations)
```

### 3. Download Required Models

Download these essential models from Google Drive and place them in `pth_models/`:

#### Core Models (Required)
- `teacher_resnet50_cifar100.pth` - Teacher model (80.75% accuracy)
- `teacher_resnet50_cifar100_with_projection.pth` - Teacher with trained 64-dim cosine projection
- `student_baseline_supcon_resnet18_cifar100.pth` - Baseline SupCon student (69.08%)

#### Best Model (Recommended) 🏆
- `student_alpha_1.0_beta_10.0_temp_0.07_resnet18_cifar100.pth` - **73.35% accuracy** (best original)
- `student_lwsupcrd_bank4096_resnet18_cifar100.pth` - **75.63% accuracy** (best with memory bank)

#### Baselines & Ablations (Optional)
- `student_baseline_crd_resnet18_cifar100.pth` - Baseline CRD (68.05%)
- `student_undistilled_resnet18_cifar100.pth` - Undistilled student (67.93%)
- Various α, β, temperature, and memory bank configurations

---

## Results Summary

### **Main Results - Best Configurations**

| Method | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Key Features |
|--------|----------|-------------|-------------|--------------|--------------|
| **🏆 LW-SupCRD + Bank4096** | **75.63%** | **+6.55%** | 0.8336 | **-3.5073** | Best overall - momentum memory bank |
| LW-SupCRD (no bank) | 74.76% | +5.68% | 1.0835 | **-3.7314** | Best uniformity |
| **LW-SupCRD (τ=0.07)** | **73.35%** | **+4.27%** | 1.1990 | -3.7104 | Original best (no bank) |
| Baseline CRD + Bank4096 | 69.56% | +0.48% | 0.8098 | -2.0739 | Memory bank helps baseline |
| **Baseline SupCon** | 69.08% | - | 0.4377 | -2.5665 | Strong alignment, weak uniformity |
| Baseline CRD (no bank) | 68.15% | -0.93% | 0.9162 | -2.2804 | Poor both metrics |
| Baseline CRD (original) | 68.05% | -1.03% | 0.9008 | -2.2358 | In-batch only |
| Undistilled Student | 67.93% | -1.15% | 0.6631 | -1.7332 | Terrible uniformity |
| **Teacher (ResNet-50)** | 80.75% | +11.67% | 0.5928 | -3.4649 | Reference upper bound |

**Key Takeaway:** LW-SupCRD with momentum memory bank achieves **75.63%**, a **+6.55%** improvement over baseline SupCon. The semantic weighting combined with 4,096 momentum-updated negatives provides optimal balance between alignment (0.8336) and uniformity (-3.5073).

---

### **Comprehensive Experimental Results**

#### 1. **Alpha (α) Sweep - Pull Force Weighting**
*Configuration: β=10, τ=0.07, adaptive β, no memory bank*

| α | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Observation |
|---|----------|-------------|-------------|--------------|-------------|
| **1.0** | **73.19%** | **+4.11%** | 1.1518 | **-3.7027** | ✅ Optimal balance |
| 2.0 | 71.78% | +2.70% | **1.1129** | -3.6744 | Tighter clusters → worse uniformity |
| 5.0 | 70.67% | +1.59% | 1.1589 | -3.6712 | Over-clustering |
| 10.0 | 70.07% | +0.99% | 1.2754 | -3.6728 | Severe over-clustering |

**Finding:** α=1 optimal - higher α causes tighter clusters, sacrificing uniformity and causing overfitting.

---

#### 2. **Beta (β) Sweep - Push Force Strength**
*Configuration: α=1, τ=0.07, adaptive β, no memory bank*

| β | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Observation |
|---|----------|-------------|-------------|--------------|-------------|
| **10.0** | **73.19%** | **+4.11%** | **1.1518** | **-3.7027** | ✅ Optimal - strong push |
| 12.0 | 71.31% | +2.23% | 1.2068 | -3.6654 | Too strong → degradation |
| 5.0 | 70.63% | +1.55% | 1.2487 | -3.6785 | Weak push → poor separation |
| 1.0 | 70.46% | +1.38% | 1.1862 | -3.6390 | Very weak push |

**Finding:** β=10 optimal - balances strong class separation with stable training. Too high causes instability, too low fails to separate classes.

---

#### 3. **Temperature (τ) Sweep - Gradient Sharpness**
*Configuration: α=1, β=10, adaptive β, no memory bank*

| τ | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Observation |
|---|----------|-------------|-------------|--------------|-------------|
| **0.07** | **73.35%** | **+4.27%** | 1.1990 | **-3.7104** | ✅ Optimal - balanced spread |
| 0.05 | 68.08% | -1.00% | 1.3503 | -3.6645 | Too sharp → poor alignment |

**Finding:** τ=0.07 provides optimal gradient flow - τ=0.05 too sharp, only closest pairs contribute.

---

#### 4. **Memory Bank Ablation - Momentum-Based Negative Sampling**
*Configuration: Batch=128, WD=1e-4, 50 epochs with cosine annealing*

| Method | Bank Size | Test Acc | Train Acc | Gap | Alignment ↓ | Uniformity ↓ | Observation |
|--------|-----------|----------|-----------|-----|-------------|--------------|-------------|
| **LW-SupCRD** | **4096** | **75.63%** | **95.72%** | 20.09% | **0.8336** | -3.5073 | ✅ Best overall |
| LW-SupCRD | None | 74.76% | 89.80% | 14.96% | 1.0835 | **-3.7314** | Best uniformity |
| Baseline CRD | 4096 | 69.56% | 78.57% | 9.01% | 0.8098 | -2.0739 | Bank helps baseline |
| Baseline CRD | None | 68.15% | 73.70% | 5.55% | 0.9162 | -2.2804 | Weak both metrics |

**Memory Bank Impact:**
- **LW-SupCRD:** +0.87% (4,096 negatives vs 127 in-batch)
- **Baseline CRD:** +1.41% (larger relative improvement)
- **Key Finding:** Momentum memory bank (K=4,096, momentum=0.5) provides measurable gains by improving uniformity through diverse negative sampling

**Implementation Details:**
- **Index-based:** One slot per training image (50,000 entries)
- **Momentum update:** `memory[i] = 0.5 × memory_old[i] + 0.5 × feature_new[i]`
- **Random sampling:** Sample K=4,096 negatives from full memory per batch
- **Gradient scaling:** 4,223 total negatives (127 in-batch + 4,096 bank) = 33× more gradient terms

**Why Banks Help:**
- More negatives → better uniformity (wider hypersphere coverage)
- Momentum smoothing → prevents staleness (vs failed FIFO queue approach)
- Complements semantic weighting → LW-SupCRD captures most benefit already, bank adds marginal gains

---

## Key Findings

### **1. The Alignment-Uniformity Trade-off for CIFAR-100** 📊

For fine-grained classification (100 classes), **uniformity is more critical than tight alignment:**

**Best Methods (74-76%):**
- Alignment: ~0.83-1.08 (moderate clusters)
- Uniformity: ~-3.50 to -3.73 (excellent spread)
- Strategy: Trade cluster tightness for class separation

**Baseline Methods (68-69%):**
- Alignment: ~0.44-0.91 (tight to moderate clusters)
- Uniformity: ~-2.07 to -2.57 (poor spread)
- Problem: Insufficient hypersphere coverage

**Counter-intuitive Insight:** Student's looser alignment (0.83-1.08 vs teacher's 0.59) actually helps generalization by maintaining better class separation on the hypersphere.

---

### **2. Hyperparameter Roles & Interactions**

**α (Pull Weight) - Semantic Confidence:**
- Controls cluster tightness via teacher probabilities
- α=1 optimal: Minimal semantic weighting
- Higher α → tighter clusters → worse uniformity → overfitting
- Effect: Primarily degrades uniformity

**β (Push Weight) - Negative Force Strength:**
- Controls class separation strength
- β=10 optimal: Strong push forces
- Critical discovery: Affects **both** alignment AND uniformity simultaneously
- Unlike α, strong β improves both metrics

**τ (Temperature) - Gradient Sharpness:**
- Controls exponential scaling in similarity
- τ=0.07 optimal: Balanced gradient flow
- τ=0.05 too sharp: Only nearest neighbors contribute
- Effect: Primarily affects uniformity

**Adaptive β - Curriculum Learning:**
- Early epochs (uncertain): β_eff = 1.25β (stronger push)
- Late epochs (confident): β_eff = 0.71β (weaker push)
- Provides natural hard negative mining

---

### **3. Memory Banks as Uniformity Boosters** 🚀

**Three Failed Approaches:**
1. **FIFO Queue (MoCo-style):** Stale features from 30+ batches → NaN losses
2. **Aggressive LR + FIFO:** Loss spikes to 26+, accuracy stuck at ~7%
3. **Strong WD (5e-4) + Batch128:** Feature collapse after epoch 11

**Successful Approach - Momentum Memory Bank:**
- **Index-based with momentum:** Prevents staleness while maintaining freshness
- **Random sampling:** 4,096 negatives from 50,000-entry memory
- **Optimization challenge:** 33× more gradient terms requires careful tuning
- **Best config:** WD=1e-4, cosine annealing, 50 epochs

**Key Insight:** Memory banks improve uniformity (more negatives = better coverage), but LW-SupCRD's semantic weighting already captures most of the benefit. Bank provides marginal +0.87% gain vs +1.41% for baseline CRD.

---

### **4. Student Surpasses Teacher in Uniformity** 🎯

| Metric | Teacher | LW-SupCRD (no bank) | LW-SupCRD + Bank |
|--------|---------|---------------------|------------------|
| Alignment | **0.5928** | 1.0835 | 0.8336 |
| Uniformity | -3.4649 | **-3.7314** | -3.5073 |
| Accuracy | 80.75% | 74.76% | 75.63% |

**Key Insight:** No-bank variant achieves best uniformity (-3.7314), even surpassing teacher (-3.4649). Adding memory bank trades some uniformity for better accuracy through improved alignment.

---

### **5. Gradient Normalization Critical** ⚙️

The `/α` normalization in the loss prevents gradient saturation:
- Without: α=2 causes exponentials ~exp(20) = 4.8×10⁸
- With: Allows proper α scaling without optimization collapse
- Enables exploration of α>1 configurations

This fix was essential for all α sweep experiments to work.

---

## Technical Details

### Model Architecture
- **Teacher:** ResNet-50 (23.5M parameters, 80.75% accuracy)
- **Student:** ResNet-18 (11.2M parameters)
- **Projection:** 2048-dim backbone → 64-dim contrastive space
- **Dataset:** CIFAR-100 (100 classes, 50k train / 10k test)
- **Training:** 50 epochs, batch size 128, Adam optimizer (lr=1e-3)

### Loss Functions

#### 1. **Baseline SupCon** (Khosla et al., 2020)
Standard supervised contrastive learning - pull positives only.

#### 2. **Baseline CRD** (Tian et al., 2020)
Contrastive Representation Distillation - instance matching (adapted as in-batch contrastive).

#### 3. **LW-SupCRD** (Ours)
Logit-weighted supervised contrastive with adaptive forces:

```python
# Pull weight (semantic confidence)
w_pull = α × p_teacher(correct_class)

# Push weight (inverse adaptive)
if adaptive_beta:
    β_effective = β / (p_target + 0.5)
    w_push = β_effective × (1 - p_teacher(negative_class))
else:
    w_push = β × (1 - p_teacher(negative_class))

# Gradient normalization
loss = -log((w_pull × pos_exp) / (w_pull × pos_exp + w_push × neg_exp))
loss = loss / α  # CRITICAL: prevents gradient saturation
```

#### 4. **Momentum Memory Bank**
```python
# Index-based storage (50,000 entries)
memory[i] = 0.5 × memory_old[i] + 0.5 × feature_new[i]

# Random sampling (4,096 negatives per batch)
sample_idx = randperm(50000)[:4096]
bank_negatives = memory[sample_idx]
```

---

## Visualization & Analysis

### Alignment & Uniformity Metrics (Wang & Isola, 2020)

**Alignment Loss (↓ better):**
```
L_align = E[||f(x) - f(x+)||²]
```
Measures positive pair distance - lower = tighter clusters.

**Uniformity Loss (↓ better, more negative):**
```
L_uniform = log(E[exp(-2||f(x) - f(y)||²)])
```
Measures hypersphere coverage - more negative = better spread.

### Available Visualizations

All experiments include:
- **t-SNE plots:** 2D projection of learned representations (20 classes)
  - Static PNG files: `plots/tsne_*.png`
  - Open directly in any image viewer
  
- **3D Hypersphere:** Interactive Plotly visualizations (`.html` files)
  - Interactive HTML files: `plots/*_hypersphere.html`
  - **How to view:** Simply double-click the `.html` file to open in your web browser
  - Features: Rotate with mouse, zoom with scroll, click legend to toggle classes
  - Shows both "All Classes" and "10 Random Classes" views side-by-side
  - Example files:
    - `plots/lwsupcrd_bank4096_hypersphere.html` - Best model (75.63%)
    - `plots/temp_0.07_hypersphere.html` - Best no-bank (73.35%)
    - `plots/baseline_supcon_hypersphere.html` - Baseline comparison
  
- **Alignment/Uniformity:** Comprehensive Wang & Isola analysis
  - Static PNG files: `plots/*_alignment.png`
  - Shows histograms, density plots, and per-class angular distributions
  
- **Training logs:** JSON files with per-epoch metrics
  - Location: `json_results/training_logs/*.json`
  - Contains: epoch-by-epoch loss, accuracy, learning rate

---

## Dependencies

```bash
pip install torch torchvision
pip install numpy matplotlib scikit-learn scipy
pip install plotly  # For interactive 3D visualizations
pip install tqdm
```

---

## Viewing Interactive Visualizations

### 3D Hypersphere Plots (HTML Files)

The repository includes **interactive 3D hypersphere visualizations** that show how representations are distributed on the unit sphere:

**To view:**
1. Navigate to the `plots/` folder
2. Double-click any `*_hypersphere.html` file
3. Your default web browser will open the visualization

**Interactive controls:**
- **Rotate:** Click and drag with mouse
- **Zoom:** Scroll wheel
- **Pan:** Right-click and drag (or Shift + drag)
- **Toggle classes:** Click legend items to show/hide classes
- **Reset view:** Double-click the plot

**Recommended visualizations to explore:**
```
plots/lwsupcrd_bank4096_hypersphere.html       # 🏆 Best model (75.63%)
plots/temp_0.07_hypersphere.html               # Best no-bank (73.35%)
plots/baseline_supcon_hypersphere.html         # Compare with baseline (69.08%)
plots/undistilled_hypersphere.html             # See the improvement
plots/baseline_crd_bank4096_hypersphere.html   # Memory bank impact on baseline
```

**What to look for:**
- **Left panel:** All 100 classes - should show good spread across sphere
- **Right panel:** 10 random classes - shows cluster separation clearly
- **Color gradient:** Represents class labels (0-99)
- **Point density:** Tighter clusters = better alignment, spread = better uniformity

**Browser compatibility:** Works best in Chrome/Firefox/Edge. Safari may have minor rendering differences.

---

## Usage

### Running the Code

1. Open `DeSupCon.ipynb` in Jupyter or run `desupcon.py` directly
2. Download required models from Google Drive
3. Place models in `pth_models/` directory
4. Run all cells sequentially

**Training Control:**
- Set `FORCE_RETRAIN = True` to retrain models (ignores cached weights)
- Set `FORCE_RETRAIN = False` to load pre-trained models (default)
- Set `FORCE_RETRAIN_BANK_ABLATION = True` to retrain memory bank experiments

### Loading Best Model

```python
import torch
from models import ModelWrapper

# Load best model (with memory bank)
model = ModelWrapper(num_classes=100, arch='resnet18')
checkpoint = torch.load('pth_models/student_lwsupcrd_bank4096_resnet18_cifar100.pth')
model.load_state_dict(checkpoint)
model.eval()

# Inference
with torch.no_grad():
    features, projections, logits = model(images)
```

---

## Experimental Protocol

### Teacher Training
1. Train ResNet-50 on CIFAR-100 → 80.75% accuracy
2. Train cosine similarity projection head (2048→64D)
3. Joint training: projection adapts during student training (CRD-style)

### Student Training
1. Multi-view augmentation (2 views per sample)
2. Contrastive loss on encoder projections
3. Separate linear classifier on frozen features (standard evaluation)
4. 50 epochs, batch size 128, Adam (lr=1e-3)
5. Optional: Momentum memory bank (K=4096, momentum=0.5)

### Comprehensive Analysis Per Experiment
- t-SNE visualizations (20 sample classes)
- 3D hypersphere distribution (interactive HTML)
- Wang & Isola alignment-uniformity metrics
- Intra/inter-class distance analysis
- Separation ratio computation
- Training curves (JSON logs)
- Model checkpointing for reproducibility

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{lw_supcrd2025,
  title={Logit-Weighted Supervised Contrastive Representation Distillation: 
         Achieving Superior Knowledge Transfer through Semantic Force Weighting 
         and Momentum Memory Banks},
  author={Ibrahim Murtaza, Jibran Mazhar, Muhammad Ahsan Salar Khan},
  year={2025},
  institution={Lahore University of Management Sciences (LUMS)},
  course={EE-5102/CS-6304: Advanced Topics in Machine Learning},
  instructor={Professor Muhammad Tahir},
  note={Best Configuration: α=1.0, β=10.0, τ=0.07, Bank=4096 achieving 75.63% on CIFAR-100}
}
```

### Key References
- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
- Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere", ICML 2020
- Tian et al., "Contrastive Representation Distillation", ICLR 2020
- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020

---

## Reproducibility

### Hardware
- **GPU:** RTX Pro 6000 Blackwell Edition
- **VRAM:** 96GB
- **Compute:** 119 TFLOPs
- **Training Time:** ~2-3 hours per configuration

### Random Seeds
All experiments use fixed random seeds for reproducibility:
```python
torch.manual_seed(42)
np.random.seed(42)
```

### Model Weights Distribution
All trained models available on Google Drive with:
- Model checkpoints (`.pth` files)
- Training logs (`.json` files)
- Comprehensive visualizations (`.png`, `.html`)

---

## Future Work

1. **Extended Architectures:** Test on deeper networks (ResNet-101, WideResNet)
2. **Larger Datasets:** Evaluate on ImageNet, iNaturalist
3. **Larger Memory Banks:** Test K=16,384 with optimized training
4. **Multi-Teacher:** Ensemble knowledge from multiple teachers
5. **Theoretical Analysis:** Formal proof of alignment-uniformity trade-off
6. **Publication:** Prepare for submission to WACV/BMVC

---

## Acknowledgments

- **Course Instructor:** Professor Muhammad Tahir
- **Team Members:** Ibrahim Murtaza, Jibran Mazhar, Muhammad Ahsan Salar Khan
- **Institution:** Lahore University of Management Sciences (LUMS)
- **Hardware Support:** RTX Pro 6000 Blackwell Edition (96GB VRAM)

Special thanks to:
- Khosla et al. for Supervised Contrastive Learning
- Wang & Isola for the alignment-uniformity framework
- Tian et al. for Contrastive Representation Distillation
- He et al. for MoCo momentum-based memory banks

---

## License
This project is for academic purposes as part of the ATML course at LUMS.

## Contact
For questions or issues, please open an issue on the repository or contact the team members.

---

**Last Updated:** December 27, 2025

**Status:** ✅ All experiments completed | 📊 Results finalized | 🎯 Best model: 75.63% accuracy (with memory bank)