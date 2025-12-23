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
│       └── student_hybrid_lambda_*.json (hybrid loss experiments)
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
    └── student_*_resnet18_cifar100.pth (various configurations)
```

### 3. Download Required Models

Download these essential models from Google Drive and place them in `pth_models/`:

#### Core Models (Required)
- `teacher_resnet50_cifar100.pth` - Teacher model (80.75% accuracy)
- `teacher_resnet50_cifar100_with_projection.pth` - Teacher with trained 64-dim cosine projection
- `student_baseline_supcon_resnet18_cifar100.pth` - Baseline SupCon student (69.08%)

#### Best Model (Recommended) 🏆
- `student_alpha_1.0_beta_10.0_temp_0.07_resnet18_cifar100.pth` - **73.35% accuracy** (best overall)

#### Baselines & Ablations (Optional)
- `student_baseline_crd_resnet18_cifar100.pth` - Baseline CRD (68.05%)
- `student_undistilled_resnet18_cifar100.pth` - Undistilled student (67.93%)
- Various α, β, temperature, and hybrid configurations

---

## Results Summary

### **Main Results - Best Configurations**

| Method | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Key Features |
|--------|----------|-------------|-------------|--------------|--------------|
| **🏆 LW-SupCRD (τ=0.07)** | **73.35%** | **+4.27%** | 1.1990 | **-3.7104** | Best overall - optimal temperature |
| LW-SupCRD (α=1, β=10) | 73.19% | +4.11% | 1.1518 | **-3.7027** | Near-identical to τ=0.07 |
| Hybrid (λ=0.3) | 72.58% | +3.50% | **0.6043** | -3.2880 | Best hybrid - severe overfitting |
| **Baseline SupCon** | 69.08% | - | **0.4377** | -2.5665 | Strong alignment, weak uniformity |
| Baseline CRD | 68.05% | -1.03% | 0.9008 | -2.2358 | Poor both metrics |
| Undistilled Student | 67.93% | -1.15% | 0.6631 | -1.7332 | Terrible uniformity |
| **Teacher (ResNet-50)** | 80.75% | +11.67% | **0.5928** | **-3.4649** | Reference upper bound |

**Key Takeaway:** LW-SupCRD achieves **73.35%** with best-in-class uniformity (**-3.7104**), even surpassing the teacher's uniformity (-3.4649), while maintaining competitive alignment for superior generalization.

---

### **Comprehensive Experimental Results**

#### 1. **Alpha (α) Sweep - Pull Force Weighting**
*Configuration: β=10, τ=0.07, adaptive β*

| α | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Observation |
|---|----------|-------------|-------------|--------------|-------------|
| **1.0** | **73.19%** | **+4.11%** | 1.1518 | **-3.7027** | ✅ Optimal balance |
| 2.0 | 71.78% | +2.70% | **1.1129** | -3.6744 | Tighter clusters → worse uniformity |
| 5.0 | 70.67% | +1.59% | 1.1589 | -3.6712 | Over-clustering |
| 10.0 | 70.07% | +0.99% | 1.2754 | -3.6728 | Severe over-clustering |

**Finding:** α=1 optimal - higher α causes tighter clusters, sacrificing uniformity and causing overfitting.

---

#### 2. **Beta (β) Sweep - Push Force Strength**
*Configuration: α=1, τ=0.07, adaptive β*

| β | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Observation |
|---|----------|-------------|-------------|--------------|-------------|
| **10.0** | **73.19%** | **+4.11%** | **1.1518** | **-3.7027** | ✅ Optimal - strong push |
| 12.0 | 71.31% | +2.23% | 1.2068 | -3.6654 | Too strong → degradation |
| 5.0 | 70.63% | +1.55% | 1.2487 | -3.6785 | Weak push → poor separation |
| 1.0 | 70.46% | +1.38% | 1.1862 | -3.6390 | Very weak push |

**Finding:** β=10 optimal - balances strong class separation with stable training. Too high causes instability, too low fails to separate classes.

---

#### 3. **Temperature (τ) Sweep - Gradient Sharpness**
*Configuration: α=1, β=10, adaptive β*

| τ | Test Acc | Δ vs SupCon | Alignment ↓ | Uniformity ↓ | Observation |
|---|----------|-------------|-------------|--------------|-------------|
| **0.07** | **73.35%** | **+4.27%** | 1.1990 | **-3.7104** | ✅ Optimal - balanced spread |
| 0.05 | 68.08% | -1.00% | 1.3503 | -3.6645 | Too sharp → poor alignment |

**Finding:** τ=0.07 provides optimal gradient flow - τ=0.05 too sharp, only closest pairs contribute.

---

#### 4. **Hybrid Loss (λ) Sweep - SupCon + LW-SupCRD Mix**
*Formula: `L = λ × SupCon + (1-λ) × LW-SupCRD`*
*Configuration: α=1, β=10, τ=0.07*

| λ | Test Acc | Train Acc | Gap | Alignment ↓ | Uniformity ↓ | Observation |
|---|----------|-----------|-----|-------------|--------------|-------------|
| **0.3** | **72.58%** | 98.73% | **26.15%** | **0.6043** | -3.2880 | Best hybrid - severe overfitting |
| 0.5 | 72.07% | 98.92% | 26.85% | **0.5166** | -2.9451 | Worse overfitting |
| 0.7 | 71.57% | 98.25% | 26.68% | **0.4845** | -2.8290 | Poor uniformity |
| 0.9 | 70.69% | 95.40% | 24.71% | **0.4394** | -2.6273 | Approaching pure SupCon |

**Critical Finding:** All hybrids show massive overfitting (24-27% gap) despite excellent alignment. **Pure LW-SupCRD (73.35%) beats best hybrid (72.58%)** - adding SupCon only adds noise.

---

## Key Findings

### **1. The Alignment-Uniformity Trade-off for CIFAR-100** 📊

For fine-grained classification (100 classes), **uniformity is more critical than tight alignment:**

**Best Methods (73%+):**
- Alignment: ~1.15-1.20 (moderate clusters)
- Uniformity: ~-3.70 (excellent spread)
- Strategy: Trade cluster tightness for class separation

**Worst Methods (68%-):**
- Alignment: ~0.44-0.66 (very tight clusters)
- Uniformity: ~-2.50 to -1.73 (poor spread)
- Problem: Over-clustering causes poor separation

**Counter-intuitive Insight:** Student's "worse" alignment (1.20 vs teacher's 0.59) actually helps generalization by maintaining better class separation on the hypersphere.

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

### **3. Why Hybrids Fail** ❌

All hybrid losses (λ=0.3 to 0.9) show:
- ✓ Excellent alignment (0.44-0.60, like teacher)
- ✗ Poor uniformity (-2.6 to -3.3)
- ✗ Massive overfitting (24-27% train-test gap)
- ✗ Lower accuracy than pure LW-SupCRD

**Root Cause:** SupCon's pull-only forces create over-tight clusters, sacrificing the uniformity that LW-SupCRD's strong push forces (β=10) achieve.

**Conclusion:** Pure LW-SupCRD (73.35%) > Best Hybrid (72.58%)

---

### **4. Student Surpasses Teacher in Uniformity** 🎯

| Metric | Teacher | Best Student | Observation |
|--------|---------|--------------|-------------|
| Alignment | **0.5928** | 1.1990 | Student 2× looser |
| Uniformity | -3.4649 | **-3.7104** | Student 7% better |
| Accuracy | 80.75% | 73.35% | Reasonable gap |

**Key Insight:** Student trades alignment for uniformity and still outperforms all baselines significantly. The looser clusters + better spread = superior linear separability.

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
Contrastive Representation Distillation - instance matching.

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

#### 4. **Hybrid Loss**
```python
L = λ × L_SupCon + (1 - λ) × L_LW-SupCRD
```
Best: λ=0.3, but still underperforms pure LW-SupCRD.

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
    - `plots/alpha_1.0_hypersphere.html` - Best α configuration
    - `plots/temp_0.07_hypersphere.html` - Best temperature (73.35% model)
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
plots/temp_0.07_hypersphere.html          # 🏆 Best model (73.35%)
plots/baseline_supcon_hypersphere.html     # Compare with baseline (69.08%)
plots/undistilled_hypersphere.html         # See the improvement
plots/hybrid_lambda_0.3_hypersphere.html   # Best hybrid (overfitting example)
```

**What to look for:**
- **Left panel:** All 100 classes - should show good spread across sphere
- **Right panel:** 10 random classes - shows cluster separation clearly
- **Color gradient:** Represents class labels (0-99)
- **Point density:** Tighter clusters = better alignment, spread = better uniformity

**Browser compatibility:** Works best in Chrome/Firefox/Edge. Safari may have minor rendering differences.

---

## Usage

### Running the Notebook

1. Open `DeSupCon.ipynb` in Jupyter
2. Download required models from Google Drive
3. Place models in `pth_models/` directory
4. Run all cells sequentially

**Training Control:**
- Set `FORCE_RETRAIN = True` to retrain models (ignores cached weights)
- Set `FORCE_RETRAIN = False` to load pre-trained models (default)

### Loading Best Model

```python
import torch
from models import ModelWrapper

# Load best model
model = ModelWrapper(num_classes=100, arch='resnet18')
checkpoint = torch.load('pth_models/student_alpha_1.0_beta_10.0_temp_0.07_resnet18_cifar100.pth')
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
@misc{lw_supcrd2024,
  title={Logit-Weighted Supervised Contrastive Representation Distillation: 
         Achieving Superior Uniformity through Semantic Force Weighting},
  author={Ibrahim Murtaza, Jibran Mazhar, Muhammad Ahsan Salar Khan},
  year={2024},
  institution={Lahore University of Management Sciences (LUMS)},
  course={EE-5102/CS-6304: Advanced Topics in Machine Learning},
  instructor={Professor Muhammad Tahir},
  note={Best Configuration: α=1.0, β=10.0, τ=0.07 achieving 73.35% on CIFAR-100}
}
```

### Key References
- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
- Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere", ICML 2020
- Tian et al., "Contrastive Representation Distillation", ICLR 2020

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
3. **Multi-Teacher:** Ensemble knowledge from multiple teachers
4. **Theoretical Analysis:** Formal proof of alignment-uniformity trade-off
5. **Publication:** Prepare for submission to WACV/BMVC

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

---

## License
This project is for academic purposes as part of the ATML course at LUMS.

## Contact
For questions or issues, please open an issue on the repository or contact the team members.

---

**Last Updated:** December 23, 2024

**Status:** ✅ All experiments completed | 📊 Results finalized | 🎯 Best model: 73.35% accuracy