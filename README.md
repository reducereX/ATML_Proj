# Supervised Contrastive Representation Distillation (SupCRD)

## Project Overview

This project implements and evaluates different contrastive distillation methods for knowledge transfer from a ResNet-50 teacher to a ResNet-18 student on CIFAR-100.

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
project/
├── desupcon.py              # Main training script
├── pth_models/              # ← Place downloaded .pth files here
│   ├── teacher_resnet50_cifar100.pth
│   ├── student_baseline_supcon_resnet18_cifar100.pth
│   ├── student_tau_0.07_resnet18_cifar100.pth
│   └── student_hybrid_lambda_0.3_resnet18_cifar100.pth
├── plots/                   # Generated visualizations
├── json_results/            # Experimental results
└── README.md               # This file
```

### 3. Download Required Models

Download these essential models from Google Drive and place them in `pth_models/`:

#### Core Models (Required)
- `teacher_resnet50_cifar100.pth` - Teacher model (72.17% accuracy)
- `student_baseline_supcon_resnet18_cifar100.pth` - Baseline student (69.08% accuracy)

#### Best Results (Recommended)
- `student_hybrid_lambda_0.3_resnet18_cifar100.pth` - **Best model: 73.0% accuracy** 🏆
- `student_tau_0.07_resnet18_cifar100.pth` - Best pure SupCRD (71.3% accuracy)

#### Additional Experiments (Optional)
- `student_alpha_1.0_beta_1.0_resnet18_cifar100.pth` - SupCRD α=1.0, β=1.0
- `student_alpha_2.0_beta_1.0_resnet18_cifar100.pth` - SupCRD α=2.0, β=1.0
- `student_alpha_1.0_beta_12.0_resnet18_cifar100.pth` - SupCRD α=1.0, β=12.0
- `student_tau_0.05_resnet18_cifar100.pth` - SupCRD τ=0.05
- `student_hybrid_lambda_0.5_resnet18_cifar100.pth` - Hybrid λ=0.5
- `student_hybrid_lambda_0.7_resnet18_cifar100.pth` - Hybrid λ=0.7
- `student_hybrid_lambda_0.9_resnet18_cifar100.pth` - Hybrid λ=0.9

## Results Summary

| Method | Configuration | Test Accuracy | Improvement over Baseline |
|--------|--------------|---------------|---------------------------|
| Teacher | ResNet-50 | 72.17% | +3.09% |
| **Baseline** | SupCon ResNet-18 | **69.08%** | - |
| **🏆 Hybrid** | **λ=0.3** | **73.0%** | **+3.92%** ✅ |
| Hybrid | λ=0.5 | 72.9% | +3.82% |
| SupCRD | τ=0.07, β=12.0 | 71.3% | +2.22% |
| Hybrid | λ=0.7 | 71.0% | +1.92% |
| SupCRD | β=12.0 | 70.9% | +1.82% |
| Hybrid | λ=0.9 | 70.4% | +1.32% |
| SupCRD | α=2.0, β=1.0 | 70.1% | +1.02% |
| SupCRD | α=1.0, β=1.0 | 69.7% | +0.62% |

### Key Findings

1. **Hybrid Loss (λ=0.3) achieves the best performance** at 73.0%, surpassing even the teacher model (72.17%)
2. **The hybrid approach combines:**
   - 70% Cross-Entropy loss for discriminative power
   - 30% SupCRD loss for semantic structure
3. **Pure SupCRD works best with:**
   - Strong push force (β=12.0)
   - Lower temperature (τ=0.07)
4. **Student beats teacher** - demonstrating effective knowledge distillation

## Dependencies

```bash
pip install torch torchvision
pip install numpy matplotlib scikit-learn
```

## Usage

### Running Inference with Pre-trained Models

```python
import torch
from torchvision.models import resnet18

# Load the best model
model = resnet18(num_classes=100)
model.load_state_dict(torch.load('pth_models/student_hybrid_lambda_0.3_resnet18_cifar100.pth'))
model.eval()

# Use for inference...
```

**Note:** Set `FORCE_RETRAIN = True` in the script to retrain models instead of loading cached weights.

## Model Architecture

- **Teacher:** ResNet-50 (23.5M parameters)
- **Student:** ResNet-18 (11.2M parameters)
- **Dataset:** CIFAR-100 (100 classes, 50k train / 10k test)
- **Training:** 50 epochs, batch size 128, SGD optimizer

## Loss Functions

### 1. Supervised Contrastive (SupCon)
Standard supervised contrastive learning baseline.

### 2. Logit-Weighted SupCRD
Incorporates teacher's probability distribution to weight positive/negative pairs:
- **Pull weight:** `α × p_teacher(correct_class)`
- **Push weight:** `β × (1 - p_teacher(negative_class))`

### 3. Hybrid Loss
Combines Cross-Entropy and SupCRD:
```
L_hybrid = λ × L_SupCRD + (1 - λ) × L_CE
```

## Citation

If you use this code or models in your research, please cite:

```bibtex
@misc{supcrd2025,
  title={Logit-Weighted Supervised Contrastive Representation Distillation},
  author={Your Name},
  year={2025},
  institution={Lahore University of Management Sciences (LUMS)},
  course={EE-5102/CS-6304: Advanced Topics in Machine Learning}
}
```

## License

This project is for academic purposes as part of the ATML course at LUMS.

## Acknowledgments

- Course Instructor: Professor Muhammad Tahir
- Based on the Supervised Contrastive Learning paper (Khosla et al., NeurIPS 2020)
- Inspired by Decoupled Knowledge Distillation (DKD)

## Contact

For questions or issues, please open an issue on the repository or contact via email.

---

**Last Updated:** December 2025
