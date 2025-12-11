# %% [markdown]
# # Contrastive Representation Distillation Experiments
# ## Comparing Baseline SupCon vs Logit-Weighted SupCRD
# 
# **Goal**: Evaluate whether using teacher logits to weight contrastive forces improves student representation quality.
# 
# **Methods**:
# - **Baseline**: Standard Supervised Contrastive Learning (SupCon)
# - **Proposed**: Logit-Weighted Supervised Contrastive Representation Distillation (SupCRD)
#   - Pull weight: `α × P_teacher(target_class)`
#   - Push weight: `β × (1 - P_teacher(negative_class))`

# %% [markdown]
# ---
# ## Setup & Imports

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ---
# ## Hyperparameters

# %%
# Training config
BATCH_SIZE = 128
LR = 1e-3
EPOCHS_TEACHER = 10
EPOCHS_STUDENT = 20  # Using 20 epochs for fair comparison

# Contrastive config
TEMP = 0.07

# Distillation config (for SupCRD)
ALPHA = 1.0  # Pull force weight
BETA = 10.0  # Push force weight

# %% [markdown]
# ---
# ## Data Loading (CIFAR-10)

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")

# %% [markdown]
# ---
# ## Model Architecture

# %%
class ConvEncoder(nn.Module):
    """Simple CNN Encoder outputting a flat feature vector."""
    def __init__(self, feature_dim=128):
        super().__init__()
        # Input: 3 x 32 x 32
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2) # -> 4x4
        )
        self.flat_dim = 128 * 4 * 4
        self.fc = nn.Linear(self.flat_dim, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ModelWrapper(nn.Module):
    """Wraps Encoder, Projection Head, and Classifier."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = ConvEncoder(feature_dim=128)
        
        # Projection Head (for Contrastive Loss)
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64) 
        )
        
        # Classifier (for Teacher Supervision / Eval)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feats = self.encoder(x)       # (Batch, 128)
        proj = self.projector(feats)  # (Batch, 64)
        logits = self.classifier(feats) # (Batch, 10)
        return feats, proj, logits

# %% [markdown]
# ---
# ## Loss Functions

# %%
class LogitWeightedSupCRDLoss(nn.Module):
    """
    Logit-Weighted Supervised Contrastive Representation Distillation.
    
    Uses teacher probabilities to weight contrastive forces:
    - Pull: α × P_teacher(target_class)
    - Push: β × (1 - P_teacher(negative_class))
    """
    def __init__(self, alpha=1.0, beta=1.0, temperature=0.07, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = temperature
        self.eps = eps

    def forward(self, student_features, teacher_features, teacher_logits, labels):
        batch_size = student_features.shape[0]
        device = student_features.device

        # 1. Normalize features
        s_norm = F.normalize(student_features, dim=1)
        t_norm = F.normalize(teacher_features, dim=1)

        # 2. Similarity Matrix
        sim_matrix = torch.matmul(s_norm, t_norm.T) / self.tau
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        exp_sim = torch.exp(sim_matrix)

        # 3. Teacher Probabilities
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # 4. Construct Masks
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_neg = 1.0 - mask_pos

        # 5. Compute Weights
        # Pull weight
        p_target = torch.gather(teacher_probs, 1, labels).view(-1)
        w_pull = self.alpha * p_target
        
        # Push weight
        target_labels_expand = labels.view(1, -1).expand(batch_size, -1)
        p_negative_class = torch.gather(teacher_probs, 1, target_labels_expand)
        w_push = self.beta * (1.0 - p_negative_class)

        # 6. Compute Loss
        sum_pos_exp = (exp_sim * mask_pos).sum(dim=1)
        numerator_term = w_pull * sum_pos_exp
        weighted_neg_exp = (exp_sim * w_push * mask_neg).sum(dim=1)
        denominator_term = numerator_term + weighted_neg_exp

        loss = -torch.log((numerator_term + self.eps) / (denominator_term + self.eps))
        return loss.mean()


class SupConLoss(nn.Module):
    """BASELINE: Standard Supervised Contrastive Loss (Student vs Student)"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, student_proj, labels):
        feats = F.normalize(student_proj, dim=1)
        sim_matrix = torch.matmul(feats, feats.T) / self.temp
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast
        logits_mask = torch.scatter(torch.ones_like(mask), 1, 
                                    torch.arange(feats.shape[0]).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()

# %% [markdown]
# ---
# ## Visualization: t-SNE Latent Space

# %%
def visualize_latents(model, loader, title="Latent Space", save_path=None):
    """Runs t-SNE on model features and plots them."""
    model.eval()
    features_list = []
    labels_list = []
    num_samples = 2000
    count = 0
    
    print(f"[{title}] Extracting features...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats, _, _ = model(images)
            feats = feats.view(feats.size(0), -1).cpu()
            
            features_list.append(feats)
            labels_list.append(labels)
            
            count += images.size(0)
            if count >= num_samples:
                break

    X = torch.cat(features_list, dim=0).numpy()[:num_samples]
    y = torch.cat(labels_list, dim=0).numpy()[:num_samples]

    print(f"[{title}] Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    cifar_classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], 
        hue=y, palette='tab10', legend='full', alpha=0.7
    )
    plt.legend(title='Classes', labels=cifar_classes, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{title} (t-SNE)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

# %% [markdown]
# ---
# ## Analysis: Pull vs Push Forces

# %%
def analyze_pull_push_forces(model, teacher_model, loader, loss_fn, num_batches=10):
    """
    Analyze pull vs push forces for the loss function.
    Returns statistics on force magnitudes.
    """
    model.eval()
    teacher_model.eval()
    
    pull_forces = []
    push_forces = []
    pull_weights = []
    push_weights = []
    
    batch_count = 0
    
    with torch.no_grad():
        for images, labels in loader:
            if batch_count >= num_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            s_feats, s_proj, _ = model(images)
            _, t_proj, t_logits = teacher_model(images)
            
            # Compute similarity and weights (mimic loss computation)
            batch_size = s_proj.shape[0]
            
            s_norm = F.normalize(s_proj, dim=1)
            t_norm = F.normalize(t_proj, dim=1)
            
            sim_matrix = torch.matmul(s_norm, t_norm.T) / loss_fn.tau
            exp_sim = torch.exp(sim_matrix)
            
            # Teacher probs
            teacher_probs = F.softmax(t_logits, dim=1)
            
            # Masks
            labels_view = labels.view(-1, 1)
            mask_pos = torch.eq(labels_view, labels_view.T).float()
            mask_neg = 1.0 - mask_pos
            
            # Weights
            p_target = torch.gather(teacher_probs, 1, labels_view).view(-1)
            w_pull = loss_fn.alpha * p_target
            
            target_labels_expand = labels_view.view(1, -1).expand(batch_size, -1)
            p_negative_class = torch.gather(teacher_probs, 1, target_labels_expand)
            w_push = loss_fn.beta * (1.0 - p_negative_class)
            
            # Compute forces
            sum_pos_exp = (exp_sim * mask_pos).sum(dim=1)
            pull_force = w_pull * sum_pos_exp
            
            weighted_neg_exp = (exp_sim * w_push * mask_neg).sum(dim=1)
            push_force = weighted_neg_exp
            
            # Store
            pull_forces.extend(pull_force.cpu().numpy())
            push_forces.extend(push_force.cpu().numpy())
            pull_weights.extend(w_pull.cpu().numpy())
            push_weights.extend(w_push.mean(dim=1).cpu().numpy())  # Average push weight per anchor
            
            batch_count += 1
    
    pull_forces = np.array(pull_forces)
    push_forces = np.array(push_forces)
    pull_weights = np.array(pull_weights)
    push_weights = np.array(push_weights)
    
    return {
        'pull_force': {
            'mean': pull_forces.mean(),
            'std': pull_forces.std(),
            'min': pull_forces.min(),
            'max': pull_forces.max()
        },
        'push_force': {
            'mean': push_forces.mean(),
            'std': push_forces.std(),
            'min': push_forces.min(),
            'max': push_forces.max()
        },
        'pull_weight': {
            'mean': pull_weights.mean(),
            'std': pull_weights.std()
        },
        'push_weight': {
            'mean': push_weights.mean(),
            'std': push_weights.std()
        },
        'ratio_pull_to_push': pull_forces.mean() / push_forces.mean()
    }

# %% [markdown]
# ---
# ## Training Functions

# %%
def train_teacher(epochs=10):
    """Train teacher model with standard cross-entropy."""
    model = ModelWrapper(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"TRAINING TEACHER MODEL ({epochs} epochs)")
    print(f"{'='*60}")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.3f} | Acc={acc:.1f}%")
    
    print(f"\n✓ Teacher training complete: {acc:.1f}% accuracy\n")
    return model


def train_student(teacher_model, mode="supcon", epochs=20, alpha=1.0, beta=1.0, temperature=0.07):
    """
    Train student with contrastive loss.
    
    Args:
        mode: 'supcon' (baseline) or 'supcrd' (logit-weighted)
    """
    student = ModelWrapper(num_classes=10).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    
    # Online Linear Probe
    probe_head = nn.Linear(128, 10).to(device) 
    probe_opt = torch.optim.Adam(probe_head.parameters(), lr=LR)
    probe_crit = nn.CrossEntropyLoss()

    # Select loss
    if mode == "supcrd":
        criterion = LogitWeightedSupCRDLoss(alpha=alpha, beta=beta, temperature=temperature)
        print(f"\n{'='*60}")
        print(f"TRAINING: Logit-Weighted SupCRD")
        print(f"α={alpha} (pull) | β={beta} (push) | τ={temperature}")
        print(f"{'='*60}\n")
    else:
        criterion = SupConLoss(temperature=temperature)
        print(f"\n{'='*60}")
        print(f"TRAINING: Baseline SupCon")
        print(f"τ={temperature}")
        print(f"{'='*60}\n")

    training_log = {
        'epochs': [],
        'loss': [],
        'probe_acc': []
    }

    for epoch in range(epochs):
        student.train()
        probe_head.train()
        total_loss = 0
        probe_acc = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward Student
            s_feats, s_proj, _ = student(images)
            
            # Compute Loss
            if mode == "supcrd":
                with torch.no_grad():
                    _, t_proj, t_logits = teacher_model(images)
                loss = criterion(s_proj, t_proj, t_logits, labels)
            else:
                loss = criterion(s_proj, labels)

            # Optimize Student
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Optimize Linear Probe
            probe_logits = probe_head(s_feats.detach()) 
            p_loss = probe_crit(probe_logits, labels)
            
            probe_opt.zero_grad()
            p_loss.backward()
            probe_opt.step()
            
            total_loss += loss.item()
            _, preds = probe_logits.max(1)
            probe_acc += preds.eq(labels).sum().item()
            total_samples += labels.size(0)
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * probe_acc / total_samples
        
        training_log['epochs'].append(epoch + 1)
        training_log['loss'].append(avg_loss)
        training_log['probe_acc'].append(avg_acc)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f} | Probe Acc={avg_acc:.2f}%")
    
    print(f"\n✓ Student training complete: {avg_acc:.2f}% accuracy\n")
    return student, training_log

# %% [markdown]
# ---
# ---
# # EXPERIMENTS
# ---

# %% [markdown]
# ## Experiment 0: Train Teacher Model

# %%
teacher = train_teacher(epochs=EPOCHS_TEACHER)
teacher.eval()

# Visualize teacher's latent space
visualize_latents(teacher, test_loader, 
                  title="Teacher Model",
                  save_path="teacher_latent.png")

# %% [markdown]
# ---
# ## Experiment 1: Baseline SupCon (Student vs Student)

# %%
baseline_student, baseline_log = train_student(
    teacher_model=teacher,
    mode="supcon",
    epochs=EPOCHS_STUDENT,
    temperature=TEMP
)

visualize_latents(baseline_student, test_loader,
                  title="Baseline SupCon",
                  save_path="baseline_supcon_latent.png")

# %% [markdown]
# ---
# ## Experiment 2: Logit-Weighted SupCRD (α=1.0, β=10.0, τ=0.07)

# %%
supcrd_student_1, supcrd_log_1 = train_student(
    teacher_model=teacher,
    mode="supcrd",
    epochs=EPOCHS_STUDENT,
    alpha=1.0,
    beta=10.0,
    temperature=0.07
)

visualize_latents(supcrd_student_1, test_loader,
                  title="SupCRD (α=1.0, β=10.0, τ=0.07)",
                  save_path="supcrd_a1_b10_t007_latent.png")

# %% [markdown]
# ---
# ## Experiment 3: Logit-Weighted SupCRD (α=1.0, β=10.0, τ=0.007)

# %%
supcrd_student_2, supcrd_log_2 = train_student(
    teacher_model=teacher,
    mode="supcrd",
    epochs=EPOCHS_STUDENT,
    alpha=1.0,
    beta=10.0,
    temperature=0.007
)

visualize_latents(supcrd_student_2, test_loader,
                  title="SupCRD (α=1.0, β=10.0, τ=0.007)",
                  save_path="supcrd_a1_b10_t0007_latent.png")

# %% [markdown]
# ---
# ## Experiment 4: Alpha Sweep (β=1.0, τ=0.07)

# %%
alpha_sweep_results = {}

for alpha in [1.0, 10.0, 50.0]:
    print(f"\n{'#'*60}")
    print(f"Alpha Sweep: α={alpha}")
    print(f"{'#'*60}")
    
    student, log = train_student(
        teacher_model=teacher,
        mode="supcrd",
        epochs=EPOCHS_STUDENT,
        alpha=alpha,
        beta=1.0,
        temperature=0.07
    )
    
    alpha_sweep_results[f"alpha_{alpha}"] = {
        'model': student,
        'log': log
    }
    
    visualize_latents(student, test_loader,
                      title=f"SupCRD (α={alpha}, β=1.0)",
                      save_path=f"supcrd_a{int(alpha)}_b1_latent.png")

# %% [markdown]
# ---
# ## Experiment 5: Beta Sweep (α=1.0, τ=0.07)

# %%
beta_sweep_results = {}

for beta in [1.0, 10.0, 50.0]:
    print(f"\n{'#'*60}")
    print(f"Beta Sweep: β={beta}")
    print(f"{'#'*60}")
    
    student, log = train_student(
        teacher_model=teacher,
        mode="supcrd",
        epochs=EPOCHS_STUDENT,
        alpha=1.0,
        beta=beta,
        temperature=0.07
    )
    
    beta_sweep_results[f"beta_{beta}"] = {
        'model': student,
        'log': log
    }
    
    visualize_latents(student, test_loader,
                      title=f"SupCRD (α=1.0, β={beta})",
                      save_path=f"supcrd_a1_b{int(beta)}_latent.png")

# %% [markdown]
# ## Pull vs Push Force Analysis
# 
# This analysis measures the actual magnitude of pull forces (toward same-class samples) vs push forces (away from different-class samples) to verify if they're mathematically coupled.

# %%
# Create loss instances for analysis
supcrd_a1_b1_loss = LogitWeightedSupCRDLoss(alpha=1.0, beta=1.0, temperature=0.07)
supcrd_a1_b10_loss = LogitWeightedSupCRDLoss(alpha=1.0, beta=10.0, temperature=0.07)
supcrd_a10_b1_loss = LogitWeightedSupCRDLoss(alpha=10.0, beta=1.0, temperature=0.07)
supcrd_a50_b1_loss = LogitWeightedSupCRDLoss(alpha=50.0, beta=1.0, temperature=0.07)
supcrd_a1_b50_loss = LogitWeightedSupCRDLoss(alpha=1.0, beta=50.0, temperature=0.07)

# Get the models from sweep results
supcrd_a1_b1_student = alpha_sweep_results['alpha_1.0']['model']
supcrd_a10_b1_student = alpha_sweep_results['alpha_10.0']['model']
supcrd_a50_b1_student = alpha_sweep_results['alpha_50.0']['model']
supcrd_a1_b10_student = beta_sweep_results['beta_10.0']['model']
supcrd_a1_b50_student = beta_sweep_results['beta_50.0']['model']

print("="*70)
print("PULL vs PUSH FORCE ANALYSIS")
print("="*70)

configs = [
    ('SupCRD α=1.0, β=1.0', supcrd_a1_b1_student, supcrd_a1_b1_loss),
    ('SupCRD α=1.0, β=10.0', supcrd_a1_b10_student, supcrd_a1_b10_loss),
    ('SupCRD α=1.0, β=50.0', supcrd_a1_b50_student, supcrd_a1_b50_loss),
    ('SupCRD α=10.0, β=1.0', supcrd_a10_b1_student, supcrd_a10_b1_loss),
    ('SupCRD α=50.0, β=1.0', supcrd_a50_b1_student, supcrd_a50_b1_loss),
]

for name, student, loss_fn in configs:
    print(f"\n{name}:")
    print("-" * 70)
    
    stats = analyze_pull_push_forces(student, teacher, test_loader, loss_fn)
    
    print(f"  Pull Force: mean={stats['pull_force']['mean']:.4f}, "
          f"std={stats['pull_force']['std']:.4f}")
    print(f"  Push Force: mean={stats['push_force']['mean']:.4f}, "
          f"std={stats['push_force']['std']:.4f}")
    print(f"  Pull Weight (α×p_target): mean={stats['pull_weight']['mean']:.4f}")
    print(f"  Push Weight (β×(1-p_neg)): mean={stats['push_weight']['mean']:.4f}")
    print(f"  **Ratio (Pull/Push): {stats['ratio_pull_to_push']:.4f}**")

print("\n" + "="*70)

# %% [markdown]
# ---
# ---
# # RESULTS & ANALYSIS
# ---

# %% [markdown]
# ## Training Curves Comparison

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(baseline_log['epochs'], baseline_log['loss'], 'o-', label='Baseline SupCon', linewidth=2)
ax1.plot(supcrd_log_1['epochs'], supcrd_log_1['loss'], 's-', label='SupCRD (τ=0.07)', linewidth=2)
ax1.plot(supcrd_log_2['epochs'], supcrd_log_2['loss'], '^-', label='SupCRD (τ=0.007)', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(baseline_log['epochs'], baseline_log['probe_acc'], 'o-', label='Baseline SupCon', linewidth=2)
ax2.plot(supcrd_log_1['epochs'], supcrd_log_1['probe_acc'], 's-', label='SupCRD (τ=0.07)', linewidth=2)
ax2.plot(supcrd_log_2['epochs'], supcrd_log_2['probe_acc'], '^-', label='SupCRD (τ=0.007)', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Probe Accuracy (%)', fontsize=12)
ax2.set_title('Linear Probe Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Alpha Sweep Results

# %%
plt.figure(figsize=(10, 6))

for name, data in alpha_sweep_results.items():
    alpha_val = name.split('_')[1]
    plt.plot(data['log']['epochs'], data['log']['probe_acc'], 
             'o-', label=f'α={alpha_val}', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Probe Accuracy (%)', fontsize=12)
plt.title('Alpha Sweep: Effect on Student Performance (β=1.0)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alpha_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Beta Sweep Results

# %%
plt.figure(figsize=(10, 6))

for name, data in beta_sweep_results.items():
    beta_val = name.split('_')[1]
    plt.plot(data['log']['epochs'], data['log']['probe_acc'], 
             's-', label=f'β={beta_val}', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Probe Accuracy (%)', fontsize=12)
plt.title('Beta Sweep: Effect on Student Performance (α=1.0)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('beta_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Final Summary Table

# %%
import pandas as pd

summary_data = {
    'Method': [
        'Teacher',
        'Baseline SupCon',
        'SupCRD (α=1, β=10, τ=0.07)',
        'SupCRD (α=1, β=10, τ=0.007)'
    ],
    'Final Accuracy (%)': [
        88.0,  # Teacher
        baseline_log['probe_acc'][-1],
        supcrd_log_1['probe_acc'][-1],
        supcrd_log_2['probe_acc'][-1]
    ],
    'Final Loss': [
        0.342,  # Teacher
        baseline_log['loss'][-1],
        supcrd_log_1['loss'][-1],
        supcrd_log_2['loss'][-1]
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(summary_df.to_string(index=False))
print("="*60)

# %% [markdown]
# ---
# ## Key Findings
# 
# **TODO**: Document your observations here:
# 
# 1. **Temperature Effect**: 
#    - τ=0.07 vs τ=0.007 comparison
#    - Impact on cluster tightness
# 
# 2. **Alpha (Pull Force)**:
#    - Does varying α significantly affect results?
#    - Hypothesis: Should not matter much if pull=push coupling exists
# 
# 3. **Beta (Push Force)**:
#    - Strong effect expected on inter-class separation
#    - Tradeoff: semantic similarity vs clear boundaries
# 
# 4. **Baseline Comparison**:
#    - Does SupCRD outperform standard SupCon?
#    - Evidence of semantic structure (dog closer to wolf than car)?
# 
# 5. **Open Questions**:
#    - Pull = Push coupling?
#    - Need for explicit decoupling?
#    - Optimal α/β balance?


