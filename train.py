"""
LEAVE-ONE-DONOR-OUT CROSS-VALIDATION 
==============================
D5: Holdout (Final generalization test)
D1, D2, D3, D4, D6: Used for Cross-Validation

For Each Donor in [D1, D2, D3, D4, D6]:
    ├─ One Donor = Test Donor
    ├─ Remaining 4 Donors = Training Donors
    └─ Train model with fixed hyperparameters → Test on Test Donor

Fixed Hyperparameters:
    - Learning Rate: 0.001
    - Batch Size: 16
    - Weight Decay: 0

Final Test:
1. Find Best Performing Fold
2. Train on ALL 5 CV Donors (D1-D4, D6)
3. Test on D5 (completely held-out)

-- Basic Description -- 
Leave-One-Donor-Out CV: Tests patient generalization
Fixed hyperparameters: Fast iteration, validated defaults
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch.nn.functional as F
# DISABLED: AMP causes NaN gradients with Conv3d on Ampere/Ada GPUs
# from torch.amp import autocast, GradScaler  
from torch.utils.data import WeightedRandomSampler
import pickle
import hashlib

# Enable TF32 for faster matmul on A100 GPUs (10-20% speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import model
from resnet import create_model
print("📊 Using standard ResNet3D (proven 84-85% configuration)")

# ============================================================================
# MIXUP DATA AUGMENTATION
# ============================================================================
def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda
    
    Mixup creates virtual training examples by linearly interpolating between
    random pairs of samples. This smooths the decision boundary and prevents
    the model from memorizing donor-specific textures.
    
    Args:
        x: Input batch (B, C, T, H, W)
        y: Labels (B,)
        alpha: Beta distribution parameter (0.4 is optimal from literature)
    
    Returns:
        mixed_x: Interpolated inputs
        y_a, y_b: Original label pairs
        lam: Interpolation coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss as weighted combination of two labels"""
    return lam * criterion(pred, y_a.unsqueeze(1)) + (1 - lam) * criterion(pred, y_b.unsqueeze(1))

# Cache directory for preprocessed data
CACHE_DIR = Path('./cache')
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# ENTROPY FILTERING (TPSF Paper Method)
# ============================================================================

def compute_entropy(array_2d):
    """
    Compute Shannon entropy of a 2D photon distribution.
    
    Args:
        array_2d: (H, W) spatial photon distribution at peak frame
    
    Returns:
        entropy: Shannon entropy in bits
    """
    total = array_2d.sum()
    if total == 0:
        return 0.0
    
    p = array_2d / total
    p = p[p > 0]  # Only non-zero bins
    
    entropy = -np.sum(p * np.log2(p))
    return entropy

def is_valid_cell_entropy(file_path, entropy_threshold_sigma=2.0):
    """
    Check if a cell passes entropy filtering.
    
    Based on TPSF paper: Filter cells outside ±threshold_sigma from mean entropy.
    Precomputed stats from analyze_entropy_filtering.py:
      Global mean: 7.285, std: 0.883
    
    Args:
        file_path: Path to cell .npy file
        entropy_threshold_sigma: Number of standard deviations (default: 2.0)
    
    Returns:
        bool: True if cell passes filter, False otherwise
    """
    # Precomputed global entropy statistics
    GLOBAL_ENTROPY_MEAN = 7.285
    GLOBAL_ENTROPY_STD = 0.883
    
    lower_bound = GLOBAL_ENTROPY_MEAN - entropy_threshold_sigma * GLOBAL_ENTROPY_STD
    upper_bound = GLOBAL_ENTROPY_MEAN + entropy_threshold_sigma * GLOBAL_ENTROPY_STD
    
    try:
        # Load cell and compute entropy at peak frame
        array = np.load(file_path)
        
        # Handle shape variations
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        elif array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        
        # Ensure (H, W, T) format
        if array.shape[2] < array.shape[0]:
            array = np.transpose(array, (1, 2, 0))
        
        # Find peak photon frame
        temporal_sum = array.sum(axis=(0, 1))
        peak_frame = temporal_sum.argmax()
        
        # Get spatial distribution at peak
        peak_spatial = array[:, :, peak_frame]
        
        # Compute entropy
        entropy = compute_entropy(peak_spatial)
        
        # Check if within bounds
        return lower_bound <= entropy <= upper_bound
    
    except Exception as e:
        # If error, keep the cell (don't filter out due to technical issues)
        return True

# ============================================================================
# DATASET CLASS
# ============================================================================
GLOBAL_INTENSITY_MIN = 0.0
GLOBAL_INTENSITY_MAX = 15.0  

class CellDataset(Dataset):
    """Custom Dataset that loads data on-the-fly"""

    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def apply_augmentation(self, array):
        """Apply random augmentations to 3D array"""
        
        # ===== TEMPORAL JITTER (IRF Drift Correction) =====
        # FLIM hardware drift: laser trigger timing shifts between donors
        # This causes the decay peak to shift by 2-3 time bins
        # Randomly shift time axis to make model learn curve SHAPE, not absolute position
        if np.random.rand() > 0.2:  # 80% probability - very important
            shift = np.random.randint(-5, 6)  # ±5 time bins (~2-3 typical drift)
            if shift != 0:
                array = np.roll(array, shift, axis=2)
                # Zero out wrap-around artifacts
                if shift < 0:
                    array[:, :, shift:] = 0  # Zero out end
                elif shift > 0:
                    array[:, :, :shift] = 0  # Zero out start
        
        # ===== INTENSITY JITTER (Brightness Robustness) =====
        # Augment brightness to make model invariant to intensity variations
        # This helps generalize across different imaging conditions/donors
        if np.random.rand() > 0.2:  # 80% probability
            scale_factor = np.random.uniform(0.6, 1.4)  # ±40% brightness
            array = array * scale_factor
        
        # Rotations
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])
            array = np.rot90(array, k=k, axes=(0, 1)).copy()
        
        # Flips
        if np.random.rand() > 0.5:
            array = np.flip(array, axis=0).copy()  
        if np.random.rand() > 0.5:
            array = np.flip(array, axis=1).copy()  
        
        # Poisson Noise (photon shot noise - realistic for FLIM)
        if np.random.rand() > 0.5:
            # Scale up to simulate photon counts, add Poisson noise, scale back
            scale = 1000  # Simulate ~1000 photons per voxel on average
            photon_counts = array * scale
            noisy_counts = np.random.poisson(photon_counts)
            array = noisy_counts / scale

        # Spatial Scaling
        if np.random.rand() > 0.5:
            from scipy.ndimage import zoom
            scale_factor = np.random.uniform(0.85, 1.15)
            zoomed = zoom(array, (scale_factor, scale_factor, 1.0), order=1)
            h, w, t = array.shape
            zh, zw, _ = zoomed.shape
            if zh > h: # Crop
                start_h = (zh - h) // 2
                zoomed = zoomed[start_h:start_h+h, :, :]
            elif zh < h: # Pad
                pad_h = (h - zh) // 2
                zoomed = np.pad(zoomed, ((pad_h, h-zh-pad_h), (0, 0), (0, 0)), 
                          mode='constant', constant_values=0)
            if zw > w: # Crop
                start_w = (zw - w) // 2
                zoomed = zoomed[:, start_w: start_w+w, :]
            elif zw < w: # Pad
                pad_w = (w - zw) // 2
                zoomed = np.pad(zoomed, ((0, 0), (pad_w, w-zw-pad_w), (0, 0)), 
                          mode='constant', constant_values=0)
            array = zoomed

        # Elastic Deformation
        if np.random.rand() > 0.5:
            from scipy.ndimage import gaussian_filter, map_coordinates
            alpha = 5 # Deformation strength
            sigma = 2 # Smoothness
            random_state = np.random.RandomState(None)
            dx = gaussian_filter((random_state.rand(array.shape[0], array.shape[1]) * 2 - 1), 
                            sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(array.shape[0], array.shape[1]) * 2 - 1), 
                            sigma, mode="constant", cval=0) * alpha
            x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            deformed = np.zeros_like(array)
            for t in range(array.shape[2]):
                deformed[:, :, t] = map_coordinates(array[:, :, t], indices, 
                                               order=1, mode='reflect').reshape(array.shape[:2])
            array = deformed

        # Gaussian Blur
        if np.random.rand() > 0.5:
            from scipy.ndimage import gaussian_filter
            sigma = np.random.uniform(0.3, 0.8)
            # Only blur spatial dimensions
            for t in range(array.shape[2]):
                array[:, :, t] = gaussian_filter(array[:, :, t], sigma=sigma)

        return np.ascontiguousarray(array)
    
    def __getitem__(self, idx):
        import hashlib
        from pathlib import Path
        
        file_path = self.file_paths[idx]
        
        # Create cache key based on file path
        cache_dir = Path('./cache')
        cache_key = hashlib.md5(str(file_path).encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache (only for non-augmented data)
        if not self.augment and cache_file.exists():
            with open(cache_file, 'rb') as f:
                array = pickle.load(f)
        else:
            # Load file
            array = np.load(file_path)
            
            # Cache the raw array if not augmenting
            if not self.augment:
                with open(cache_file, 'wb') as f:
                    pickle.dump(array, f)
        
        if self.augment:
            array = self.apply_augmentation(array)
        
        # =============================================================================
        # DUAL-CHANNEL: LOG-TRANSFORMED FLIM + BINARY MASK
        # =============================================================================
        # Channel 0: Log-transformed FLIM (proven method, simple and effective)
        # Channel 1: Binary mask (explicit region of interest)
        # 
        # Why this works:
        # - Log1p preserves 0.0 background (no "glowing background")
        # - Mask explicitly signals: "Look here for biology, ignore padding"
        # - CNN learns: Weight_FLIM * Decay + Weight_Mask * Geometry
        # - Solves geometric sparsity WITHOUT complex transforms
        # =============================================================================

        array = array.astype(np.float32)
        
        # ===== LOG TRANSFORMATION (Original, proven method) =====
        # Simple log1p handles Poisson noise well enough
        # Preserves zero background, computationally efficient
        signal = np.log1p(array)  # log(1 + x)

        # Binary Mask
        spatial_sum = np.sum(array, axis=2)
        mask = (spatial_sum > 0).astype(np.float32)
        
        # Normalize log-transformed signal
        data = signal / 10.0  # Simple scaling
        
        # Transpose FLIM to (T, H, W)
        data = np.transpose(data, (2, 0, 1))  
        
        # Expand mask to match temporal dimension
        mask_3d = np.tile(mask[np.newaxis, :, :], (data.shape[0], 1, 1)) 
        
        # Stack as 2 channels: (2, T, H, W)
        X = np.stack([data, mask_3d], axis=0) 
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return X, y

# ============================================================================
# DATA COLLECTION BY FOLDER
# ============================================================================

def collect_data_by_folder(use_entropy_filter=False, entropy_threshold_sigma=2.0):
    """
    Collects all data files by their dataset folder (D1-D6).
    
    Args:
        use_entropy_filter: Whether to apply entropy filtering (TPSF paper method)
        entropy_threshold_sigma: Threshold for entropy filtering (default: 2.0σ)
    """

    data_dir = Path("3d_cnn/3d_cnn/data/")
    dataset_dict = {}

    for i in range(1, 7):

        # Get folder name
        folder_name = f'isolated_cells_D{i}'

        # Find folder 
        folder = data_dir / folder_name
        if not folder.exists():
            print(f"Warning: {folder_name} does not exist, skipping...")
            continue

        # Empty list for files and labels
        files = []
        labels = []
        filtered_count = 0

        # Iterate through each folder
        for file in folder.glob('*.npy'):
            
            # Check each label
            if '_in' in file.name:
                label = 0
            elif '_act' in file.name:
                label = 1
            else:
                continue
            
            # Apply entropy filter if enabled
            if use_entropy_filter:
                if not is_valid_cell_entropy(file, entropy_threshold_sigma):
                    filtered_count += 1
                    continue  # Skip this cell

            files.append(file)
            labels.append(label)

        # Store files, labels in dict with folder_name as key
        dataset_dict[folder_name] = (files, labels)
        
        if use_entropy_filter:
            total_before = len(files) + filtered_count
            print(f"{folder_name}: {len(files)} files (inactive={labels.count(0)}, active={labels.count(1)}) | Filtered: {filtered_count}/{total_before} ({100*filtered_count/total_before:.1f}%)")
        else:
            print(f"{folder_name}: {len(files)} files (inactive={labels.count(0)}, active={labels.count(1)})")
    
    return dataset_dict


# ============================================================================
# CREATE TRAIN/VALIDATION SPLIT FOR ONE FOLD
# ============================================================================

def create_fold_data(dataset_dict, val_fold):
    """Splits data into train and validation sets for a specific fold"""
    
    # Create empty lists
    train_files = []
    train_labels = []
    val_files = []
    val_labels = []
    
    # Loop through each dataset in the dict
    for fold_name, (files, labels) in dataset_dict.items():
        if fold_name == val_fold:
            # This fold is the validation/test set
            val_files.extend(files)
            val_labels.extend(labels)
        else:
            # All other folds are training set
            train_files.extend(files)
            train_labels.extend(labels)
        
    return train_files, train_labels, val_files, val_labels


# ============================================================================
# TRAIN ONE FOLD
# ============================================================================

def train_one_fold(fold_name, train_files, train_labels, val_files, val_labels, test_files, test_labels,
                   device, num_epochs=20, patience=5, batch_size=16, learning_rate=0.00005, weight_decay=0, num_workers=8):
    """Trains and evaluates a single fold"""
    
    # Print fold header and data summary
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD: {fold_name}")
    print(f"{'='*80}")
    print(f"Training samples: {len(train_files)}")
    print(f"  - Inactive: {train_labels.count(0)} ({train_labels.count(0)/len(train_labels)*100:.1f}%)")
    print(f"  - Active:   {train_labels.count(1)} ({train_labels.count(1)/len(train_labels)*100:.1f}%)")
    print(f"Validation samples: {len(val_files)}")
    print(f"  - Inactive: {val_labels.count(0)} ({val_labels.count(0)/len(val_labels)*100:.1f}%)")
    print(f"  - Active:   {val_labels.count(1)} ({val_labels.count(1)/len(val_labels)*100:.1f}%)")
    
    # Create CellDataset objects for train and validation
    train_dataset = CellDataset(train_files, train_labels, augment=True) # model gets trained on 100% augmented images
    val_dataset = CellDataset(val_files, val_labels, augment=False) # model validated on raw, true images

    # Print class distribution
    print(f"\n📊 Class distribution:")
    active_count = sum(1 for label in train_labels if label == 1)
    inactive_count = len(train_labels) - active_count
    train_active_ratio = active_count / len(train_labels)
    print(f"  - Active cells: {active_count} ({train_active_ratio*100:.1f}%)")
    print(f"  - Inactive cells: {inactive_count} ({inactive_count/len(train_labels)*100:.1f}%)") 
    print(f"  - Using CLASS-BALANCED WeightedRandomSampler")
    print(f"  - Creates 50/50 active/inactive batches regardless of class distribution\n")

    # CLASS-BALANCED WeightedRandomSampler
    # Now that we've fixed the geometric sparsity bug (Anscombe + Mask),
    # we don't need donor stratification - the model can learn true biology
    # Simple class balancing is sufficient to handle imbalance
    
    # Calculate sample weights by class only
    sample_weights = []
    
    for label in train_labels:
        if label == 1:  # Active
            # Upweight active samples to match inactive count
            sample_weights.append(inactive_count / active_count)
        else:  # Inactive
            sample_weights.append(1.0)
    
    # Create WeightedRandomSampler
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"  ✓ Active cells upweighted {inactive_count/active_count:.2f}x to match inactive count")
    print(f"  ✓ Expected batch composition: ~50% active, ~50% inactive")
    print(f"  ✓ Geometric sparsity fixed by Anscombe + Mask (no need for donor stratification)\n")

    # Create DataLoaders with stratified sampler
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,  # Use stratified sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    )

    # Create fresh model (2 channels: density-normalized FLIM + binary mask)
    model = create_model(in_channels=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    # DISABLED: Mixed precision causes NaN gradients with Conv3d
    # scaler = GradScaler(device.type)  
    
    # Use standard cosine annealing (NO restarts for stability)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,    # Decay smoothly over full training
        eta_min=1e-6         # Minimum LR at end
    )
    print("  ✓ Using CosineAnnealingLR (stable decay, no restarts)")

    # BCE Loss - NO pos_weight (would double-count with WeightedRandomSampler)
    active_count = sum(1 for label in train_labels if label == 1)
    inactive_count = len(train_labels) - active_count
    
    # Standard BCE loss - WeightedRandomSampler already handles class balance
    # Adding pos_weight would create 2.26 × 2.26 ≈ 5.1x bias (DOUBLE COUNTING)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Label smoothing parameter (reduce from 0.1 to 0.05 for sharper boundaries)
    label_smoothing = 0.05
    
    print(f"  ✓ BCE Loss with Label Smoothing ({label_smoothing}) initialized")
    print(f"    • Class-balanced sampling: WeightedRandomSampler creates 50/50 batches")
    print(f"    • NO pos_weight: Would double-count with sampler (2.26 × 2.26 = 5.1x bias)")
    print(f"    • Label smoothing: targets [{label_smoothing}, {1-label_smoothing}] instead of [0, 1]")
    print(f"    • Mixup (alpha=0.4): interpolates between samples for smooth boundaries")
    print(f"    • Intensity augmentation (±40%) handles brightness variations")
    print(f"    • Dataset: {active_count} active ({active_count/len(train_labels)*100:.1f}%), {inactive_count} inactive ({inactive_count/len(train_labels)*100:.1f}%)")

    # Tracking variables
    best_val_f1 = 0.0  
    epochs_without_improvement = 0
    best_epoch = 0
    best_val_acc = 0
    best_val_precision = 0 
    best_val_recall = 0
    best_val_loss = float('inf')  # Track best validation loss for reporting

    # Epoch extension
    original_num_epochs = num_epochs
    max_extensions = 2  
    extension_count = 0
    epochs_to_extend = 10 
    
    # Lists for metrics history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Debugging statistics
    debug_stats = {
        'grad_norms': [],
        'param_norms': [],
        'batch_losses': [],
        'learning_rates': [],
        'activation_stats': {},
        'per_layer_grads': {},
        'weight_updates': [],
        'batch_input_stats': [],
        'batch_output_stats': [],
        'learning_dynamics': []
    }
    
    print(f"\n🔍 EXTENSIVE DEBUGGING ENABLED")
    print(f"   Tracking: per-layer gradients, activations, weight updates, loss components, batch stats")
    
    # Register hooks for activation tracking only (no backward hooks to avoid in-place issues)
    activation_values = {}
    
    def get_activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_values[name] = {
                    'mean': output.detach().mean().item(),
                    'std': output.detach().std().item(),
                    'min': output.detach().min().item(),
                    'max': output.detach().max().item(),
                    'sparsity': (output.detach().abs() < 1e-6).float().mean().item()
                }
        return hook
    
    # Register forward hooks only on key layers
    hook_handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear, torch.nn.BatchNorm3d)):
            hook_handles.append(module.register_forward_hook(get_activation_hook(name)))
    
    # Main training loop
    epoch = 0
    while epoch < num_epochs:
        
        # ===== TRAINING PHASE =====
        model.train()

        # Initialize metrics
        running_loss = 0.0 
        correct_train = 0 
        total_train = 0
        
        # Epoch-level debugging
        epoch_grad_norms = []
        epoch_batch_losses = []

        # Progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        # Loop through training batches
        batch_idx = 0
        for inputs, labels in train_pbar:
            
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device) 
            
            # Track input statistics
            batch_input_stats = {
                'mean': inputs.mean().item(),
                'std': inputs.std().item(),
                'min': inputs.min().item(),
                'max': inputs.max().item(),
                'zeros': (inputs == 0).float().mean().item()
            }
            debug_stats['batch_input_stats'].append(batch_input_stats)

            # Zero gradients
            optimizer.zero_grad()
            
            # Store pre-update weights for tracking updates
            if batch_idx % 50 == 0:  # Sample every 50 batches
                pre_weights = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
            
            batch_idx += 1

            # Apply Mixup augmentation (30% probability - reduced for gradient stability)
            if np.random.rand() > 0.7:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)
                
                # Apply label smoothing to mixup targets
                labels_a_smooth = labels_a.float() * (1 - label_smoothing) + label_smoothing / 2
                labels_b_smooth = labels_b.float() * (1 - label_smoothing) + label_smoothing / 2
                
                # DISABLED: autocast causes NaN with Conv3d - using full FP32
                # with autocast('cuda'):
                outputs = model(inputs)
                loss = mixup_criterion(loss_fn, outputs, labels_a_smooth, labels_b_smooth, lam)
            else:
                # Regular training (70% of batches for more stable gradients)
                # Apply label smoothing: 0 → 0.05, 1 → 0.95
                labels_smooth = labels.float() * (1 - label_smoothing) + label_smoothing / 2
                
                # DISABLED: autocast causes NaN with Conv3d - using full FP32
                # with autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels_smooth.unsqueeze(1))

            # Backward pass
            loss.backward()  # Direct backward - no scaler
            
            # Track per-layer gradient statistics (sample every 50 batches)
            if (batch_idx - 1) % 50 == 0:
                layer_grad_stats = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Handle scalar gradients (bias terms)
                        grad_flat = param.grad.flatten()
                        if grad_flat.numel() > 1:
                            layer_grad_stats[name] = {
                                'mean': param.grad.mean().item(),
                                'std': param.grad.std().item(),
                                'norm': param.grad.norm().item(),
                                'max': param.grad.abs().max().item()
                            }
                        else:
                            layer_grad_stats[name] = {
                                'mean': param.grad.item(),
                                'std': 0.0,
                                'norm': abs(param.grad.item()),
                                'max': abs(param.grad.item())
                            }
                debug_stats['per_layer_grads'][f'epoch_{epoch}_batch_{batch_idx-1}'] = layer_grad_stats

            # Gradient clipping (back to 35.0 - bottleneck removed)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            epoch_grad_norms.append(grad_norm.item())
            
            # Track gradient flow through network
            grad_flow = {}
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_flow[name] = param.grad.abs().mean().item()

            # Update weights - direct step, no scaler
            optimizer.step()
            # scaler.update()  # DISABLED - not using mixed precision
            
            # Track weight updates (sample every 50 batches)
            if (batch_idx - 1) % 50 == 0:
                weight_updates = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and name in pre_weights:
                        update = (param.data - pre_weights[name]).abs().mean().item()
                        weight_updates[name] = update
                debug_stats['weight_updates'].append(weight_updates)
            
            # Track output statistics
            with torch.no_grad():
                output_probs = torch.sigmoid(outputs)
                batch_output_stats = {
                    'mean_prob': output_probs.mean().item(),
                    'std_prob': output_probs.std().item(),
                    'min_prob': output_probs.min().item(),
                    'max_prob': output_probs.max().item(),
                    'entropy': -(output_probs * torch.log(output_probs + 1e-8) + 
                                 (1 - output_probs) * torch.log(1 - output_probs + 1e-8)).mean().item()
                }
                debug_stats['batch_output_stats'].append(batch_output_stats)

            # Update running metrics
            batch_loss = loss.item()
            running_loss += batch_loss
            epoch_batch_losses.append(batch_loss)
            # Track accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()
        
        # Calculate epoch training metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        
        # Debugging: Check for training issues
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0
        max_grad_norm = np.max(epoch_grad_norms) if epoch_grad_norms else 0
        min_grad_norm = np.min(epoch_grad_norms) if epoch_grad_norms else 0
        batch_loss_std = np.std(epoch_batch_losses) if epoch_batch_losses else 0
        batch_loss_mean = np.mean(epoch_batch_losses) if epoch_batch_losses else 0
        
        # Store debugging stats
        debug_stats['grad_norms'].append(avg_grad_norm)
        debug_stats['batch_losses'].extend(epoch_batch_losses)
        debug_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Analyze activation patterns (if we tracked any this epoch)
        if activation_values:
            avg_activation_stats = {
                'mean_activation': np.mean([v['mean'] for v in activation_values.values()]),
                'mean_sparsity': np.mean([v['sparsity'] for v in activation_values.values()]),
                'dead_neurons': sum(1 for v in activation_values.values() if v['sparsity'] > 0.95)
            }
            debug_stats['activation_stats'][f'epoch_{epoch}'] = avg_activation_stats
        
        # Analyze learning dynamics
        if len(debug_stats['batch_output_stats']) > 0:
            recent_outputs = debug_stats['batch_output_stats'][-len(train_loader):]
            learning_dynamics = {
                'avg_entropy': np.mean([o['entropy'] for o in recent_outputs]),
                'prob_std': np.mean([o['std_prob'] for o in recent_outputs]),
                'confidence': 1 - np.mean([o['entropy'] for o in recent_outputs])  # High confidence = low entropy
            }
            debug_stats['learning_dynamics'].append(learning_dynamics)
        
        # Print detailed debugging info every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n  🔍 DEBUG EPOCH {epoch+1}:")
            print(f"     Grad: avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f}, min={min_grad_norm:.4f}")
            print(f"     Loss: mean={batch_loss_mean:.4f}, std={batch_loss_std:.4f}")
            if activation_values:
                print(f"     Activations: mean={avg_activation_stats['mean_activation']:.4f}, sparsity={avg_activation_stats['mean_sparsity']:.4f}")
                if avg_activation_stats['dead_neurons'] > 0:
                    print(f"     ⚠️  Dead neurons detected: {avg_activation_stats['dead_neurons']} layers")
            if learning_dynamics:
                print(f"     Output: entropy={learning_dynamics['avg_entropy']:.4f}, confidence={learning_dynamics['confidence']:.4f}")
            
            # Check for specific bottlenecks
            if batch_loss_std / batch_loss_mean > 0.5:
                print(f"     ⚠️  High loss variance (CV={batch_loss_std/batch_loss_mean:.2f}) - inconsistent batches")
            if learning_dynamics and learning_dynamics['avg_entropy'] < 0.3:
                print(f"     ⚠️  Very low entropy - model might be overconfident")
            if learning_dynamics and learning_dynamics['avg_entropy'] > 0.6:
                print(f"     ⚠️  High entropy - model is uncertain")
        
        # Warnings for potential issues
        if avg_grad_norm < 0.001 and epoch > 5:
            print(f"  ⚠️  WARNING: Very small gradients ({avg_grad_norm:.6f}) - possible vanishing gradients")
        if max_grad_norm > 4.5 and epoch > 5:
            print(f"  ⚠️  WARNING: Gradients hitting clip threshold ({max_grad_norm:.2f}) - possible exploding gradients")
        if batch_loss_std > 0.5 and epoch > 10:
            print(f"  ⚠️  WARNING: High batch loss variance ({batch_loss_std:.3f}) - possible data quality issues")
        if min_grad_norm < 1e-7 and epoch > 5:
            print(f"  ⚠️  WARNING: Near-zero gradients detected ({min_grad_norm:.2e}) - learning stalled")
        
        # Append to history lists
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # ===== VALIDATION PHASE =====
        model.eval()
        
        # Validation metrics
        running_val_loss = 0.0
        all_val_preds = []  
        all_val_labels = [] 

        # Progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]  ', 
                        leave=False, ncols=100)

        # Disable gradient computation
        with torch.no_grad():

            # Loop through validation batches
            for inputs, labels in val_pbar:

                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = loss_fn(outputs, labels.unsqueeze(1))

                running_val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                # Collect predictions and labels for metrics
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Calculate epoch validation metrics
            epoch_val_loss = running_val_loss / len(val_loader)

            # Calculate precision, recall, f1
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            epoch_val_acc = accuracy_score(all_val_labels, all_val_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_val_labels, all_val_preds, average='binary', zero_division=0
            )

            epoch_val_f1 = f1

            # Append to history lists
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

        # Step scheduler (CosineAnnealingLR - smooth decay each epoch)
        scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | "
            f"Val F1: {epoch_val_f1:.4f} | Val Prec: {precision:.4f} | Val Rec: {recall:.4f}")
    
        # ===== EARLY STOPPING & MODEL CHECKPOINTING =====
        if epoch_val_f1 > best_val_f1:

            # Update best metrics
            best_val_f1 = epoch_val_f1 
            best_epoch = epoch + 1
            best_val_acc = epoch_val_acc
            best_val_precision = precision 
            best_val_recall = recall
            best_val_loss = epoch_val_loss  # Track best loss when model is saved
            epochs_without_improvement = 0

            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': epoch_val_acc,
                'val_f1': epoch_val_f1,      
                'val_precision': precision,   
                'val_recall': recall,        
                'fold': fold_name,
            }, f'saved_models/best_model_fold_{fold_name}.pth')

            print(f"  ✓ New best model saved! (Val F1: {epoch_val_f1:.4f})")

            if (epoch + 1 >= num_epochs - 2 and 
                extension_count < max_extensions):
                num_epochs += epochs_to_extend
                extension_count += 1
                print(f"  🔄 Model still improving! Extending training by {epochs_to_extend} epochs")
                print(f"     New max epochs: {num_epochs} (extension {extension_count}/{max_extensions})")

        else:
            epochs_without_improvement += 1
            print(f"  ⚠ No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
                print(f"   Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
                break
            
        epoch += 1

    print(f"\n✅ Training completed after {epoch} epochs")
    print(f"   Original budget: {original_num_epochs} epochs")
    if extension_count > 0:
        print(f"   Extended {extension_count} time(s) for {extension_count * epochs_to_extend} extra epochs")
    print(f"Best model was at epoch {best_epoch} with Val F1: {best_val_f1:.4f} (Precision: {best_val_precision:.4f}, Recall: {best_val_recall:.4f})")

    # ===== FINAL TEST EVALUATION (AFTER TRAINING COMPLETES) =====
    print(f"\n{'='*80}")
    print(f"FINAL TEST EVALUATION ON {fold_name}")
    print(f"{'='*80}")

    # Load best model
    checkpoint = torch.load(f'saved_models/best_model_fold_{fold_name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test dataset and loader
    test_dataset = CellDataset(test_files, test_labels, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 4,  # 4x larger batch for testing (no gradients needed)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Test-Time Augmentation (TTA) Helper
    def predict_tta(model, inputs):
        """Apply test-time augmentation and average predictions"""
        model.eval()
        predictions = []
        
        # Original
        with torch.no_grad():
            pred = torch.sigmoid(model(inputs))
            predictions.append(pred)
        
        # Rotate 90°
        with torch.no_grad():
            rotated = torch.rot90(inputs, k=1, dims=[3, 4])  # Rotate H×W plane
            pred = torch.sigmoid(model(rotated))
            predictions.append(pred)
        
        # Flip Horizontal
        with torch.no_grad():
            flipped_h = torch.flip(inputs, dims=[4])  # Flip width
            pred = torch.sigmoid(model(flipped_h))
            predictions.append(pred)
        
        # Flip Vertical
        with torch.no_grad():
            flipped_v = torch.flip(inputs, dims=[3])  # Flip height
            pred = torch.sigmoid(model(flipped_v))
            predictions.append(pred)
        
        # Average all predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred

    # Dynamic Threshold Tuning with K-Means
    def tune_threshold(model, val_dataset, device, batch_size, train_active_ratio):
        """
        Find optimal threshold using K-Means clustering on validation probabilities.
        
        Automatically estimates test set's class distribution from cluster sizes
        and adjusts threshold for prior probability shift - NO LABELS NEEDED.
        
        Args:
            model: Trained model
            val_dataset: Validation/test dataset (labels not used)
            device: Computation device
            batch_size: Batch size for inference
            train_active_ratio: Known training set active ratio (from training labels)
        
        Returns:
            Adjusted threshold accounting for estimated class distribution shift
        """
        from sklearn.cluster import KMeans
        
        print("\n🔧 Tuning decision threshold with K-Means...")
        
        # Create fresh val_loader for threshold tuning
        val_loader_tune = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        model.eval()
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(val_loader_tune, desc='Collecting probabilities', leave=False):
                inputs = inputs.to(device)
                # Use TTA for more robust probabilities
                probs = predict_tta(model, inputs)
                all_probs.extend(probs.cpu().numpy().flatten())
        
        # Reshape for K-Means (needs 2D array)
        probs_array = np.array(all_probs).reshape(-1, 1)
        
        # K-Means with k=2 (active vs inactive clusters)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(probs_array)
        
        # Get cluster centers and assignments
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        
        # Identify which cluster is "active" (higher probability center)
        if centers[0] > centers[1]:
            active_cluster_id = 0
            inactive_cluster_id = 1
        else:
            active_cluster_id = 1
            inactive_cluster_id = 0
        
        inactive_center = centers[inactive_cluster_id]
        active_center = centers[active_cluster_id]
        
        # Base threshold = midpoint between clusters
        base_threshold = (inactive_center + active_center) / 2.0
        
        # CRITICAL: Estimate test set's active ratio from cluster sizes (NO LABELS NEEDED!)
        active_cluster_size = np.sum(labels == active_cluster_id)
        total_samples = len(labels)
        estimated_test_active_ratio = active_cluster_size / total_samples
        
        # Adjust threshold based on estimated prior probability shift
        train_log_odds = np.log(train_active_ratio / (1 - train_active_ratio))
        test_log_odds = np.log(estimated_test_active_ratio / (1 - estimated_test_active_ratio))
        adjustment = test_log_odds - train_log_odds
        
        # Apply adjustment (scale by 0.1 for smoother correction)
        adjusted_threshold = base_threshold - adjustment * 0.1
        
        print(f"  Inactive cluster: center={inactive_center:.4f}, size={total_samples - active_cluster_size}")
        print(f"  Active cluster:   center={active_center:.4f}, size={active_cluster_size}")
        print(f"  Base threshold:   {base_threshold:.4f}")
        print(f"\n  📐 Prior probability adjustment:")
        print(f"     Training active ratio:  {train_active_ratio:.1%} (from labels)")
        print(f"     Estimated test ratio:   {estimated_test_active_ratio:.1%} (from clusters)")
        print(f"     Log-odds adjustment:    {adjustment:.4f}")
        print(f"  ✓ Adjusted threshold:     {adjusted_threshold:.4f}")
        
        return adjusted_threshold

    # Tune threshold on validation set (uses K-means clustering + automatic prior adjustment)
    optimal_threshold = tune_threshold(
        model=model, 
        val_dataset=val_dataset, 
        device=device, 
        batch_size=batch_size,
        train_active_ratio=train_active_ratio  # From training labels
    )

    # Testing with TTA and Adaptive Threshold
    print(f"\nTesting on holdout set (with TTA + adaptive threshold={optimal_threshold:.4f}):")
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_predictions = []
    all_labels = []
    all_probs = []  # Store probabilities for analysis
    
    model.eval()
    for inputs, labels in tqdm(test_loader, desc='Testing (TTA)'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Use TTA for predictions
        probs = predict_tta(model, inputs)
        predicted = (probs > optimal_threshold).float()  # Use dynamic threshold
        
        # Calculate loss on original (non-augmented) prediction
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            test_loss += loss.item()
        
        total_test += labels.size(0)
        correct_test += (predicted == labels.unsqueeze(1)).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    final_test_loss = test_loss / len(test_loader)
    final_test_acc = correct_test / total_test

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm = confusion_matrix(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    print(f"\n📊 Test Results:")
    print(f"  Decision Threshold: {optimal_threshold:.4f} (K-Means optimized)")
    print(f"  Test Loss: {final_test_loss:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")

    # ===== ERROR ANALYSIS =====
    print(f"\n{'='*80}")
    print("ERROR ANALYSIS: Understanding the Performance Ceiling")
    print(f"{'='*80}")
    
    # Categorize predictions
    fp_indices = [i for i, (pred, true) in enumerate(zip(all_predictions, all_labels)) 
                  if pred == 1 and true == 0]
    tp_indices = [i for i, (pred, true) in enumerate(zip(all_predictions, all_labels)) 
                  if pred == 1 and true == 1]
    fn_indices = [i for i, (pred, true) in enumerate(zip(all_predictions, all_labels)) 
                  if pred == 0 and true == 1]
    tn_indices = [i for i, (pred, true) in enumerate(zip(all_predictions, all_labels)) 
                  if pred == 0 and true == 0]
    
    # Extract probabilities for each category
    fp_probs = [all_probs[i][0] for i in fp_indices]
    tp_probs = [all_probs[i][0] for i in tp_indices]
    fn_probs = [all_probs[i][0] for i in fn_indices]
    tn_probs = [all_probs[i][0] for i in tn_indices]
    
    print(f"\n🔴 False Positives (Predicted Active, Actually Inactive):")
    print(f"  Count: {len(fp_indices)} ({len(fp_indices)/(cm[0,0]+cm[0,1])*100:.1f}% of all inactive)")
    print(f"  Average Confidence: {np.mean(fp_probs):.3f} ± {np.std(fp_probs):.3f}")
    print(f"  Median Confidence: {np.median(fp_probs):.3f}")
    print(f"  Quartiles: [{np.percentile(fp_probs, 25):.3f}, {np.percentile(fp_probs, 50):.3f}, {np.percentile(fp_probs, 75):.3f}]")
    print(f"  Highly confident errors (>0.7): {sum(1 for p in fp_probs if p > 0.7)} ({sum(1 for p in fp_probs if p > 0.7)/len(fp_probs)*100:.1f}%)")
    print(f"  Borderline errors (0.5-0.6): {sum(1 for p in fp_probs if 0.5 <= p <= 0.6)} ({sum(1 for p in fp_probs if 0.5 <= p <= 0.6)/len(fp_probs)*100:.1f}%)")
    
    print(f"\n🟢 True Positives (Predicted Active, Actually Active):")
    print(f"  Count: {len(tp_indices)}")
    print(f"  Average Confidence: {np.mean(tp_probs):.3f} ± {np.std(tp_probs):.3f}")
    print(f"  Median Confidence: {np.median(tp_probs):.3f}")
    
    print(f"\n🟠 False Negatives (Predicted Inactive, Actually Active):")
    print(f"  Count: {len(fn_indices)} ({len(fn_indices)/(cm[1,0]+cm[1,1])*100:.1f}% of all active)")
    print(f"  Average Confidence: {np.mean(fn_probs):.3f} ± {np.std(fn_probs):.3f}")
    print(f"  Median Confidence: {np.median(fn_probs):.3f}")
    print(f"  Close to threshold (0.4-0.5): {sum(1 for p in fn_probs if 0.4 <= p <= 0.5)} ({sum(1 for p in fn_probs if 0.4 <= p <= 0.5)/len(fn_probs)*100:.1f}%)")
    
    print(f"\n⚪ True Negatives (Predicted Inactive, Actually Inactive):")
    print(f"  Count: {len(tn_indices)}")
    print(f"  Average Confidence: {np.mean(tn_probs):.3f} ± {np.std(tn_probs):.3f}")
    
    # Diagnosis
    print(f"\n🔬 DIAGNOSIS:")
    fp_median = np.median(fp_probs)
    fn_median = np.median(fn_probs)
    
    # False Positive Analysis
    if fp_median < 0.55:
        print(f"  ✓ FP median confidence ({fp_median:.3f}) is near threshold")
        print(f"    → These are genuinely ambiguous cells (biological overlap)")
        print(f"    → Model is correctly uncertain about borderline cases")
        print(f"    → Ceiling is likely due to overlapping lifetime distributions")
    elif fp_median < 0.70:
        print(f"  ⚠ FP median confidence ({fp_median:.3f}) is moderate")
        print(f"    → Mix of borderline cases and systematic errors")
        print(f"    → Some false positives might be mislabeled in ground truth")
        print(f"    → Consider manual review of high-confidence errors")
    else:
        print(f"  ❌ FP median confidence ({fp_median:.3f}) is high!")
        print(f"    → Model is confidently wrong on many inactive cells")
        print(f"    → Suggests systematic bias or missing features")
        print(f"    → Need to investigate what makes these cells look 'active'")
    
    # False Negative Analysis
    if fn_median > 0.45:
        print(f"  ✓ FN median confidence ({fn_median:.3f}) is close to threshold")
        print(f"    → Borderline active cells that are hard to classify")
    else:
        print(f"  ⚠ FN median confidence ({fn_median:.3f}) is low")
        print(f"    → Model is confidently wrong about some active cells")
        print(f"    → These might be dim/low-SNR active cells")
    
    # Separation quality
    tp_median = np.median(tp_probs)
    tn_median = np.median(tn_probs)
    separation = tp_median - tn_median
    print(f"\n📏 CLASS SEPARATION:")
    print(f"  TP median: {tp_median:.3f}, TN median: {tn_median:.3f}")
    print(f"  Separation: {separation:.3f}")
    if separation > 0.5:
        print(f"  ✓ Excellent separation - model has strong discriminative power")
    elif separation > 0.3:
        print(f"  ✓ Good separation - classes are distinguishable")
    else:
        print(f"  ⚠ Weak separation - significant overlap between classes")
    
    # Create error analysis plot
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Confidence distributions
    plt.subplot(1, 3, 1)
    bins = np.linspace(0, 1, 30)
    plt.hist(fp_probs, bins=bins, alpha=0.6, label=f'False Positives (n={len(fp_indices)})', color='red', edgecolor='darkred')
    plt.hist(tp_probs, bins=bins, alpha=0.6, label=f'True Positives (n={len(tp_indices)})', color='green', edgecolor='darkgreen')
    plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Prediction Confidence (Probability of Active)')
    plt.ylabel('Count')
    plt.title('Confidence Distribution: Active Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(fn_probs, bins=bins, alpha=0.6, label=f'False Negatives (n={len(fn_indices)})', color='orange', edgecolor='darkorange')
    plt.hist(tn_probs, bins=bins, alpha=0.6, label=f'True Negatives (n={len(tn_indices)})', color='blue', edgecolor='darkblue')
    plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Prediction Confidence (Probability of Active)')
    plt.ylabel('Count')
    plt.title('Confidence Distribution: Inactive Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error summary
    plt.subplot(1, 3, 3)
    categories = ['False\nPositives', 'True\nPositives', 'False\nNegatives', 'True\nNegatives']
    counts = [len(fp_indices), len(tp_indices), len(fn_indices), len(tn_indices)]
    colors_plot = ['red', 'green', 'orange', 'blue']
    bars = plt.bar(categories, counts, color=colors_plot, alpha=0.7, edgecolor='black')
    plt.ylabel('Count')
    plt.title('Prediction Categories')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'saved_models/error_analysis_{fold_name}.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Error analysis plot saved to: saved_models/error_analysis_{fold_name}.png")
    plt.close()
    
    # ===== TRAINING DIAGNOSTICS SUMMARY =====
    print(f"\n{'='*80}")
    print("TRAINING DIAGNOSTICS SUMMARY")
    print(f"{'='*80}")
    
    avg_grad_norm_all = np.mean(debug_stats['grad_norms'])
    max_grad_norm_all = np.max(debug_stats['grad_norms'])
    min_grad_norm_all = np.min(debug_stats['grad_norms'])
    
    print(f"\n🔧 Gradient Flow:")
    print(f"  Average gradient norm: {avg_grad_norm_all:.4f}")
    print(f"  Max gradient norm: {max_grad_norm_all:.4f}")
    print(f"  Min gradient norm: {min_grad_norm_all:.4f}")
    if avg_grad_norm_all < 0.01:
        print(f"  ⚠️  Very small gradients - might benefit from higher learning rate")
    elif max_grad_norm_all > 4.0:
        print(f"  ⚠️  Large gradients frequently clipped - might benefit from lower LR or stronger regularization")
    else:
        print(f"  ✓ Gradient flow looks healthy")
    
    # Plot training diagnostics
    fig = plt.figure(figsize=(15, 10))
    
    # Gradient norms over epochs
    plt.subplot(2, 3, 1)
    plt.plot(debug_stats['grad_norms'], color='purple', alpha=0.7)
    plt.axhline(y=avg_grad_norm_all, color='red', linestyle='--', label=f'Mean: {avg_grad_norm_all:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Norm')
    plt.title('Gradient Flow During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate schedule
    plt.subplot(2, 3, 2)
    plt.plot(debug_stats['learning_rates'], color='blue', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Batch loss distribution
    plt.subplot(2, 3, 3)
    plt.hist(debug_stats['batch_losses'], bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Batch Loss')
    plt.ylabel('Frequency')
    plt.title('Batch Loss Distribution')
    plt.grid(True, alpha=0.3)
    
    # Training curves
    plt.subplot(2, 3, 4)
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    plt.axvline(best_epoch-1, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(train_accuracies, label='Train Acc', color='blue', alpha=0.7)
    plt.plot(val_accuracies, label='Val Acc', color='red', alpha=0.7)
    plt.axvline(best_epoch-1, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Overfitting analysis
    plt.subplot(2, 3, 6)
    train_val_gap = [train - val for train, val in zip(train_accuracies, val_accuracies)]
    plt.plot(train_val_gap, color='purple', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=np.mean(train_val_gap), color='red', linestyle='--', label=f'Mean Gap: {np.mean(train_val_gap):.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Train - Val Accuracy')
    plt.title('Overfitting Monitor (Train-Val Gap)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'saved_models/training_diagnostics_{fold_name}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Training diagnostics plot saved to: saved_models/training_diagnostics_{fold_name}.png")
    plt.close()
    
    # Overfitting check
    final_train_val_gap = train_accuracies[-1] - val_accuracies[-1]
    print(f"\n📈 Overfitting Analysis:")
    print(f"  Final train-val gap: {final_train_val_gap:.3f}")
    print(f"  Average train-val gap: {np.mean(train_val_gap):.3f}")
    if final_train_val_gap > 0.1:
        print(f"  ⚠️  Significant overfitting detected - consider:")
        print(f"     • Increasing dropout (current: 0.25)")
        print(f"     • Reducing model capacity")
        print(f"     • More aggressive data augmentation")
    elif final_train_val_gap < -0.05:
        print(f"  ⚠️  Underfitting detected - val accuracy > train accuracy")
        print(f"     • Model might be too regularized")
        print(f"     • Consider reducing dropout or weight decay")
    else:
        print(f"  ✓ Good train-val balance - no significant overfitting")
    
    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
    
    # Save extensive debugging data
    debug_save_path = f'saved_models/debug_data_{fold_name}.pkl'
    with open(debug_save_path, 'wb') as f:
        pickle.dump(debug_stats, f)
    print(f"\n📊 Debugging data saved to: {debug_save_path}")
    
    # Generate detailed diagnostic summary
    print(f"\n{'='*80}")
    print(f"BOTTLENECK ANALYSIS FOR {fold_name}")
    print(f"{'='*80}")
    
    # 1. Gradient flow analysis
    if debug_stats['grad_norms']:
        grad_stats = debug_stats['grad_norms']
        print(f"\n1️⃣ GRADIENT FLOW:")
        print(f"   Mean: {np.mean(grad_stats):.4f}")
        print(f"   Std: {np.std(grad_stats):.4f}")
        print(f"   Range: [{np.min(grad_stats):.4f}, {np.max(grad_stats):.4f}]")
        if np.mean(grad_stats) > 0:
            print(f"   Coefficient of Variation: {np.std(grad_stats)/np.mean(grad_stats):.4f}")
    
    # 2. Learning dynamics
    if debug_stats['learning_dynamics']:
        entropy_vals = [d['avg_entropy'] for d in debug_stats['learning_dynamics']]
        confidence_vals = [d['confidence'] for d in debug_stats['learning_dynamics']]
        print(f"\n2️⃣ LEARNING DYNAMICS:")
        print(f"   Avg Entropy: {np.mean(entropy_vals):.4f} (lower = more confident)")
        print(f"   Avg Confidence: {np.mean(confidence_vals):.4f}")
        print(f"   Entropy trend: {entropy_vals[0]:.4f} → {entropy_vals[-1]:.4f}")
    
    # 3. Input data quality
    if debug_stats['batch_input_stats']:
        input_means = [s['mean'] for s in debug_stats['batch_input_stats']]
        input_stds = [s['std'] for s in debug_stats['batch_input_stats']]
        input_zeros = [s['zeros'] for s in debug_stats['batch_input_stats']]
        print(f"\n3️⃣ INPUT DATA QUALITY:")
        print(f"   Mean intensity: {np.mean(input_means):.4f} ± {np.std(input_means):.4f}")
        print(f"   Mean std: {np.mean(input_stds):.4f}")
        print(f"   Avg sparsity: {np.mean(input_zeros)*100:.1f}% zeros")
    
    # 4. Activation health
    if debug_stats['activation_stats']:
        all_sparsity = [v['mean_sparsity'] for v in debug_stats['activation_stats'].values()]
        all_dead = [v['dead_neurons'] for v in debug_stats['activation_stats'].values()]
        print(f"\n4️⃣ ACTIVATION HEALTH:")
        print(f"   Avg sparsity: {np.mean(all_sparsity):.4f}")
        print(f"   Dead neurons: {np.mean(all_dead):.1f} layers/epoch")
    
    # 5. Weight update magnitude
    if debug_stats['weight_updates']:
        all_updates = []
        for update_dict in debug_stats['weight_updates']:
            all_updates.extend(update_dict.values())
        print(f"\n5️⃣ WEIGHT UPDATE MAGNITUDE:")
        print(f"   Mean: {np.mean(all_updates):.6f}")
        print(f"   Range: [{np.min(all_updates):.6f}, {np.max(all_updates):.6f}]")
        if np.mean(all_updates) < 1e-5:
            print(f"   ⚠️  Very small updates - learning is slow")
    
    # 6. Identify bottleneck
    print(f"\n🔍 BOTTLENECK DIAGNOSIS:")
    bottlenecks = []
    
    if debug_stats['grad_norms'] and np.mean(debug_stats['grad_norms']) < 0.01:
        bottlenecks.append("Vanishing gradients - consider higher LR or skip connections")
    if debug_stats['grad_norms'] and np.max(debug_stats['grad_norms']) > 10:
        bottlenecks.append("Exploding gradients - lower LR or stronger clipping")
    if debug_stats['activation_stats']:
        avg_dead = np.mean([v['dead_neurons'] for v in debug_stats['activation_stats'].values()])
        if avg_dead > 5:
            bottlenecks.append(f"Too many dead neurons ({avg_dead:.0f}) - ReLU dying issue")
    if debug_stats['weight_updates'] and np.mean([np.mean(list(u.values())) for u in debug_stats['weight_updates']]) < 1e-6:
        bottlenecks.append("Minimal weight updates - learning plateau")
    if debug_stats['learning_dynamics']:
        final_entropy = debug_stats['learning_dynamics'][-1]['avg_entropy']
        if final_entropy > 0.6:
            bottlenecks.append("High output entropy - model is uncertain/confused")
    
    if bottlenecks:
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"   {i}. {bottleneck}")
    else:
        print(f"   ✓ No major bottlenecks detected - training appears healthy")
    
    print(f"\n{'='*80}\n")

    print(f"\nFold {fold_name} Complete!")
    print(f"Best Validation Epoch: {best_epoch} | Val F1: {best_val_f1:.4f} | Val Acc: {best_val_acc:.4f}")
    print(f"Test: Acc={final_test_acc:.4f} | F1={f1:.4f}")

    # Return results
    return {
        'fold': fold_name,
        'best_epoch': best_epoch,
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,          
        'val_precision': best_val_precision,  
        'val_recall': best_val_recall,
        # Test results
        'test_loss': final_test_loss,
        'test_acc': final_test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        # Training history
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        # Metadata
        'num_train_samples': len(train_files),
        'num_val_samples': len(val_files),
        'num_test_samples': len(test_files),
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cross_validation_results(all_results, save_path='saved_models/cross_validation_results.png'):
    """Create comprehensive plots for all folds"""
    num_folds = len(all_results)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Accuracy comparison across folds
    plt.subplot(2, 3, 1)
    folds = [r['fold'] for r in all_results]
    test_accs = [r['test_acc'] for r in all_results]
    plt.bar(folds, test_accs, color='skyblue', edgecolor='navy')
    plt.axhline(y=np.mean(test_accs), color='r', linestyle='--', 
                label=f'Mean: {np.mean(test_accs):.4f}')
    plt.title('Test Accuracy per Fold')
    plt.ylabel('Accuracy')
    plt.xlabel('Test Fold')
    plt.legend()
    plt.ylim([0, 1])
    
    # 2. Loss comparison across folds
    plt.subplot(2, 3, 2)
    test_losses = [r['test_loss'] for r in all_results]
    plt.bar(folds, test_losses, color='lightcoral', edgecolor='darkred')
    plt.axhline(y=np.mean(test_losses), color='b', linestyle='--', 
                label=f'Mean: {np.mean(test_losses):.4f}')
    plt.title('Test Loss per Fold')
    plt.ylabel('Loss')
    plt.xlabel('Test Fold')
    plt.legend()
    
    # 3. Training curves for all folds (Accuracy)
    plt.subplot(2, 3, 3)
    for result in all_results:
        plt.plot(result['val_accuracies'], label=f"{result['fold']}", alpha=0.7)
    plt.title('Validation Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Training curves for all folds (Loss)
    plt.subplot(2, 3, 4)
    for result in all_results:
        plt.plot(result['val_losses'], label=f"{result['fold']}", alpha=0.7)
    plt.title('Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Sample distribution
    plt.subplot(2, 3, 5)
    train_samples = [r['num_train_samples'] for r in all_results]
    test_samples = [r['num_test_samples'] for r in all_results]
    x = np.arange(len(folds))
    width = 0.35
    plt.bar(x - width/2, train_samples, width, label='Train', color='lightgreen')
    plt.bar(x + width/2, test_samples, width, label='Test', color='orange')
    plt.xlabel('Fold')
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution per Fold')
    plt.xticks(x, folds)
    plt.legend()
    
    # 6. Summary statistics text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    test_accs = [r['test_acc'] for r in all_results]
    test_losses = [r['test_loss'] for r in all_results]
    summary_text = f"""
    6-Fold Cross-Validation Summary
    ================================
    
    Number of Folds: {num_folds}
    
    Test Accuracy:
      Mean: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}
      Min:  {np.min(test_accs):.4f} ({folds[np.argmin(test_accs)]})
      Max:  {np.max(test_accs):.4f} ({folds[np.argmax(test_accs)]})
    
    Test Loss:
      Mean: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}
      Min:  {np.min(test_losses):.4f} ({folds[np.argmin(test_losses)]})
      Max:  {np.max(test_losses):.4f} ({folds[np.argmax(test_losses)]})
    
    Average Training Time:
      {np.mean([r['best_epoch'] for r in all_results]):.1f} epochs
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {save_path}")
    plt.show()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    
    import time
    start_time = time.time()
    
    # L4 GPU Optimizations (24GB VRAM)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Auto-optimize CUDA kernels
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for speed
        torch.backends.cudnn.allow_tf32 = True
    
    # Configuration parameters
    NUM_EPOCHS = 75
    PATIENCE = 20  # Increased to survive 20-epoch restart cycles
    LR = 0.0001  # Restored to 87% baseline value (was incorrectly lowered to 5e-5)
    BATCH_SIZE = 64  # Optimized for L4 GPU (24GB) - 2x increase from 32
    WEIGHT_DECAY = 1e-3
    NUM_WORKERS = 8  # Increased for better data pipeline throughput
    
    QUICK_TEST = True
    USE_ENTROPY_FILTER = True  # Enable TPSF entropy filtering
    ENTROPY_THRESHOLD_SIGMA = 2.0  # Standard deviations for filtering (2.0σ = 5% filtered)
    
    # Set random seed for reproducibility
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"🎲 Random seed: {RANDOM_SEED}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Optimizations: CUDNN benchmark=True, TF32=True\n")
    else:
        print()
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Directory for saving models
    os.makedirs('saved_models', exist_ok=True)
    
    # Collect all data organized by folder
    print("="*80)
    print("COLLECTING DATA")
    print("="*80)
    
    if USE_ENTROPY_FILTER:
        print(f"🔬 ENTROPY FILTERING ENABLED (±{ENTROPY_THRESHOLD_SIGMA}σ threshold)")
        print(f"   Based on TPSF paper: removes poorly segmented cells")
        print(f"   Expected to filter ~5% of cells globally")
        print(f"   D4 expected: ~22.8% (high entropy variance)")
        print()
    
    dataset_dict = collect_data_by_folder(
        use_entropy_filter=USE_ENTROPY_FILTER,
        entropy_threshold_sigma=ENTROPY_THRESHOLD_SIGMA
    )
    
    # Check if any datasets were found
    if len(dataset_dict) == 0:
        print("Error: No datasets found!")
        return
    
    # Print total file count
    total_files = sum(len(files) for files, _ in dataset_dict.values())
    print(f"\nTotal files across all datasets: {total_files}")
    
    # Holdout and validation datasets
    d5_data = dataset_dict.pop('isolated_cells_D5')
    d6_data = dataset_dict.pop('isolated_cells_D6')

    cv_donor_names = ['isolated_cells_D1', 'isolated_cells_D2', 
                      'isolated_cells_D3', 'isolated_cells_D4'] 

    # Configure based on test mode
    if QUICK_TEST:
        print("\n" + "⚠️ " * 20)
        print("⚠️  QUICK TEST MODE ENABLED")
        print("⚠️  - Skipping cross-validation entirely")
        print("⚠️  - Training directly on D1+D2+D3+D4")
        print("⚠️  - Validating on D6")
        print("⚠️  - Testing on D5 (complete holdout)")
        print(f"⚠️  - Using validated hyperparameters (LR={LR}, BS={BATCH_SIZE}, WD={WEIGHT_DECAY})")
        print("⚠️  - Expected time: ~30 minutes")
        print("⚠️  - Set QUICK_TEST = False for full 4-fold CV")
        print("⚠️ " * 20 + "\n")
        
        print(f"Expected models to train: 1 (D5 final test only)")
    else:
        print(f"\nFull 4-fold CV with fixed hyperparameters")
        print(f"Expected models to train: {len(cv_donor_names)} outer folds + 1 (D5 final)")
    
    # Initialize list to store all results
    all_results = []
    
    # ==== Leave-One-Donor-Out Cross-Validation ====
    if not QUICK_TEST:
        print("\n" + "="*80)
        print("CROSS-VALIDATION")
        print("="*80)

        # Loop through each dataset as the held-out test fold
        for fold_idx, test_donor in enumerate(cv_donor_names):
            
            fold_start_time = time.time()
            
            # Split into train/val/test
            available_training_donors = [donor for donor in cv_donor_names if donor != test_donor]
            
            # Need at least 2 donors for train+val split
            if len(available_training_donors) < 2:
                print(f"\n⚠️  Skipping fold {test_donor} - not enough donors for proper train/val/test split")
                print(f"    (Need at least 3 total donors, have {len(cv_donor_names)})")
                continue
            
            # Use first available donor as validation, rest for training
            val_donor = available_training_donors[0]
            train_donors = available_training_donors[1:]
            

            print(f"\n>>> Test Donor: {test_donor} (Fold {fold_idx+1}/{len(cv_donor_names)})")
            print(f"    Validation Donor: {val_donor}")
            print(f"    Training Donors: {train_donors}")
            print(f"    Using fixed hyperparameters: LR={LR}, BS={BATCH_SIZE}, WD={WEIGHT_DECAY}")

            # Collect data
            train_files = []
            train_labels = []
            for donor in train_donors:
                files, labels = dataset_dict[donor]
                train_files.extend(files)
                train_labels.extend(labels)
            val_files, val_labels = dataset_dict[val_donor]
            test_files, test_labels = dataset_dict[test_donor]


            # Train fold with fixed hyperparameters
            fold_result = train_one_fold(
                fold_name=test_donor,
                train_files=train_files,
                train_labels=train_labels,
                val_files=val_files,
                val_labels=val_labels,
                test_files=test_files,
                test_labels=test_labels,
                device=device,
                num_epochs=NUM_EPOCHS,
                patience=PATIENCE,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                weight_decay=WEIGHT_DECAY,
                num_workers=NUM_WORKERS
            )
            
            # Store hyperparameters used in this fold
            fold_result['hyperparameters'] = {
                'learning_rate': LR,
                'batch_size': BATCH_SIZE,
                'weight_decay': WEIGHT_DECAY
            }
            all_results.append(fold_result)
            
            # Print fold completion time
            fold_time = time.time() - fold_start_time
            print(f"\n✓ Fold {fold_idx+1}/{len(cv_donor_names)} completed in {fold_time/60:.1f} minutes")
            
            # Estimate remaining time
            if fold_idx < len(cv_donor_names) - 1:
                avg_time_per_fold = (time.time() - start_time) / (fold_idx + 1)
                remaining_folds = len(cv_donor_names) - (fold_idx + 1)
                estimated_remaining = avg_time_per_fold * remaining_folds
                print(f"⏱️  Estimated time remaining: {estimated_remaining/3600:.2f} hours ({estimated_remaining/60:.1f} minutes)")
    else:
        print("\n⚠️  Quick test mode: Skipping cross-validation")
        print("   Going directly to D5 final test\n")
    
    # ==== Final D5 Generalization Test ====
    print("\n" + "="*80)
    print("FINAL GENERALIZATION TEST ON D5 (COMPLETE HOLDOUT)")
    print("="*80)
    
    # In quick test mode, use default hyperparameters
    if QUICK_TEST or len(all_results) == 0:
        print("\nUsing default hyperparameters (validated in literature):")
        best_overall_hyperparams = {
            'learning_rate': LR,
            'batch_size': BATCH_SIZE, 
            'weight_decay': WEIGHT_DECAY
        }
        print(f"  LR: {LR}, BS: {BATCH_SIZE}, WD: {WEIGHT_DECAY}")
    else:
        # Find best overall hyperparameters from outer CV results
        best_fold = max(all_results, key=lambda x: x['test_acc'])
        best_overall_hyperparams = best_fold['hyperparameters']
        
        print(f"\nBest hyperparameters from outer CV: {best_overall_hyperparams}")
        print(f"(From fold: {best_fold['fold']} with accuracy: {best_fold['test_acc']:.4f})")
    
    # Combine all 5 CV donors for training
    print(f"\nTraining on: D1, D2, D3, D4")
    print(f"Validation on: D6 (completely held out from CV)") 
    print(f"Testing on: D5 (never seen before)")
    
    final_train_files = []
    final_train_labels = [] 
    for donor in cv_donor_names:
        files, labels = dataset_dict[donor]
        final_train_files.extend(files)
        final_train_labels.extend(labels)
    
    # Use D6 for validation
    val_files, val_labels = d6_data

    # Get D5 test data
    d5_test_files, d5_test_labels = d5_data
    
    # Train final model on all CV donors with best hyperparameters
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL ON ALL CV DATA (D1+D2+D3+D4)")
    print("Testing on D5 (Complete Holdout)")
    print("="*80)
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL ON ALL CV DATA (D1+D2+D3+D4)")
    print("Testing on D5 (Complete Holdout)")
    print("="*80)
    
    d5_result = train_one_fold(
            fold_name="D5_FINAL_TEST",
            train_files=final_train_files,
            train_labels=final_train_labels,
            val_files=val_files,
            val_labels=val_labels,
            test_files=d5_test_files,  
            test_labels=d5_test_labels,
            device=device,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            batch_size=best_overall_hyperparams['batch_size'],
            learning_rate=best_overall_hyperparams['learning_rate'],
            weight_decay=best_overall_hyperparams['weight_decay'],
            num_workers=NUM_WORKERS
    )
    
    # Print final D5 results
    print("\n" + "="*80)
    print("FINAL D5 GENERALIZATION RESULTS")
    print("="*80)
    print(f"\n📊 Testing Results:")
    print(f"  D5 Test Accuracy: {d5_result['test_acc']:.4f}")
    print(f"  D5 Test F1:       {d5_result['f1']:.4f}")
    print(f"  D5 Test Loss:     {d5_result['test_loss']:.4f}")
    print(f"\n⚙️  Training Details:")
    print(f"  Best Epoch: {d5_result['best_epoch']}")
    print(f"  Hyperparameters: {best_overall_hyperparams}")
    
    # Save D5 results separately
    d5_results_file = 'saved_models/d5_final_test_results.json'
    with open(d5_results_file, 'w') as f:
        d5_json = {
            'timestamp': datetime.now().isoformat(),
            # Test results
            'test_accuracy': float(d5_result['test_acc']),
            'test_f1': float(d5_result['f1']),
            'test_precision': float(d5_result['precision']),
            'test_recall': float(d5_result['recall']),
            'test_loss': float(d5_result['test_loss']),
            # Training details
            'best_epoch': int(d5_result['best_epoch']),
            'hyperparameters': best_overall_hyperparams,
            'training_donors': cv_donor_names,
            'num_train_samples': d5_result['num_train_samples'],
            'num_test_samples': d5_result['num_test_samples'],
            'train_losses': [float(x) for x in d5_result['train_losses']],
            'val_losses': [float(x) for x in d5_result['val_losses']],
            'train_accuracies': [float(x) for x in d5_result['train_accuracies']],
            'val_accuracies': [float(x) for x in d5_result['val_accuracies']]
        }
        json.dump(d5_json, f, indent=2)
    
    print(f"\nD5 results saved to: {d5_results_file}")

    # ==== Summary ====
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    if len(all_results) > 0:
        # Only show CV results if we did CV
        print("\nCross-Validation Results Summary:")
        print("-" * 100)
        print(f"{'Test Fold':<20} {'Best Epoch':<12} {'Val Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 100)
        
        for result in all_results:
            print(f"{result['fold']:<20} {result['best_epoch']:<12} "
                  f"{result['val_acc']:<12.4f} {result['test_acc']:<12.4f} "
                  f"{result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1']:<12.4f}")
        
        print("-" * 100)
        
        # Calculate overall statistics
        mean_test_acc = np.mean([r['test_acc'] for r in all_results])
        std_test_acc = np.std([r['test_acc'] for r in all_results])
        mean_test_loss = np.mean([r['test_loss'] for r in all_results])
        std_test_loss = np.std([r['test_loss'] for r in all_results])
        
        print(f"\nOverall Cross-Validation Performance:")
        print(f"  Accuracy:  {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"  Loss:      {mean_test_loss:.4f} ± {std_test_loss:.4f}")
        print(f"  Precision: {np.mean([r['precision'] for r in all_results]):.4f} ± {np.std([r['precision'] for r in all_results]):.4f}")
        print(f"  Recall:    {np.mean([r['recall'] for r in all_results]):.4f} ± {np.std([r['recall'] for r in all_results]):.4f}")
        print(f"  F1-Score:  {np.mean([r['f1'] for r in all_results]):.4f} ± {np.std([r['f1'] for r in all_results]):.4f}")
        
        # Save results to JSON
        results_file = 'saved_models/cross_validation_results.json'
        with open(results_file, 'w') as f:
            json_results = []
            for r in all_results:
                json_r = r.copy()
                json_r['train_losses'] = [float(x) for x in r['train_losses']]
                json_r['val_losses'] = [float(x) for x in r['val_losses']]
                json_r['train_accuracies'] = [float(x) for x in r['train_accuracies']]
                json_r['val_accuracies'] = [float(x) for x in r['val_accuracies']]
                json_results.append(json_r)
            
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'mean_test_accuracy': float(mean_test_acc),
                'std_test_accuracy': float(std_test_acc),
                'mean_test_loss': float(mean_test_loss),
                'std_test_loss': float(std_test_loss),
                'folds': json_results
            }, f, indent=2)
        
        print(f"\nCV results saved to: {results_file}")
        
        # Create visualization
        plot_cross_validation_results(all_results)
    else:
        print("\n⚠️  Quick test mode: Cross-validation was skipped")
        print("   Only D5 final test was performed")
        print("   Set QUICK_TEST=False to run full 4-fold CV")
    
    # Calculate and display total time
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"⏰ Started at:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"⏰ Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"⏱️  Total time:  {hours}h {minutes}m {seconds}s ({total_time/3600:.2f} hours)")
    print("="*80)


# ============================================================================
# SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()
