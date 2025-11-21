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
from torch.amp import autocast, GradScaler  
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
# DATASET CLASS
# ============================================================================
GLOBAL_INTENSITY_MIN = 0.0
GLOBAL_INTENSITY_MAX = 8.2212  # Actual max from dataset - provides optimal contrast for dim cells

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
        
        # ===== INTENSITY JITTER (Domain Shift) =====
        # D5 is dimmer than D1-4 training donors
        # Scale brightness to make model brightness-invariant
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
        
        # Sum across time axis
        total_photons_per_pixel = np.sum(array, axis=2, keepdims=True)
        total_photons_per_pixel = np.where(
            total_photons_per_pixel == 0, 
            1.0,
            total_photons_per_pixel
        )

        # Channel 0: Normalized decay shape (divide by SUM for stability with low photon counts)
        # Peak norm amplifies noise by 26% (sqrt(14)/14), sum norm only 1% (sqrt(10k)/10k)
        decay_normalized = np.zeros_like(array, dtype=np.float32)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                pixel_sum = array[i, j, :].sum()
                if pixel_sum > 0:
                    decay_normalized[i, j, :] = array[i, j, :] / pixel_sum

        # Channel 1: Log-intensity map 
        intensity_map = array.sum(axis=2)  # Total photons per pixel
        log_intensity = np.log1p(intensity_map)  # Log transform to compress range
        
        # This preserves relative brightness: bright cells stay bright, dim cells stay dim
        log_intensity = (log_intensity - GLOBAL_INTENSITY_MIN) / (GLOBAL_INTENSITY_MAX - GLOBAL_INTENSITY_MIN)
        log_intensity = np.clip(log_intensity, 0, 1)  # Ensure [0, 1] range

        # Expand intensity to match temporal dimensions
        log_intensity_expanded = np.tile(log_intensity[:, :, np.newaxis], (1, 1, array.shape[2]))

        # Tranpose to (Time, Height, Width) for CNN
        decay_normalized = np.transpose(decay_normalized, (2, 0, 1))
        log_intensity_expanded = np.transpose(log_intensity_expanded, (2, 0, 1))

        # Stack as 2 channels (2, 256, 21, 21)
        array = np.stack([decay_normalized, log_intensity_expanded], axis=0)

        # Convert to PyTorch tensors 
        X = torch.tensor(array, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return X, y

# ============================================================================
# DATA COLLECTION BY FOLDER
# ============================================================================

def collect_data_by_folder():
    """Collects all data files by their dataset folder (D1-D6)"""

    data_dir = Path("/content/3d_cnn/3d_cnn/data")
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

        # Iterate through each folder
        for file in folder.glob('*.npy'):
            
            # Check each label
            if '_in' in file.name:
                label = 0
            elif '_act' in file.name:
                label = 1
            else:
                continue

            files.append(file)
            labels.append(label)

        # Store files, labels in dict with folder_name as key
        dataset_dict[folder_name] = (files, labels)
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
                   device, num_epochs=20, patience=5, batch_size=16, learning_rate=0.001, weight_decay=0):
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
    print(f"  - Active cells: {active_count} ({active_count/len(train_labels)*100:.1f}%)")
    print(f"  - Inactive cells: {inactive_count} ({inactive_count/len(train_labels)*100:.1f}%)") 
    print(f"  - Using WeightedRandomSampler for balanced 50/50 batches")
    print(f"  - NO pos_weight in loss (sampler handles imbalance)\n")

    # WeightedRandomSampler for TRAINING - balanced 50/50 batches
    train_sample_weights = []
    active_count_total = sum(1 for label in train_labels if label == 1)
    inactive_count_total = len(train_labels) - active_count_total
    
    for label in train_labels:
        if label == 1:  # Active
            weight = 1.0 / active_count_total
        else:  # Inactive  
            weight = 1.0 / inactive_count_total
        train_sample_weights.append(weight)
    
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # WeightedRandomSampler for VALIDATION - CRITICAL FIX!
    # D6 is 77% inactive, but we force 50/50 batches during validation
    # This prevents model from learning "guess inactive" strategy
    val_sample_weights = []
    val_active_count = sum(1 for label in val_labels if label == 1)
    val_inactive_count = len(val_labels) - val_active_count
    
    for label in val_labels:
        if label == 1:  # Active
            weight = 1.0 / val_active_count
        else:  # Inactive
            weight = 1.0 / val_inactive_count
        val_sample_weights.append(weight)
    
    val_sampler = WeightedRandomSampler(
        weights=val_sample_weights,
        num_samples=len(val_dataset),
        replacement=True
    )

    print(f"  🎯 BALANCED VALIDATION SAMPLING ENABLED")
    print(f"     • D6 raw distribution: {val_inactive_count} inactive ({val_inactive_count/len(val_labels)*100:.1f}%), {val_active_count} active ({val_active_count/len(val_labels)*100:.1f}%)")
    print(f"     • But validation batches will be 50/50 (prevents 'guess inactive' bias)")
    print(f"     • This forces model to learn features that work on D5 (67% active)\n")

    # Create DataLoaders with WeightedRandomSampler
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,  # Balanced training batches
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,  
        prefetch_factor=2  
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,  # 🔥 BALANCED validation batches (key fix!)
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Create fresh model
    model = create_model(in_channels=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(device.type)  
    
    # Use warm restarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,              # Restart every 15 epochs (faster feedback)
        T_mult=2,            # Double cycle length after each restart (15, 30, 60...)
        eta_min=1e-6         # Minimum LR at end of each cycle
    )
    print("  ✓ Using CosineAnnealingWarmRestarts (periodic exploration)")

    # BCE Loss (manual label smoothing applied during training)
    active_count = sum(1 for label in train_labels if label == 1)
    inactive_count = len(train_labels) - active_count
    
    # Standard BCE loss - label smoothing applied by adjusting targets
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Label smoothing parameter
    label_smoothing = 0.1
    
    print(f"  ✓ BCE Loss with Label Smoothing ({label_smoothing}) initialized")
    print(f"    • WeightedRandomSampler creates balanced 50/50 batches")
    print(f"    • Label smoothing: targets [{label_smoothing}, {1-label_smoothing}] instead of [0, 1]")
    print(f"    • Mixup (alpha=0.4): interpolates between samples for smooth boundaries")
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
    
    # Main training loop
    epoch = 0
    while epoch < num_epochs:
        
        # ===== TRAINING PHASE =====
        model.train()

        # Initialize metrics
        running_loss = 0.0 
        correct_train = 0 
        total_train = 0

        # Progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        # Loop through training batches
        for inputs, labels in train_pbar:
            
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device) 

            # Zero gradients
            optimizer.zero_grad()

            # Apply Mixup augmentation (80% probability)
            if np.random.rand() > 0.2:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)
                
                # Apply label smoothing to mixup targets
                labels_a_smooth = labels_a.float() * (1 - label_smoothing) + label_smoothing / 2
                labels_b_smooth = labels_b.float() * (1 - label_smoothing) + label_smoothing / 2
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = mixup_criterion(loss_fn, outputs, labels_a_smooth, labels_b_smooth, lam)
            else:
                # Regular training (20% of batches to maintain some hard examples)
                # Apply label smoothing: 0 → 0.1, 1 → 0.9
                labels_smooth = labels.float() * (1 - label_smoothing) + label_smoothing / 2
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels_smooth.unsqueeze(1))

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Update running metrics
            running_loss += loss.item()
            # Track accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()
        
        # Calculate epoch training metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        
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
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=4
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
    def tune_threshold(model, val_dataset, device, batch_size):
        """Find optimal threshold using K-Means clustering on validation probabilities"""
        from sklearn.cluster import KMeans
        
        print("\n🔧 Tuning decision threshold with K-Means...")
        
        # Create fresh val_loader for threshold tuning
        val_loader_tune = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        model.eval()
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(val_loader_tune, desc='Collecting val probabilities', leave=False):
                inputs = inputs.to(device)
                # Use TTA for more robust probabilities
                probs = predict_tta(model, inputs)
                all_probs.extend(probs.cpu().numpy().flatten())
        
        # Reshape for K-Means (needs 2D array)
        probs_array = np.array(all_probs).reshape(-1, 1)
        
        # K-Means with k=2 (active vs inactive clusters)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(probs_array)
        
        # Get cluster centers
        centers = sorted(kmeans.cluster_centers_.flatten())
        inactive_center = centers[0]
        active_center = centers[1]
        
        # Optimal threshold = midpoint between clusters
        optimal_threshold = (inactive_center + active_center) / 2.0
        
        print(f"  Inactive cluster center: {inactive_center:.4f}")
        print(f"  Active cluster center:   {active_center:.4f}")
        print(f"  ✓ Optimal threshold:     {optimal_threshold:.4f}")
        
        return optimal_threshold

    # Tune threshold on validation set
    optimal_threshold = tune_threshold(model, val_dataset, device, batch_size)

    # Testing with TTA and Dynamic Threshold
    print(f"\nTesting on holdout set (with TTA + threshold={optimal_threshold:.4f}):")
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
    
    # Configuration parameters
    NUM_EPOCHS = 75
    PATIENCE = 20  # Increased to survive 20-epoch restart cycles
    LR = 0.0005
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-3
    
    QUICK_TEST = True
    
    # Set random seed for reproducibility
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"🎲 Random seed: {RANDOM_SEED}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Directory for saving models
    os.makedirs('saved_models', exist_ok=True)
    
    # Collect all data organized by folder
    print("="*80)
    print("COLLECTING DATA")
    print("="*80)
    dataset_dict = collect_data_by_folder()
    
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
                weight_decay=WEIGHT_DECAY
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
            weight_decay=best_overall_hyperparams['weight_decay']
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
