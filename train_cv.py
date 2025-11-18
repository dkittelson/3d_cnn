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

from resnet import create_model

# Cache directory for preprocessed data
CACHE_DIR = Path('./cache')
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATASET CLASS
# ============================================================================

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

        # Normalize
        array = array / total_photons_per_pixel

        # Transpose to (Time, Height, Width) for 3D CNN
        array = np.transpose(array, (2, 0, 1))

        # Add channel dimension
        array = np.expand_dims(array, axis=0)

        # Convert to PyTorch tensors 
        X = torch.tensor(array, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return X, y
    

# ============================================================================
# FOCAL LOSS FUNCTION
# ============================================================================

class FocalLoss(nn.Module):
    "Focal Loss for handling class imbalances and overal generalization"
    def __init__(self, gamma=2.0, auto_balance=True, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.auto_balance = auto_balance
        self.alpha = None
        self.label_smoothing = label_smoothing

    def set_class_weights(self, pos_samples, neg_samples):
        total = pos_samples + neg_samples
        self.alpha = neg_samples / total

        print(f"  ✓ Dynamic alpha computed: {self.alpha:.3f}")
        print(f"    (Positive weight: {self.alpha:.3f}, Negative weight: {1-self.alpha:.3f})")
        print(f"    Dataset: {pos_samples} active ({pos_samples/total*100:.1f}%), "
              f"{neg_samples} inactive ({neg_samples/total*100:.1f}%)")

    def forward(self, inputs, targets):
            """Compute focal loss."""

            # Ensure targets have correct shape
            targets = targets.view(-1, 1).float()
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

              # Compute BCE Loss 
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )

            # Get probability of CORRECT class (sigmoid)
            probs = torch.sigmoid(inputs)

            # Calculate probability of the SPECIFIC class
            # If target=1 (active), p_t = prob
            # If target=0 (inactive), p_t = 1 - prob
            p_t = probs * targets + (1 - probs) * (1 - targets)

            # Calculate focal weights
            # - Easy samples (p_t close to 1): weight → 0 (ignored)
            # - Hard samples (p_t close to 0): weight → 1 (focused on)
            focal_weight = (1 - p_t) ** self.gamma

            # Calculate alpha weight to balance classes
            if self.auto_balance and self.alpha is not None:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_t = 0.5

            # Combines all components
            focal_loss = alpha_t * focal_weight * bce_loss

            # Return mean loss
            return focal_loss.mean()

    def __repr__(self):
        alpha_str = f"{self.alpha:.3f}" if self.alpha else "auto"
        return f"FocalLoss(alpha={alpha_str}, gamma={self.gamma})" 

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
    print(f"  - Class-Weighted BCE Loss handles imbalance (pos_weight will be calculated below)")
    print(f"  - Using natural batch distribution (no resampling)\n")

    # Create DataLoaders with simple random shuffling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Simple random shuffling (no resampling)  
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,  
        prefetch_factor=2  
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,  # Optimal for Colab
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,  # Keeps workers alive between epochs (much faster)
        prefetch_factor=2  # Lower prefetch for persistent workers
    )

    # Create fresh model
    model = create_model().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(device.type)  
    
    # ReduceLROnPlateau Scheduler (validation-based, stable for domain shift)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Maximize validation F1
        factor=0.5,           # Reduce LR by half
        patience=5,           # Wait 5 epochs before reducing
        min_lr=1e-6           # Don't go below this
    )   

    # Class-Weighted BCE Loss (standard for moderate class imbalance)
    active_count = sum(1 for label in train_labels if label == 1)
    inactive_count = len(train_labels) - active_count
    
    # Calculate pos_weight: ratio of negative to positive samples
    # This makes the model care equally about both classes
    pos_weight = torch.tensor([inactive_count / active_count]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"  ✓ Class-weighted BCE Loss initialized")
    print(f"    • Positive weight: {pos_weight.item():.3f}")
    print(f"    • Dataset: {active_count} active ({active_count/len(train_labels)*100:.1f}%), {inactive_count} inactive ({inactive_count/len(train_labels)*100:.1f}%)")
    print(f"    • This weight ensures equal importance for both classes")

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

            with autocast('cuda'):
                # Mixup augmentation
                if np.random.rand() < 0.5:  # 50% probability
                    
                    # Mixup ratio
                    lam = np.random.beta(1.0, 1.0)  # Uniform mixing
                    
                    # Random permutation of batch
                    batch_size_actual = inputs.size(0)
                    index = torch.randperm(batch_size_actual).to(device)

                    # Mix inputs
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

                    # Get both labels
                    labels_a = labels
                    labels_b = labels[index]

                    # Forward pass on mixed inputs
                    outputs = model(mixed_inputs)

                    # Mixed loss
                    loss = lam * loss_fn(outputs, labels_a.unsqueeze(1)) + (1 - lam) * loss_fn(outputs, labels_b.unsqueeze(1))

                else:
                    # Normal training
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels.unsqueeze(1))

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

        # Step scheduler based on validation F1 (ReduceLROnPlateau)
        scheduler.step(epoch_val_f1)
        
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

    # Standard Testing
    print("\nTesting on holdout set:")
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = loss_fn(outputs, labels.unsqueeze(1))
            test_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels.unsqueeze(1)).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_test_loss = test_loss / len(test_loader)
    final_test_acc = correct_test / total_test

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm = confusion_matrix(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    print(f"\n📊 Test Results:")
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
    PATIENCE = 15
    LR = 0.001
    BATCH_SIZE = 16
    WEIGHT_DECAY = 1e-3
    
    QUICK_TEST = True
    
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
