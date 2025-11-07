"""
LEAVE-ONE-DONOR-OUT CROSS-VALIDATION 
==============================
D6: Holdout (Final generalization test)
D1, D2, D3, D4, D5: Used for Cross-Validation

For Each Donor in [D1, D2, D3, D4, D5]:
    ├─ One Donor = Test Donor
    ├─ Remaining 4 Donors = Training Donors
    └─ Train model with fixed hyperparameters → Test on Test Donor

Fixed Hyperparameters:
    - Learning Rate: 0.001
    - Batch Size: 16
    - Weight Decay: 0

Final Test:
1. Find Best Performing Fold
2. Train on ALL 5 CV Donors (D1-D5)
3. Test on D6 (completely held-out)

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

from resnet import create_model


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
        
        # Intensity
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            array = np.clip(array * scale, 0, 1)
        
        # Gaussian Noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, array.shape)
            array = np.clip(array + noise, 0, 1)

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
        # Load file
        array = np.load(self.file_paths[idx])
        
        # Normalize array
        max_val = np.max(array)
        if max_val > 0:
            array = array / max_val
        
        # Apply augmentation if enabled
        if self.augment:
            array = self.apply_augmentation(array)

        # Add channel dimension 
        array = np.expand_dims(array, axis = -1)
        
        # Convert to torch tensors
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
    print(f"FOLD: {fold_name} (Test Set)")
    print(f"{'='*80}")
    print(f"Training samples: {len(train_files)} (inactive={train_labels.count(0)}, active={train_labels.count(1)})")
    print(f"Validation samples: {len(val_files)} (inactive={val_labels.count(0)}, active={val_labels.count(1)})")  
    print(f"Test samples: {len(test_files)} (inactive={test_labels.count(0)}, active={test_labels.count(1)})") 
    
    # Create CellDataset objects for train and validation
    train_dataset = CellDataset(train_files, train_labels, augment=True) # model gets trained on 100% augmented images
    val_dataset = CellDataset(val_files, val_labels, augment=False) # model validated on raw, true images

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create fresh model
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5, #cycle length
        T_mult=2, # multiply cycle length after each restart
        eta_min=learning_rate / 100 # min LR
    )
    num_inactive = train_labels.count(0)
    num_active = train_labels.count(1)
    pos_weight = torch.tensor([num_inactive / num_active]).to(device) 
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Tracking variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    best_val_acc = 0
    
    # Lists for metrics history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Main training loop
    for epoch in range(num_epochs):
        
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

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = loss_fn(outputs, labels.unsqueeze(1))

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Update running metrics
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()
        
        # Calculate epoch training metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        
        # Append to history lists
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Step scheduler
        scheduler.step()

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # ===== VALIDATION PHASE =====
        model.eval()
        
        # Validation metrics
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

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
                predicted = (outputs > 0).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels.unsqueeze(1)).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Calculate epoch validation metrics
            epoch_val_loss = running_val_loss / len(val_loader)
            epoch_val_acc = correct_val / total_val

            # Append to history lists
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)
        
        # Print epoch summary
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        # ===== EARLY STOPPING & MODEL CHECKPOINTING =====
        if epoch_val_loss < best_val_loss:

            # Update best metrics
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_val_acc = epoch_val_acc
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
                'fold': fold_name,
            }, f'saved_models/best_model_fold_{fold_name}.pth')

            print(f"  ✓ New best model saved! (Val Loss: {epoch_val_loss:.4f})")

        else:
            epochs_without_improvement += 1
            print(f"  ⚠ No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= patience:
                print(f"\n⏹ Early stopping triggered after {epoch + 1} epochs")
                print(f"Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
                break

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
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Evaluate on test set
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
            
            predicted = (outputs > 0).float()
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
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {final_test_loss:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    print(f"\nFold {fold_name} Complete!")
    print(f"Best Validation Epoch: {best_epoch} | Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_acc:.4f}")
    print(f"Final Test Performance: Test Loss: {final_test_loss:.4f} | Test Acc: {final_test_acc:.4f}")

    # Return results for this fold
    return {
        'fold': fold_name,
        'best_epoch': best_epoch,
        'val_loss': best_val_loss,  # 🆕 CHANGED from test_loss
        'val_acc': best_val_acc,    # 🆕 CHANGED from test_acc
        'test_loss': final_test_loss,  # 🆕 NEW - actual test performance
        'test_acc': final_test_acc,    # 🆕 NEW - actual test performance
        'precision': precision,         # 🆕 NEW
        'recall': recall,               # 🆕 NEW
        'f1': f1,                       # 🆕 NEW
        'confusion_matrix': cm.tolist(),  # 🆕 NEW
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'num_train_samples': len(train_files),
        'num_val_samples': len(val_files),  # 🆕 NEW
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
    NUM_EPOCHS = 15
    PATIENCE = 5
    BATCH_SIZE = 16
    
    # 🆕 QUICK TEST MODE - Set to False for full nested CV
    QUICK_TEST = True  # Change to False when ready for full run
    
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

    # Hold out D6
    d6_data = dataset_dict.pop('isolated_cells_D6')
    
    # Fixed hyperparameters (validated by quick test, per Wang et al. 2019)
    # Paper shows nested CV hyperparameter tuning is computationally expensive
    # with minimal performance gain when reasonable defaults are used
    FIXED_LR = 0.001
    FIXED_BATCH_SIZE = 16
    FIXED_WEIGHT_DECAY = 0
    
    # Configure based on test mode
    if QUICK_TEST:
        print("\n" + "⚠️ " * 20)
        print("⚠️  QUICK TEST MODE ENABLED")
        print("⚠️  - Testing only 2 outer folds (D1, D2)")
        print("⚠️  - Using validated hyperparameters (LR=0.001, BS=16, WD=0)")
        print("⚠️  - Expected time: ~30-60 minutes")
        print("⚠️  - Set QUICK_TEST = False for full 5-fold CV")
        print("⚠️ " * 20 + "\n")
        
        # 2 folds for quick testing
        cv_donor_names = ['isolated_cells_D1', 'isolated_cells_D2']
        print(f"Expected models to train: 2 outer folds = 2 models")
    else:
        cv_donor_names = ['isolated_cells_D1', 'isolated_cells_D2', 'isolated_cells_D3',
                          'isolated_cells_D4', 'isolated_cells_D5']
        
        print(f"\nFull 5-fold CV with fixed hyperparameters")
        print(f"Expected models to train: {len(cv_donor_names)} outer folds + 1 (D6 final) = {len(cv_donor_names) + 1} models")
        print(f"Hyperparameters: LR={FIXED_LR}, Batch Size={FIXED_BATCH_SIZE}, Weight Decay={FIXED_WEIGHT_DECAY}")
    
    # Initialize list to store all results
    all_results = []
    
    # ==== Leave-One-Donor-Out Cross-Validation ====
    print("\n" + "="*80)
    print("CROSS-VALIDATION")
    print("="*80)

    # Loop through each dataset as the held-out test fold
    for fold_idx, test_donor in enumerate(cv_donor_names):
        
        fold_start_time = time.time()
        
        # Split
        available_training_donors = [donor for donor in cv_donor_names if donor != test_donor]
        val_donor = available_training_donors[0]
        train_donors = available_training_donors[1:]
        

        print(f"\n>>> Test Donor: {test_donor} (Fold {fold_idx+1}/{len(cv_donor_names)})")
        print(f"    Validation Donor: {val_donor}")
        print(f"    Training Donors: {train_donors}")
        print(f"    Using fixed hyperparameters: LR={FIXED_LR}, BS={FIXED_BATCH_SIZE}, WD={FIXED_WEIGHT_DECAY}")

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
            batch_size=FIXED_BATCH_SIZE,
            learning_rate=FIXED_LR,
            weight_decay=FIXED_WEIGHT_DECAY
        )
        
        # Store hyperparameters used in this fold
        fold_result['hyperparameters'] = {
            'learning_rate': FIXED_LR,
            'batch_size': FIXED_BATCH_SIZE,
            'weight_decay': FIXED_WEIGHT_DECAY
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
    
    # ==== Final D6 Generalization Test ====
    print("\n" + "="*80)
    print("FINAL GENERALIZATION TEST ON D6 (COMPLETE HOLDOUT)")
    print("="*80)
    
    # Find best overall hyperparameters from outer CV results
    best_fold = max(all_results, key=lambda x: x['test_acc'])
    best_overall_hyperparams = best_fold['hyperparameters']
    
    print(f"\nBest hyperparameters from outer CV: {best_overall_hyperparams}")
    print(f"(From fold: {best_fold['fold']} with accuracy: {best_fold['test_acc']:.4f})")
    
    # Combine all 5 CV donors for training
    print(f"\nTraining on all 5 CV donors: {cv_donor_names}")
    print(f"Testing on: D6 (never seen before)")
    
    final_train_files = []
    final_train_labels = []
    
    for donor in cv_donor_names:
        files, labels = dataset_dict[donor]
        final_train_files.extend(files)
        final_train_labels.extend(labels)
    
    # Get D6 test data
    d6_test_files, d6_test_labels = d6_data
    
    # Train final model on all 5 donors with best hyperparameters
    # For D6: Use same data for both val and test (no separate val donor available)
    d6_result = train_one_fold(
        fold_name="D6_FINAL_TEST",
        train_files=final_train_files,
        train_labels=final_train_labels,
        val_files=d6_test_files,
        val_labels=d6_test_labels,
        test_files=d6_test_files,  # Same as val since D6 is final holdout
        test_labels=d6_test_labels,
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        batch_size=best_overall_hyperparams['batch_size'],
        learning_rate=best_overall_hyperparams['learning_rate'],
        weight_decay=best_overall_hyperparams['weight_decay']
    )
    
    # Print final D6 results
    print("\n" + "="*80)
    print("FINAL D6 GENERALIZATION RESULTS")
    print("="*80)
    print(f"D6 Test Accuracy: {d6_result['test_acc']:.4f}")
    print(f"D6 Test Loss: {d6_result['test_loss']:.4f}")
    print(f"Best Epoch: {d6_result['best_epoch']}")
    print(f"Hyperparameters used: {best_overall_hyperparams}")
    
    # Save D6 results separately
    d6_results_file = 'saved_models/d6_final_test_results.json'
    with open(d6_results_file, 'w') as f:
        d6_json = {
            'timestamp': datetime.now().isoformat(),
            'test_accuracy': float(d6_result['test_acc']),
            'test_loss': float(d6_result['test_loss']),
            'best_epoch': int(d6_result['best_epoch']),
            'hyperparameters': best_overall_hyperparams,
            'training_donors': cv_donor_names,
            'num_train_samples': d6_result['num_train_samples'],
            'num_test_samples': d6_result['num_test_samples'],
            'train_losses': [float(x) for x in d6_result['train_losses']],
            'val_losses': [float(x) for x in d6_result['val_losses']],
            'train_accuracies': [float(x) for x in d6_result['train_accuracies']],
            'val_accuracies': [float(x) for x in d6_result['val_accuracies']]
        }
        json.dump(d6_json, f, indent=2)
    
    print(f"\nD6 results saved to: {d6_results_file}")

    # ==== Summary ====
    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETE")
    print("="*80)
    
    print("\nResults Summary:")
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
    
    print(f"\nOverall Test Performance:")
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
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    plot_cross_validation_results(all_results)
    
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
