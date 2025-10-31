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

from resnet import create_model


# ============================================================================
# SECTION 1: DATASET CLASS
# ============================================================================

class CellDataset(Dataset):
    """Custom Dataset that loads data on-the-fly"""

    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load file
        array = np.load(self.file_paths[idx])

        # Normalize array
        max_val = np.max(array)
        if max_val > 0:
            array = array / max_val

        # Add channel dimension 
        array = np.expand_dims(array, axis = -1)
        
        # Convert to torch tensors
        X = torch.tensor(array, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return X, y


# ============================================================================
# SECTION 2: DATA COLLECTION BY FOLDER
# ============================================================================

def collect_data_by_folder():
    """Collects all data files by their dataset folder (D1-D6)"""

    data_dir = Path("data")
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
# SECTION 3: CREATE TRAIN/VALIDATION SPLIT FOR ONE FOLD
# ============================================================================

def create_fold_data(dataset_dict, test_fold, val_fold):
    """Splits data into train and validation sets for a specific fold"""
    
    # Create four empty lists
    train_files = []
    train_labels = []
    val_files = []
    val_labels = []
    test_files = []
    test_labels = []
    
    # Loop through each dataset in the dict
    for fold_name, (files, labels) in dataset_dict.items():
        
        # Test set
        if fold_name == test_fold:
            test_files.extend(files)
            test_labels.extend(labels)

        # Validation set
        elif fold_name == val_fold:
            val_files.extend(files)
            val_labels.extend(labels)

        # Train set
        else:
            train_files.extend(files)
            train_labels.extend(labels)

        
    return train_files, train_labels, val_files, val_labels, test_files, test_labels
# ============================================================================
# SECTION 4: TRAIN ONE FOLD
# ============================================================================

def train_one_fold(fold_name, train_files, train_labels, val_files, val_labels, test_files, 
                   test_labels, device, num_epochs=20, patience=5, batch_size=16):
    """Trains and evaluates a single fold"""
    
    # Print fold header and data summary
    print(f"\n{'='*80}")
    print(f"FOLD: {fold_name} (Validation Set)")
    print(f"{'='*80}")
    print(f"Training samples: {len(train_files)} (inactive={train_labels.count(0)}, active={train_labels.count(1)})")
    print(f"Validation samples: {len(val_files)} (inactive={val_labels.count(0)}, active={val_labels.count(1)})")
    print(f"Test samples: {len(test_files)} (inactive={test_labels.count(0)}, active={test_labels.count(1)})")
    
    # Create CellDataset objects for train and validation
    train_dataset = CellDataset(train_files, train_labels)
    val_dataset = CellDataset(val_files, val_labels)
    test_dataset = CellDataset(test_files, test_labels)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create fresh model
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_inactive = train_labels.count(0)
    num_active = train_labels.count(1)
    pos_weight = torch.tensor([num_active / num_inactive]).to(device)
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
            running_loss += loss.item() # total loss across all batches
            predicted = (outputs > 0).float() # convert probabilities to actual predictions (0, 1)
            total_train += labels.size(0) # count how many samples processed
            correct_train += (predicted == labels.unsqueeze(1)).sum().item() # count correct predictions
        
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

    # ==== TESTING PHASE ====
    print(f"\n{'='*80}")
    print(f"TESTING ON HELD-OUT SET: {fold_name}")
    print(f"{'='*80}")

    # Load best model
    checkpoint = torch.load(f'saved_models/best_model_fold_{fold_name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0

    test_pbar = tqdm(test_loader, desc='Testing')

    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.unsqueeze[1])

            running_test_loss += loss.item()
            predicted = (outputs > 0).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels.unsqueeze(1)).sum().item()

            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    test_loss = running_test_loss / len(test_loader)
    test_acc = correct_test / total_test

    print(f"\nFold {fold_name} Complete!")
    print(f"Best Validation - Epoch: {best_epoch} | Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_acc:.4f}")
    print(f"Test Performance - Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Return results for this fold
    return {
        'fold': fold_name,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'num_train_samples': len(train_files),
        'num_val_samples': len(val_files),
    }    

# ============================================================================
# SECTION 5: VISUALIZATION
# ============================================================================

def plot_cross_validation_results(all_results, save_path='saved_models/cross_validation_results.png'):
    """Create comprehensive plots for all folds"""
    num_folds = len(all_results)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Accuracy comparison across folds
    plt.subplot(2, 3, 1)
    folds = [r['fold'] for r in all_results]
    best_accs = [r['best_val_acc'] for r in all_results]
    plt.bar(folds, best_accs, color='skyblue', edgecolor='navy')
    plt.axhline(y=np.mean(best_accs), color='r', linestyle='--', 
                label=f'Mean: {np.mean(best_accs):.4f}')
    plt.title('Best Validation Accuracy per Fold')
    plt.ylabel('Accuracy')
    plt.xlabel('Test Fold')
    plt.legend()
    plt.ylim([0, 1])
    
    # 2. Loss comparison across folds
    plt.subplot(2, 3, 2)
    best_losses = [r['best_val_loss'] for r in all_results]
    plt.bar(folds, best_losses, color='lightcoral', edgecolor='darkred')
    plt.axhline(y=np.mean(best_losses), color='b', linestyle='--', 
                label=f'Mean: {np.mean(best_losses):.4f}')
    plt.title('Best Validation Loss per Fold')
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
    val_samples = [r['num_val_samples'] for r in all_results]
    x = np.arange(len(folds))
    width = 0.35
    plt.bar(x - width/2, train_samples, width, label='Train', color='lightgreen')
    plt.bar(x + width/2, val_samples, width, label='Validation', color='orange')
    plt.xlabel('Fold')
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution per Fold')
    plt.xticks(x, folds)
    plt.legend()
    
    # 6. Summary statistics text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
    Cross-Validation Summary
    ========================
    
    Number of Folds: {num_folds}
    
    Validation Accuracy:
      Mean: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}
      Min:  {np.min(best_accs):.4f} ({folds[np.argmin(best_accs)]})
      Max:  {np.max(best_accs):.4f} ({folds[np.argmax(best_accs)]})
    
    Validation Loss:
      Mean: {np.mean(best_losses):.4f} ± {np.std(best_losses):.4f}
      Min:  {np.min(best_losses):.4f} ({folds[np.argmin(best_losses)]})
      Max:  {np.max(best_losses):.4f} ({folds[np.argmax(best_losses)]})
    
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
# SECTION 6: MAIN FUNCTION
# ============================================================================

def main():
    
    # Configuration parameters
    NUM_EPOCHS = 20
    PATIENCE = 5
    BATCH_SIZE = 16
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
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

    # Get sorted folder names
    folder_names = sorted(dataset_dict.keys())
    
    # Initialize list to store all results
    all_results = []
    
    # ==== Leave-One-Patient-Out Cross-Validation ====
    print("\n" + "="*80)
    print("LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION")
    print("="*80)


    for i, test_fold in enumerate(folder_names):

        # Choose next folder as validation fold
        val_fold = folder_names[(i + 1) % len(folder_names)]

        print(f"\n>>> Configuration for this fold:")
        print(f"    Test Fold: {test_fold}")
        print(f"    Validation Fold: {val_fold}")
        print(f"    Training Folds: {[f for f in folder_names if f not in [test_fold, val_fold]]}")

        # Create train/val/test split
        train_files, train_labels, val_files, val_labels, test_files, test_labels = create_fold_data(dataset_dict, test_fold=test_fold, val_fold=val_fold)
        
        # Train fold
        fold_result = train_one_fold(
            fold_name=test_fold,
            train_files=train_files,
            train_labels=train_labels,
            val_files=val_files,
            val_labels=val_labels,
            device=device,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE
        )
        
        # Append result to all_results
        all_results.append(fold_result)
    
    # ==== Summary ====
    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETE")
    print("="*80)
    
    print("\nResults Summary:")
    print("-" * 100)
    print(f"{'Test Fold':<20} {'Val Fold':<20} {'Epoch':<8} {'Val Acc':<10} {'Test Acc':<10} {'Samples (train/val/test)'}")
    print("-" * 100)
    
    for i, result in enumerate(all_results):
        val_fold = folder_names[(i + 1) % len(folder_names)]
        print(f"{result['fold']:<20} {val_fold:<20} {result['best_epoch']:<8} "
              f"{result['best_val_acc']:<10.4f} {result['test_acc']:<10.4f} "
              f"{result['num_train_samples']}/{result['num_val_samples']}/{result['num_test_samples']}")
    
    print("-" * 100)
    
    # Calculate overall statistics
    mean_test_acc = np.mean([r['test_acc'] for r in all_results])
    std_test_acc = np.std([r['test_acc'] for r in all_results])
    mean_val_acc = np.mean([r['best_val_acc'] for r in all_results])
    std_val_acc = np.std([r['best_val_acc'] for r in all_results])
    
    print(f"\nOverall Performance:")
    print(f"  Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"  Test Accuracy:       {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    
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
            'mean_val_accuracy': float(mean_val_acc),
            'std_val_accuracy': float(std_val_acc),
            'mean_test_accuracy': float(mean_test_acc),
            'std_test_accuracy': float(std_test_acc),
            'folds': json_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    plot_cross_validation_results(all_results)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


# ============================================================================
# SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()





