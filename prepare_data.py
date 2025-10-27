import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# --- Custom Dataset class for lazy loading --- #
class CellDataset(Dataset):
    """Custom Dataset that loads data on-the-fly instead of loading all into memory"""
    
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the file
        array = np.load(self.file_paths[idx])
        
        # Standardize
        max_val = np.max(array)
        if max_val > 0:
            array = array / max_val
        
        # Add channel dimension at end of array
        array = np.expand_dims(array, axis=-1)
        
        # Convert to tensor
        X = torch.tensor(array, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return X, y


# --- Collect all files and labels --- #
data_dir = Path('data')
all_files = []
all_labels = []

for i in range(1, 7):
    folder = data_dir / f'isolated_cells_D{i}'
    if folder.exists():
        files = list(folder.glob('*.npy'))
        
        for file in files:
            # Find label
            if '_in' in file.name:
                label = 0
            elif '_act' in file.name:
                label = 1
            else:
                continue
            
            all_files.append(file)
            all_labels.append(label)

print(f"Total files found: {len(all_files)}")
print(f"Labels: inactive={all_labels.count(0)}, active={all_labels.count(1)}")

# Split data (only file paths and labels, not the actual data)
files_train, files_test, labels_train, labels_test = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

print(f"Training samples: {len(files_train)}")
print(f"Test samples: {len(files_test)}")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create PyTorch datasets (these don't load data yet, just store file paths)
train_dataset = CellDataset(files_train, labels_train)
test_dataset = CellDataset(files_test, labels_test)

# Create DataLoaders (data will be loaded in batches during training)
BATCH_SIZE = 16
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
    pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

print("\nData preparation complete! DataLoaders ready for training.")







