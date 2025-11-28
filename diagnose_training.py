"""
Quick diagnostic to understand why model predicts everything as active
"""
import torch
import numpy as np
from pathlib import Path
from train_cv import CellDataset, collect_data_by_folder

print("="*70)
print("DIAGNOSTIC: Input Data & Model Behavior")
print("="*70)

# Load a few samples
data_dict = collect_data_by_folder()
train_files = data_dict['isolated_cells_D1'][:10]  # First 10 samples

dataset = CellDataset(train_files, augment=False)

print(f"\nLoaded {len(dataset)} samples")
print("\nAnalyzing first 5 samples:")

for i in range(min(5, len(dataset))):
    X, y = dataset[i]
    
    flim_channel = X[0]  # (256, 21, 21)
    mask_channel = X[1]  # (256, 21, 21)
    
    print(f"\nSample {i} (Label: {'Active' if y == 1 else 'Inactive'}):")
    print(f"  Input shape: {X.shape}")
    print(f"  FLIM channel (0):")
    print(f"    - Min: {flim_channel.min():.6f}")
    print(f"    - Max: {flim_channel.max():.6f}")
    print(f"    - Mean (non-zero): {flim_channel[flim_channel > 0].mean():.6f}")
    print(f"    - % zeros: {(flim_channel == 0).float().mean()*100:.1f}%")
    
    print(f"  Mask channel (1):")
    print(f"    - Unique values: {mask_channel.unique()}")
    print(f"    - % ones: {(mask_channel == 1).float().mean()*100:.1f}%")
    print(f"    - % zeros: {(mask_channel == 0).float().mean()*100:.1f}%")

# Check model forward pass
print("\n" + "="*70)
print("TESTING MODEL FORWARD PASS")
print("="*70)

from resnet import create_model

model = create_model(in_channels=2, dropout_rate=0.25)
model.eval()

# Create mini batch
batch_X = torch.stack([dataset[i][0] for i in range(4)])
batch_y = torch.stack([dataset[i][1] for i in range(4)])

print(f"\nBatch input shape: {batch_X.shape}")
print(f"Batch labels: {batch_y}")

with torch.no_grad():
    logits = model(batch_X)
    probs = torch.sigmoid(logits)
    
print(f"\nModel outputs:")
print(f"  Logits: {logits.squeeze()}")
print(f"  Probabilities: {probs.squeeze()}")
print(f"  Predictions (>0.5): {(probs > 0.5).long().squeeze()}")

print("\n" + "="*70)
print("CHECKING FOR ISSUES:")
print("="*70)

issues = []

# Check 1: Are inputs reasonable?
if batch_X.isnan().any():
    issues.append("❌ NaN values in input!")
if batch_X.isinf().any():
    issues.append("❌ Inf values in input!")
if (batch_X == 0).float().mean() > 0.95:
    issues.append(f"⚠️  Input is {(batch_X == 0).float().mean()*100:.1f}% zeros (too sparse?)")

# Check 2: Are model weights initialized?
first_layer_weight = model.temporal_compress[0].weight
if first_layer_weight.std() < 0.001:
    issues.append("❌ Model weights not properly initialized (too small variance)")

# Check 3: Are outputs reasonable?
if logits.std() < 0.1:
    issues.append(f"⚠️  Model outputs have low variance (std={logits.std():.4f})")
if probs.mean() > 0.9 or probs.mean() < 0.1:
    issues.append(f"⚠️  Model is biased (mean prob={probs.mean():.3f})")

if not issues:
    print("✅ No obvious issues detected")
else:
    for issue in issues:
        print(issue)

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)

if (batch_X == 0).float().mean() > 0.9:
    print("⚠️  Input is very sparse - model may struggle to learn features")
    print("   Consider: Using more aggressive augmentation or different normalization")

if probs.mean() > 0.9:
    print("⚠️  Model predicts mostly 'active' - check:")
    print("   1. Label distribution in training data")
    print("   2. Loss function (pos_weight may be too high)")
    print("   3. WeightedRandomSampler configuration")
