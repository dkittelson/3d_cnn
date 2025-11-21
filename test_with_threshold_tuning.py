"""
Quick script to test saved model with K-Means threshold tuning
No retraining needed - just loads best model and optimizes threshold
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import sys
import os

# Add current directory to path (works in both script and notebook)
if '__file__' in globals():
    sys.path.append(str(Path(__file__).parent))
else:
    # Running in notebook/interactive environment
    sys.path.append(os.getcwd())

from resnet import create_model
from train_cv import CellDataset, collect_data_by_folder

def predict_tta(model, inputs, device):
    """Apply test-time augmentation and average predictions"""
    model.eval()
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(inputs))
        predictions.append(pred)
    
    # Rotate 90°
    with torch.no_grad():
        rotated = torch.rot90(inputs, k=1, dims=[3, 4])
        pred = torch.sigmoid(model(rotated))
        predictions.append(pred)
    
    # Flip Horizontal
    with torch.no_grad():
        flipped_h = torch.flip(inputs, dims=[4])
        pred = torch.sigmoid(model(flipped_h))
        predictions.append(pred)
    
    # Flip Vertical
    with torch.no_grad():
        flipped_v = torch.flip(inputs, dims=[3])
        pred = torch.sigmoid(model(flipped_v))
        predictions.append(pred)
    
    # Average all predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred

def tune_threshold_kmeans(model, val_dataset, device, batch_size=32):
    """Find optimal threshold using K-Means clustering"""
    print("\n🔧 Tuning decision threshold with K-Means...")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc='Collecting val probabilities'):
            inputs = inputs.to(device)
            probs = predict_tta(model, inputs, device)
            all_probs.extend(probs.cpu().numpy().flatten())
    
    # Reshape for K-Means
    probs_array = np.array(all_probs).reshape(-1, 1)
    
    # K-Means with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(probs_array)
    
    # Get cluster centers
    centers = sorted(kmeans.cluster_centers_.flatten())
    inactive_center = centers[0]
    active_center = centers[1]
    
    # Optimal threshold = midpoint
    optimal_threshold = (inactive_center + active_center) / 2.0
    
    print(f"  Inactive cluster center: {inactive_center:.4f}")
    print(f"  Active cluster center:   {active_center:.4f}")
    print(f"  ✓ Optimal threshold:     {optimal_threshold:.4f}")
    
    return optimal_threshold

def test_with_threshold(model, test_dataset, device, threshold, batch_size=64):
    """Test model with specific threshold"""
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print(f"\nTesting with threshold={threshold:.4f}...")
    for inputs, labels in tqdm(test_loader, desc='Testing (TTA)'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        probs = predict_tta(model, inputs, device)
        predicted = (probs > threshold).float()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    return acc, precision, recall, f1, cm, all_probs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading datasets...")
    dataset_dict = collect_data_by_folder()
    
    # Get D5 (test) and D6 (validation for tuning)
    d5_files, d5_labels = dataset_dict['isolated_cells_D5']
    d6_files, d6_labels = dataset_dict['isolated_cells_D6']
    
    print(f"D5 (Test): {len(d5_files)} samples")
    print(f"D6 (Val):  {len(d6_files)} samples")
    
    # Create datasets
    val_dataset = CellDataset(d6_files, d6_labels, augment=False)
    test_dataset = CellDataset(d5_files, d5_labels, augment=False)
    
    # Load trained model
    print("\nLoading trained model...")
    model = create_model(in_channels=2).to(device)
    checkpoint = torch.load('saved_models/best_model_fold_D5_FINAL_TEST.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}" if 'val_f1' in checkpoint else "")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}" if 'val_acc' in checkpoint else "")
    
    # Tune threshold using K-Means on validation set
    optimal_threshold = tune_threshold_kmeans(model, val_dataset, device)
    
    # Compare results with different thresholds
    print("\n" + "="*80)
    print("COMPARING THRESHOLDS")
    print("="*80)
    
    thresholds_to_test = [0.5, optimal_threshold]
    results = {}
    
    for thresh in thresholds_to_test:
        print(f"\n--- Testing with threshold={thresh:.4f} ---")
        acc, prec, rec, f1, cm, probs = test_with_threshold(
            model, test_dataset, device, thresh
        )
        
        results[thresh] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cm': cm
        }
        
        print(f"\nResults:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: THRESHOLD COMPARISON")
    print("="*80)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for thresh in thresholds_to_test:
        r = results[thresh]
        print(f"{thresh:<12.4f} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
              f"{r['recall']:<12.4f} {r['f1']:<12.4f}")
    
    # Show improvement
    baseline = results[0.5]
    optimized = results[optimal_threshold]
    
    print("\n" + "="*80)
    print("IMPROVEMENT FROM K-MEANS THRESHOLD TUNING")
    print("="*80)
    acc_improvement = (optimized['accuracy'] - baseline['accuracy']) * 100
    f1_improvement = (optimized['f1'] - baseline['f1']) * 100
    
    print(f"Accuracy:  {baseline['accuracy']:.4f} → {optimized['accuracy']:.4f} "
          f"({acc_improvement:+.2f}%)")
    print(f"F1-Score:  {baseline['f1']:.4f} → {optimized['f1']:.4f} "
          f"({f1_improvement:+.2f}%)")
    print(f"Precision: {baseline['precision']:.4f} → {optimized['precision']:.4f}")
    print(f"Recall:    {baseline['recall']:.4f} → {optimized['recall']:.4f}")
    
    # Show FP/FN changes
    baseline_fp = baseline['cm'][0,1]
    baseline_fn = baseline['cm'][1,0]
    optimized_fp = optimized['cm'][0,1]
    optimized_fn = optimized['cm'][1,0]
    
    print(f"\nFalse Positives: {baseline_fp} → {optimized_fp} ({optimized_fp - baseline_fp:+d})")
    print(f"False Negatives: {baseline_fn} → {optimized_fn} ({optimized_fn - baseline_fn:+d})")
    
    if acc_improvement > 0:
        print(f"\n✅ K-Means threshold tuning improved accuracy by {acc_improvement:.2f}%!")
    else:
        print(f"\n⚠️  Threshold 0.5 was already optimal for this dataset")

if __name__ == "__main__":
    main()
