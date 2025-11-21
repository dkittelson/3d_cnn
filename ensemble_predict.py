"""
Ensemble Prediction Script
===========================
Loads multiple trained models and averages their predictions.
Uses Test-Time Augmentation (TTA) and K-Means threshold tuning.

Expected: Ensemble accuracy ~85% (vs individual models ~82%)
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.cluster import KMeans
from tqdm import tqdm
import sys

# Import from train_cv
sys.path.append(str(Path(__file__).parent))
from train_cv import CellDataset, collect_data_by_folder
from resnet import create_model


def predict_tta(model, dataset, device, batch_size=128):
    """
    Test-Time Augmentation: Apply 4 augmentations and average predictions
    """
    from torch.utils.data import DataLoader

    model.eval()
    all_probs = []

    augmentations = [
        ('original', lambda x: x),
        ('rot90', lambda x: torch.rot90(x, k=1, dims=[3, 4])),
        ('fliph', lambda x: torch.flip(x, dims=[3])),
        ('flipv', lambda x: torch.flip(x, dims=[4]))
    ]

    with torch.no_grad():
        for aug_name, aug_fn in tqdm(augmentations, desc='  TTA', leave=False):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True)
            probs_list = []

            for inputs, _ in loader:
                inputs = inputs.to(device)
                inputs = aug_fn(inputs)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                probs_list.append(probs)

            all_probs.append(np.concatenate(probs_list))# Average across augmentations
    avg_probs = np.mean(all_probs, axis=0).squeeze()
    return avg_probs


def tune_threshold_kmeans(models, val_dataset, device, batch_size=128):
    """
    Find optimal threshold using K-Means clustering on ensemble predictions
    """
    print("\n🔍 Tuning decision threshold with K-Means...")
    
    # Get ensemble predictions on validation set
    ensemble_probs = []
    for i, model_path in enumerate(models):
        print(f"  Loading model {i+1}/{len(models)}...")
        model = create_model(in_channels=2).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        probs = predict_tta(model, val_dataset, device, batch_size)
        ensemble_probs.append(probs)
    
    # Average predictions across models
    avg_probs = np.mean(ensemble_probs, axis=0)
    
    # K-Means clustering to find threshold
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(avg_probs.reshape(-1, 1))
    
    # Cluster centers (inactive and active)
    centers = sorted(kmeans.cluster_centers_.flatten())
    optimal_threshold = np.mean(centers)
    
    print(f"  Inactive cluster center: {centers[0]:.4f}")
    print(f"  Active cluster center:   {centers[1]:.4f}")
    print(f"  ✓ Optimal threshold:     {optimal_threshold:.4f}")
    
    return optimal_threshold


def test_ensemble(model_paths, test_dataset, test_labels, device, threshold=0.5, batch_size=128):
    """
    Test ensemble by averaging predictions from multiple models
    """
    print(f"\n{'='*80}")
    print("TESTING ENSEMBLE")
    print(f"{'='*80}")
    print(f"Number of models: {len(model_paths)}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Collect predictions from each model
    all_model_probs = []

    for i, model_path in enumerate(model_paths):
        model_id = chr(65 + i)  # A, B, C, ...
        print(f"\n🔄 Model {model_id}: {model_path.name}")
        
        # Load model
        model = create_model(in_channels=2).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get predictions with TTA
        print(f"   Running predictions with TTA...")
        probs = predict_tta(model, test_dataset, device, batch_size)
        all_model_probs.append(probs)
        
        # Individual model accuracy
        preds = (probs >= threshold).astype(int)
        acc = np.mean(preds == test_labels)
        print(f"  Individual accuracy: {acc:.4f}")
    
    # Ensemble: Average predictions
    ensemble_probs = np.mean(all_model_probs, axis=0)
    ensemble_preds = (ensemble_probs >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(ensemble_preds == test_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, ensemble_preds, average='binary', zero_division=0
    )
    
    cm = confusion_matrix(test_labels, ensemble_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print(f"\n{'='*80}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*80}")
    print(f"\n📊 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"  True Negatives:  {tn:4d}")
    print(f"  False Positives: {fp:4d}")
    print(f"  False Negatives: {fn:4d}")
    print(f"  True Positives:  {tp:4d}")
    
    print(f"\n🎯 Classification Report:")
    inactive_total = tn + fp
    active_total = fn + tp
    print(f"  Inactive (0): {tn}/{inactive_total} correct ({tn/inactive_total*100:.1f}%)")
    print(f"  Active (1):   {tp}/{active_total} correct ({tp/active_total*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find ensemble models
    model_dir = Path('saved_models')
    model_paths = sorted(model_dir.glob('ensemble_model_*.pth'))
    
    if len(model_paths) == 0:
        print("\n❌ No ensemble models found!")
        print("   Expected files: saved_models/ensemble_model_A.pth, ensemble_model_B.pth, ...")
        print("   Run train_cv.py with ENABLE_ENSEMBLE=True first")
        return
    
    print(f"\n✓ Found {len(model_paths)} ensemble models:")
    for p in model_paths:
        print(f"  - {p}")
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    dataset_dict = collect_data_by_folder()
    d5_files, d5_labels = dataset_dict['isolated_cells_D5']
    d6_files, d6_labels = dataset_dict['isolated_cells_D6']
    
    # Create datasets
    val_dataset = CellDataset(d6_files, d6_labels, augment=False)
    test_dataset = CellDataset(d5_files, d5_labels, augment=False)
    
    print(f"Validation (D6): {len(val_dataset)} samples")
    print(f"Test (D5): {len(test_dataset)} samples")
    
    # Tune threshold on validation set
    optimal_threshold = tune_threshold_kmeans(model_paths, val_dataset, device)
    
    # Test ensemble
    results = test_ensemble(model_paths, test_dataset, d5_labels, device, optimal_threshold)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
