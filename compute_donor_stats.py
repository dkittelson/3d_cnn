"""
Compute per-donor intensity statistics for normalization

This script calculates the mean and std of log-intensity for each donor (D1-D6).
These statistics are used to standardize intensity across donors, removing
systematic brightness differences.

Usage:
    python compute_donor_stats.py

Output:
    Prints DONOR_INTENSITY_STATS dictionary to copy into train_cv.py
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_donor_statistics(data_dir="/content/3d_cnn/3d_cnn/data"):
    """Compute mean and std of log-intensity for each donor"""
    
    data_dir = Path(data_dir)
    donor_stats = {}
    
    for donor_id in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        folder = data_dir / f'isolated_cells_{donor_id}'
        
        if not folder.exists():
            print(f"⚠️  {folder} not found, skipping...")
            continue
        
        files = list(folder.glob('*.npy'))
        if not files:
            print(f"⚠️  No .npy files in {folder}, skipping...")
            continue
        
        print(f"\n📊 Processing {donor_id}: {len(files)} files")
        
        log_intensities = []
        
        for file in tqdm(files[:500], desc=f"  Loading {donor_id}"):  # Sample 500 files per donor
            try:
                array = np.load(file)
                intensity_map = array.sum(axis=2)  # Total photons per pixel
                log_intensity = np.log1p(intensity_map)  # Log transform
                log_intensities.append(log_intensity.flatten())
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")
                continue
        
        if log_intensities:
            all_log_intensities = np.concatenate(log_intensities)
            mean = np.mean(all_log_intensities)
            std = np.std(all_log_intensities)
            
            donor_stats[donor_id] = {
                'mean': float(mean),
                'std': float(std),
                'n_files': len(log_intensities),
                'n_pixels': len(all_log_intensities)
            }
            
            print(f"  ✓ {donor_id}: mean={mean:.3f}, std={std:.3f}")
    
    return donor_stats

if __name__ == "__main__":
    print("="*80)
    print("COMPUTING PER-DONOR INTENSITY STATISTICS")
    print("="*80)
    
    stats = compute_donor_statistics()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nCopy this into train_cv.py:\n")
    print("DONOR_INTENSITY_STATS = {")
    for donor_id in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        if donor_id in stats:
            s = stats[donor_id]
            print(f"    '{donor_id}': {{'mean': {s['mean']:.2f}, 'std': {s['std']:.2f}}},  # {s['n_files']} files")
    print("}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if stats:
        means = [s['mean'] for s in stats.values()]
        stds = [s['std'] for s in stats.values()]
        
        print(f"\nMean log-intensity:")
        print(f"  Average: {np.mean(means):.3f}")
        print(f"  Range: [{np.min(means):.3f}, {np.max(means):.3f}]")
        print(f"  Variation: {(np.max(means) - np.min(means)) / np.mean(means) * 100:.1f}%")
        
        print(f"\nStd log-intensity:")
        print(f"  Average: {np.mean(stds):.3f}")
        print(f"  Range: [{np.min(stds):.3f}, {np.max(stds):.3f}]")
        
        # Find outliers
        mean_avg = np.mean(means)
        for donor_id, s in stats.items():
            diff_pct = (s['mean'] - mean_avg) / mean_avg * 100
            if abs(diff_pct) > 10:
                print(f"\n⚠️  {donor_id} is {'brighter' if diff_pct > 0 else 'dimmer'} than average by {abs(diff_pct):.1f}%")
                print(f"   This explains why model struggles on {donor_id}!")
