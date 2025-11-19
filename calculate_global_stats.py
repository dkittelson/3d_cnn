"""
Calculate global intensity statistics across all datasets.
Run this once to get GLOBAL_MIN and GLOBAL_MAX for proper normalization.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_global_intensity_stats():
    """Calculate min/max log-intensity across all cells"""
    
    data_dir = Path("data")
    
    all_log_intensities = []
    
    dataset_names = [f'isolated_cells_D{i}' for i in range(1, 7)]
    
    print("Scanning all datasets for intensity statistics...")
    print("="*80)
    
    for dataset_name in dataset_names:
        dataset_path = data_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"⚠️  {dataset_name} not found, skipping...")
            continue
        
        files = list(dataset_path.glob('*.npy'))
        print(f"\n{dataset_name}: {len(files)} files")
        
        for file_path in tqdm(files, desc=f"  Processing {dataset_name}"):
            # Load cell data
            array = np.load(file_path)  # Shape: (21, 21, 256)
            
            # Calculate total photons per pixel
            intensity_map = array.sum(axis=2)  # Shape: (21, 21)
            
            # Log transform
            log_intensity = np.log1p(intensity_map)
            
            # Store all values
            all_log_intensities.extend(log_intensity.flatten())
    
    # Convert to array
    all_log_intensities = np.array(all_log_intensities)
    
    # Calculate statistics
    global_min = all_log_intensities.min()
    global_max = all_log_intensities.max()
    global_mean = all_log_intensities.mean()
    global_std = all_log_intensities.std()
    
    # Percentiles for robustness
    p1 = np.percentile(all_log_intensities, 1)
    p99 = np.percentile(all_log_intensities, 99)
    
    print("\n" + "="*80)
    print("GLOBAL INTENSITY STATISTICS (log-scale)")
    print("="*80)
    print(f"Min:        {global_min:.4f}")
    print(f"Max:        {global_max:.4f}")
    print(f"Mean:       {global_mean:.4f}")
    print(f"Std:        {global_std:.4f}")
    print(f"1st %ile:   {p1:.4f}")
    print(f"99th %ile:  {p99:.4f}")
    print(f"\nTotal pixels analyzed: {len(all_log_intensities):,}")
    print(f"Total cells analyzed:  {len(all_log_intensities) // (21*21):,}")
    print("="*80)
    
    print("\n💡 Recommendation:")
    print(f"   Use GLOBAL_MIN = {global_min:.4f}")
    print(f"   Use GLOBAL_MAX = {p99:.4f}  # Using 99th percentile for robustness")
    print("\n   This preserves relative brightness differences between cells!")
    
    return {
        'min': global_min,
        'max': global_max,
        'mean': global_mean,
        'std': global_std,
        'p1': p1,
        'p99': p99
    }

if __name__ == "__main__":
    stats = calculate_global_intensity_stats()
