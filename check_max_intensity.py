"""
Verify the true maximum log-intensity value in the dataset.
This checks if GLOBAL_INTENSITY_MAX = 8.2212 is accurate or if we're clipping bright cells.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_true_max():
    """Find the actual maximum log-intensity across all datasets"""
    
    data_dir = Path("data")
    
    max_val = 0
    max_file = None
    
    dataset_names = [f'isolated_cells_D{i}' for i in range(1, 7)]
    
    all_files = []
    for dataset_name in dataset_names:
        dataset_path = data_dir / dataset_name
        if dataset_path.exists():
            all_files.extend(list(dataset_path.glob('*.npy')))
    
    print(f"Scanning {len(all_files)} files for true maximum log-intensity...")
    print("="*80)
    
    for file_path in tqdm(all_files, desc="Scanning"):
        arr = np.load(file_path)
        
        # Calculate log-intensity exactly as training code does
        intensity = arr.sum(axis=2)  # Sum over time dimension
        log_intensity = np.log1p(intensity)  # log(1 + intensity)
        
        curr_max = log_intensity.max()
        
        if curr_max > max_val:
            max_val = curr_max
            max_file = file_path
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"TRUE DATASET MAX:     {max_val:.4f}")
    print(f"Current GLOBAL_MAX:   8.2212")
    print(f"File with max value:  {max_file.name if max_file else 'None'}")
    print()
    
    if max_val > 8.2212:
        diff = max_val - 8.2212
        percent_higher = (diff / 8.2212) * 100
        print(f"⚠️  CRITICAL: You are clipping {percent_higher:.1f}% of the range!")
        print(f"   Update GLOBAL_INTENSITY_MAX to {max_val:.4f}")
        print(f"   This could be causing your {202} false negatives (missed active cells)")
    else:
        print("✅  Your GLOBAL_MAX is safe - no clipping occurring")
    
    print("="*80)
    
    return max_val

if __name__ == "__main__":
    true_max = check_true_max()
