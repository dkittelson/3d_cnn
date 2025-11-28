"""
ENTROPY FILTERING ANALYSIS
===========================
Analyzes how many cells would be filtered out using TPSF entropy thresholding.
Also checks if cell sizes differ between donors (D4/D5 vs others).

Based on the paper's approach:
1. Compute Shannon entropy at peak photon frame
2. Fit Gaussian to entropy distribution
3. Filter cells outside ±2σ (or adjustable threshold)
"""

import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def compute_entropy(array_2d):
    """
    Compute Shannon entropy of a 2D photon distribution.
    
    Args:
        array_2d: (H, W) spatial photon distribution
    
    Returns:
        entropy: Shannon entropy in bits
    """
    # Normalize to probability distribution
    total = array_2d.sum()
    if total == 0:
        return 0.0
    
    p = array_2d / total
    p = p[p > 0]  # Only non-zero bins
    
    entropy = -np.sum(p * np.log2(p))
    return entropy


def compute_cell_size(array_4d):
    """
    Compute effective cell size (number of non-zero spatial pixels).
    
    Args:
        array_4d: (H, W, T) or (21, 21, 256) TPSF image
    
    Returns:
        cell_size: Number of pixels with photons
    """
    spatial_sum = array_4d.sum(axis=2)  # (H, W)
    cell_size = (spatial_sum > 0).sum()
    return cell_size


def load_and_analyze_cell(file_path):
    """
    Load cell file and compute entropy + size metrics.
    
    Returns:
        dict with entropy, cell_size, peak_frame, total_photons
    """
    array = np.load(file_path)
    
    # Handle shape variations
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    elif array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    
    # Ensure (H, W, T) format
    if array.shape[2] < array.shape[0]:
        array = np.transpose(array, (1, 2, 0))
    
    # Find peak photon frame
    temporal_sum = array.sum(axis=(0, 1))  # Sum over space
    peak_frame = temporal_sum.argmax()
    
    # Get spatial distribution at peak
    peak_spatial = array[:, :, peak_frame]
    
    # Compute metrics
    entropy = compute_entropy(peak_spatial)
    cell_size = compute_cell_size(array)
    total_photons = array.sum()
    
    return {
        'entropy': entropy,
        'cell_size': cell_size,
        'peak_frame': peak_frame,
        'total_photons': total_photons,
        'peak_photons': peak_spatial.sum()
    }


def analyze_folder(folder_path, max_files=None):
    """
    Analyze all cells in a folder.
    
    Args:
        folder_path: Path to isolated_cells_DX folder
        max_files: Limit number of files (None = all)
    
    Returns:
        list of dicts with metrics per cell
    """
    files = sorted(Path(folder_path).glob("*.npy"))
    
    if max_files:
        files = files[:max_files]
    
    results = []
    for file in tqdm(files, desc=f"Processing {folder_path.name}"):
        try:
            metrics = load_and_analyze_cell(file)
            metrics['file'] = file.name
            
            # Extract activity label from filename
            filename = file.name.lower()
            if '_act_cell_' in filename or '_active_' in filename:
                metrics['label'] = 1  # Active
            elif '_in_cell_' in filename or '_inactive_' in filename:
                metrics['label'] = 0  # Inactive
            else:
                metrics['label'] = -1  # Unknown
            
            results.append(metrics)
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    return results


def main():
    print("="*80)
    print("ENTROPY FILTERING & CELL SIZE ANALYSIS")
    print("="*80)
    
    # Data directory
    data_dir = Path("dataset")
    
    # Collect data from all donors
    all_data = defaultdict(list)
    
    for donor in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        folder = data_dir / f"isolated_cells_{donor}"
        
        if not folder.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue
        
        print(f"\n📂 Analyzing {donor}...")
        results = analyze_folder(folder, max_files=None)  # Analyze all files
        all_data[donor] = results
        
        # Quick summary
        entropies = [r['entropy'] for r in results]
        sizes = [r['cell_size'] for r in results]
        active_count = sum(1 for r in results if r['label'] == 1)
        inactive_count = sum(1 for r in results if r['label'] == 0)
        
        print(f"   Total cells: {len(results)}")
        print(f"   Active: {active_count}, Inactive: {inactive_count}")
        print(f"   Entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
        print(f"   Cell size: {np.mean(sizes):.1f} ± {np.std(sizes):.1f} pixels")
    
    # ========================================================================
    # ENTROPY FILTERING ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("ENTROPY FILTERING ANALYSIS")
    print("="*80)
    
    # Combine all entropies
    all_entropies = []
    all_donors = []
    all_labels = []
    
    for donor, results in all_data.items():
        for r in results:
            all_entropies.append(r['entropy'])
            all_donors.append(donor)
            all_labels.append(r['label'])
    
    all_entropies = np.array(all_entropies)
    
    # Fit Gaussian to entropy distribution
    mu, sigma = norm.fit(all_entropies)
    
    print(f"\n📊 Global Entropy Distribution:")
    print(f"   Mean: {mu:.3f}")
    print(f"   Std:  {sigma:.3f}")
    
    # Test different threshold levels
    for threshold_sigma in [1.5, 2.0, 2.5, 3.0]:
        lower = mu - threshold_sigma * sigma
        upper = mu + threshold_sigma * sigma
        
        valid_mask = (all_entropies >= lower) & (all_entropies <= upper)
        filtered_count = len(all_entropies) - valid_mask.sum()
        filtered_pct = 100 * filtered_count / len(all_entropies)
        
        print(f"\n   Threshold: ±{threshold_sigma}σ [{lower:.3f}, {upper:.3f}]")
        print(f"   → Would filter: {filtered_count}/{len(all_entropies)} cells ({filtered_pct:.1f}%)")
        print(f"   → Remaining: {valid_mask.sum()} cells")
        
        # Breakdown by donor
        print(f"   Filtered per donor:")
        for donor in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
            donor_mask = np.array(all_donors) == donor
            donor_filtered = donor_mask & ~valid_mask
            if donor_mask.sum() > 0:
                donor_pct = 100 * donor_filtered.sum() / donor_mask.sum()
                print(f"      {donor}: {donor_filtered.sum()}/{donor_mask.sum()} ({donor_pct:.1f}%)")
    
    # ========================================================================
    # CELL SIZE ANALYSIS (D4/D5 vs others)
    # ========================================================================
    print("\n" + "="*80)
    print("CELL SIZE ANALYSIS: D4/D5 vs D1/D2/D3/D6")
    print("="*80)
    
    large_donors = ['D4', 'D5']
    small_donors = ['D1', 'D2', 'D3', 'D6']
    
    large_sizes = []
    small_sizes = []
    
    for donor, results in all_data.items():
        sizes = [r['cell_size'] for r in results]
        
        if donor in large_donors:
            large_sizes.extend(sizes)
        else:
            small_sizes.extend(sizes)
    
    large_sizes = np.array(large_sizes)
    small_sizes = np.array(small_sizes)
    
    print(f"\n📏 Cell Size Distribution:")
    print(f"   LARGE donors (D4, D5):")
    print(f"      Mean: {large_sizes.mean():.1f} ± {large_sizes.std():.1f} pixels")
    print(f"      Median: {np.median(large_sizes):.1f}")
    print(f"      Range: [{large_sizes.min()}, {large_sizes.max()}]")
    print(f"      Total cells: {len(large_sizes)}")
    
    print(f"\n   SMALL donors (D1, D2, D3, D6):")
    print(f"      Mean: {small_sizes.mean():.1f} ± {small_sizes.std():.1f} pixels")
    print(f"      Median: {np.median(small_sizes):.1f}")
    print(f"      Range: [{small_sizes.min()}, {small_sizes.max()}]")
    print(f"      Total cells: {len(small_sizes)}")
    
    size_ratio = large_sizes.mean() / small_sizes.mean()
    print(f"\n   📊 SIZE RATIO: D4/D5 cells are {size_ratio:.2f}x larger than D1/D2/D3/D6")
    
    # Statistical test
    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(large_sizes, small_sizes, alternative='two-sided')
    print(f"   Mann-Whitney U test: p={pval:.2e} {'✓ SIGNIFICANT' if pval < 0.001 else ''}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING PLOTS...")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Entropy distribution by donor
    ax = axes[0, 0]
    for donor in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        if donor in all_data:
            entropies = [r['entropy'] for r in all_data[donor]]
            ax.hist(entropies, bins=50, alpha=0.5, label=donor)
    
    # Add Gaussian fit
    x = np.linspace(all_entropies.min(), all_entropies.max(), 100)
    ax.plot(x, len(all_entropies) * norm.pdf(x, mu, sigma) * (all_entropies.max() - all_entropies.min()) / 50, 
            'k--', linewidth=2, label='Gaussian fit')
    
    # Add threshold lines
    threshold = 2.0
    ax.axvline(mu - threshold*sigma, color='red', linestyle='--', label=f'±{threshold}σ')
    ax.axvline(mu + threshold*sigma, color='red', linestyle='--')
    
    ax.set_xlabel('Entropy (bits)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Entropy Distribution by Donor', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cell size distribution (D4/D5 vs others)
    ax = axes[0, 1]
    ax.hist(small_sizes, bins=50, alpha=0.6, label='D1/D2/D3/D6 (small)', color='blue')
    ax.hist(large_sizes, bins=50, alpha=0.6, label='D4/D5 (large)', color='red')
    ax.axvline(small_sizes.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean small: {small_sizes.mean():.1f}')
    ax.axvline(large_sizes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean large: {large_sizes.mean():.1f}')
    ax.set_xlabel('Cell Size (pixels)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Cell Size: Large cells {size_ratio:.2f}x bigger', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cell size by donor
    ax = axes[1, 0]
    donor_names = []
    donor_sizes = []
    colors = []
    for donor in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        if donor in all_data:
            sizes = [r['cell_size'] for r in all_data[donor]]
            donor_names.append(donor)
            donor_sizes.append(sizes)
            colors.append('red' if donor in large_donors else 'blue')
    
    bp = ax.boxplot(donor_sizes, labels=donor_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Cell Size (pixels)', fontsize=12)
    ax.set_title('Cell Size by Donor', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Entropy vs Cell Size (scatter)
    ax = axes[1, 1]
    for donor in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        if donor in all_data:
            entropies = [r['entropy'] for r in all_data[donor]]
            sizes = [r['cell_size'] for r in all_data[donor]]
            color = 'red' if donor in large_donors else 'blue'
            ax.scatter(sizes, entropies, alpha=0.3, s=10, label=donor, color=color)
    
    ax.set_xlabel('Cell Size (pixels)', fontsize=12)
    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title('Entropy vs Cell Size', fontsize=14, fontweight='bold')
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entropy_and_size_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: entropy_and_size_analysis.png")
    
    # ========================================================================
    # SAVE DETAILED RESULTS
    # ========================================================================
    import json
    
    summary = {
        'entropy_stats': {
            'mean': float(mu),
            'std': float(sigma),
            'total_cells': len(all_entropies)
        },
        'filtering_results': {},
        'cell_size_stats': {
            'large_donors': {
                'donors': large_donors,
                'mean': float(large_sizes.mean()),
                'std': float(large_sizes.std()),
                'count': len(large_sizes)
            },
            'small_donors': {
                'donors': small_donors,
                'mean': float(small_sizes.mean()),
                'std': float(small_sizes.std()),
                'count': len(small_sizes)
            },
            'size_ratio': float(size_ratio),
            'mann_whitney_p': float(pval)
        },
        'per_donor_stats': {}
    }
    
    # Add filtering results
    for threshold_sigma in [1.5, 2.0, 2.5, 3.0]:
        lower = mu - threshold_sigma * sigma
        upper = mu + threshold_sigma * sigma
        valid_mask = (all_entropies >= lower) & (all_entropies <= upper)
        
        summary['filtering_results'][f'{threshold_sigma}sigma'] = {
            'threshold_range': [float(lower), float(upper)],
            'filtered_count': int(len(all_entropies) - valid_mask.sum()),
            'filtered_percent': float(100 * (len(all_entropies) - valid_mask.sum()) / len(all_entropies)),
            'remaining_count': int(valid_mask.sum())
        }
    
    # Add per-donor stats
    for donor, results in all_data.items():
        entropies = [r['entropy'] for r in results]
        sizes = [r['cell_size'] for r in results]
        active_count = sum(1 for r in results if r['label'] == 1)
        
        summary['per_donor_stats'][donor] = {
            'total_cells': len(results),
            'active_cells': active_count,
            'inactive_cells': len(results) - active_count,
            'entropy_mean': float(np.mean(entropies)),
            'entropy_std': float(np.std(entropies)),
            'cell_size_mean': float(np.mean(sizes)),
            'cell_size_std': float(np.std(sizes))
        }
    
    with open('entropy_analysis_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Detailed results saved to: entropy_analysis_results.json")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
