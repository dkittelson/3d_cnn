"""
BRIGHTNESS BIAS ANALYSIS
========================
Analyzes photon intensity differences between donors to detect hidden biases.

The model might be learning:
- "Bright cells = active" (if D4/D5 are brighter + more active)
- "Dim cells = inactive" (if D1/D2/D3/D6 are dimmer + less active)

This script computes multiple brightness metrics per donor:
1. Total photon count (sum of all photons)
2. Peak intensity (maximum value in decay curve)
3. Mean intensity (average across space-time)
4. SNR (signal-to-noise ratio)
5. Active vs Inactive brightness difference WITHIN each donor
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp

def compute_brightness_metrics(array):
    """
    Compute comprehensive brightness metrics for a TPSF image.
    
    Args:
        array: (H, W, T) or (21, 21, 256) TPSF image
    
    Returns:
        dict with multiple brightness metrics
    """
    # Total photon count
    total_photons = array.sum()
    
    # Peak intensity (brightest pixel-timepoint)
    peak_intensity = array.max()
    
    # Mean intensity (excluding zeros)
    nonzero = array[array > 0]
    mean_intensity = nonzero.mean() if len(nonzero) > 0 else 0
    
    # Temporal decay curve (sum over space)
    decay_curve = array.sum(axis=(0, 1))
    peak_time = decay_curve.argmax()
    peak_photons = decay_curve[peak_time]
    
    # SNR (peak / background noise)
    # Background = mean of lowest 10% of spatial pixels at peak time
    peak_spatial = array[:, :, peak_time]
    background = np.percentile(peak_spatial[peak_spatial > 0], 10) if (peak_spatial > 0).sum() > 0 else 1
    snr = peak_photons / (background + 1e-10)
    
    # Spatial extent (number of active pixels)
    spatial_sum = array.sum(axis=2)
    active_pixels = (spatial_sum > 0).sum()
    
    # Density (photons per active pixel)
    density = total_photons / active_pixels if active_pixels > 0 else 0
    
    return {
        'total_photons': total_photons,
        'peak_intensity': peak_intensity,
        'mean_intensity': mean_intensity,
        'peak_photons': peak_photons,
        'snr': snr,
        'active_pixels': active_pixels,
        'density': density,
        'peak_time': peak_time
    }


def load_and_analyze_cell(file_path):
    """
    Load cell file and compute brightness metrics.
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
    
    metrics = compute_brightness_metrics(array)
    
    # Extract label from filename (multiple possible patterns)
    filename = file_path.name.lower()
    
    # Pattern 1: _act_cell_ or _active_ (ACTIVE cells)
    if '_act_cell_' in filename or '_active_' in filename or filename.startswith('act_'):
        metrics['label'] = 1  # Active
        metrics['label_str'] = 'active'
    # Pattern 2: _in_cell_ or _inactive_ (INACTIVE cells)
    elif '_in_cell_' in filename or '_inactive_' in filename or filename.startswith('in_'):
        metrics['label'] = 0  # Inactive
        metrics['label_str'] = 'inactive'
    else:
        # Try to infer from parent directory or filename structure
        # Some datasets use different naming conventions
        metrics['label'] = -1
        metrics['label_str'] = 'unknown'
        print(f"⚠️  Unknown label pattern in: {file_path.name[:50]}")
    
    metrics['file'] = file_path.name
    
    return metrics


def analyze_folder(folder_path, max_files=None):
    """
    Analyze all cells in a folder.
    """
    files = sorted(Path(folder_path).glob("*.npy"))
    
    if max_files:
        files = files[:max_files]
    
    results = []
    for file in tqdm(files, desc=f"Processing {folder_path.name}"):
        try:
            metrics = load_and_analyze_cell(file)
            results.append(metrics)
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    return results


def main():
    print("="*80)
    print("BRIGHTNESS BIAS ANALYSIS")
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
    
    # ========================================================================
    # BRIGHTNESS ANALYSIS PER DONOR
    # ========================================================================
    print("\n" + "="*80)
    print("BRIGHTNESS STATISTICS PER DONOR")
    print("="*80)
    
    donor_summary = {}
    
    for donor in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        if donor not in all_data:
            continue
        
        results = all_data[donor]
        
        # Overall stats
        total_photons = [r['total_photons'] for r in results]
        mean_intensity = [r['mean_intensity'] for r in results]
        snr = [r['snr'] for r in results]
        density = [r['density'] for r in results]
        
        # Active vs Inactive
        active_results = [r for r in results if r['label'] == 1]
        inactive_results = [r for r in results if r['label'] == 0]
        
        active_photons = [r['total_photons'] for r in active_results]
        inactive_photons = [r['total_photons'] for r in inactive_results]
        
        active_mean = [r['mean_intensity'] for r in active_results]
        inactive_mean = [r['mean_intensity'] for r in inactive_results]
        
        print(f"\n📊 {donor} (n={len(results)}):")
        print(f"   Total cells: {len(results)}")
        print(f"   Active: {len(active_results)} ({100*len(active_results)/len(results):.1f}%)")
        print(f"   Inactive: {len(inactive_results)} ({100*len(inactive_results)/len(results):.1f}%)")
        
        print(f"\n   Overall Brightness:")
        print(f"      Total photons: {np.mean(total_photons):.1f} ± {np.std(total_photons):.1f}")
        print(f"      Mean intensity: {np.mean(mean_intensity):.3f} ± {np.std(mean_intensity):.3f}")
        print(f"      SNR: {np.mean(snr):.1f} ± {np.std(snr):.1f}")
        print(f"      Density: {np.mean(density):.1f} ± {np.std(density):.1f}")
        
        if len(active_photons) > 0 and len(inactive_photons) > 0:
            print(f"\n   Active vs Inactive:")
            print(f"      Active total photons:   {np.mean(active_photons):.1f} ± {np.std(active_photons):.1f}")
            print(f"      Inactive total photons: {np.mean(inactive_photons):.1f} ± {np.std(inactive_photons):.1f}")
            ratio = np.mean(active_photons) / np.mean(inactive_photons)
            print(f"      Ratio (active/inactive): {ratio:.2f}x")
            
            print(f"\n      Active mean intensity:   {np.mean(active_mean):.3f} ± {np.std(active_mean):.3f}")
            print(f"      Inactive mean intensity: {np.mean(inactive_mean):.3f} ± {np.std(inactive_mean):.3f}")
            
            # Statistical test
            stat, pval = mannwhitneyu(active_photons, inactive_photons, alternative='two-sided')
            print(f"      Mann-Whitney p-value: {pval:.2e} {'✓ SIGNIFICANT' if pval < 0.001 else ''}")
        
        donor_summary[donor] = {
            'total_cells': len(results),
            'active_count': len(active_results),
            'inactive_count': len(inactive_results),
            'active_ratio': len(active_results) / len(results),
            'total_photons_mean': float(np.mean(total_photons)),
            'total_photons_std': float(np.std(total_photons)),
            'mean_intensity_mean': float(np.mean(mean_intensity)),
            'mean_intensity_std': float(np.std(mean_intensity)),
            'snr_mean': float(np.mean(snr)),
            'active_photons_mean': float(np.mean(active_photons)) if active_photons else 0,
            'inactive_photons_mean': float(np.mean(inactive_photons)) if inactive_photons else 0,
            'active_inactive_ratio': float(np.mean(active_photons) / np.mean(inactive_photons)) if active_photons and inactive_photons else 0
        }
    
    # ========================================================================
    # CROSS-DONOR BRIGHTNESS COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("CROSS-DONOR BRIGHTNESS COMPARISON")
    print("="*80)
    
    large_donors = ['D4', 'D5']
    small_donors = ['D1', 'D2', 'D3', 'D6']
    
    # Collect brightness by donor group
    large_photons = []
    small_photons = []
    large_mean_int = []
    small_mean_int = []
    
    for donor, results in all_data.items():
        photons = [r['total_photons'] for r in results]
        mean_int = [r['mean_intensity'] for r in results]
        
        if donor in large_donors:
            large_photons.extend(photons)
            large_mean_int.extend(mean_int)
        else:
            small_photons.extend(photons)
            small_mean_int.extend(mean_int)
    
    large_photons = np.array(large_photons)
    small_photons = np.array(small_photons)
    large_mean_int = np.array(large_mean_int)
    small_mean_int = np.array(small_mean_int)
    
    print(f"\n💡 LARGE donors (D4, D5):")
    print(f"   Total photons: {large_photons.mean():.1f} ± {large_photons.std():.1f}")
    print(f"   Mean intensity: {large_mean_int.mean():.3f} ± {large_mean_int.std():.3f}")
    
    print(f"\n💡 SMALL donors (D1, D2, D3, D6):")
    print(f"   Total photons: {small_photons.mean():.1f} ± {small_photons.std():.1f}")
    print(f"   Mean intensity: {small_mean_int.mean():.3f} ± {small_mean_int.std():.3f}")
    
    photon_ratio = large_photons.mean() / small_photons.mean()
    intensity_ratio = large_mean_int.mean() / small_mean_int.mean()
    
    print(f"\n📊 BRIGHTNESS RATIO (large/small):")
    print(f"   Total photons: {photon_ratio:.2f}x")
    print(f"   Mean intensity: {intensity_ratio:.2f}x")
    
    stat, pval = mannwhitneyu(large_photons, small_photons, alternative='two-sided')
    print(f"\n   Mann-Whitney U test: p={pval:.2e} {'✓ SIGNIFICANT' if pval < 0.001 else ''}")
    
    if photon_ratio > 1.2 or photon_ratio < 0.8:
        print(f"\n   ⚠️  WARNING: Large brightness difference detected!")
        print(f"   Model may learn 'bright = active' instead of true metabolic features")
    
    # ========================================================================
    # CORRELATION ANALYSIS: Brightness vs Activity
    # ========================================================================
    print("\n" + "="*80)
    print("CORRELATION: BRIGHTNESS vs ACTIVITY RATIO")
    print("="*80)
    
    donor_names = []
    donor_brightness = []
    donor_active_ratios = []
    
    for donor in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        if donor not in donor_summary:
            continue
        
        donor_names.append(donor)
        donor_brightness.append(donor_summary[donor]['total_photons_mean'])
        donor_active_ratios.append(donor_summary[donor]['active_ratio'])
    
    donor_brightness = np.array(donor_brightness)
    donor_active_ratios = np.array(donor_active_ratios)
    
    # Compute correlation
    correlation = np.corrcoef(donor_brightness, donor_active_ratios)[0, 1]
    
    print(f"\n📈 Correlation between brightness and active ratio:")
    print(f"   Pearson r = {correlation:.3f}")
    
    if abs(correlation) > 0.7:
        print(f"\n   🚨 STRONG CORRELATION DETECTED!")
        print(f"   Brighter donors have {'higher' if correlation > 0 else 'lower'} active ratios")
        print(f"   Model may use brightness as proxy for activity!")
    elif abs(correlation) > 0.4:
        print(f"\n   ⚠️  MODERATE CORRELATION")
        print(f"   Some confounding between brightness and activity")
    else:
        print(f"\n   ✓ Low correlation - brightness not strongly confounded with activity")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING PLOTS...")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Total photons by donor
    ax1 = fig.add_subplot(gs[0, 0])
    donor_list = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    photon_means = [donor_summary[d]['total_photons_mean'] for d in donor_list if d in donor_summary]
    colors = ['red' if d in large_donors else 'blue' for d in donor_list if d in donor_summary]
    
    ax1.bar(range(len(photon_means)), photon_means, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(photon_means)))
    ax1.set_xticklabels([d for d in donor_list if d in donor_summary])
    ax1.set_ylabel('Total Photons', fontsize=12)
    ax1.set_title('Average Total Photons by Donor', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Mean intensity by donor
    ax2 = fig.add_subplot(gs[0, 1])
    intensity_means = [donor_summary[d]['mean_intensity_mean'] for d in donor_list if d in donor_summary]
    
    ax2.bar(range(len(intensity_means)), intensity_means, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(intensity_means)))
    ax2.set_xticklabels([d for d in donor_list if d in donor_summary])
    ax2.set_ylabel('Mean Intensity', fontsize=12)
    ax2.set_title('Average Mean Intensity by Donor', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Active ratio by donor
    ax3 = fig.add_subplot(gs[0, 2])
    active_ratios = [donor_summary[d]['active_ratio'] * 100 for d in donor_list if d in donor_summary]
    
    ax3.bar(range(len(active_ratios)), active_ratios, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(active_ratios)))
    ax3.set_xticklabels([d for d in donor_list if d in donor_summary])
    ax3.set_ylabel('Active Ratio (%)', fontsize=12)
    ax3.set_title('Active Cell Percentage by Donor', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Brightness correlation with activity
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(donor_brightness, donor_active_ratios * 100, s=200, c=colors, alpha=0.7)
    for i, name in enumerate(donor_names):
        ax4.annotate(name, (donor_brightness[i], donor_active_ratios[i] * 100), 
                    fontsize=12, ha='center', va='bottom')
    
    # Add trend line
    z = np.polyfit(donor_brightness, donor_active_ratios * 100, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(donor_brightness.min(), donor_brightness.max(), 100)
    ax4.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2)
    
    ax4.set_xlabel('Average Total Photons', fontsize=12)
    ax4.set_ylabel('Active Ratio (%)', fontsize=12)
    ax4.set_title(f'Brightness vs Activity (r={correlation:.3f})', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Distribution comparison (large vs small donors)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(small_photons, bins=50, alpha=0.6, label='Small donors (D1/D2/D3/D6)', color='blue', density=True)
    ax5.hist(large_photons, bins=50, alpha=0.6, label='Large donors (D4/D5)', color='red', density=True)
    ax5.axvline(small_photons.mean(), color='blue', linestyle='--', linewidth=2)
    ax5.axvline(large_photons.mean(), color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Total Photons', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title(f'Brightness Distribution ({photon_ratio:.2f}x difference)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Active vs Inactive within each donor
    ax6 = fig.add_subplot(gs[1, 2])
    x_pos = np.arange(len(donor_list))
    active_means = []
    inactive_means = []
    
    for donor in donor_list:
        if donor not in all_data:
            active_means.append(0)
            inactive_means.append(0)
            continue
        
        results = all_data[donor]
        active = [r['total_photons'] for r in results if r['label'] == 1]
        inactive = [r['total_photons'] for r in results if r['label'] == 0]
        
        active_means.append(np.mean(active) if active else 0)
        inactive_means.append(np.mean(inactive) if inactive else 0)
    
    width = 0.35
    ax6.bar(x_pos - width/2, active_means, width, label='Active', color='green', alpha=0.7)
    ax6.bar(x_pos + width/2, inactive_means, width, label='Inactive', color='gray', alpha=0.7)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(donor_list)
    ax6.set_ylabel('Total Photons', fontsize=12)
    ax6.set_title('Active vs Inactive Brightness by Donor', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Box plot of photons by donor
    ax7 = fig.add_subplot(gs[2, :])
    donor_data = []
    donor_labels = []
    
    for donor in donor_list:
        if donor in all_data:
            photons = [r['total_photons'] for r in all_data[donor]]
            donor_data.append(photons)
            donor_labels.append(f"{donor}\n(n={len(photons)})")
    
    bp = ax7.boxplot(donor_data, labels=donor_labels, patch_artist=True, showfliers=False)
    for patch, donor in zip(bp['boxes'], donor_list):
        if donor in all_data:
            color = 'red' if donor in large_donors else 'blue'
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax7.set_ylabel('Total Photons', fontsize=12)
    ax7.set_title('Brightness Distribution by Donor (box plots)', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('brightness_bias_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: brightness_bias_analysis.png")
    
    # ========================================================================
    # SAVE DETAILED RESULTS
    # ========================================================================
    import json
    
    summary = {
        'cross_donor_comparison': {
            'large_donors': {
                'names': large_donors,
                'total_photons_mean': float(large_photons.mean()),
                'mean_intensity_mean': float(large_mean_int.mean()),
                'count': len(large_photons)
            },
            'small_donors': {
                'names': small_donors,
                'total_photons_mean': float(small_photons.mean()),
                'mean_intensity_mean': float(small_mean_int.mean()),
                'count': len(small_photons)
            },
            'photon_ratio': float(photon_ratio),
            'intensity_ratio': float(intensity_ratio),
            'mann_whitney_p': float(pval)
        },
        'brightness_activity_correlation': {
            'pearson_r': float(correlation),
            'interpretation': 'STRONG' if abs(correlation) > 0.7 else 'MODERATE' if abs(correlation) > 0.4 else 'WEAK'
        },
        'per_donor_summary': donor_summary
    }
    
    with open('brightness_analysis_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Detailed results saved to: brightness_analysis_results.json")
    
    # ========================================================================
    # FINAL DIAGNOSIS
    # ========================================================================
    print("\n" + "="*80)
    print("🔬 BIAS DIAGNOSIS")
    print("="*80)
    
    if abs(correlation) > 0.7:
        print("\n🚨 CRITICAL ISSUE DETECTED:")
        print(f"   - Strong correlation (r={correlation:.3f}) between brightness and activity")
        print(f"   - Large donors {photon_ratio:.2f}x brighter than small donors")
        print(f"   - Model likely learning: 'Bright = Active' instead of metabolic features")
        print("\n   RECOMMENDATION:")
        print("   1. Apply per-donor intensity normalization")
        print("   2. Use donor-stratified sampling in training")
        print("   3. Add brightness augmentation (±40%)")
    elif abs(correlation) > 0.4:
        print("\n⚠️  MODERATE BIAS DETECTED:")
        print(f"   - Moderate correlation (r={correlation:.3f}) between brightness and activity")
        print(f"   - May partially confound model learning")
        print("\n   RECOMMENDATION:")
        print("   1. Monitor false positives on bright inactive cells")
        print("   2. Consider brightness augmentation")
    else:
        print("\n✓ NO SIGNIFICANT BRIGHTNESS BIAS:")
        print(f"   - Low correlation (r={correlation:.3f}) between brightness and activity")
        print(f"   - Brightness differences exist but don't strongly correlate with labels")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
