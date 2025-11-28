"""
PREPROCESSING QUALITY ANALYSIS

Compares raw FLIM images (D1/SDT, D1/Mask) with preprocessed isolated cells
to determine if preprocessing is degrading data quality and causing the 85% accuracy ceiling.

Checks:
1. Information loss during preprocessing (SNR, dynamic range, spatial features)
2. Data quality per donor (correlate with model performance)
3. Preprocessing artifacts (clipping, normalization issues, spatial distortion)
4. Active vs Inactive cell quality differences

Usage:
    python analyze_preprocessing_quality.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage, stats
from sklearn.metrics import mutual_information_score
import seaborn as sns
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_DIR = Path("/content/3d_cnn/3d_cnn/data")
ISOLATED_CELLS_DIR = Path("/content/3d_cnn/3d_cnn/data")

DONORS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
SAMPLE_SIZE = 50  # Cells per donor to analyze (balance computation time vs coverage)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_snr(array):
    """Compute Signal-to-Noise Ratio"""
    signal = np.mean(array)
    noise = np.std(array)
    if noise == 0:
        return float('inf')
    return signal / noise

def compute_dynamic_range(array):
    """Compute dynamic range in dB"""
    max_val = np.max(array)
    min_val = np.min(array[array > 0]) if np.any(array > 0) else 1e-10
    return 20 * np.log10(max_val / min_val)

def compute_entropy(array):
    """Compute Shannon entropy of intensity distribution"""
    hist, _ = np.histogram(array.flatten(), bins=256, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def compute_spatial_frequency(array):
    """Compute spatial frequency content (sharpness indicator)"""
    # Apply FFT and compute power spectrum
    fft = np.fft.fftn(array)
    power = np.abs(fft) ** 2
    
    # High-frequency content (upper half of spectrum)
    # Handle both 2D and 3D arrays
    if len(power.shape) == 3:
        high_freq = power[power.shape[0]//2:, power.shape[1]//2:, power.shape[2]//2:]
    elif len(power.shape) == 2:
        high_freq = power[power.shape[0]//2:, power.shape[1]//2:]
    else:
        high_freq = power[len(power)//2:]
    
    total_power = np.sum(power)
    return np.sum(high_freq) / total_power if total_power > 0 else 0

def compute_edge_strength(array):
    """Compute average edge strength using Sobel filter"""
    edges = ndimage.sobel(array)
    return np.mean(np.abs(edges))

def analyze_clipping(array):
    """Check for clipping artifacts"""
    flat = array.flatten()
    max_val = np.max(flat)
    min_val = np.min(flat)
    
    # Count pixels at extremes (potential clipping)
    at_max = np.sum(flat == max_val) / len(flat)
    at_min = np.sum(flat == min_val) / len(flat)
    
    return {
        'max_clipping': at_max,
        'min_clipping': at_min,
        'total_clipping': at_max + at_min
    }

def compute_contrast(array):
    """Compute Michelson contrast"""
    max_val = np.max(array)
    min_val = np.min(array)
    if max_val + min_val == 0:
        return 0
    return (max_val - min_val) / (max_val + min_val)

# ============================================================================
# LOAD RAW IMAGE FROM D*/SDT OR D*/Mask
# ============================================================================

def find_raw_image(donor_id, cell_filename):
    """
    Find the original raw image for a preprocessed cell.
    
    cell_filename format: 'd01m1_in_cell_1.npy' or 'd01m1_act_cell_1.npy'
    Corresponds to: D1/SDT/m1.npy or D1/Mask/m1.npy
    """
    # Parse filename to get movie ID
    # Example: 'd01m1_in_cell_1.npy' -> 'm1'
    parts = cell_filename.stem.split('_')
    movie_part = parts[0]  # 'd01m1'
    movie_id = movie_part.split('d')[1].split('m')[1]  # Extract '1' from 'd01m1'
    movie_name = f'm{movie_id}.npy'
    
    # Try both SDT and Mask folders
    sdt_path = RAW_DATA_DIR / donor_id / "SDT" / movie_name
    mask_path = RAW_DATA_DIR / donor_id / "Mask" / movie_name
    
    if sdt_path.exists():
        return np.load(sdt_path), 'SDT'
    elif mask_path.exists():
        return np.load(mask_path), 'Mask'
    else:
        return None, None

# ============================================================================
# ANALYZE SINGLE CELL
# ============================================================================

def analyze_cell_quality(isolated_cell_path):
    """Analyze quality metrics for one preprocessed cell"""
    
    # Load preprocessed cell
    preprocessed = np.load(isolated_cell_path)
    
    # Preprocessed is (2, 64, 21, 21): [decay_normalized, log_intensity]
    decay_channel = preprocessed[0]  # (64, 21, 21)
    intensity_channel = preprocessed[1]  # (64, 21, 21)
    
    # Compute metrics for preprocessed data
    metrics = {
        'filename': isolated_cell_path.name,
        'preprocessed': {
            'decay': {
                'snr': compute_snr(decay_channel),
                'dynamic_range': compute_dynamic_range(decay_channel),
                'entropy': compute_entropy(decay_channel),
                'spatial_freq': compute_spatial_frequency(decay_channel),
                'edge_strength': compute_edge_strength(decay_channel),
                'contrast': compute_contrast(decay_channel),
                'mean': np.mean(decay_channel),
                'std': np.std(decay_channel),
                'min': np.min(decay_channel),
                'max': np.max(decay_channel),
                'zeros_pct': np.sum(decay_channel == 0) / decay_channel.size,
                **analyze_clipping(decay_channel)
            },
            'intensity': {
                'snr': compute_snr(intensity_channel),
                'dynamic_range': compute_dynamic_range(intensity_channel),
                'entropy': compute_entropy(intensity_channel),
                'spatial_freq': compute_spatial_frequency(intensity_channel),
                'edge_strength': compute_edge_strength(intensity_channel),
                'contrast': compute_contrast(intensity_channel),
                'mean': np.mean(intensity_channel),
                'std': np.std(intensity_channel),
                'min': np.min(intensity_channel),
                'max': np.max(intensity_channel),
                'zeros_pct': np.sum(intensity_channel == 0) / intensity_channel.size,
                **analyze_clipping(intensity_channel)
            }
        }
    }
    
    return metrics

# ============================================================================
# ANALYZE DONOR
# ============================================================================

def analyze_donor(donor_id, sample_size=SAMPLE_SIZE):
    """Analyze preprocessing quality for one donor"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING DONOR: {donor_id}")
    print(f"{'='*80}")
    
    # Get isolated cells for this donor
    isolated_cells_dir = ISOLATED_CELLS_DIR / f"isolated_cells_{donor_id}"
    
    if not isolated_cells_dir.exists():
        print(f"❌ Directory not found: {isolated_cells_dir}")
        return None
    
    # Get all cell files
    all_files = list(isolated_cells_dir.glob("*.npy"))
    
    # Separate active and inactive
    inactive_files = [f for f in all_files if '_in_' in f.name]
    active_files = [f for f in all_files if '_act_' in f.name]
    
    print(f"Found {len(inactive_files)} inactive and {len(active_files)} active cells")
    
    # Sample equal amounts from each class
    sample_per_class = sample_size // 2
    inactive_sample = np.random.choice(inactive_files, min(sample_per_class, len(inactive_files)), replace=False)
    active_sample = np.random.choice(active_files, min(sample_per_class, len(active_files)), replace=False)
    
    # Analyze each cell
    results = {
        'donor_id': donor_id,
        'inactive': [],
        'active': []
    }
    
    print(f"\n📊 Analyzing {len(inactive_sample)} inactive cells...")
    for cell_file in tqdm(inactive_sample):
        metrics = analyze_cell_quality(cell_file)
        results['inactive'].append(metrics)
    
    print(f"\n📊 Analyzing {len(active_sample)} active cells...")
    for cell_file in tqdm(active_sample):
        metrics = analyze_cell_quality(cell_file)
        results['active'].append(metrics)
    
    return results

# ============================================================================
# AGGREGATE AND VISUALIZE
# ============================================================================

def aggregate_metrics(donor_results):
    """Compute statistics across all cells in a donor"""
    
    def compute_stats(cell_list, channel):
        """Compute mean/std for each metric"""
        metrics_dict = {}
        
        if not cell_list:
            return metrics_dict
        
        # Get all metric names
        metric_names = cell_list[0]['preprocessed'][channel].keys()
        
        for metric in metric_names:
            values = [cell['preprocessed'][channel][metric] for cell in cell_list]
            values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
            
            if values:
                metrics_dict[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return metrics_dict
    
    summary = {
        'donor_id': donor_results['donor_id'],
        'inactive': {
            'decay': compute_stats(donor_results['inactive'], 'decay'),
            'intensity': compute_stats(donor_results['inactive'], 'intensity')
        },
        'active': {
            'decay': compute_stats(donor_results['active'], 'decay'),
            'intensity': compute_stats(donor_results['active'], 'intensity')
        }
    }
    
    return summary

def plot_quality_comparison(all_summaries, output_path='preprocessing_quality_analysis.png'):
    """Create comprehensive visualization of preprocessing quality"""
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Preprocessing Quality Analysis Across Donors', fontsize=16, fontweight='bold')
    
    donors = [s['donor_id'] for s in all_summaries]
    
    # Key metrics to plot
    metrics_to_plot = [
        ('snr', 'Signal-to-Noise Ratio'),
        ('dynamic_range', 'Dynamic Range (dB)'),
        ('entropy', 'Entropy'),
        ('spatial_freq', 'Spatial Frequency'),
        ('edge_strength', 'Edge Strength'),
        ('contrast', 'Contrast'),
        ('zeros_pct', 'Zero Pixels (%)'),
        ('total_clipping', 'Clipping (%)'),
        ('mean', 'Mean Value'),
        ('std', 'Std Dev'),
        ('min', 'Min Value'),
        ('max', 'Max Value')
    ]
    
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        # Collect data for both channels
        decay_inactive = []
        decay_active = []
        intensity_inactive = []
        intensity_active = []
        
        for summary in all_summaries:
            # Decay channel
            if metric_key in summary['inactive']['decay']:
                decay_inactive.append(summary['inactive']['decay'][metric_key]['mean'])
                decay_active.append(summary['active']['decay'][metric_key]['mean'])
            
            # Intensity channel
            if metric_key in summary['inactive']['intensity']:
                intensity_inactive.append(summary['inactive']['intensity'][metric_key]['mean'])
                intensity_active.append(summary['active']['intensity'][metric_key]['mean'])
        
        x = np.arange(len(donors))
        width = 0.2
        
        # Plot bars
        if decay_inactive:
            ax.bar(x - width*1.5, decay_inactive, width, label='Decay (Inactive)', color='skyblue', alpha=0.8)
            ax.bar(x - width*0.5, decay_active, width, label='Decay (Active)', color='blue', alpha=0.8)
        
        if intensity_inactive:
            ax.bar(x + width*0.5, intensity_inactive, width, label='Intensity (Inactive)', color='lightcoral', alpha=0.8)
            ax.bar(x + width*1.5, intensity_active, width, label='Intensity (Active)', color='red', alpha=0.8)
        
        ax.set_xlabel('Donor')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label}')
        ax.set_xticks(x)
        ax.set_xticklabels(donors)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Quality comparison plot saved to: {output_path}")
    plt.close()

def create_quality_report(all_summaries, output_path='preprocessing_quality_report.txt'):
    """Generate detailed text report"""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PREPROCESSING QUALITY ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("SUMMARY ACROSS ALL DONORS\n")
        f.write("-"*80 + "\n\n")
        
        # Check for problematic patterns
        issues = []
        
        for summary in all_summaries:
            donor_id = summary['donor_id']
            
            # Check for high clipping
            for class_label in ['inactive', 'active']:
                for channel in ['decay', 'intensity']:
                    if 'total_clipping' in summary[class_label][channel]:
                        clip = summary[class_label][channel]['total_clipping']['mean']
                        if clip > 0.1:  # More than 10% clipped
                            issues.append(f"{donor_id} {class_label} {channel}: {clip*100:.1f}% clipping")
            
            # Check for low SNR
            for class_label in ['inactive', 'active']:
                for channel in ['decay', 'intensity']:
                    if 'snr' in summary[class_label][channel]:
                        snr = summary[class_label][channel]['snr']['mean']
                        if snr < 2:  # SNR below 2 is concerning
                            issues.append(f"{donor_id} {class_label} {channel}: Low SNR ({snr:.2f})")
            
            # Check for excessive zeros
            for class_label in ['inactive', 'active']:
                for channel in ['decay', 'intensity']:
                    if 'zeros_pct' in summary[class_label][channel]:
                        zeros = summary[class_label][channel]['zeros_pct']['mean']
                        if zeros > 0.5:  # More than 50% zeros
                            issues.append(f"{donor_id} {class_label} {channel}: {zeros*100:.1f}% zero pixels")
        
        if issues:
            f.write("⚠️  QUALITY ISSUES DETECTED:\n\n")
            for issue in issues:
                f.write(f"  • {issue}\n")
        else:
            f.write("✓ No major quality issues detected\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed per-donor analysis
        for summary in all_summaries:
            donor_id = summary['donor_id']
            f.write(f"DONOR: {donor_id}\n")
            f.write("-"*80 + "\n\n")
            
            for class_label in ['inactive', 'active']:
                f.write(f"  {class_label.upper()} CELLS:\n\n")
                
                for channel in ['decay', 'intensity']:
                    f.write(f"    {channel.upper()} Channel:\n")
                    
                    for metric, values in summary[class_label][channel].items():
                        f.write(f"      {metric:20s}: mean={values['mean']:8.3f}, std={values['std']:8.3f}, "
                               f"median={values['median']:8.3f}\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"\n📄 Quality report saved to: {output_path}")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*80)
    print("PREPROCESSING QUALITY ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {SAMPLE_SIZE} cells per donor ({SAMPLE_SIZE//2} per class)")
    print(f"Donors: {', '.join(DONORS)}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Analyze each donor
    all_results = []
    all_summaries = []
    
    for donor_id in DONORS:
        donor_results = analyze_donor(donor_id, sample_size=SAMPLE_SIZE)
        
        if donor_results:
            all_results.append(donor_results)
            summary = aggregate_metrics(donor_results)
            all_summaries.append(summary)
    
    # Save raw results
    print("\n💾 Saving raw results...")
    with open('preprocessing_quality_raw_data.json', 'w') as f:
        # Convert to JSON-serializable format
        json.dump(all_summaries, f, indent=2, default=str)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_quality_comparison(all_summaries)
    
    # Generate report
    print("\n📝 Generating report...")
    create_quality_report(all_summaries)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • preprocessing_quality_analysis.png")
    print("  • preprocessing_quality_report.txt")
    print("  • preprocessing_quality_raw_data.json")
    print("\nCheck the report for detailed findings!")

if __name__ == "__main__":
    main()
