"""
FALSE POSITIVE ANALYSIS TOOL
=============================
Extracts and visualizes the 206 false positives from D5 test set
to understand what makes the model systematically label inactive cells as active.

This script:
1. Loads the trained model and test data
2. Identifies all false positive predictions
3. Creates detailed visualizations of FLIM decay curves
4. Compares FP characteristics vs true negatives
5. Generates statistical analysis of what went wrong

Usage:
    python analyze_false_positives.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans

# Import model and dataset
from resnet import create_model
from train_cv import CellDataset

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "models/best_model_fold_D5_FINAL_TEST.pth"
OUTPUT_DIR = Path("false_positive_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

def load_model_and_data():
    """Load trained model and D5 test dataset"""
    
    print("\n" + "="*80)
    print("LOADING MODEL AND DATA")
    print("="*80)
    
    # Load model
    print(f"\n📦 Loading model from {MODEL_PATH}")
    model = create_model(in_channels=2).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Try loading with strict=False to handle architecture mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Model loaded (epoch {checkpoint['epoch']})")
    except RuntimeError as e:
        print(f"   ⚠️  Architecture mismatch detected - attempting flexible load")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   ✓ Model loaded with architecture differences (epoch {checkpoint['epoch']})")
        print(f"   Note: Model may have been trained with different normalization layers")
    
    model.eval()
    
    # Load D5 test data
    print("\n📁 Loading D5 test data")
    data_dir = Path("dataset/isolated_cells_D5")
    
    test_files = []
    test_labels = []
    
    for file in data_dir.glob('*.npy'):
        if '_in' in file.name:
            label = 0  # Inactive
        elif '_act' in file.name:
            label = 1  # Active
        else:
            continue
        
        test_files.append(file)
        test_labels.append(label)
    
    print(f"   ✓ Loaded {len(test_files)} files")
    print(f"     - Inactive: {test_labels.count(0)} ({test_labels.count(0)/len(test_labels)*100:.1f}%)")
    print(f"     - Active: {test_labels.count(1)} ({test_labels.count(1)/len(test_labels)*100:.1f}%)")
    
    # Create dataset (no augmentation for analysis)
    test_dataset = CellDataset(test_files, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    return model, test_dataset, test_loader, test_files, test_labels

# ============================================================================
# IDENTIFY FALSE POSITIVES
# ============================================================================

def identify_false_positives(model, test_loader, test_labels, threshold=0.4910):
    """Run inference and identify all false positive predictions"""
    
    print("\n" + "="*80)
    print("IDENTIFYING FALSE POSITIVES")
    print("="*80)
    print(f"Using threshold: {threshold:.4f}\n")
    
    model.eval()
    all_probs = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Running inference"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs)
    all_predictions = (all_probs > threshold).astype(int)
    
    # Find false positives (predicted=1, actual=0)
    fp_indices = [i for i in range(len(test_labels)) 
                  if all_predictions[i] == 1 and test_labels[i] == 0]
    
    # Find true negatives for comparison (predicted=0, actual=0)
    tn_indices = [i for i in range(len(test_labels)) 
                  if all_predictions[i] == 0 and test_labels[i] == 0]
    
    # Find true positives and false negatives for context
    tp_indices = [i for i in range(len(test_labels)) 
                  if all_predictions[i] == 1 and test_labels[i] == 1]
    fn_indices = [i for i in range(len(test_labels)) 
                  if all_predictions[i] == 0 and test_labels[i] == 1]
    
    print(f"\n📊 Prediction breakdown:")
    print(f"   False Positives: {len(fp_indices)} ({len(fp_indices)/test_labels.count(0)*100:.1f}% of inactive)")
    print(f"   True Negatives:  {len(tn_indices)} ({len(tn_indices)/test_labels.count(0)*100:.1f}% of inactive)")
    print(f"   True Positives:  {len(tp_indices)}")
    print(f"   False Negatives: {len(fn_indices)}")
    
    # Extract probabilities for each category
    fp_probs = all_probs[fp_indices]
    tn_probs = all_probs[tn_indices]
    tp_probs = all_probs[tp_indices]
    fn_probs = all_probs[fn_indices]
    
    print(f"\n📈 Confidence statistics:")
    print(f"   FP median: {np.median(fp_probs):.3f} (should be near threshold)")
    print(f"   TN median: {np.median(tn_probs):.3f}")
    print(f"   Separation (FP - TN): {np.median(fp_probs) - np.median(tn_probs):.3f}")
    
    return {
        'fp_indices': fp_indices,
        'tn_indices': tn_indices,
        'tp_indices': tp_indices,
        'fn_indices': fn_indices,
        'fp_probs': fp_probs,
        'tn_probs': tn_probs,
        'tp_probs': tp_probs,
        'fn_probs': fn_probs,
        'all_probs': all_probs
    }

# ============================================================================
# EXTRACT RAW FLIM DATA
# ============================================================================

def extract_raw_flim_data(test_files, indices, max_samples=None):
    """Load raw FLIM data (before preprocessing) for specified indices"""
    
    print(f"\n📥 Loading raw FLIM data for {len(indices)} cells...")
    if max_samples:
        indices = indices[:max_samples]
        print(f"   (Limited to {max_samples} samples)")
    
    raw_data = []
    filenames = []
    
    for idx in tqdm(indices, desc="Loading files"):
        file_path = test_files[idx]
        array = np.load(file_path)  # Shape: (21, 21, 256) - spatial × temporal
        raw_data.append(array)
        filenames.append(file_path.name)
    
    return raw_data, filenames

# ============================================================================
# ANALYZE FLIM CHARACTERISTICS
# ============================================================================

def analyze_flim_characteristics(raw_data_fp, raw_data_tn, raw_data_tp=None):
    """Compare FLIM characteristics between false positives and true negatives"""
    
    print("\n" + "="*80)
    print("ANALYZING FLIM CHARACTERISTICS")
    print("="*80)
    
    def compute_statistics(data_list):
        """Compute FLIM statistics for a list of cells"""
        stats_dict = {
            'total_photons': [],
            'peak_time': [],
            'peak_intensity': [],
            'decay_tau': [],
            'background_level': [],
            'signal_to_noise': [],
            'spatial_variance': [],
            'center_intensity': []
        }
        
        for array in data_list:
            # Spatial average (average over 21×21 pixels)
            avg_decay = array.mean(axis=(0, 1))  # Shape: (256,) temporal bins
            
            # Total photons
            total_photons = array.sum()
            stats_dict['total_photons'].append(total_photons)
            
            # Peak location and intensity
            peak_idx = np.argmax(avg_decay)
            stats_dict['peak_time'].append(peak_idx)
            stats_dict['peak_intensity'].append(avg_decay[peak_idx])
            
            # Background level (average of first 20 bins before peak)
            bg_level = avg_decay[:20].mean()
            stats_dict['background_level'].append(bg_level)
            
            # Signal-to-noise (peak / background)
            snr = avg_decay[peak_idx] / (bg_level + 1e-6)
            stats_dict['signal_to_noise'].append(snr)
            
            # Decay time constant (exponential fit)
            # Fit exp decay after peak: I(t) = I0 * exp(-t/tau)
            post_peak = avg_decay[peak_idx:]
            if len(post_peak) > 10 and post_peak[0] > 0:
                try:
                    # Normalize and take log
                    normalized = post_peak / post_peak[0]
                    normalized = np.clip(normalized, 1e-6, 1.0)
                    log_decay = np.log(normalized)
                    
                    # Linear fit: log(I) = -t/tau
                    times = np.arange(len(log_decay))
                    slope, _ = np.polyfit(times, log_decay, 1)
                    tau = -1.0 / (slope + 1e-6)
                    stats_dict['decay_tau'].append(tau)
                except:
                    stats_dict['decay_tau'].append(np.nan)
            else:
                stats_dict['decay_tau'].append(np.nan)
            
            # Spatial variance (how uniform is the cell)
            spatial_sum = array.sum(axis=2)  # Sum over time -> (21, 21)
            stats_dict['spatial_variance'].append(spatial_sum.std())
            
            # Center pixel intensity (is photon concentration in center?)
            center_intensity = array[10, 10, :].sum()  # Center pixel
            stats_dict['center_intensity'].append(center_intensity)
        
        # Convert to arrays and remove NaNs
        for key in stats_dict:
            arr = np.array(stats_dict[key])
            stats_dict[key] = arr[~np.isnan(arr)]
        
        return stats_dict
    
    print("\n🔬 Computing statistics...")
    fp_stats = compute_statistics(raw_data_fp)
    tn_stats = compute_statistics(raw_data_tn)
    
    if raw_data_tp:
        tp_stats = compute_statistics(raw_data_tp)
    else:
        tp_stats = None
    
    # Statistical comparison
    print("\n📊 Statistical Comparison (FP vs TN):")
    print(f"{'Metric':<20} {'FP Mean':<12} {'TN Mean':<12} {'p-value':<12} {'Significant?'}")
    print("-" * 70)
    
    for key in fp_stats.keys():
        fp_mean = np.mean(fp_stats[key])
        tn_mean = np.mean(tn_stats[key])
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(fp_stats[key], tn_stats[key], equal_var=False)
        
        significant = "✓✓✓" if p_value < 0.001 else "✓✓" if p_value < 0.01 else "✓" if p_value < 0.05 else ""
        
        print(f"{key:<20} {fp_mean:<12.2f} {tn_mean:<12.2f} {p_value:<12.2e} {significant}")
    
    return fp_stats, tn_stats, tp_stats

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_false_positives(raw_data_fp, raw_data_tn, raw_data_tp, 
                              fp_probs, tn_probs, tp_probs,
                              fp_probs_full, tn_probs_full, tp_probs_full,
                              fp_stats, tn_stats, tp_stats, filenames_fp):
    """Create comprehensive visualizations"""
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Sample decay curves comparison
    print("\n📈 Creating decay curve comparisons...")
    
    # Determine how many samples we can show
    n_samples_fp = min(12, len(raw_data_fp))
    n_samples_tn = min(12, len(raw_data_tn))
    n_samples = max(n_samples_fp, n_samples_tn)
    
    if n_samples_tn < 3:
        print(f"   ⚠️  Warning: Only {n_samples_tn} true negative(s) available - comparing FP vs TP instead")
        # Use true positives for comparison if we don't have enough true negatives
        raw_data_comparison = raw_data_tp if raw_data_tp and len(raw_data_tp) >= 12 else raw_data_fp
        comparison_probs = tp_probs if raw_data_tp and len(raw_data_tp) >= 12 else fp_probs
        comparison_label = "TP" if raw_data_tp and len(raw_data_tp) >= 12 else "FP"
        comparison_color = "green" if comparison_label == "TP" else "orange"
        n_samples_comparison = min(12, len(raw_data_comparison))
    else:
        raw_data_comparison = raw_data_tn
        comparison_probs = tn_probs
        comparison_label = "TN"
        comparison_color = "blue"
        n_samples_comparison = n_samples_tn
    
    fig = plt.figure(figsize=(20, 12))
    
    for i in range(n_samples):
        # False Positive
        if i < n_samples_fp:
            plt.subplot(4, 6, i+1)
            array_fp = raw_data_fp[i]
            avg_decay_fp = array_fp.mean(axis=(0, 1))
            plt.plot(avg_decay_fp, color='red', linewidth=2, alpha=0.8)
            plt.title(f'FP {i+1} (conf={fp_probs[i]:.2f})', fontsize=9, color='red')
            plt.xlabel('Time Bin', fontsize=8)
            plt.ylabel('Photons', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(avg_decay_fp) * 1.1)
        
        # Comparison (TN or TP)
        if i < n_samples_comparison:
            plt.subplot(4, 6, i+13)
            array_comp = raw_data_comparison[i]
            avg_decay_comp = array_comp.mean(axis=(0, 1))
            plt.plot(avg_decay_comp, color=comparison_color, linewidth=2, alpha=0.8)
            plt.title(f'{comparison_label} {i+1} (conf={comparison_probs[i]:.2f})', fontsize=9, color=comparison_color)
            plt.xlabel('Time Bin', fontsize=8)
            plt.ylabel('Photons', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(avg_decay_comp) * 1.1)
    
    plt.suptitle('Decay Curves: False Positives (top 2 rows) vs True Negatives (bottom 2 rows)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(OUTPUT_DIR / 'decay_curves_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {OUTPUT_DIR / 'decay_curves_comparison.png'}")
    plt.close()
    
    # 2. Statistical distributions
    print("\n📊 Creating statistical distribution plots...")
    fig = plt.figure(figsize=(20, 12))
    
    metrics = ['total_photons', 'peak_time', 'peak_intensity', 'decay_tau', 
               'signal_to_noise', 'spatial_variance']
    
    for idx, metric in enumerate(metrics):
        plt.subplot(2, 3, idx+1)
        
        # Plot distributions
        plt.hist(fp_stats[metric], bins=30, alpha=0.6, color='red', 
                label=f'FP (n={len(fp_stats[metric])})', density=True, edgecolor='darkred')
        plt.hist(tn_stats[metric], bins=30, alpha=0.6, color='blue', 
                label=f'TN (n={len(tn_stats[metric])})', density=True, edgecolor='darkblue')
        
        if tp_stats and metric in tp_stats:
            plt.hist(tp_stats[metric], bins=30, alpha=0.4, color='green', 
                    label=f'TP (n={len(tp_stats[metric])})', density=True, edgecolor='darkgreen')
        
        # Add median lines
        plt.axvline(np.median(fp_stats[metric]), color='red', linestyle='--', linewidth=2, 
                   label=f'FP median: {np.median(fp_stats[metric]):.1f}')
        plt.axvline(np.median(tn_stats[metric]), color='blue', linestyle='--', linewidth=2,
                   label=f'TN median: {np.median(tn_stats[metric]):.1f}')
        
        plt.xlabel(metric.replace('_', ' ').title(), fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Add t-test result
        t_stat, p_value = stats.ttest_ind(fp_stats[metric], tn_stats[metric], equal_var=False)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        plt.title(f'{metric.replace("_", " ").title()} (p={p_value:.2e} {significance})', fontsize=10)
    
    plt.suptitle('FLIM Characteristic Distributions: FP vs TN vs TP', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'statistical_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {OUTPUT_DIR / 'statistical_distributions.png'}")
    plt.close()
    
    # 3. Spatial intensity maps (show what FP cells look like spatially)
    print("\n🗺️  Creating spatial intensity maps...")
    fig = plt.figure(figsize=(20, 8))
    
    for i in range(16):
        if i >= len(raw_data_fp):
            break
        
        plt.subplot(2, 8, i+1)
        array_fp = raw_data_fp[i]
        spatial_sum = array_fp.sum(axis=2)  # Sum over time
        
        plt.imshow(spatial_sum, cmap='hot', interpolation='nearest')
        plt.colorbar(shrink=0.6)
        plt.title(f'FP {i+1}\n({filenames_fp[i][:15]}...)', fontsize=7)
        plt.axis('off')
    
    plt.suptitle('Spatial Intensity Maps: False Positive Cells', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spatial_maps_false_positives.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {OUTPUT_DIR / 'spatial_maps_false_positives.png'}")
    plt.close()
    
    # 4. Confidence vs characteristics scatter plots
    print("\n🔍 Creating confidence correlation plots...")
    fig = plt.figure(figsize=(20, 10))
    
    for idx, metric in enumerate(metrics):
        plt.subplot(2, 3, idx+1)
        
        # Use full arrays for correlation analysis
        metric_values = fp_stats[metric]
        probs_values = fp_probs_full[:len(metric_values)]
        
        # Scatter plot: confidence vs metric
        plt.scatter(metric_values, probs_values, 
                   alpha=0.5, color='red', s=30, label='False Positives')
        
        # Add trend line
        z = np.polyfit(metric_values, probs_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(metric_values.min(), metric_values.max(), 100)
        plt.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8)
        
        # Correlation
        corr = np.corrcoef(metric_values, probs_values)[0, 1]
        
        plt.xlabel(metric.replace('_', ' ').title(), fontsize=10)
        plt.ylabel('Model Confidence', fontsize=10)
        plt.title(f'{metric.replace("_", " ").title()}\n(correlation={corr:.3f})', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
        plt.legend(fontsize=8)
    
    plt.suptitle('What FLIM Features Drive False Positive Confidence?', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_correlations.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {OUTPUT_DIR / 'confidence_correlations.png'}")
    plt.close()
    
    # 5. High-confidence FP analysis (>0.8 confidence)
    print("\n🚨 Creating high-confidence error analysis...")
    high_conf_mask = fp_probs_full > 0.8
    high_conf_indices = np.where(high_conf_mask)[0]
    
    if len(high_conf_indices) > 0:
        fig = plt.figure(figsize=(20, 10))
        
        n_high_conf = min(20, len(high_conf_indices))
        n_plotted = 0
        
        for i in range(n_high_conf):
            idx = high_conf_indices[i]
            
            # Only plot if we have the raw data (within first 50 samples)
            if idx >= len(raw_data_fp):
                continue
            
            n_plotted += 1
            plt.subplot(4, 5, n_plotted)
            array_fp = raw_data_fp[idx]
            avg_decay = array_fp.mean(axis=(0, 1))
            
            plt.plot(avg_decay, color='darkred', linewidth=2)
            plt.title(f'Conf={fp_probs_full[idx]:.3f}\n{filenames_fp[idx][:20] if idx < len(filenames_fp) else "..."}', fontsize=7)
            plt.xlabel('Time', fontsize=7)
            plt.ylabel('Photons', fontsize=7)
            plt.grid(True, alpha=0.3)
            
            if n_plotted >= 20:
                break
        
        plt.suptitle(f'HIGH CONFIDENCE FALSE POSITIVES (>{0.8:.1f}): Why is model so wrong?', 
                     fontsize=14, fontweight='bold', color='darkred')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'high_confidence_errors.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {OUTPUT_DIR / 'high_confidence_errors.png'}")
        plt.close()
    else:
        print("   ℹ️  No high-confidence errors (>0.8) found")

# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

def cluster_false_positives(fp_stats, fp_probs, filenames_fp):
    """Use clustering to find subgroups of false positives"""
    
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS")
    print("="*80)
    print("\nFinding subgroups within false positives...\n")
    
    # Create feature matrix (normalize each feature)
    features = []
    feature_names = ['total_photons', 'peak_time', 'peak_intensity', 
                    'decay_tau', 'signal_to_noise', 'spatial_variance']
    
    for fname in feature_names:
        feat = fp_stats[fname]
        # Normalize to [0, 1]
        feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
        features.append(feat_norm)
    
    # Stack features (ensure same length)
    min_len = min(len(f) for f in features)
    features = [f[:min_len] for f in features]
    X = np.column_stack(features)
    
    # K-means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    print("📊 Cluster statistics:")
    for cluster_id in range(3):
        mask = clusters == cluster_id
        n_samples = mask.sum()
        avg_conf = fp_probs[:min_len][mask].mean()
        
        print(f"\n   Cluster {cluster_id + 1} (n={n_samples}):")
        print(f"      Avg confidence: {avg_conf:.3f}")
        
        for i, fname in enumerate(feature_names):
            cluster_mean = X[mask, i].mean()
            print(f"      {fname}: {cluster_mean:.3f}")
    
    # Visualize clusters
    fig = plt.figure(figsize=(15, 10))
    
    # PCA-like: plot top 2 features with highest variance
    feature_vars = X.var(axis=0)
    top_2_features = np.argsort(feature_vars)[-2:]
    
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(X[:, top_2_features[0]], X[:, top_2_features[1]], 
                         c=clusters, cmap='viridis', alpha=0.6, s=50)
    plt.xlabel(f'{feature_names[top_2_features[0]]}', fontsize=10)
    plt.ylabel(f'{feature_names[top_2_features[1]]}', fontsize=10)
    plt.title('Cluster Visualization (2D projection)', fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # Confidence distribution per cluster
    plt.subplot(2, 2, 2)
    for cluster_id in range(3):
        mask = clusters == cluster_id
        cluster_conf = fp_probs[:min_len][mask]
        plt.hist(cluster_conf, bins=20, alpha=0.5, label=f'Cluster {cluster_id+1}')
    
    plt.xlabel('Model Confidence', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Confidence Distribution by Cluster', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature importance (which features separate clusters best)
    plt.subplot(2, 2, 3)
    cluster_separations = []
    for i in range(X.shape[1]):
        # Variance across cluster means
        cluster_means = [X[clusters == c, i].mean() for c in range(3)]
        separation = np.var(cluster_means)
        cluster_separations.append(separation)
    
    plt.barh(feature_names, cluster_separations, color='steelblue')
    plt.xlabel('Separation Variance', fontsize=10)
    plt.title('Feature Importance for Clustering', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'clustering_analysis.png'}")
    plt.close()
    
    return clusters

# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_report(results, fp_stats, tn_stats, tp_stats, clusters=None):
    """Generate a text report summarizing findings"""
    
    report_path = OUTPUT_DIR / 'analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FALSE POSITIVE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total D5 test samples: {len(results['all_probs'])}\n")
        f.write(f"False Positives: {len(results['fp_indices'])} ({len(results['fp_indices'])/len(results['all_probs'])*100:.1f}%)\n")
        f.write(f"True Negatives: {len(results['tn_indices'])}\n")
        f.write(f"True Positives: {len(results['tp_indices'])}\n")
        f.write(f"False Negatives: {len(results['fn_indices'])}\n\n")
        
        f.write("CONFIDENCE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"FP median confidence: {np.median(results['fp_probs']):.3f}\n")
        f.write(f"TN median confidence: {np.median(results['tn_probs']):.3f}\n")
        f.write(f"TP median confidence: {np.median(results['tp_probs']):.3f}\n")
        f.write(f"FN median confidence: {np.median(results['fn_probs']):.3f}\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Identify which metrics differ most
        differences = []
        for key in fp_stats.keys():
            fp_mean = np.mean(fp_stats[key])
            tn_mean = np.mean(tn_stats[key])
            _, p_val = stats.ttest_ind(fp_stats[key], tn_stats[key], equal_var=False)
            
            pct_diff = abs(fp_mean - tn_mean) / (tn_mean + 1e-6) * 100
            differences.append((key, pct_diff, p_val))
        
        differences.sort(key=lambda x: x[2])  # Sort by p-value
        
        f.write("Top discriminating features (FP vs TN):\n\n")
        for i, (metric, pct_diff, p_val) in enumerate(differences[:5]):
            f.write(f"{i+1}. {metric.replace('_', ' ').title()}\n")
            f.write(f"   Percent difference: {pct_diff:.1f}%\n")
            f.write(f"   Statistical significance: p={p_val:.2e}\n")
            f.write(f"   FP mean: {np.mean(fp_stats[metric]):.2f}\n")
            f.write(f"   TN mean: {np.mean(tn_stats[metric]):.2f}\n\n")
        
        if clusters is not None:
            f.write("\nCLUSTERING RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write("False positives were grouped into 3 clusters:\n\n")
            
            for cluster_id in range(3):
                mask = clusters == cluster_id
                f.write(f"Cluster {cluster_id + 1}: {mask.sum()} samples\n")
                f.write(f"   Characteristics: [See clustering_analysis.png]\n\n")
        
        f.write("\nCONCLUSIONS\n")
        f.write("-" * 80 + "\n")
        
        if np.median(results['fp_probs']) > 0.75:
            f.write("⚠️  HIGH CONFIDENCE ERRORS: Model is very confident about FPs (median > 0.75)\n")
            f.write("   This suggests systematic bias or missing features rather than\n")
            f.write("   borderline ambiguity. Possible causes:\n")
            f.write("   • D5 inactive cells have different characteristics (domain shift)\n")
            f.write("   • Ground truth labeling issues (some 'inactive' cells are weakly active)\n")
            f.write("   • Model learned donor-specific artifacts from D1-D4 training data\n\n")
        else:
            f.write("✓ Borderline uncertainty: Most FPs are near decision threshold\n")
            f.write("   This suggests genuine biological overlap between active/inactive\n")
            f.write("   populations rather than systematic model error.\n\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Manually inspect high-confidence errors (>0.8) to check ground truth\n")
        f.write("2. If labeling is correct, consider:\n")
        f.write("   • Ensemble methods to reduce bias\n")
        f.write("   • Explicit donor encoding as input feature\n")
        f.write("   • Training with more diverse donor data\n")
        f.write("3. For deployment, adjust threshold based on clinical cost of FP vs FN\n\n")
    
    print(f"\n✓ Report saved: {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete false positive analysis"""
    
    print("\n" + "="*80)
    print("FALSE POSITIVE ANALYSIS TOOL")
    print("="*80)
    
    # 1. Load model and data
    model, test_dataset, test_loader, test_files, test_labels = load_model_and_data()
    
    # 2. Identify false positives
    results = identify_false_positives(model, test_loader, test_labels)
    
    # 3. Extract raw FLIM data
    print("\n" + "="*80)
    print("EXTRACTING RAW FLIM DATA")
    print("="*80)
    
    # Limit to first 50 samples for visualization (full dataset for stats)
    raw_data_fp, filenames_fp = extract_raw_flim_data(
        test_files, results['fp_indices'], max_samples=50
    )
    raw_data_tn, _ = extract_raw_flim_data(
        test_files, results['tn_indices'], max_samples=50
    )
    raw_data_tp, _ = extract_raw_flim_data(
        test_files, results['tp_indices'], max_samples=50
    )
    
    # For statistics, use all samples
    print("\n📊 Loading full datasets for statistical analysis...")
    raw_data_fp_full, _ = extract_raw_flim_data(test_files, results['fp_indices'])
    raw_data_tn_full, _ = extract_raw_flim_data(test_files, results['tn_indices'])
    raw_data_tp_full, _ = extract_raw_flim_data(test_files, results['tp_indices'])
    
    # 4. Analyze FLIM characteristics
    fp_stats, tn_stats, tp_stats = analyze_flim_characteristics(
        raw_data_fp_full, raw_data_tn_full, raw_data_tp_full
    )
    
    # 5. Create visualizations
    visualize_false_positives(
        raw_data_fp, raw_data_tn, raw_data_tp,
        results['fp_probs'][:50], results['tn_probs'][:50], results['tp_probs'][:50],
        results['fp_probs'], results['tn_probs'], results['tp_probs'],
        fp_stats, tn_stats, tp_stats, filenames_fp
    )
    
    # 6. Clustering analysis
    clusters = cluster_false_positives(fp_stats, results['fp_probs'], filenames_fp)
    
    # 7. Generate report
    generate_report(results, fp_stats, tn_stats, tp_stats, clusters)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • decay_curves_comparison.png      - Visual comparison of FP vs TN decay curves")
    print("  • statistical_distributions.png    - Distribution plots of FLIM metrics")
    print("  • spatial_maps_false_positives.png - Spatial intensity maps of FP cells")
    print("  • confidence_correlations.png      - Which features drive model confidence")
    print("  • high_confidence_errors.png       - Detailed view of worst mistakes")
    print("  • clustering_analysis.png          - Subgroups within false positives")
    print("  • analysis_report.txt              - Comprehensive text summary")
    print("\n✅ Next steps:")
    print("  1. Review high_confidence_errors.png - these are the most suspicious")
    print("  2. Check if ground truth labeling might be incorrect")
    print("  3. If labels are correct, consider ensemble methods or domain adaptation")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
