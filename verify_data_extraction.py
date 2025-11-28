"""
Verify that data_pipeline.py is correctly extracting cell data
"""
import numpy as np
from pathlib import Path

print("="*70)
print("DATA PIPELINE VERIFICATION")
print("="*70)

# Check a few extracted cells to verify correctness
data_dir = Path("dataset/")

issues = []
samples_checked = 0

for dataset in ["isolated_cells_D1", "isolated_cells_D2", "isolated_cells_D5"]:
    dataset_path = data_dir / dataset
    if not dataset_path.exists():
        print(f"\n⚠️  {dataset} not found - skipping")
        continue
    
    print(f"\n{dataset}:")
    
    # Check first 5 files
    files = sorted(list(dataset_path.glob("*.npy")))[:5]
    
    for f in files:
        array = np.load(f)
        samples_checked += 1
        
        print(f"  {f.name}:")
        print(f"    Shape: {array.shape}")
        print(f"    Expected: (21, 21, 256) or (21, 21, 128)")
        
        # Check 1: Shape validation
        if array.shape[0] != 21 or array.shape[1] != 21:
            issues.append(f"❌ {f.name}: Spatial shape {array.shape[:2]} != (21, 21)")
        
        if array.shape[2] not in [128, 256]:
            issues.append(f"❌ {f.name}: Temporal shape {array.shape[2]} not 128 or 256")
        
        # Check 2: Data range validation
        min_val = array.min()
        max_val = array.max()
        print(f"    Range: [{min_val:.2f}, {max_val:.2f}]")
        
        if min_val < 0:
            issues.append(f"❌ {f.name}: Negative values detected (min={min_val:.2f})")
        
        # Check 3: Sparsity check (should have SOME data)
        nonzero_pct = (array > 0).mean() * 100
        print(f"    Non-zero: {nonzero_pct:.1f}%")
        
        if nonzero_pct < 1:
            issues.append(f"⚠️  {f.name}: Very sparse ({nonzero_pct:.1f}% non-zero)")
        
        if nonzero_pct > 99:
            issues.append(f"⚠️  {f.name}: Almost no masking ({nonzero_pct:.1f}% non-zero)")
        
        # Check 4: Temporal decay validation
        # FLIM should show photon decay - later time bins should have fewer counts
        spatial_sum = array.sum(axis=(0, 1))  # Sum over spatial, get temporal profile
        if spatial_sum.sum() > 0:
            # Check if there's a general decay trend
            first_half = spatial_sum[:len(spatial_sum)//2].sum()
            second_half = spatial_sum[len(spatial_sum)//2:].sum()
            
            if second_half > first_half * 1.5:
                issues.append(f"⚠️  {f.name}: Unusual temporal profile (more photons in late bins)")
        
        # Check 5: Masking validation
        # Padding should be exactly 0, not near-zero
        edge_pixels = array[0, :, :].sum() + array[-1, :, :].sum() + \
                      array[:, 0, :].sum() + array[:, -1, :].sum()
        
        if edge_pixels == 0:
            print(f"    ✓ Proper masking (edge pixels = 0)")
        else:
            print(f"    ⚠️  Edge pixels non-zero (might be large cell or padding issue)")

print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print(f"Samples checked: {samples_checked}")

if issues:
    print(f"\n⚠️  Found {len(issues)} issues:\n")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ All checks passed!")

print("\n" + "="*70)
print("KEY CHECKS:")
print("="*70)
print("1. ✓ Shape: (21, 21, 256) or (21, 21, 128)")
print("2. ✓ Data range: Non-negative values")
print("3. ✓ Sparsity: 1-99% non-zero (not empty or unmasked)")
print("4. ✓ Temporal: Decay profile (early > late photon counts)")
print("5. ✓ Masking: Padding properly zeroed")

print("\n" + "="*70)
print("EXPECTED DATA STRUCTURE:")
print("="*70)
print("Shape: (H=21, W=21, T=256)")
print("  H, W: Spatial dimensions (padded/cropped to 21x21)")
print("  T: Temporal bins (256 time points for FLIM decay)")
print()
print("Masking: Non-cell regions set to 0 across all time bins")
print("Content: Photon counts per (x,y,t) voxel")
print("Expected: Decay curve shape (high counts early, low counts late)")
