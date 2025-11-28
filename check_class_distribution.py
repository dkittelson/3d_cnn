"""
Check class distribution (active vs inactive) for each donor

This script counts the number of active and inactive cells in each donor folder
to identify class imbalance patterns.

Usage:
    python check_class_distribution.py
"""

from pathlib import Path

def check_class_distribution(data_dir="/content/3d_cnn/3d_cnn/data"):
    """Count active and inactive cells per donor"""
    
    data_dir = Path(data_dir)
    
    print("="*80)
    print("CLASS DISTRIBUTION BY DONOR")
    print("="*80)
    
    all_donors = {}
    
    for donor_id in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        folder = data_dir / f'isolated_cells_{donor_id}'
        
        if not folder.exists():
            print(f"\n⚠️  {folder} not found, skipping...")
            continue
        
        inactive_count = 0
        active_count = 0
        
        for file in folder.glob('*.npy'):
            if '_in' in file.name or '_in_' in file.name:
                inactive_count += 1
            elif '_act' in file.name or '_act_' in file.name:
                active_count += 1
        
        total = inactive_count + active_count
        
        if total > 0:
            inactive_pct = inactive_count / total * 100
            active_pct = active_count / total * 100
            
            all_donors[donor_id] = {
                'inactive': inactive_count,
                'active': active_count,
                'total': total,
                'inactive_pct': inactive_pct,
                'active_pct': active_pct
            }
            
            print(f"\n📊 {donor_id}:")
            print(f"   Total: {total:,} cells")
            print(f"   Inactive: {inactive_count:,} ({inactive_pct:.1f}%)")
            print(f"   Active:   {active_count:,} ({active_pct:.1f}%)")
            
            if active_pct > 55:
                print(f"   ⚠️  ACTIVE-MAJORITY donor")
            elif inactive_pct > 55:
                print(f"   ✓ Inactive-majority donor")
            else:
                print(f"   ≈ Balanced donor")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_donors:
        # Training donors (D1-D4, D6)
        train_donors = [d for d in ['D1', 'D2', 'D3', 'D4', 'D6'] if d in all_donors]
        test_donor = 'D5'
        
        if train_donors:
            train_inactive = sum(all_donors[d]['inactive'] for d in train_donors)
            train_active = sum(all_donors[d]['active'] for d in train_donors)
            train_total = train_inactive + train_active
            
            print(f"\n🎓 TRAINING SET (D1-D4, D6):")
            print(f"   Total: {train_total:,} cells")
            print(f"   Inactive: {train_inactive:,} ({train_inactive/train_total*100:.1f}%)")
            print(f"   Active:   {train_active:,} ({train_active/train_total*100:.1f}%)")
            
        if test_donor in all_donors:
            test_data = all_donors[test_donor]
            print(f"\n🧪 TEST SET (D5):")
            print(f"   Total: {test_data['total']:,} cells")
            print(f"   Inactive: {test_data['inactive']:,} ({test_data['inactive_pct']:.1f}%)")
            print(f"   Active:   {test_data['active']:,} ({test_data['active_pct']:.1f}%)")
            
            if train_donors:
                train_active_pct = train_active/train_total*100
                test_active_pct = test_data['active_pct']
                diff = test_active_pct - train_active_pct
                
                print(f"\n⚡ CLASS DISTRIBUTION SHIFT:")
                print(f"   Training active%: {train_active_pct:.1f}%")
                print(f"   Test active%:     {test_active_pct:.1f}%")
                print(f"   Difference:       {diff:+.1f}%")
                
                if abs(diff) > 10:
                    print(f"\n   🚨 MAJOR CLASS IMBALANCE SHIFT!")
                    print(f"   This confounds intensity with class label:")
                    print(f"   - Test set has {'more' if diff > 0 else 'fewer'} active cells")
                    print(f"   - Active cells are typically brighter")
                    print(f"   - Model may have learned donor-specific intensity patterns")
        
        print("\n" + "="*80)
        print("CORRELATION WITH INTENSITY")
        print("="*80)
        
        # Import intensity stats if available
        try:
            from train_cv import DONOR_INTENSITY_STATS
            
            print("\nDonor | Active% | Mean Intensity | Correlation?")
            print("-" * 60)
            
            for donor_id in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
                if donor_id in all_donors and donor_id in DONOR_INTENSITY_STATS:
                    active_pct = all_donors[donor_id]['active_pct']
                    intensity = DONOR_INTENSITY_STATS[donor_id]['mean']
                    print(f"{donor_id:4s}  | {active_pct:6.1f}% | {intensity:14.2f} | ", end="")
                    
                    if active_pct > 55 and intensity > 2.8:
                        print("✓ High active% → High intensity")
                    elif active_pct < 45 and intensity < 2.6:
                        print("✓ Low active% → Low intensity")
                    else:
                        print("≈ No clear pattern")
            
            # Calculate correlation
            import numpy as np
            active_pcts = [all_donors[d]['active_pct'] for d in all_donors if d in DONOR_INTENSITY_STATS]
            intensities = [DONOR_INTENSITY_STATS[d]['mean'] for d in all_donors if d in DONOR_INTENSITY_STATS]
            
            if len(active_pcts) >= 3:
                correlation = np.corrcoef(active_pcts, intensities)[0, 1]
                print(f"\n📈 Correlation coefficient: {correlation:.3f}")
                
                if abs(correlation) > 0.5:
                    print(f"   🚨 STRONG CORRELATION! Intensity is confounded with class label!")
                    print(f"   Model cannot distinguish:")
                    print(f"   - 'Bright because active' vs 'Bright because active-rich donor'")
                elif abs(correlation) > 0.3:
                    print(f"   ⚠️  Moderate correlation - some confounding present")
                else:
                    print(f"   ✓ Weak correlation - minimal confounding")
                    
        except ImportError:
            print("\n(Run after updating DONOR_INTENSITY_STATS in train_cv.py)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    check_class_distribution()
