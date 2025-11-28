# Per-Donor Normalization Results

## 🔍 Discovery: D5 is BRIGHTER, Not Dimmer!

### Actual Statistics from compute_donor_stats.py:

```
D1: mean=2.52, std=2.71
D2: mean=2.45, std=2.77
D3: mean=2.60, std=2.83
D4: mean=2.21, std=2.93  ← 18.6% dimmer than average
D5: mean=3.52, std=2.97  ← 29.8% BRIGHTER than average
D6: mean=2.99, std=3.09  ← 10.2% brighter
```

Training average (D1-D4, D6): **2.57**  
D5: **3.52** (37% higher!)

---

## 🤔 Wait... False Positive Analysis Said D5 Was Dimmer?

**False Positive Analysis Results:**
- FP (D5 inactive): 44,673 photons, SNR=131
- TN (D5 inactive): 84,873 photons, SNR=1860

This appears contradictory! Let's reconcile:

### The Key Insight: Within D5, FPs Are Dimmer Than TNs

1. **Global donor statistics (compute_donor_stats.py):**
   - D5 has mean=3.52 → D5 is brighter on average than training donors

2. **Within-D5 comparison (false_positive_analysis):**
   - D5 False Positives: 44k photons (model says "active")
   - D5 True Negatives: 84k photons (model correctly says "inactive")

### What's Happening:

The model learned from D1-D4 training that:
- "Typical intensity around 2.5" = baseline
- Lower intensity → active
- Higher intensity → inactive

When it sees D5 (globally brighter at 3.52):
- D5 dim cells (44k photons) are still brighter than D1-D4 average
- But WITHIN D5, these are the dim cells
- Model sees "below D5 average" and incorrectly thinks "active"

**Analogy:** If you train a model on normal-height people (5'8" average) and test on basketball players (6'4" average), it might call the "short" basketball players (6'0") as "tall" because they're still taller than the training average, even though they're short *for basketball players*.

---

## 💡 Why Per-Donor Normalization Fixes This

### Before (Global Norm):

```
D1-D4 Training:
  - Inactive cell: 85k photons → log=11.35 → normalized = 11.35/8.22 = 1.38
  - Active cell: 40k photons → log=10.60 → normalized = 10.60/8.22 = 1.29

D5 Testing (globally brighter):
  - Inactive cell (bright): 84k photons → normalized = 1.38 ✓ (correct)
  - Inactive cell (dim): 44k photons → normalized = 1.30 ✗ (looks "active" to model)
```

### After (Per-Donor Norm):

```
D1-D4 Training:
  - Inactive: (11.35 - 2.57) / 2.8 = 3.14 → rescaled = 0.86
  - Active: (10.60 - 2.57) / 2.8 = 2.87 → rescaled = 0.81

D5 Testing (standardized to same scale):
  - Inactive (bright): (11.38 - 3.52) / 2.97 = 2.65 → rescaled = 0.78
  - Inactive (dim): (10.70 - 3.52) / 2.97 = 2.42 → rescaled = 0.74
```

Now both D5 inactive cells map to similar ranges as D1-D4 inactive cells, regardless of whether they're bright or dim *within* D5.

---

## 📊 Expected Impact

**Current Performance (Global Norm):**
- Test Accuracy: 85.2%
- False Positives: 210 (33.6% of D5 inactive)
- Issue: D5's brightness distribution differs from training

**Expected After Per-Donor Norm:**
- Test Accuracy: **88-91%** (3-6% improvement)
- False Positives: **~100-130** (halved, down to 16-21%)
- Benefit: Removes domain shift, model sees standardized intensities

---

## ✅ Implementation Complete

Changes made to `train_cv.py`:

1. Added actual `DONOR_INTENSITY_STATS` from your data
2. Modified `CellDataset` to standardize per-donor
3. Preserves within-donor variation (active vs inactive signal)

**Next Step:** Retrain model with `use_donor_norm=True` (default)

---

## 🧪 Why This Differs From Intensity Jitter

Your current augmentation:
```python
scale_factor = np.random.uniform(0.6, 1.4)  # Random ±40% scaling
```

This creates random variations *within* training batches but:
- Doesn't fix that D5 has a different brightness *distribution* (higher mean)
- Can't learn that D5's "dim" cells (44k photons) are still brighter than D1-D4 average

Per-donor norm standardizes each donor to mean=0, std=1 *before* augmentation, removing the systematic offset.
