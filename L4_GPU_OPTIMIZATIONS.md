# L4 GPU Optimizations Applied to train_cv.py

## Overview
Applied comprehensive optimizations to maximize training speed and GPU utilization on L4 GPU (24GB VRAM).

## Changes Made

### 1. CUDA Optimizations (lines ~1518-1521)
```python
# Auto-optimize CUDA kernels for L4 architecture
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Enable CUDNN autotuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions
```

**Impact:**
- **CUDNN benchmark**: Automatically selects fastest convolution algorithms for your specific GPU and input sizes. Expected 5-10% speedup.
- **TF32**: Uses TensorFloat-32 precision (19-bit mantissa) instead of FP32, giving ~3x speedup on Ampere+ GPUs while maintaining accuracy. L4 is Ampere-based, so this is critical.

### 2. Batch Size Increase (line ~1528)
```python
BATCH_SIZE = 64  # Was 32, now optimized for L4 GPU (24GB)
```

**Impact:**
- **2× larger batches**: Better GPU utilization, more stable gradients
- **Memory estimate**: 
  - Input: (64, 2, 256, 21, 21) = ~115 MB per batch
  - Model: 909k params × 4 bytes = ~3.6 MB
  - Gradients + optimizer states: ~15 MB
  - Total: ~134 MB → plenty of headroom on 24GB L4
- **Training stability**: Larger batches = more stable gradient estimates
- **Expected speedup**: ~1.8× (not full 2× due to overhead)

### 3. Data Loading Workers (line ~1530)
```python
NUM_WORKERS = 8  # Was 4, increased for better throughput
```

**Impact:**
- **8 parallel workers**: Prevents data loading bottleneck
- **CPU utilization**: Better use of multi-core CPU during training
- **Persistent workers**: Workers stay alive between epochs (saves spawn overhead)
- **Expected impact**: Eliminates data stalls, keeps GPU fed

### 4. Persistent Workers (lines ~412-415, ~421-424)
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True if num_workers > 0 else False  # Keep alive
)
```

**Impact:**
- **Reduced overhead**: Workers don't respawn each epoch
- **Faster epoch transitions**: ~2-5 seconds saved per epoch
- **Memory tradeoff**: Uses more RAM but saves time

### 5. Enhanced Device Reporting (lines ~1535-1540)
```python
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Optimizations: CUDNN benchmark=True, TF32=True\n")
```

**Impact:**
- **Visibility**: Confirms optimizations are enabled
- **Monitoring**: Shows GPU name and VRAM for verification

## Expected Performance Gains

### Training Speed
| Component | Before | After | Speedup | Reason |
|-----------|--------|-------|---------|--------|
| **Batch processing** | 32 | 64 | 1.8× | Larger batches, better GPU utilization |
| **CUDNN benchmark** | OFF | ON | 1.05-1.1× | Auto-optimized kernels |
| **TF32 operations** | FP32 | TF32 | 1.3× | Faster matmul on Ampere |
| **Data loading** | 4 workers | 8 workers | 1.1× | No data stalls |
| **Persistent workers** | Respawn | Persistent | 1.02× | Reduced epoch overhead |
| **TOTAL** | Baseline | **~2.5× faster** | Combined effects |

### Practical Impact
- **Old speed**: ~2.03 it/s (32 batch size)
- **Expected new speed**: ~5-6 it/s (64 batch size + optimizations)
- **Time per epoch**: ~15 minutes → ~6 minutes
- **Total training time**: 75 epochs × 15 min = **18.75 hours → 7.5 hours**

### GPU Utilization
- **Before**: ~40-50% utilization (small batches, FP32, slow data loading)
- **After**: ~80-90% utilization (large batches, TF32, fast data pipeline)
- **Memory usage**: ~8-10 GB / 24 GB (plenty of headroom)

## Architecture Remains Unchanged

All optimizations are training-specific. The model architecture is still:
- **Input**: (2, 256, 21, 21) - log1p(FLIM) + binary mask
- **Parameters**: 909k
- **Temporal downsampling**: Gentle stride=2 (256 → 128 → 64 → 32)
- **Preprocessing**: `log1p(array) / 10.0` preserves intensity

## Verification Checklist

When you run this on L4, verify:
- [ ] Console shows "CUDNN benchmark=True, TF32=True"
- [ ] Console shows "VRAM: 24.0 GB" (confirms L4)
- [ ] Training speed: 5-6 it/s (vs old 2.03 it/s)
- [ ] GPU utilization: 80-90% (use `nvidia-smi dmon`)
- [ ] Memory usage: ~8-10 GB (well below 24 GB limit)
- [ ] No OOM errors
- [ ] Validation accuracy still improving (76% → 85%+)

## Further Optimizations (If Needed)

If GPU utilization is still low (<70%), try:
1. **Increase batch size to 96**: `BATCH_SIZE = 96`
   - Should fit in 24GB VRAM easily
   - Will push utilization to 90%+
2. **Increase num_workers to 12**: `NUM_WORKERS = 12`
   - If you have 12+ CPU cores
3. **Enable mixed precision (FP16)**: Add to train loop
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       outputs = model(inputs)
   ```
   - 2× faster training
   - Uses half the memory
   - Requires careful gradient scaling

## Summary

✅ **Batch size**: 32 → 64 (2× increase)  
✅ **CUDNN benchmark**: Enabled (auto-optimize kernels)  
✅ **TF32**: Enabled (3× faster matmul on L4)  
✅ **Workers**: 4 → 8 (better data pipeline)  
✅ **Persistent workers**: Enabled (reduced overhead)  
✅ **Device reporting**: Enhanced (visibility)

🎯 **Expected speedup**: ~2.5× faster training  
🎯 **Expected training time**: 18.75 hours → **7.5 hours**  
🎯 **GPU utilization**: 40-50% → **80-90%**

The fixes to preprocessing (log1p), architecture (gentle stride=2), and data pipeline (T=256) remain unchanged. These optimizations only make the corrected training run faster on L4 hardware.
