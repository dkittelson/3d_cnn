#!/usr/bin/env python3
"""
Cache Cleanup Script for Colab
Run this before starting training to ensure no corrupted cached data exists.
"""

import shutil
from pathlib import Path

def clear_cache(cache_dir="./cache"):
    """
    Safely delete the cache directory and all its contents.
    
    Args:
        cache_dir: Path to cache directory (default: "./cache")
    """
    cache_path = Path(cache_dir)
    
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
            print(f"✓ Cache directory '{cache_dir}' deleted successfully")
            print(f"  Removed: {cache_path.absolute()}")
        except Exception as e:
            print(f"✗ Error deleting cache: {e}")
            return False
    else:
        print(f"✓ Cache directory '{cache_dir}' does not exist (already clean)")
    
    return True

if __name__ == "__main__":
    # For Colab: Run this cell before training
    clear_cache("./cache")
    print("\n🚀 Ready to start training with clean cache!")
