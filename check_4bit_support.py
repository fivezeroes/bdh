#!/usr/bin/env python3
"""
Example script demonstrating 4-bit training with BDH.

This script shows how to configure and run 4-bit training with different
memory optimization levels.
"""

import os
import sys

# Check if bitsandbytes is available
try:
    import bitsandbytes as bnb
    print("✓ bitsandbytes is installed")
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    print("✗ bitsandbytes is NOT installed")
    print("\nTo enable 4-bit training, install bitsandbytes:")
    print("  pip install bitsandbytes")
    print("\nNote: Requires CUDA-capable GPU")
    BITSANDBYTES_AVAILABLE = False
    sys.exit(1)

import torch

# Check CUDA availability
if not torch.cuda.is_available():
    print("\n✗ CUDA is not available")
    print("bitsandbytes requires a CUDA-capable GPU")
    print(f"Current device: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')}")
    sys.exit(1)
else:
    print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")

print("\n" + "="*60)
print("4-bit Training Configuration Examples")
print("="*60)

print("\n1. Standard Training (Baseline)")
print("-" * 60)
print("""
USE_4BIT = False
USE_4BIT_OPTIMIZER = False
COMPUTE_DTYPE = "float32"

Memory Usage: ~100% (baseline)
Training Speed: Baseline
Recommended For: Small models, abundant GPU memory
""")

print("\n2. 4-bit Optimizer Only (Recommended)")
print("-" * 60)
print("""
USE_4BIT = False
USE_4BIT_OPTIMIZER = True
COMPUTE_DTYPE = "bfloat16"

Memory Usage: ~50-60% of baseline
Training Speed: Similar to baseline
Recommended For: Most use cases, good balance of memory/speed
Best Feature: Works with any model architecture
""")

print("\n3. Full 4-bit Training")
print("-" * 60)
print("""
USE_4BIT = True
USE_4BIT_OPTIMIZER = True
COMPUTE_DTYPE = "bfloat16"
USE_DOUBLE_QUANT = True

Memory Usage: ~40-50% of baseline
Training Speed: Slightly faster due to mixed precision
Recommended For: Large models, limited GPU memory
Note: Best memory savings
""")

print("\n" + "="*60)
print("Quick Start")
print("="*60)
print("""
To enable 4-bit optimizer training, edit train.py:

1. Open train.py
2. Find the "4-bit training configuration" section
3. Set: USE_4BIT_OPTIMIZER = True
4. Optionally set: COMPUTE_DTYPE = "bfloat16"
5. Run: python train.py

That's it! You'll see ~50% memory reduction in optimizer states.
""")

print("\n" + "="*60)
print("Memory Estimation")
print("="*60)

# Simple memory estimation
def estimate_memory(num_params_millions, batch_size=32, seq_len=512, use_4bit_opt=False, mixed_precision=False):
    """Rough memory estimation for training."""
    
    # Model parameters (MB)
    param_bytes = num_params_millions * 1_000_000 * 4  # FP32
    if mixed_precision:
        param_bytes = param_bytes / 2  # BF16/FP16
    
    # Optimizer states (usually 2x params for Adam)
    opt_bytes = param_bytes * 2
    if use_4bit_opt:
        opt_bytes = opt_bytes / 4  # 8-bit quantization reduces to ~1/4
    
    # Activations and gradients (rough estimate)
    activation_bytes = batch_size * seq_len * 512 * 4 * 12  # Rough estimate for 12 layers
    
    total_mb = (param_bytes + opt_bytes + activation_bytes) / (1024 * 1024)
    
    return {
        'params_mb': param_bytes / (1024 * 1024),
        'optimizer_mb': opt_bytes / (1024 * 1024),
        'activations_mb': activation_bytes / (1024 * 1024),
        'total_mb': total_mb,
        'total_gb': total_mb / 1024
    }

configs = [
    ("Standard (FP32)", False, False),
    ("4-bit Optimizer", True, False),
    ("4-bit Opt + Mixed Precision", True, True),
]

print("\nMemory estimates for 100M parameter model:")
print("(Batch size: 32, Sequence length: 512)\n")

baseline_total = None
for name, use_4bit, mixed_prec in configs:
    mem = estimate_memory(100, use_4bit_opt=use_4bit, mixed_precision=mixed_prec)
    
    if baseline_total is None:
        baseline_total = mem['total_gb']
        savings = 0
    else:
        savings = ((baseline_total - mem['total_gb']) / baseline_total) * 100
    
    print(f"{name}:")
    print(f"  Total: {mem['total_gb']:.2f} GB (savings: {savings:.1f}%)")
    print(f"    - Parameters: {mem['params_mb']:.0f} MB")
    print(f"    - Optimizer: {mem['optimizer_mb']:.0f} MB")
    print(f"    - Activations: {mem['activations_mb']:.0f} MB")
    print()

print("\n" + "="*60)
print("For more details, see: 4BIT_TRAINING.md")
print("="*60)
