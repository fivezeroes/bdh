# Learning Rate Scheduler Guide

## Overview

The BDH training system now supports adaptive learning rate scheduling to improve training convergence and performance. This feature is fully configurable, backward compatible (defaults to no scheduling), and integrates seamlessly with the existing training infrastructure.

## Features

- **Multiple scheduler types**: Cosine annealing, linear decay, exponential decay, reduce on plateau
- **Warmup support**: Optional warmup phase with linear or constant warmup
- **Validation-based scheduling**: ReduceLROnPlateau uses validation loss to adaptively reduce learning rate
- **TensorBoard integration**: Logs learning rate, validation loss, and optional scheduler internals
- **Checkpoint support**: Scheduler state is saved and restored from checkpoints
- **Gradient accumulation aware**: Scheduler steps only after optimizer steps (respects gradient accumulation)

## Configuration

### Basic Example (Cosine Annealing with Warmup)

```yaml
scheduler:
  type: "cosine"
  warmup_steps: 1000
  warmup_type: "linear"
  min_lr: 1e-6
  T_max: 99000  # max_iters (100000) - warmup_steps (1000)
```

### Scheduler Types

#### 1. None (Default - Constant Learning Rate)
```yaml
scheduler:
  type: "none"
```
- No learning rate adjustment
- Backward compatible with existing training
- Use when you want constant learning rate

#### 2. Cosine Annealing
```yaml
scheduler:
  type: "cosine"
  T_max: 100000  # REQUIRED - period of cosine annealing
  min_lr: 1e-6
  warmup_steps: 1000
  warmup_type: "linear"
```
- Smoothly decreases LR following cosine curve
- **T_max is REQUIRED** - typically `max_iters - warmup_steps`
- Good for long training runs and fine-tuning
- Learning rate oscillates between initial LR and min_lr

#### 3. Linear Decay
```yaml
scheduler:
  type: "linear"
  min_lr: 1e-6
  warmup_steps: 1000
  warmup_type: "linear"
```
- Linearly decreases LR from initial to min_lr
- Simple decay strategy
- Good for straightforward training schedules

#### 4. Exponential Decay
```yaml
scheduler:
  type: "exponential"
  gamma: 0.95  # LR *= gamma each step
  warmup_steps: 1000
  warmup_type: "linear"
```
- LR multiplied by gamma each optimizer step
- Aggressive decay (e.g., gamma=0.95 = 5% reduction per step)
- Good for rapid learning rate reduction

#### 5. Reduce on Plateau
```yaml
scheduler:
  type: "plateau"
  patience: 10  # Evaluations without improvement before reducing
  factor: 0.1  # Multiply LR by this when reducing
  threshold: 1e-4  # Minimum change to count as improvement
  min_lr: 1e-6
  warmup_steps: 1000
  warmup_type: "linear"
```
- Reduces LR when validation loss stops improving
- Adaptive based on training progress
- Requires validation loss computation (uses `val_batches` for efficiency)
- Good for avoiding overfitting and adaptive training

### Warmup Configuration

All scheduler types support an optional warmup phase:

```yaml
scheduler:
  warmup_steps: 1000  # Number of steps to warm up (0 = no warmup)
  warmup_type: "linear"  # "linear" or "constant"
```

- **Linear warmup**: Gradually increases LR from ~0 to initial LR over warmup_steps
- **Constant warmup**: Immediately jumps to initial LR for warmup_steps, then starts main schedule
- Recommended: 1000-5000 steps for large models
- Helps stabilize early training

### Validation Configuration

For ReduceLROnPlateau scheduler and validation loss tracking:

```yaml
dataset:
  val_split: 0.1  # Fraction of data files for validation (0.0-1.0)
  val_batches: 100  # Number of validation batches to evaluate
```

- `val_split`: Controls train/validation split (default: 0.1 = 10% validation)
- `val_batches`: Subsamples validation data for efficiency (default: 100 batches)
- Validation loss is computed at `test_freq` intervals

### TensorBoard Logging

```yaml
tensorboard:
  enabled: true
  log_scheduler_state: false  # Log scheduler internals (can be expensive)
```

The following metrics are logged to TensorBoard:
- `Learning_Rate`: Current learning rate at each logging step
- `Loss/validation`: Validation loss (when computed)
- `Scheduler/*`: Internal scheduler state (if `log_scheduler_state: true`)

## Usage Examples

### Example 1: Cosine Annealing for Long Training

Ideal for training from scratch with 1M iterations:

```yaml
training:
  max_iters: 1000000
  learning_rate: 3.0e-4

scheduler:
  type: "cosine"
  warmup_steps: 5000
  warmup_type: "linear"
  min_lr: 3.0e-6  # 1% of initial LR
  T_max: 995000  # 1000000 - 5000
```

### Example 2: Plateau Scheduler for Adaptive Training

Automatically reduces LR when validation loss plateaus:

```yaml
training:
  max_iters: 100000
  learning_rate: 3.0e-4
  test_freq: 500  # Evaluate every 500 steps

scheduler:
  type: "plateau"
  warmup_steps: 1000
  warmup_type: "linear"
  patience: 5  # Reduce after 5 evaluations without improvement
  factor: 0.5  # Cut LR in half
  threshold: 1e-4
  min_lr: 1e-6

dataset:
  val_split: 0.1
  val_batches: 100
```

### Example 3: Simple Linear Decay

Straightforward linear decay without bells and whistles:

```yaml
scheduler:
  type: "linear"
  warmup_steps: 0  # No warmup
  min_lr: 1e-6
```

### Example 4: No Scheduler (Backward Compatible)

Maintains constant learning rate (default behavior):

```yaml
scheduler:
  type: "none"
```

## Implementation Details

### Gradient Accumulation

The scheduler is gradient-accumulation aware:
- Scheduler steps **only** after optimizer steps
- With `gradient_accumulation_steps: 8`, the scheduler steps every 8 micro-batches
- This ensures the learning rate schedule aligns with effective batches, not micro-batches

### Checkpoint Compatibility

- Scheduler state is automatically saved in checkpoints
- Old checkpoints without scheduler state are fully compatible
- When resuming, the scheduler continues from its saved state

### Validation Loss Computation

For ReduceLROnPlateau scheduler:
- Validation loss is computed at `test_freq` intervals
- Only `val_batches` batches are evaluated (subsampling for efficiency)
- Uses same FP8/mixed precision settings as training
- Validation data comes from the `val_split` portion of parquet files

## Monitoring Training

Use TensorBoard to monitor the learning rate schedule:

```bash
tensorboard --logdir=runs
```

Look for:
- `Learning_Rate`: Should follow your expected schedule
- `Loss/train` vs `Loss/validation`: Check for overfitting
- `Scheduler/*`: Internal state (if enabled)

## Common Issues

### Issue: "T_max is required for cosine scheduler"

**Solution**: Set `T_max` in your config:
```yaml
scheduler:
  type: "cosine"
  T_max: 95000  # Usually max_iters - warmup_steps
```

### Issue: Scheduler not stepping

**Cause**: Likely the scheduler type is "none" or validation loss not being computed for plateau.

**Solution**: 
- Check `scheduler.type` is not "none"
- For plateau scheduler, ensure `test_freq` is set and validation data is available

### Issue: Learning rate not changing during warmup

**Cause**: Warmup steps may be too short or warmup hasn't started yet.

**Solution**: 
- Check TensorBoard `Learning_Rate` graph
- Ensure `warmup_steps > 0`
- Verify training has progressed past step 0

## Best Practices

1. **Start simple**: Begin with `type: "none"` to establish a baseline, then experiment with schedulers
2. **Tune warmup**: Use 0.5-1% of total iterations for warmup (e.g., 1000-5000 steps for 1M iterations)
3. **Monitor validation**: Always track validation loss to detect overfitting
4. **Cosine for long runs**: Use cosine annealing for training from scratch
5. **Plateau for fine-tuning**: Use ReduceLROnPlateau when fine-tuning or adapting a model
6. **Log scheduler state**: Enable `log_scheduler_state: true` during debugging, disable for production

## References

- PyTorch Learning Rate Schedulers: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- Cosine Annealing: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2016)
- Warmup: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Goyal et al., 2017)
