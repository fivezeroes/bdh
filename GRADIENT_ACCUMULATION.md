# Gradient Accumulation

## Overview

Gradient accumulation allows you to train with larger effective batch sizes while using smaller micro-batches. This is useful when:

- Your GPU memory is limited
- You want to use larger batch sizes for better convergence without running out of memory
- You need to simulate larger batch training on smaller hardware

## How It Works

Instead of updating the model parameters after every batch, gradient accumulation:

1. Processes multiple small batches (micro-batches)
2. Accumulates gradients from each micro-batch
3. Updates the model parameters once after N micro-batches

**Effective Batch Size = `batch_size` × `gradient_accumulation_steps`**

## Configuration

Edit `config.yaml` to set the gradient accumulation steps:

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size = 8 × 4 = 32
```

### Examples

| batch_size | gradient_accumulation_steps | Effective Batch Size |
|------------|----------------------------|---------------------|
| 1          | 8                          | 8                   |
| 2          | 4                          | 8                   |
| 4          | 4                          | 16                  |
| 8          | 4                          | 32                  |

## Memory vs. Effective Batch Size Trade-off

- **Lower `batch_size`**: Uses less GPU memory per step, but needs more accumulation steps
- **Higher `gradient_accumulation_steps`**: Achieves larger effective batch size without increasing memory

For example, if `batch_size=32` causes OOM (Out of Memory), you can use:
- `batch_size=8` with `gradient_accumulation_steps=4` (same effective batch size of 32)
- `batch_size=4` with `gradient_accumulation_steps=8` (same effective batch size of 32)

## Implementation Details

### Loss Scaling

The implementation automatically scales the loss by `1 / gradient_accumulation_steps` to ensure correct gradient magnitudes:

```python
loss = loss / self.gradient_accumulation_steps
loss.backward()
```

This ensures that averaged gradients from N micro-batches equal the gradients from one large batch.

### Optimizer Updates

The optimizer only steps every N micro-batches:

```python
self.accum_step += 1
if self.accum_step >= self.gradient_accumulation_steps:
    self.optimizer.step()
    self.optimizer.zero_grad()
    self.accum_step = 0
```

### Checkpoint Resumption

Gradient accumulation state is saved in checkpoints, so training can resume mid-accumulation without disruption.

## Learning Rate Considerations

When increasing effective batch size via gradient accumulation, you may need to adjust the learning rate:

- **Linear Scaling Rule**: When doubling batch size, consider doubling the learning rate
- **Example**: If you change from `batch_size=8` (no accumulation) to `batch_size=8, gradient_accumulation_steps=4` (effective batch size 32), you might scale learning rate by 4×

However, this is not automatic - you need to adjust `learning_rate` in `config.yaml` manually based on your training dynamics.

## Performance Notes

- **Computation**: Gradient accumulation does NOT reduce computation - you still process the same number of samples
- **Memory**: Reduces peak memory usage by using smaller batches
- **Speed**: May be slightly slower due to multiple forward/backward passes, but enables training that wouldn't otherwise fit in memory
- **Convergence**: Larger effective batch sizes often improve training stability and convergence

## Testing

Run the test suite to verify gradient accumulation works correctly:

```bash
source .venv/bin/activate
python test_gradient_accumulation.py
```

This verifies that:
1. Gradients from accumulated micro-batches match gradients from one large batch
2. The Trainer correctly manages accumulation state
3. Checkpoint save/restore works with accumulation state
