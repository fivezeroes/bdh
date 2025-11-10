# TensorBoard Integration

This project includes full TensorBoard support for visualizing training metrics and diagnostics.

## Installation

TensorBoard is included in the requirements:

```bash
pip install tensorboard
```

## Configuration

TensorBoard settings are configured in `config.yaml`:

```yaml
tensorboard:
  enabled: true  # Enable TensorBoard logging
  log_dir: "runs"  # Directory for TensorBoard logs
  log_gradients: false  # Log gradient histograms (can be expensive)
  log_weights: false  # Log weight histograms (can be expensive)
```

### Configuration Options

- **enabled**: Set to `true` to enable TensorBoard logging, `false` to disable
- **log_dir**: Base directory where TensorBoard logs will be stored. Each training run creates a timestamped subdirectory
- **log_gradients**: When `true`, logs gradient histograms for all model parameters. Note: This can slow down training and increase disk usage
- **log_weights**: When `true`, logs weight/parameter histograms. Note: This can slow down training and increase disk usage

## What Gets Logged

The trainer automatically logs the following metrics:

### Scalars
- **Loss/train**: Average training loss at each logging interval
- **Learning_Rate**: Current learning rate from the optimizer

### Optional Histograms (when enabled)
- **Gradients/**: Gradient distributions for each parameter (if `log_gradients: true`)
- **Weights/**: Weight/parameter distributions (if `log_weights: true`)

## Viewing TensorBoard

### Start TensorBoard

After starting training, TensorBoard will print instructions. To view logs:

```bash
tensorboard --logdir=runs
```

Then open your browser to: `http://localhost:6006`

### Multiple Runs

TensorBoard automatically organizes multiple training runs by timestamp:

```
runs/
├── run_20251109-143022/
├── run_20251109-145533/
└── run_20251109-152144/
```

You can compare multiple runs side-by-side in the TensorBoard interface.

### Remote Training

If training on a remote server, use SSH port forwarding:

```bash
ssh -L 6006:localhost:6006 user@remote-server
```

Then start TensorBoard on the remote server and access it locally at `http://localhost:6006`.

## Custom Logging

The Trainer class provides methods for custom logging:

```python
# Log a scalar value
trainer.log_scalar('Custom/metric', value, step)

# Log text
trainer.log_text('Generated/sample', generated_text, step)
```

## Performance Considerations

- Basic scalar logging (loss, learning rate) has minimal overhead
- Histogram logging (`log_gradients`, `log_weights`) can slow down training by 10-20%
- TensorBoard log files grow over time; clean up old runs periodically

## Disabling TensorBoard

To disable TensorBoard logging, set in `config.yaml`:

```yaml
tensorboard:
  enabled: false
```

Training will continue normally without TensorBoard overhead.
