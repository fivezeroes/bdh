# Configuration File Usage

## Overview

The training configuration has been refactored to use a YAML configuration file instead of hardcoded constants. This makes it easier to manage different training configurations and experiment with different hyperparameters.

## Checkpoint Management

Each training run automatically creates a timestamped directory to organize checkpoints:

### Directory Structure
```
checkpoints/
├── run_20251108_143022/
│   ├── config.yaml          # Copy of the config used for this run
│   ├── checkpoint_500.pt
│   ├── checkpoint_1000.pt
│   └── checkpoint_1500.pt
└── run_20251108_150345/
    ├── config.yaml
    ├── checkpoint_500.pt
    └── ...
```

### Features
- **Timestamped directories**: Each run gets a unique `run_YYYYMMDD_HHMMSS` folder
- **Lazy creation**: Run directory is only created when the first checkpoint is saved (prevents empty folders from aborted runs)
- **Config preservation**: The config file is automatically saved with the first checkpoint
- **Resume-friendly**: When resuming, checkpoints continue in the same directory
- **Reproducibility**: Each run's config is saved, making it easy to reproduce experiments

## Configuration Files

### `config.yaml`
The main configuration file containing all training parameters, organized into sections:
- **model**: Model architecture configuration (n_layer, n_embd, dropout, etc.)
- **training**: Training hyperparameters (batch_size, learning_rate, max_iters, etc.)
- **dataloader**: DataLoader settings (num_workers, pin_memory, prefetch_factor)
- **dataset**: Dataset configuration (parquet_dir, max_cached_files)
- **low_precision**: Low-precision training settings (use_fp8, use_4bit, compute_dtype, etc.)
- **fp8**: FP8-specific configuration (format, amax_history_len, amax_compute_algo)

### `config.py`
Python module for loading and validating configuration:
- Defines dataclasses for each configuration section
- Provides `Config.from_yaml()` to load configuration from file
- Includes `Config.to_bdh_config()` to convert to BDHConfig for the model

## Usage

### Training with default config
```bash
python train.py
```

### Training with custom config
```bash
python train.py --config my_config.yaml
```

### Creating a custom config
Copy `config.yaml` to a new file and modify the parameters:
```bash
cp config.yaml my_experiment.yaml
# Edit my_experiment.yaml with your parameters
python train.py --config my_experiment.yaml
```

### Resuming Training
To resume from a checkpoint, edit your config file or create a new one:
```yaml
training:
  resume_from_checkpoint: "checkpoints/run_20251108_143022/checkpoint_1000.pt"
```

Then run:
```bash
python train.py --config my_config.yaml
```

The training will continue in the same `run_YYYYMMDD_HHMMSS` directory.

## Configuration Parameters

### Model Configuration
- `n_layer`: Number of transformer layers (default: 6)
- `n_embd`: Embedding dimension (default: 256)
- `dropout`: Dropout rate (default: 0.1)
- `n_head`: Number of attention heads (default: 4)
- `mlp_internal_dim_multiplier`: MLP internal dimension multiplier (default: 128)
- `vocab_size`: Vocabulary size (default: 256)

### Training Configuration
- `block_size`: Sequence length (default: 512)
- `batch_size`: Batch size (default: 8)
- `max_iters`: Maximum training iterations (default: 3000)
- `learning_rate`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay for AdamW (default: 0.1)
- `log_freq`: Logging frequency (default: 100)
- `test_freq`: Test/evaluation frequency (default: 500)
- `checkpoint_freq`: Checkpoint save frequency (default: 500)
- `checkpoint_dir`: Directory to save checkpoints (default: "checkpoints")
- `resume_from_checkpoint`: Path to checkpoint to resume from (default: null)
- `debug`: Enable debug mode (default: false)

### Low-Precision Training
- `use_fp8`: Enable FP8 training for Hopper/Blackwell GPUs (default: false)
- `use_4bit`: Enable 4-bit training preparation (default: true)
- `use_4bit_optimizer`: Use 4-bit AdamW optimizer (default: false)
- `quantization_type`: "nf4" or "fp4" (default: "nf4")
- `use_double_quant`: Use nested quantization (default: true)
- `compute_dtype`: "bfloat16", "float16", or "float32" (default: "bfloat16")

## Requirements

Install the required dependency for YAML support:
```bash
pip install pyyaml
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Migration from Old Code

The old hardcoded configuration constants have been replaced with the config object:
- `BLOCK_SIZE` → `config.training.block_size`
- `BATCH_SIZE` → `config.training.batch_size`
- `LEARNING_RATE` → `config.training.learning_rate`
- `USE_FP8` → `config.low_precision.use_fp8`
- etc.

Checkpoints saved with the new code will include the full configuration, making it easier to reproduce experiments.
