# Copyright Pathway Technology, Inc.

"""Configuration loader for BDH training."""

import os
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    block_size: int = 512
    batch_size: int = 8
    max_iters: int = 3000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    log_freq: int = 100
    test_freq: int = 500
    checkpoint_freq: int = 500
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    debug: bool = False


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    parquet_dir: str = "/Volumes/Data/fineweb/data/CC-MAIN-2025-26"
    max_cached_files: int = 5


@dataclass
class LowPrecisionConfig:
    """Low-precision training configuration."""
    use_fp8: bool = False
    use_4bit: bool = True
    use_4bit_optimizer: bool = False
    quantization_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class FP8Config:
    """FP8-specific configuration."""
    format: str = "hybrid"
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"


@dataclass
class TensorBoardConfig:
    """TensorBoard configuration."""
    enabled: bool = True
    log_dir: str = "runs"
    log_gradients: bool = False
    log_weights: bool = False


@dataclass
class Config:
    """Complete configuration for BDH training."""
    model: ModelConfig
    training: TrainingConfig
    dataloader: DataLoaderConfig
    dataset: DatasetConfig
    low_precision: LowPrecisionConfig
    fp8: FP8Config
    tensorboard: TensorBoardConfig

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            dataloader=DataLoaderConfig(**config_dict.get('dataloader', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            low_precision=LowPrecisionConfig(**config_dict.get('low_precision', {})),
            fp8=FP8Config(**config_dict.get('fp8', {})),
            tensorboard=TensorBoardConfig(**config_dict.get('tensorboard', {})),
        )
    
    @classmethod
    def from_defaults(cls) -> "Config":
        """Create configuration with default values."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            dataloader=DataLoaderConfig(),
            dataset=DatasetConfig(),
            low_precision=LowPrecisionConfig(),
            fp8=FP8Config(),
            tensorboard=TensorBoardConfig(),
        )
    
    def to_bdh_config(self):
        """Convert to BDHConfig for the model."""
        import bdh
        return bdh.BDHConfig(
            n_layer=self.model.n_layer,
            n_embd=self.model.n_embd,
            dropout=self.model.dropout,
            n_head=self.model.n_head,
            mlp_internal_dim_multiplier=self.model.mlp_internal_dim_multiplier,
            vocab_size=self.model.vocab_size,
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for YAML serialization."""
        import dataclasses
        return {
            'model': dataclasses.asdict(self.model),
            'training': dataclasses.asdict(self.training),
            'dataloader': dataclasses.asdict(self.dataloader),
            'dataset': dataclasses.asdict(self.dataset),
            'low_precision': dataclasses.asdict(self.low_precision),
            'fp8': dataclasses.asdict(self.fp8),
            'tensorboard': dataclasses.asdict(self.tensorboard),
        }
    
    def save_to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
