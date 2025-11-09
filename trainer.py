# Copyright Pathway Technology, Inc.

"""Trainer class for BDH model training."""

from contextlib import nullcontext
from datetime import datetime

import torch

try:
    import transformer_engine.pytorch as te
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False


class Trainer:
    """Encapsulates the training loop for BDH model."""
    
    def __init__(self, model, optimizer, config, device, dtype, scaler, fp8_recipe=None):
        """
        Initialize trainer.
        
        Args:
            model: BDH model to train
            optimizer: Optimizer for training
            config: Config object with training settings
            device: Device to train on
            dtype: Data type for training
            scaler: Gradient scaler for mixed precision
            fp8_recipe: Optional FP8 recipe for Transformer Engine
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.scaler = scaler
        self.fp8_recipe = fp8_recipe
        
        # Set up autocast context
        self.ctx = (
            torch.amp.autocast(device_type=device.type, dtype=self._get_pytorch_dtype())
            if "cuda" in device.type
            else nullcontext()
        )
        
        # Training state
        self.current_run_dir = None
        self.first_checkpoint_saved = False
        self.loss_acc = 0
        self.loss_steps = 0
    
    def _get_pytorch_dtype(self):
        """Convert dtype string to PyTorch dtype."""
        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        return dtypes[self.dtype]
    
    def train_step(self, x, y):
        """
        Execute a single training step.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Loss value for this step
        """
        # Use FP8 autocast if enabled
        if self.config.low_precision.use_fp8 and TRANSFORMER_ENGINE_AVAILABLE:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                with self.ctx:
                    logits, loss = self.model(x, y)
        else:
            with self.ctx:
                logits, loss = self.model(x, y)
        
        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss
    
    def should_log(self, step):
        """Check if should log at this step."""
        return step % self.config.training.log_freq == 0
    
    def should_checkpoint(self, step):
        """Check if should save checkpoint at this step."""
        return step % self.config.training.checkpoint_freq == 0 and step > 0
    
    def should_eval(self, step):
        """Check if should evaluate at this step."""
        return step % self.config.training.test_freq == 0 and step > 0
    
    def log_progress(self, step, max_iters):
        """Log training progress."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_loss = self.loss_acc.item() / self.loss_steps if self.loss_steps > 0 else 0
        print(f"[{timestamp}] Step: {step}/{max_iters} loss {avg_loss:.3}")
        self.loss_acc = 0
        self.loss_steps = 0
    
    def update_loss_accumulator(self, loss):
        """Update the loss accumulator."""
        self.loss_acc += loss
        self.loss_steps += 1
    
    def set_run_directory(self, run_dir):
        """Set the current run directory for checkpoints."""
        self.current_run_dir = run_dir
    
    def get_run_directory(self):
        """Get the current run directory."""
        return self.current_run_dir
    
    def mark_first_checkpoint_saved(self):
        """Mark that the first checkpoint has been saved."""
        self.first_checkpoint_saved = True
    
    def is_first_checkpoint(self):
        """Check if this is the first checkpoint."""
        return not self.first_checkpoint_saved
