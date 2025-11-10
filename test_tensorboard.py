#!/usr/bin/env python3
# Copyright Pathway Technology, Inc.

"""Test script to verify TensorBoard integration."""

import torch
import config as cfg
from trainer import Trainer


def test_tensorboard_integration():
    """Test that TensorBoard integration works correctly."""
    
    # Load config
    config = cfg.Config.from_yaml('config.yaml')
    
    # Check TensorBoard config exists
    assert hasattr(config, 'tensorboard'), "TensorBoard config missing"
    assert hasattr(config.tensorboard, 'enabled'), "TensorBoard enabled flag missing"
    assert hasattr(config.tensorboard, 'log_dir'), "TensorBoard log_dir missing"
    
    print("✓ TensorBoard configuration loaded successfully")
    print(f"  Enabled: {config.tensorboard.enabled}")
    print(f"  Log directory: {config.tensorboard.log_dir}")
    print(f"  Log gradients: {config.tensorboard.log_gradients}")
    print(f"  Log weights: {config.tensorboard.log_weights}")
    
    # Create a simple mock model and optimizer
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    device = torch.device("cpu")
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.GradScaler('cpu', enabled=False)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        dtype='float32',
        scaler=scaler,
        fp8_recipe=None
    )
    
    print("\n✓ Trainer initialized successfully")
    
    # Test that writer exists if TensorBoard is enabled
    if config.tensorboard.enabled:
        assert trainer.writer is not None, "TensorBoard writer should be initialized"
        print("✓ TensorBoard SummaryWriter initialized")
        
        # Test logging methods
        trainer.log_scalar('test/metric', 1.0, 0)
        print("✓ Scalar logging works")
        
        trainer.log_text('test/text', 'Test message', 0)
        print("✓ Text logging works")
        
        # Clean up
        trainer.close()
        print("✓ TensorBoard writer closed successfully")
    else:
        assert trainer.writer is None, "TensorBoard writer should be None when disabled"
        print("✓ TensorBoard writer correctly disabled")
    
    print("\n✅ All TensorBoard integration tests passed!")


if __name__ == "__main__":
    test_tensorboard_integration()
