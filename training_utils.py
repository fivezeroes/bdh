# Copyright Pathway Technology, Inc.

"""Training utilities for BDH model training."""

import os
from datetime import datetime

import torch


def eval_model(model, device, max_new_tokens=100, top_k=3):
    """
    Evaluate model by generating text from a prompt.
    
    Args:
        model: BDH model to evaluate
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate
        top_k: Top-k sampling parameter
    """
    model.eval()

    prompt = torch.tensor(
        bytearray("The Capital of France is ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=max_new_tokens, top_k=top_k)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )

    print(ret_decoded)


def save_checkpoint(model, optimizer, step, loss, scaler, dtype, config, 
                    checkpoint_dir=None, current_run_dir=None, is_first=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step
        loss: Current loss value
        scaler: Gradient scaler (for mixed precision)
        dtype: Data type being used
        config: Config object
        checkpoint_dir: Directory to save checkpoint in (optional)
        current_run_dir: Current run directory (will be created if None on first save)
        is_first: Whether this is the first checkpoint (saves config file)
        
    Returns:
        Tuple of (checkpoint_path, updated_run_dir)
    """
    # Create run directory on first checkpoint if needed
    if checkpoint_dir is None:
        if current_run_dir is None:
            # First checkpoint - create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_run_dir = os.path.join(config.training.checkpoint_dir, f"run_{timestamp}")
            print(f"Creating run directory: {current_run_dir}")
        checkpoint_dir = current_run_dir
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    
    # Get the underlying model if it's been compiled
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.to_bdh_config(),
        'scaler_state_dict': scaler.state_dict() if dtype == "float16" else None,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save config file with the first checkpoint (from memory, not disk)
    if is_first:
        config_copy_path = os.path.join(checkpoint_dir, "config.yaml")
        config.save_to_yaml(config_copy_path)
        print(f"Config saved: {config_copy_path}")
    
    return checkpoint_path, current_run_dir


def load_checkpoint(checkpoint_path, model, optimizer, scaler, dtype, device):
    """
    Load model checkpoint and return starting step.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scaler: Gradient scaler to load state into
        dtype: Data type being used
        device: Device to load checkpoint on
        
    Returns:
        Starting step number (checkpoint step + 1)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get the underlying model if it's been compiled
    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if checkpoint.get('scaler_state_dict') is not None and dtype == "float16":
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_step = checkpoint['step'] + 1
    print(f"Resumed from step {checkpoint['step']} (continuing from step {start_step})")
    
    return start_step
