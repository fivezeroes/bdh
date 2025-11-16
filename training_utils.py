# Copyright Pathway Technology, Inc.

"""Training utilities for BDH model training."""

import os
from contextlib import nullcontext
from datetime import datetime

import torch

try:
    import transformer_engine.pytorch as te
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False


def compute_validation_loss(model, val_loader, device, config, num_batches=100):
    """
    Compute validation loss on a subset of validation data.
    
    Args:
        model: BDH model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        config: Config object with training settings
        num_batches: Number of batches to evaluate (subsampling for efficiency)
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    batches_evaluated = 0
    
    # Set up autocast context
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    pytorch_dtype = dtype_map.get(config.low_precision.compute_dtype, torch.float32)
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=pytorch_dtype)
        if "cuda" in device.type
        else nullcontext()
    )
    
    # Get FP8 recipe if needed
    fp8_recipe = None
    if config.low_precision.use_fp8 and TRANSFORMER_ENGINE_AVAILABLE:
        fp8_recipe = te.recipe.DelayedScaling(
            fp8_format=getattr(te.common.recipe.Format, config.fp8.format.upper(), te.common.recipe.Format.HYBRID),
            amax_history_len=config.fp8.amax_history_len,
            amax_compute_algo=config.fp8.amax_compute_algo,
        )
    
    with torch.no_grad():
        val_iter = iter(val_loader)
        for _ in range(min(num_batches, len(val_loader))):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass with FP8 if enabled
            if config.low_precision.use_fp8 and TRANSFORMER_ENGINE_AVAILABLE:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    with ctx:
                        _, loss = model(x, y)
            else:
                with ctx:
                    _, loss = model(x, y)
            
            total_loss += loss.item()
            batches_evaluated += 1
    
    model.train()
    
    if batches_evaluated == 0:
        return 0.0
    
    return total_loss / batches_evaluated


def eval_model(model, device, tokenizer=None, max_new_tokens=100, top_k=3):
    """
    Evaluate model by generating text from a prompt.
    
    Args:
        model: BDH model to evaluate
        device: Device to run on
        tokenizer: Tokenizer instance (if None, uses byte-level encoding)
        max_new_tokens: Maximum number of tokens to generate
        top_k: Top-k sampling parameter
    """
    model.eval()

    prompt_text = "The Capital of France is "
    
    # Encode prompt using tokenizer
    if tokenizer is not None:
        token_ids = tokenizer.encode(prompt_text)
        prompt = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # Fallback to byte-level encoding
        prompt = torch.tensor(
            bytearray(prompt_text, "utf-8"), dtype=torch.long, device=device
        ).unsqueeze(0)
    
    ret = model.generate(prompt, max_new_tokens=max_new_tokens, top_k=top_k)
    
    # Decode output using tokenizer
    if tokenizer is not None:
        ret_tokens = ret.to("cpu").squeeze(0).tolist()
        ret_decoded = tokenizer.decode(ret_tokens)
    else:
        # Fallback to byte-level decoding
        ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
            errors="backslashreplace"
        )

    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(ret_decoded)
    print("=" * 80)


def save_checkpoint(model, optimizer, step, loss, scaler, dtype, config, 
                    run_dir, is_first=False, tokenizer_config=None, accum_step=0, scheduler=None):
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
        run_dir: Run directory to save checkpoint in (shared with TensorBoard)
        is_first: Whether this is the first checkpoint (saves config file)
        tokenizer_config: Tokenizer config dict to save in checkpoint
        accum_step: Current gradient accumulation step (for resuming mid-accumulation)
        scheduler: Optional learning rate scheduler to save state
        
    Returns:
        checkpoint_path: Path to the saved checkpoint
    """
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, f"checkpoint_{step}.pt")
    
    # Get the underlying model if it's been compiled
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.to_bdh_config(),
        'scaler_state_dict': scaler.state_dict() if dtype == "float16" else None,
        'tokenizer_config': tokenizer_config,
        'accum_step': accum_step,  # Save accumulation state for proper resumption
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save config file with the first checkpoint (from memory, not disk)
    if is_first:
        config_copy_path = os.path.join(run_dir, "config.yaml")
        config.save_to_yaml(config_copy_path)
        print(f"Config saved: {config_copy_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scaler, dtype, device, scheduler=None):
    """
    Load model checkpoint and return starting step and accumulation state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scaler: Gradient scaler to load state into
        dtype: Data type being used
        device: Device to load checkpoint on
        scheduler: Optional learning rate scheduler to load state into
        
    Returns:
        Tuple of (starting_step, accum_step)
        - starting_step: Step number to continue from (checkpoint step + 1)
        - accum_step: Gradient accumulation step (0 if not in checkpoint)
        
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
    
    # Load scheduler state if available and scheduler is provided
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Scheduler state loaded")
    
    start_step = checkpoint['step'] + 1
    accum_step = checkpoint.get('accum_step', 0)  # Default to 0 for old checkpoints
    
    print(f"Resumed from step {checkpoint['step']} (continuing from step {start_step})")
    if accum_step > 0:
        print(f"Resuming mid-accumulation at accum_step {accum_step}")
    
    return start_step, accum_step
