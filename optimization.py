# Copyright Pathway Technology, Inc.

"""Optimization utilities including 4-bit and FP8 support for BDH training."""

import torch

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False


def prepare_model_for_4bit_training(model, config):
    """
    Prepare model for 4-bit training by converting parameters to appropriate dtypes
    and enabling gradient checkpointing if available.
    
    Note: This model uses nn.Parameter instead of nn.Linear, so we can't directly
    use bitsandbytes quantization. However, we can still benefit from:
    1. 4-bit optimizers (AdamW8bit)
    2. Mixed precision training
    3. Gradient checkpointing
    
    Args:
        model: Model to prepare
        config: Config object with low_precision settings
        
    Returns:
        Prepared model
    """
    if not BITSANDBYTES_AVAILABLE:
        print("Warning: bitsandbytes not available, skipping 4-bit preparation")
        return model
    
    # Cast parameters to the compute dtype for better performance
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype_torch = compute_dtype_map.get(config.low_precision.compute_dtype, torch.bfloat16)
    
    print(f"Preparing model for 4-bit training with compute dtype: {config.low_precision.compute_dtype}")
    
    # For models with nn.Parameter, we focus on optimizer quantization
    # and ensure proper dtype handling
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(compute_dtype_torch)
    
    return model


def create_optimizer(model, config):
    """
    Create optimizer with optional 4-bit quantization.
    
    Args:
        model: Model to create optimizer for
        config: Config object with training and low_precision settings
        
    Returns:
        Optimizer instance (either AdamW8bit or standard AdamW)
    """
    if config.low_precision.use_4bit_optimizer and BITSANDBYTES_AVAILABLE:
        print("Using 4-bit AdamW optimizer (AdamW8bit)")
        # Use 8-bit optimizer which significantly reduces memory usage
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        print("Using standard AdamW optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    
    return optimizer


def create_fp8_recipe(config):
    """
    Create FP8 recipe for Transformer Engine.
    
    Args:
        config: Config object with fp8 settings
        
    Returns:
        FP8 recipe if Transformer Engine is available, None otherwise
    """
    if not TRANSFORMER_ENGINE_AVAILABLE:
        return None
    
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        interval=1,
        fp8_format=recipe.Format.HYBRID if config.fp8.format == "hybrid" else recipe.Format.E4M3,
        amax_history_len=config.fp8.amax_history_len,
        amax_compute_algo=config.fp8.amax_compute_algo,
    )
    
    return fp8_recipe


def check_low_precision_requirements(config, device):
    """
    Check if low precision training requirements are met.
    
    Args:
        config: Config object with low_precision and fp8 settings
        device: Device being used for training
        
    Raises:
        SystemExit: If requirements are not met but features are enabled
    """
    # Check FP8 configuration
    if config.low_precision.use_fp8:
        if not TRANSFORMER_ENGINE_AVAILABLE:
            print("ERROR: FP8 training requested but transformer_engine is not installed!")
            print("Please install it with: pip install transformer-engine")
            exit(1)
        if device.type != "cuda":
            print("ERROR: FP8 training requires a CUDA GPU (Hopper H100 or Blackwell)")
            exit(1)
        print(f"FP8 training configuration:")
        print(f"  - FP8_FORMAT: {config.fp8.format}")
        print(f"  - FP8_AMAX_HISTORY_LEN: {config.fp8.amax_history_len}")
        print(f"  - FP8_AMAX_COMPUTE_ALGO: {config.fp8.amax_compute_algo}")

    # Check 4-bit configuration
    if config.low_precision.use_4bit or config.low_precision.use_4bit_optimizer:
        if not BITSANDBYTES_AVAILABLE:
            print("ERROR: 4-bit training requested but bitsandbytes is not installed!")
            print("Please install it with: pip install bitsandbytes")
            exit(1)
        print(f"4-bit training configuration:")
        print(f"  - USE_4BIT: {config.low_precision.use_4bit}")
        print(f"  - USE_4BIT_OPTIMIZER: {config.low_precision.use_4bit_optimizer}")
        print(f"  - QUANTIZATION_TYPE: {config.low_precision.quantization_type}")
        print(f"  - COMPUTE_DTYPE: {config.low_precision.compute_dtype}")
        print(f"  - USE_DOUBLE_QUANT: {config.low_precision.use_double_quant}")


def get_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to create scheduler for
        config: Config object with scheduler settings
        
    Returns:
        Learning rate scheduler instance or None if scheduler type is "none"
        
    Raises:
        ValueError: If scheduler configuration is invalid
    """
    scheduler_type = config.scheduler.type.lower()
    
    if scheduler_type == "none":
        return None
    
    # Validate required parameters for specific scheduler types
    if scheduler_type == "cosine" and config.scheduler.T_max is None:
        raise ValueError("T_max is required for cosine scheduler. Set scheduler.T_max in config (typically max_iters - warmup_steps)")
    
    # Create warmup scheduler if warmup_steps > 0
    warmup_scheduler = None
    if config.scheduler.warmup_steps > 0:
        if config.scheduler.warmup_type == "linear":
            # Linear warmup from 0 to initial LR
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-10,  # Start from very small LR
                end_factor=1.0,  # End at initial LR
                total_iters=config.scheduler.warmup_steps
            )
        elif config.scheduler.warmup_type == "constant":
            # Constant warmup (immediate jump to initial LR)
            warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=config.scheduler.warmup_steps
            )
        else:
            raise ValueError(f"Invalid warmup_type: {config.scheduler.warmup_type}. Options: 'linear', 'constant'")
    
    # Create main scheduler
    main_scheduler = None
    if scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.min_lr
        )
        print(f"Created CosineAnnealingLR scheduler (T_max={config.scheduler.T_max}, min_lr={config.scheduler.min_lr})")
    
    elif scheduler_type == "linear":
        # Linear decay from initial LR to min_lr over max_iters
        # Note: This assumes the scheduler steps once per optimizer step
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.scheduler.min_lr / config.training.learning_rate if config.training.learning_rate > 0 else 0.0,
            total_iters=config.training.max_iters - config.scheduler.warmup_steps
        )
        print(f"Created LinearLR scheduler (min_lr={config.scheduler.min_lr})")
    
    elif scheduler_type == "exponential":
        main_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler.gamma
        )
        print(f"Created ExponentialLR scheduler (gamma={config.scheduler.gamma})")
    
    elif scheduler_type == "plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # Reduce LR when validation loss stops decreasing
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            threshold=config.scheduler.threshold,
            min_lr=config.scheduler.min_lr
        )
        print(f"Created ReduceLROnPlateau scheduler (patience={config.scheduler.patience}, factor={config.scheduler.factor})")
    
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}. Options: 'none', 'cosine', 'linear', 'exponential', 'plateau'")
    
    # Combine warmup and main scheduler if warmup is enabled
    if warmup_scheduler is not None and main_scheduler is not None:
        # Use SequentialLR to chain warmup -> main scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.scheduler.warmup_steps]
        )
        print(f"Chained warmup ({config.scheduler.warmup_type}) for {config.scheduler.warmup_steps} steps -> {scheduler_type} scheduler")
        return scheduler
    elif warmup_scheduler is not None:
        # Only warmup, no main scheduler
        return warmup_scheduler
    else:
        # Only main scheduler, no warmup
        return main_scheduler

