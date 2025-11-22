# Copyright Pathway Technology, Inc.

"""Main training script for BDH model."""

import argparse
import os
import dataclasses

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import bdh
import config as cfg
import torch
from torch.utils.data import DataLoader

# Import tokenizer support
from tokenizers import create_tokenizer

# Import refactored modules
from data import ParquetDataset, fetch_parquet_files, split_parquet_files
from optimization import (
    check_low_precision_requirements,
    create_fp8_recipe,
    create_optimizer,
    prepare_model_for_4bit_training,
    get_scheduler,
)
from trainer import Trainer
from training_utils import eval_model, load_checkpoint, save_checkpoint, compute_validation_loss
from logging_utils import setup_logging, teardown_logging

def main():
    """Main training function."""
    # Load configuration
    parser = argparse.ArgumentParser(description='Train BDH model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = cfg.Config.from_yaml(args.config)
    
    # Determine run directory early so we can set up logging
    run_dir = None
    if config.training.resume_from_checkpoint is not None:
        # Resuming from checkpoint - use the same directory as the checkpoint
        run_dir = os.path.dirname(config.training.resume_from_checkpoint)
    else:
        # New training run - create timestamped directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.training.runs_dir, f"run_{timestamp}")
    
    # Set up logging early to capture all output
    loggers = setup_logging(run_dir)

    # Initialize tokenizer
    print(f"Initializing tokenizer: {config.tokenizer.type}")
    tokenizer = create_tokenizer(
        tokenizer_type=config.tokenizer.type,
        tokenizer_name=config.tokenizer.name
    )
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Update model vocab_size from tokenizer if not explicitly set
    if config.model.vocab_size != tokenizer.vocab_size:
        print(f"Updating model vocab_size from {config.model.vocab_size} to {tokenizer.vocab_size}")
        config.model.vocab_size = tokenizer.vocab_size
    
    # Create tokenizer config dict for checkpointing
    tokenizer_config_dict = dataclasses.asdict(config.tokenizer)

    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    dtype = (
        "bfloat16"
        if (device.type == "cuda" and torch.cuda.is_bf16_supported()) or device.type == "mps"
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    # Setup gradient scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == "float16"))
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.fp32_precision = 'tf32'

    # Setup FP8 recipe if using Transformer Engine
    fp8_recipe = create_fp8_recipe(config) if config.low_precision.use_fp8 else None

    # Print configuration info
    print(f"Using device: {device} with dtype {dtype}")
    if fp8_recipe:
        print(f"FP8 training enabled with format: {config.fp8.format}")
    
    # Print gradient accumulation info
    effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
    print(f"Batch size: {config.training.batch_size}, Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    # Fetch and split data
    parquet_files = fetch_parquet_files(config.dataset.parquet_dir)
    train_files, test_files = split_parquet_files(parquet_files, train_ratio=1.0 - config.dataset.val_split)

    # Check low precision requirements
    check_low_precision_requirements(config, device)
    
    # Create datasets and dataloaders
    # Index files once in main process to avoid repeated indexing in workers
    print(f"Pre-indexing dataset in main process...")
    temp_dataset = ParquetDataset(train_files, config.training.block_size, tokenizer=tokenizer)
    file_lengths = temp_dataset.file_lengths
    
    # Create actual dataset with pre-computed file lengths
    train_dataset = ParquetDataset(
        train_files, 
        config.training.block_size, 
        tokenizer=tokenizer,
        file_lengths=file_lengths
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=config.dataloader.prefetch_factor,  # Prefetch batches per worker
    )
    
    # Create validation dataset and loader
    print(f"Creating validation dataset...")
    val_dataset = ParquetDataset(
        test_files,
        config.training.block_size,
        tokenizer=tokenizer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=True,
        prefetch_factor=config.dataloader.prefetch_factor,
    )

    # Create and prepare model
    model = bdh.BDH(config.to_bdh_config()).to(device)
    
    # Prepare model for 4-bit training if enabled
    if config.low_precision.use_4bit:
        model = prepare_model_for_4bit_training(model, config)
    
    # Compile model if enabled and compatible with device
    # Note: MPS backend has limited threadgroup memory and may fail with torch.compile
    if config.training.compile_model:
        if device.type == "mps":
            print("Warning: torch.compile() on MPS may cause memory errors. Consider setting compile_model: false in config.")
            print("Attempting compilation anyway...")
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    else:
        print("Model compilation disabled (running in eager mode)")
    
    # Create optimizer with optional 4-bit support
    optimizer = create_optimizer(model, config)
    
    # Create learning rate scheduler
    scheduler = get_scheduler(optimizer, config)
    if scheduler is not None:
        print(f"Learning rate scheduler created: {config.scheduler.type}")
        if config.scheduler.warmup_steps > 0:
            print(f"  Warmup: {config.scheduler.warmup_steps} steps ({config.scheduler.warmup_type})")
    else:
        print("No learning rate scheduler (type='none')")

    # Create trainer with run directory and scheduler
    trainer = Trainer(model, optimizer, config, device, dtype, scaler, fp8_recipe, run_dir=run_dir, scheduler=scheduler)

    # Resume from checkpoint if specified
    start_step = 1
    if config.training.resume_from_checkpoint is not None:
        start_step, accum_step = load_checkpoint(
            config.training.resume_from_checkpoint, 
            model, 
            optimizer, 
            scaler, 
            dtype, 
            device,
            scheduler=scheduler
        )
        trainer.set_accum_state(accum_step)

    # Training loop with DataLoader
    train_iter = iter(train_loader)
    loss = None  # Initialize for final checkpoint
    
    try:
        for step in range(start_step, config.training.max_iters):
            # Get next batch from DataLoader
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Execute training step
            loss = trainer.train_step(x, y)
            trainer.update_loss_accumulator(loss)

            # Logging
            if trainer.should_log(step):
                trainer.log_progress(step, config.training.max_iters)
            
            # Step scheduler after optimizer step (respects gradient accumulation)
            # Only step when optimizer actually stepped (accum_step == 0 after step)
            if trainer.get_accum_state() == 0 and scheduler is not None:
                # Check if we need validation loss for plateau scheduler
                val_loss = None
                if config.scheduler.type == "plateau" and trainer.should_eval(step):
                    val_loss = compute_validation_loss(
                        model, val_loader, device, config, num_batches=config.dataset.val_batches
                    )
                    trainer.last_val_loss = val_loss  # Store for logging
                    print(f"Validation loss: {val_loss:.4f}")
                
                trainer.scheduler_step(val_loss)
            
            # Checkpointing
            if trainer.should_checkpoint(step):
                checkpoint_path = save_checkpoint(
                    model, 
                    optimizer, 
                    step, 
                    loss.item(), 
                    scaler,
                    dtype,
                    config,
                    run_dir=trainer.get_run_directory(),
                    is_first=trainer.is_first_checkpoint(),
                    tokenizer_config=tokenizer_config_dict,
                    accum_step=trainer.get_accum_state(),
                    scheduler=scheduler
                )
                trainer.mark_first_checkpoint_saved()
            
            # Evaluation
            if trainer.should_eval(step):
                eval_model(model, device, tokenizer=tokenizer)
    finally:
        # Ensure TensorBoard writer is properly closed
        trainer.close()
        # Restore stdout/stderr and close log files
        teardown_logging(loggers)

    print("Training done, now generating a sample")
    eval_model(model, device, tokenizer=tokenizer)
    
    # Save final checkpoint
    final_loss = loss.item() if loss is not None else 0.0
    save_checkpoint(
        model, 
        optimizer, 
        config.training.max_iters, 
        final_loss,
        scaler,
        dtype,
        config,
        run_dir=trainer.get_run_directory(),
        tokenizer_config=tokenizer_config_dict,
        accum_step=trainer.get_accum_state(),
        scheduler=scheduler
    )


if __name__ == "__main__":
    main()
