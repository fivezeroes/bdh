# Copyright Pathway Technology, Inc.

"""Main training script for BDH model."""

import argparse
import os

import bdh
import config as cfg
import torch
from torch.utils.data import DataLoader

# Import refactored modules
from data import ParquetDataset, fetch_parquet_files, split_parquet_files
from optimization import (
    check_low_precision_requirements,
    create_fp8_recipe,
    create_optimizer,
    prepare_model_for_4bit_training,
)
from trainer import Trainer
from training_utils import eval_model, load_checkpoint, save_checkpoint


def main():
    """Main training function."""
    # Load configuration
    parser = argparse.ArgumentParser(description='Train BDH model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = cfg.Config.from_yaml(args.config)

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
    torch.backends.cudnn.fp32_precision = 'tf32'

    # Setup FP8 recipe if using Transformer Engine
    fp8_recipe = create_fp8_recipe(config) if config.low_precision.use_fp8 else None

    # Print configuration info
    print(f"Using device: {device} with dtype {dtype}")
    if fp8_recipe:
        print(f"FP8 training enabled with format: {config.fp8.format}")

    # Fetch and split data
    parquet_files = fetch_parquet_files(config.dataset.parquet_dir)
    train_files, test_files = split_parquet_files(parquet_files, train_ratio=0.9)
    
    # Determine run directory for this training session
    current_run_dir = None
    if config.training.resume_from_checkpoint is not None:
        # Resuming from checkpoint - use the same directory as the checkpoint
        current_run_dir = os.path.dirname(config.training.resume_from_checkpoint)
        print(f"Resuming training in directory: {current_run_dir}")

    # Check low precision requirements
    check_low_precision_requirements(config, device)
    
    # Create datasets and dataloaders
    train_dataset = ParquetDataset(train_files, config.training.block_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=config.dataloader.prefetch_factor,  # Prefetch batches per worker
    )

    # Create and prepare model
    model = bdh.BDH(config.to_bdh_config()).to(device)
    
    # Prepare model for 4-bit training if enabled
    if config.low_precision.use_4bit:
        model = prepare_model_for_4bit_training(model, config)
    
    model = torch.compile(model)
    
    # Create optimizer with optional 4-bit support
    optimizer = create_optimizer(model, config)

    # Create trainer
    trainer = Trainer(model, optimizer, config, device, dtype, scaler, fp8_recipe)
    if current_run_dir:
        trainer.set_run_directory(current_run_dir)

    # Resume from checkpoint if specified
    start_step = 0
    if config.training.resume_from_checkpoint is not None:
        start_step = load_checkpoint(
            config.training.resume_from_checkpoint, 
            model, 
            optimizer, 
            scaler, 
            dtype, 
            device
        )

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
            
            # Checkpointing
            if trainer.should_checkpoint(step):
                checkpoint_path, updated_run_dir = save_checkpoint(
                    model, 
                    optimizer, 
                    step, 
                    loss.item(), 
                    scaler,
                    dtype,
                    config,
                    checkpoint_dir=trainer.get_run_directory(),
                    current_run_dir=trainer.get_run_directory(),
                    is_first=trainer.is_first_checkpoint()
                )
                trainer.set_run_directory(updated_run_dir)
                trainer.mark_first_checkpoint_saved()
            
            # Evaluation
            if trainer.should_eval(step):
                eval_model(model, device)
    finally:
        # Ensure TensorBoard writer is properly closed
        trainer.close()

    print("Training done, now generating a sample")
    eval_model(model, device)
    
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
        checkpoint_dir=trainer.get_run_directory(),
        current_run_dir=trainer.get_run_directory()
    )


if __name__ == "__main__":
    main()
