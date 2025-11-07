# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext
from datetime import datetime
import glob

import bdh
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

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
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cudnn.conv.fp32_precision = 'tf32'
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
TEST_FREQ = 500
CHECKPOINT_DIR = "checkpoints"
RESUME_FROM_CHECKPOINT = "checkpoints/checkpoint_1000.pt"  # Set to checkpoint path to resume, e.g., "checkpoints/checkpoint_1000.pt"

# Parquet dataset configuration
PARQUET_DIR = "/Volumes/Data/fineweb/data/CC-MAIN-2025-26"

# Global variables for parquet file handling
parquet_files = []
parquet_file_cache = {}  # Cache opened parquet files
MAX_CACHED_FILES = 5  # Keep only a few files open


def fetch_data():
    """Initialize parquet file list."""
    global parquet_files
    
    print(f"Loading parquet file list from {PARQUET_DIR}...")
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")
    
    print(f"Found {len(parquet_files)} parquet files")


def get_parquet_file(file_path):
    """Get or open a parquet file, with caching."""
    global parquet_file_cache
    
    if file_path not in parquet_file_cache:
        # If cache is full, remove oldest entry
        if len(parquet_file_cache) >= MAX_CACHED_FILES:
            parquet_file_cache.pop(next(iter(parquet_file_cache)))
        
        parquet_file_cache[file_path] = pq.ParquetFile(file_path)
    
    return parquet_file_cache[file_path]


def get_batch(split):
    """Get a batch by randomly sampling from parquet files."""
    # Determine which files to use based on train/test split
    split_idx = int(0.9 * len(parquet_files))
    if split == "train":
        available_files = parquet_files[:split_idx]
    else:
        available_files = parquet_files[split_idx:]
    
    if not available_files:
        raise ValueError(f"No files available for split: {split}")
    
    # Collect batch samples
    x_list = []
    y_list = []
    
    for _ in range(BATCH_SIZE):
        # Randomly select a parquet file
        file_idx = int(torch.randint(len(available_files), (1,)).item())
        file_path = available_files[file_idx]
        pf = get_parquet_file(file_path)
        
        # Randomly select a row from the file
        num_rows = pf.metadata.num_rows
        row_idx = int(torch.randint(num_rows, (1,)).item())
        
        # Read the specific row
        table = pf.read_row_group(row_idx // 10000, columns=['text'])
        local_idx = row_idx % 10000
        if local_idx < len(table):
            text = table.column('text')[local_idx].as_py()
        else:
            # Fallback: read a different row group
            table = pf.read_row_group(0, columns=['text'])
            text = table.column('text')[0].as_py()
        
        if text is None:
            text = ""
        
        # Convert text to bytes
        text_bytes = text.encode('utf-8', errors='ignore')
        
        # Sample a random sequence of BLOCK_SIZE from this text
        if len(text_bytes) > BLOCK_SIZE + 1:
            start_idx = int(torch.randint(len(text_bytes) - BLOCK_SIZE - 1, (1,)).item())
            seq_bytes = text_bytes[start_idx:start_idx + BLOCK_SIZE + 1]
        else:
            # Pad if text is too short
            seq_bytes = text_bytes + b'\x00' * (BLOCK_SIZE + 1 - len(text_bytes))
        
        # Convert to tensors
        seq_array = np.frombuffer(seq_bytes, dtype=np.uint8).astype(np.int64)
        x_list.append(torch.from_numpy(seq_array[:BLOCK_SIZE]))
        y_list.append(torch.from_numpy(seq_array[1:BLOCK_SIZE + 1]))
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


def eval(model):
    model.eval()

    prompt = torch.tensor(
        bytearray("The Capital of France is ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )

    print(ret_decoded)


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir=CHECKPOINT_DIR):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    
    # Get the underlying model if it's been compiled
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': BDH_CONFIG,
        'scaler_state_dict': scaler.state_dict() if dtype == "float16" else None,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load model checkpoint and return starting step."""
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

if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Resume from checkpoint if specified
    start_step = 0
    if RESUME_FROM_CHECKPOINT is not None:
        start_step = load_checkpoint(RESUME_FROM_CHECKPOINT, model, optimizer)

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    for step in range(start_step, MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % LOG_FREQ == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = 0
            loss_steps = 0
        if step % TEST_FREQ == 0 and step > 0:
            eval(model)
            # Save checkpoint at eval frequency
            save_checkpoint(model, optimizer, step, loss.item())

    print("Training done, now generating a sample ")
    eval(model)
    # Save final checkpoint
    save_checkpoint(model, optimizer, MAX_ITERS, loss.item())

