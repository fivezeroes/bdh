# Copyright Pathway Technology, Inc.

"""Data loading and dataset utilities for BDH training."""

import glob
import os
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
    """Dataset class for efficient parquet file loading with PyTorch DataLoader."""
    
    def __init__(self, parquet_files, block_size=512, tokenizer=None):
        self.parquet_files = parquet_files
        self.block_size = block_size
        self.tokenizer = tokenizer  # Tokenizer instance for encoding text
        
        # Pre-compute total number of samples
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        print(f"Indexing {len(parquet_files)} parquet files...")
        for i, file_path in enumerate(parquet_files):
            pf = pq.ParquetFile(file_path)
            num_rows = pf.metadata.num_rows
            self.file_lengths.append(num_rows)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + num_rows)
            if (i + 1) % 100 == 0:
                print(f"  Indexed {i + 1}/{len(parquet_files)} files...")
        
        self.total_length = self.cumulative_lengths[-1]
        print(f"Dataset indexed: {self.total_length:,} total samples across {len(parquet_files)} files")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_len:
                file_idx = i
                break
        
        local_idx = idx - self.cumulative_lengths[file_idx]
        file_path = self.parquet_files[file_idx]
        
        # Read the row
        pf = pq.ParquetFile(file_path)
        row_group_idx = local_idx // 10000
        local_row_idx = local_idx % 10000
        
        try:
            table = pf.read_row_group(row_group_idx, columns=['text'])
            if local_row_idx < len(table):
                text = table.column('text')[local_row_idx].as_py()
            else:
                # Fallback to first row if index is out of bounds
                text = table.column('text')[0].as_py()
        except Exception as e:
            # Fallback for any read errors
            text = ""
        
        if text is None:
            text = ""
        
        # Tokenize text
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(text)
        else:
            # Fallback to byte-level encoding if no tokenizer provided
            text_bytes = text.encode('utf-8', errors='ignore')
            token_ids = list(text_bytes)
        
        # Sample a random sequence
        if len(token_ids) > self.block_size + 1:
            start_idx = int(torch.randint(len(token_ids) - self.block_size - 1, (1,)).item())
            seq_tokens = token_ids[start_idx:start_idx + self.block_size + 1]
        else:
            # Pad if sequence is too short
            if self.tokenizer is not None:
                # Use pad token if available, otherwise use eos token, otherwise use 0
                pad_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None 
                         else self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None 
                         else 0)
            else:
                pad_id = 0
            seq_tokens = token_ids + [pad_id] * (self.block_size + 1 - len(token_ids))
        
        # Convert to tensors
        seq_array = np.array(seq_tokens, dtype=np.int64)
        x = torch.from_numpy(seq_array[:self.block_size])
        y = torch.from_numpy(seq_array[1:self.block_size + 1])
        
        return x, y


def fetch_parquet_files(parquet_dir):
    """
    Load list of parquet files from directory.
    
    Args:
        parquet_dir: Directory containing parquet files
        
    Returns:
        Sorted list of parquet file paths
        
    Raises:
        FileNotFoundError: If no parquet files found in directory
    """
    print(f"Loading parquet file list from {parquet_dir}...")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    return parquet_files


def split_parquet_files(parquet_files, train_ratio=0.9):
    """
    Split parquet files into train and test sets.
    
    Args:
        parquet_files: List of parquet file paths
        train_ratio: Ratio of files to use for training (default: 0.9)
        
    Returns:
        Tuple of (train_files, test_files)
    """
    split_idx = int(train_ratio * len(parquet_files))
    train_files = parquet_files[:split_idx]
    test_files = parquet_files[split_idx:]
    
    print(f"Split: {len(train_files)} train files, {len(test_files)} test files")
    return train_files, test_files


class ParquetFileCache:
    """Cache for opened parquet files to avoid repeated file opening."""
    
    def __init__(self, max_cached_files=100):
        self.cache = {}
        self.max_cached_files = max_cached_files
    
    def get(self, file_path):
        """Get or open a parquet file, with caching."""
        if file_path not in self.cache:
            # If cache is full, remove oldest entry
            if len(self.cache) >= self.max_cached_files:
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[file_path] = pq.ParquetFile(file_path)
        
        return self.cache[file_path]
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


def get_batch(parquet_files, batch_size, block_size, device, file_cache=None, tokenizer=None, debug=False):
    """
    Get a batch by randomly sampling from parquet files.
    
    Args:
        parquet_files: List of parquet files to sample from
        batch_size: Number of samples in batch
        block_size: Size of each sequence block
        device: Device to place tensors on
        file_cache: Optional ParquetFileCache instance
        tokenizer: Optional tokenizer for encoding text
        debug: If True, print debug information
        
    Returns:
        Tuple of (x, y) tensors
    """
    if not parquet_files:
        raise ValueError("No parquet files available")
    
    # Create temporary cache if none provided
    if file_cache is None:
        file_cache = ParquetFileCache()
    
    # Collect batch samples
    x_list = []
    y_list = []
    
    for _ in range(batch_size):
        # Randomly select a parquet file
        file_idx = int(torch.randint(len(parquet_files), (1,)).item())
        file_path = parquet_files[file_idx]
        pf = file_cache.get(file_path)
        
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
        
        # Tokenize text
        if tokenizer is not None:
            token_ids = tokenizer.encode(text)
        else:
            # Fallback to byte-level encoding if no tokenizer provided
            text_bytes = text.encode('utf-8', errors='ignore')
            token_ids = list(text_bytes)
        
        # Sample a random sequence of block_size from this text
        if len(token_ids) > block_size + 1:
            start_idx = int(torch.randint(len(token_ids) - block_size - 1, (1,)).item())
            seq_tokens = token_ids[start_idx:start_idx + block_size + 1]
        else:
            # Pad if sequence is too short
            if tokenizer is not None:
                # Use pad token if available, otherwise use eos token, otherwise use 0
                pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None 
                         else tokenizer.eos_token_id if tokenizer.eos_token_id is not None 
                         else 0)
            else:
                pad_id = 0
            seq_tokens = token_ids + [pad_id] * (block_size + 1 - len(token_ids))
        
        # Convert to tensors
        seq_array = np.array(seq_tokens, dtype=np.int64)
        x_list.append(torch.from_numpy(seq_array[:block_size]))
        y_list.append(torch.from_numpy(seq_array[1:block_size + 1]))
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    if debug:
        print("x:")
        for i in range(len(x)):
            if tokenizer is not None:
                print(tokenizer.decode(x[i].tolist()))
            else:
                for j in range(len(x[i])):
                    print(f"{chr(int(x[i][j].item()))}", end="")
                print()

        print("y:")
        for i in range(len(y)):
            if tokenizer is not None:
                print(tokenizer.decode(y[i].tolist()))
            else:
                for j in range(len(y[i])):
                    print(f"{chr(int(y[i][j].item()))}", end="")
                print()
        input("Press the <ENTER> key to continue...")
    
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y
