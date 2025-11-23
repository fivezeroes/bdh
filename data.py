# Copyright Pathway Technology, Inc.

"""Data loading and dataset utilities for BDH training."""

import glob
import os
from typing import Optional, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
import pyarrow.parquet as pq


def _encode_text_to_tokens(text, tokenizer, block_size):
    """Encode text into token arrays (x, y) of length block_size and block_size.

    This mirrors the previous `ParquetDataset` behavior: tokens are truncated or
    padded to block_size + 1, then converted to x (first block_size tokens) and
    y (tokens 1..block_size + 1).
    """
    if text is None:
        text = ""
    if isinstance(text, (bytes, bytearray)):
        try:
            text = text.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    # Tokenize text
    if tokenizer is not None:
        token_ids = tokenizer.encode(text)
    else:
        token_ids = list(text.encode("utf-8", errors="ignore"))

    # Ensure at least one token
    if not token_ids:
        if tokenizer is not None:
            pad_id = (
                tokenizer.pad_token_id
                if getattr(tokenizer, "pad_token_id", None) is not None
                else tokenizer.eos_token_id
                if getattr(tokenizer, "eos_token_id", None) is not None
                else 0
            )
        else:
            pad_id = 0
        token_ids = [pad_id]

    # Truncate or pad so we have exactly block_size + 1 length
    target_len = block_size + 1
    if len(token_ids) >= target_len:
        seq_tokens = token_ids[:target_len]
    else:
        if tokenizer is not None:
            pad_id = (
                tokenizer.pad_token_id
                if getattr(tokenizer, "pad_token_id", None) is not None
                else tokenizer.eos_token_id
                if getattr(tokenizer, "eos_token_id", None) is not None
                else 0
            )
        else:
            pad_id = 0
        seq_tokens = token_ids + [pad_id] * (target_len - len(token_ids))

    seq_array = np.array(seq_tokens, dtype=np.int64)
    x = seq_array[:block_size]
    y = seq_array[1 : block_size + 1]
    return x, y


# _local_path_to_url removed; not needed for PyTorch-based loader


class ParquetDataset(torch.utils.data.Dataset):
    """Dataset that reads rows from a set of Parquet files using pyarrow.

    This is a minimal replacement for the custom loader previously used.
    It supports precomputed file_lengths so the main process can index once and
    worker processes avoid re-indexing.
    """
    def __init__(self, parquet_files, block_size=512, tokenizer=None, file_lengths=None):
        self.parquet_files = parquet_files
        self.block_size = block_size
        self.tokenizer = tokenizer

        if file_lengths is not None:
            self.file_lengths = file_lengths
            self.cumulative_lengths = [0]
            for length in file_lengths:
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
            self.total_length = self.cumulative_lengths[-1]
        else:
            # Index files and compute number of rows per file
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
        # Locate file index using cumulative lengths
        file_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_len:
                file_idx = i
                break

        local_idx = idx - self.cumulative_lengths[file_idx]
        file_path = self.parquet_files[file_idx]

        pf = pq.ParquetFile(file_path)
        # Find row group containing local_idx
        row_group_idx = 0
        rows_before = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            if rows_before + rg_rows > local_idx:
                row_group_idx = rg_idx
                local_row_idx = local_idx - rows_before
                break
            rows_before += rg_rows
        else:
            # fallback
            row_group_idx = pf.metadata.num_row_groups - 1
            local_row_idx = 0

        try:
            table = pf.read_row_group(row_group_idx, columns=['text'])
            if local_row_idx < len(table):
                text = table.column('text')[local_row_idx].as_py()
            else:
                text = table.column('text')[0].as_py()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            text = ""

        x_arr, y_arr = _encode_text_to_tokens(text, self.tokenizer, self.block_size)
        x = torch.from_numpy(x_arr)
        y = torch.from_numpy(y_arr)
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


def create_parquet_dataloader(
    parquet_files: Iterable[str],
    tokenizer,
    block_size: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    file_lengths: Optional[Iterable[int]] = None,
):
    """Create a PyTorch DataLoader wrapping ParquetDataset.

    Parameters are intentionally minimal to keep the wrapper small.
    """
    dataset = ParquetDataset(parquet_files, block_size=block_size, tokenizer=tokenizer, file_lengths=file_lengths)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(1, num_workers),
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=drop_last,
    )



