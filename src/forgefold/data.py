"""
Bucketed DataLoader for protein sequences
Buckets sequences by length (powers of 2) to minimize padding overhead
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import ESMCTokenizer
import random
from collections import defaultdict


class BucketedProteinDataset(Dataset):
    """Dataset that buckets sequences by length for efficient batching"""

    def __init__(self, fasta_path, tokenizer, max_length=16384, split='train', test_split=0.1, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.test_split = test_split
        self.seed = seed

        # Define power-of-2 buckets
        self.buckets = {
            128: [],
            256: [],
            512: [],
            1024: [],
            2048: [],
            4096: [],
            8192: [],
            16384: []
        }

        print(f"Scanning {fasta_path} and bucketing sequences...")
        self._bucket_sequences(fasta_path)

        # Split data into train/test
        if test_split > 0:
            self._split_data()

        # Flatten all buckets into a single list with bucket info
        self.samples = []
        for bucket_size, sequences in self.buckets.items():
            for seq in sequences:
                self.samples.append((seq, bucket_size))

        print(f"Total sequences loaded ({split}): {len(self.samples)}")
        self._print_bucket_stats()

    def _bucket_sequences(self, fasta_path):
        """Read FASTA and assign sequences to length buckets"""
        sequences = self.tokenizer.read_fasta(fasta_path)

        for seq in sequences:
            seq_len = len(seq)

            # Skip sequences longer than max_length
            if seq_len > self.max_length:
                continue

            # Find appropriate bucket (smallest power of 2 >= seq_len)
            if seq_len <= 128:
                self.buckets[128].append(seq)
            elif seq_len <= 256:
                self.buckets[256].append(seq)
            elif seq_len <= 512:
                self.buckets[512].append(seq)
            elif seq_len <= 1024:
                self.buckets[1024].append(seq)
            elif seq_len <= 2048:
                self.buckets[2048].append(seq)
            elif seq_len <= 4096:
                self.buckets[4096].append(seq)
            elif seq_len <= 8192:
                self.buckets[8192].append(seq)
            else:
                self.buckets[16384].append(seq)

    def _split_data(self):
        """Split each bucket into train/test sets"""
        random.seed(self.seed)

        for bucket_size in self.buckets.keys():
            sequences = self.buckets[bucket_size]
            if len(sequences) == 0:
                continue

            # Shuffle sequences deterministically
            shuffled = sequences.copy()
            random.shuffle(shuffled)

            # Split into train/test
            test_size = int(len(shuffled) * self.test_split)

            if self.split == 'train':
                self.buckets[bucket_size] = shuffled[test_size:]
            elif self.split == 'test':
                self.buckets[bucket_size] = shuffled[:test_size]
            else:
                raise ValueError(f"Unknown split: {self.split}")

    def _print_bucket_stats(self):
        """Print statistics about bucket distribution"""
        print("\nBucket distribution:")
        for bucket_size in sorted(self.buckets.keys()):
            count = len(self.buckets[bucket_size])
            if count > 0:
                print(f"  {bucket_size:5d}: {count:8d} sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, bucket_size = self.samples[idx]
        # Return sequence and its bucket size
        return seq, bucket_size


class BucketedBatchSampler:
    """Sampler that creates batches with constant token budget"""

    def __init__(self, dataset, target_tokens=16384, shuffle=True):
        self.dataset = dataset
        self.target_tokens = target_tokens
        self.shuffle = shuffle

        # Group indices by bucket
        self.bucket_indices = defaultdict(list)
        for idx, (seq, bucket_size) in enumerate(dataset.samples):
            self.bucket_indices[bucket_size].append(idx)

        # Calculate batch size for each bucket (to hit target_tokens)
        self.bucket_batch_sizes = {}
        for bucket_size in self.bucket_indices.keys():
            # batch_size * bucket_size ≈ target_tokens
            self.bucket_batch_sizes[bucket_size] = max(1, self.target_tokens // bucket_size)

    def __iter__(self):
        # Create list of all batches (reshuffle each time __iter__ is called)
        all_batches = []

        # Make a copy of indices to shuffle without modifying original
        bucket_indices_copy = {k: v.copy() for k, v in self.bucket_indices.items()}

        # Shuffle within buckets if requested
        if self.shuffle:
            for bucket_size in bucket_indices_copy:
                random.shuffle(bucket_indices_copy[bucket_size])

        for bucket_size, indices in bucket_indices_copy.items():
            batch_size = self.bucket_batch_sizes[bucket_size]
            # Split bucket into batches with appropriate size
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                all_batches.append(batch)

        # Shuffle batches if requested
        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for bucket_size, indices in self.bucket_indices.items():
            batch_size = self.bucket_batch_sizes[bucket_size]
            total_batches += (len(indices) + batch_size - 1) // batch_size
        return total_batches


def collate_fn(batch, tokenizer):
    """
    Collate function to tokenize and pad sequences in a batch
    All sequences in batch should be from same bucket (similar lengths)
    Pads to bucket length for consistent batch shapes

    Returns dictionary with:
        - input_ids: [num_seqs, bucket_len]
        - attention_mask: [num_seqs, bucket_len]
        - bucket_size: int
    """
    sequences = [item[0] for item in batch]
    bucket_sizes = [item[1] for item in batch]

    # All sequences in batch should have same bucket size
    bucket_length = bucket_sizes[0]
    assert all(b == bucket_length for b in bucket_sizes), "All sequences in batch must be from same bucket"

    # Tokenize all sequences
    encoded_seqs = [torch.tensor(tokenizer.encode(seq)) for seq in sequences]

    # Pad to bucket length (not just longest in batch)
    batch_size = len(encoded_seqs)
    input_ids = torch.full(
        (batch_size, bucket_length),
        fill_value=tokenizer.pad_idx,
        dtype=torch.long
    )

    # Fill in actual sequence data
    for i, seq_tensor in enumerate(encoded_seqs):
        seq_len = min(len(seq_tensor), bucket_length)  # Handle edge case
        input_ids[i, :seq_len] = seq_tensor[:seq_len]

    # Create attention mask (True for real tokens, False for padding)
    attention_mask = (input_ids != tokenizer.pad_idx)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'bucket_size': bucket_length
    }


def create_dataloader(fasta_path, target_tokens=16384, shuffle=True, num_workers=0, max_length=16384,
                      split='train', test_split=0.1, seed=42):
    """
    Create a bucketed dataloader for protein sequences with constant token budget

    Args:
        fasta_path: Path to FASTA file
        target_tokens: Target number of tokens per batch (default: 16384)
                      Batch size will be dynamically calculated: target_tokens / bucket_size
        shuffle: Whether to shuffle batches
        num_workers: Number of workers for data loading
        max_length: Maximum sequence length to include
        split: 'train' or 'test'
        test_split: Fraction of data to use for test (default: 0.1 = 10%)
        seed: Random seed for reproducible splits (default: 42)

    Returns:
        DataLoader with bucketed batching, tokenizer

    Example batch sizes for target_tokens=16384:
        - 128-token sequences: batch_size = 128
        - 256-token sequences: batch_size = 64
        - 512-token sequences: batch_size = 32
        - 1024-token sequences: batch_size = 16
    """
    tokenizer = ESMCTokenizer()
    dataset = BucketedProteinDataset(
        fasta_path,
        tokenizer,
        max_length=max_length,
        split=split,
        test_split=test_split,
        seed=seed
    )

    batch_sampler = BucketedBatchSampler(dataset, target_tokens=target_tokens, shuffle=shuffle)

    # Create collate function with tokenizer
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_wrapper,
        num_workers=num_workers
    )

    return dataloader, tokenizer


def create_train_test_dataloaders(fasta_path, target_tokens=16384, num_workers=0, max_length=16384,
                                   test_split=0.1, seed=42):
    """
    Create both train and test dataloaders with the same configuration

    Args:
        fasta_path: Path to FASTA file
        target_tokens: Target number of tokens per batch
        num_workers: Number of dataloader workers
        max_length: Maximum sequence length
        test_split: Fraction of data for test set (default: 0.1)
        seed: Random seed for reproducible splits (default: 42)

    Returns:
        train_dataloader, test_dataloader, tokenizer
    """
    print("Creating train dataloader...")
    train_dataloader, tokenizer = create_dataloader(
        fasta_path=fasta_path,
        target_tokens=target_tokens,
        shuffle=True,
        num_workers=num_workers,
        max_length=max_length,
        split='train',
        test_split=test_split,
        seed=seed
    )

    print("\nCreating test dataloader...")
    test_dataloader, _ = create_dataloader(
        fasta_path=fasta_path,
        target_tokens=target_tokens,
        shuffle=True,  # Shuffle so periodic eval samples different batches each time
        num_workers=num_workers,
        max_length=max_length,
        split='test',
        test_split=test_split,
        seed=seed
    )

    return train_dataloader, test_dataloader, tokenizer
