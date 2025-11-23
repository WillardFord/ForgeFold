"""Data loading and preprocessing utilities"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


class ProteinDataset(Dataset):
    """Dataset for protein sequences"""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def get_bucket(length):
    """Assign sequence to length bucket"""
    buckets = [2**i for i in range(7, 15)]  # [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]


def load_bucketed_data(path):
    """Load pre-bucketed data from file"""
    return torch.load(path)


def create_buckets_from_fasta(fasta_path, tokenizer):
    """Create buckets from FASTA file"""
    sequences = tokenizer.read_fasta(fasta_path)
    buckets = defaultdict(list)
    for seq in sequences:
        bucket = get_bucket(len(seq))
        buckets[bucket].append(seq)
    return dict(buckets)


def save_buckets(buckets, path):
    """Save bucketed data to file"""
    torch.save(buckets, path)


def create_dataloader(sequences, batch_size, tokenizer, shuffle=True):
    """Create DataLoader with collate function"""

    def collate_fn(batch):
        encoded = [torch.tensor(tokenizer.encode(seq)) for seq in batch]
        input_ids = pad_sequence(encoded, batch_first=True, padding_value=tokenizer.pad_idx)
        attention_mask = (input_ids != tokenizer.pad_idx)
        return input_ids, attention_mask

    dataset = ProteinDataset(sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
