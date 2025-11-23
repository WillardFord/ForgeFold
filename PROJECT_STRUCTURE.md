# ForgeFold Project Structure

## Directory Layout

```
ForgeFold/
├── src/forgefold/          # Core package
│   ├── __init__.py         # Package initialization
│   ├── tokenizer.py        # ESMCTokenizer implementation
│   ├── plm.py              # Protein language model
│   ├── loss.py             # Loss functions
│   └── data.py             # Data loading utilities
├── data/                   # Data directory
│   └── uniparc30_sample_100.fasta
├── configs/                # Configuration files
│   └── train_config.yaml
├── checkpoints/            # Model checkpoints
├── tests/                  # Test files
│   └── test_tokenizer.py
├── train.py                # Training script
└── README.md               # Project documentation
```

## Usage

### Training

**Basic training:**
```bash
python train.py
```

**With custom parameters:**
```bash
python train.py --data_path data/your_data.fasta --batch_size 64 --num_epochs 10 --lr 5e-5
```

**Using pre-bucketed data:**
```bash
python train.py --buckets_path data/bucketed_data.pt --batch_size 64
```

**With checkpointing:**
```bash
python train.py --save_checkpoints --checkpoint_dir checkpoints --save_interval 1
```

### Command-line Arguments

- `--buckets_path`: Path to pre-bucketed data (optional)
- `--data_path`: Path to FASTA file (default: data/uniparc30_sample_100.fasta)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 1e-4)
- `--mask_ratio`: Masking ratio for MLM (default: 0.15)
- `--log_interval`: Log every N batches (default: 10)
- `--save_checkpoints`: Enable checkpoint saving
- `--checkpoint_dir`: Directory for checkpoints (default: checkpoints)
- `--save_interval`: Save checkpoint every N epochs (default: 1)

## Module Structure

### `forgefold.tokenizer`
- `ESMCTokenizer`: Tokenizer for protein sequences

### `forgefold.plm`
- `ProteinLanguageModel`: 82M parameter transformer model

### `forgefold.loss`
- `apply_mask()`: Apply masking for MLM
- `mlm_loss_function()`: Masked language modeling loss

### `forgefold.data`
- `ProteinDataset`: PyTorch dataset for protein sequences
- `get_bucket()`: Assign sequences to length buckets
- `load_bucketed_data()`: Load pre-bucketed data
- `create_buckets_from_fasta()`: Create buckets from FASTA
- `save_buckets()`: Save bucketed data
- `create_dataloader()`: Create DataLoader with collate function

## Data Bucketing

Sequences are automatically grouped into length buckets:
- 128, 256, 512, 1024, 2048, 4096, 8192, 16384

This reduces padding overhead and improves training efficiency.

## Pre-processing Data

To pre-process and save bucketed data:

```python
from forgefold import ESMCTokenizer
from forgefold.data import create_buckets_from_fasta, save_buckets

tokenizer = ESMCTokenizer()
buckets = create_buckets_from_fasta("data/your_data.fasta", tokenizer)
save_buckets(buckets, "data/bucketed_data.pt")
```
