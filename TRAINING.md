# ForgeFold Training Guide

## Stage A: Encoder Pre-training

Train the 82M parameter protein language model on UniParc30 sequences.

### Basic Training

```bash
python train.py \
  --data_path /home/ubuntu/homework/ESE5460/ForgeFold/data/uniparc30_sample_10000000.fasta \
  --target_tokens 32768 \
  --max_length 16384 \
  --num_epochs 1 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --mask_ratio 0.15 \
  --use_amp \
  --use_wandb \
  --wandb_project forgefold \
  --wandb_run_name plm-10m-run1
```

### Resume from Checkpoint

```bash
python train.py \
  --data_path /home/ubuntu/homework/ESE5460/ForgeFold/data/uniparc30_sample_10000000.fasta \
  --target_tokens 32768 \
  --max_length 16384 \
  --num_epochs 3 \
  --resume checkpoints/checkpoint_epoch_1.pt \
  --use_amp \
  --use_wandb \
  --wandb_project forgefold \
  --wandb_run_name plm-10m-run2
```

### Key Settings

| Setting | Value | Notes |
|---------|-------|-------|
| `target_tokens` | 32768 | Dynamic batching targeting 32k tokens per batch |
| `max_length` | 16384 | 16k context window (8x longer than ESM-C's 2048) |
| `use_amp` | enabled | bfloat16 mixed precision with flash attention |
| `weight_decay` | 0.01 | AdamW optimizer regularization |
| `mask_ratio` | 0.15 | 15% of tokens masked for MLM |
| `lr` | 1e-4 | Learning rate |
| `eval_interval` | 900 | Evaluate on test sample every N training batches |
| `eval_batches` | 100 | Number of test batches to sample during periodic eval |
| `test_split` | 0.1 | 90% train, 10% test split |

### Results

**Epoch 1 (10M sequences):**
- Test loss: 2.1033
- Perplexity: 8.19

### Running in tmux

```bash
tmux new -s train
# run training command
# Ctrl+B, D to detach
tmux attach -t train  # to reattach
```

## Stage B: Structure Prediction Training

TODO: Folding trunk + diffusion decoder training
