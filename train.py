"""
Training script for protein language model with MLM objective
Uses bucketed dataloader with constant token budget for efficient training
"""

import argparse
import sys
sys.path.insert(0, 'src')

from forgefold import ESMCTokenizer, ProteinLanguageModel, apply_mask, mlm_loss_function
from forgefold.data import create_train_test_dataloaders
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import time
import csv
import os
from pathlib import Path


def evaluate(model, dataloader, tokenizer, device, mask_ratio=0.15, use_amp=False, dtype=torch.float32, max_batches=None):
    """
    Evaluate model on a dataset

    Args:
        max_batches: If set, only evaluate on this many batches (randomly sampled)

    Returns:
        avg_loss: Average loss over the dataset
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", total=max_batches):
            # Unpack batch dictionary and move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Apply masking
            masked_input, labels, mlm_mask = apply_mask(
                input_ids,
                mask_token_id=tokenizer.mask_idx,
                mask_ratio=mask_ratio
            )

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                logits = model(masked_input, mask=attention_mask)
                loss = mlm_loss_function(logits, labels, mlm_mask)

            total_loss += loss.item()
            num_batches += 1

            # Early exit if we've done enough batches
            if max_batches and num_batches >= max_batches:
                break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main(args):
    print("=" * 80)
    print("Protein Language Model Training")
    print("=" * 80)

    # Setup mixed precision training
    use_amp = args.use_amp and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nMixed Precision Training: {'Enabled (bfloat16)' if use_amp else 'Disabled (float32)'}")

    # Setup CSV logging
    batch_log_path = None
    epoch_log_path = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        batch_log_path = os.path.join(args.log_dir, "batch_metrics.csv")
        epoch_log_path = os.path.join(args.log_dir, "epoch_metrics.csv")

        # Initialize batch log file
        with open(batch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss', 'avg_loss', 'bucket_size',
                           'num_sequences', 'actual_tokens', 'tokens_per_sec', 'step_time'])

        # Initialize epoch log file
        with open(epoch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_loss', 'epoch_time_minutes', 'total_batches'])

        print(f"\nCSV logs will be saved to:")
        print(f"  Batch metrics: {batch_log_path}")
        print(f"  Epoch metrics: {epoch_log_path}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": "ProteinLanguageModel",
                "architecture": "ESM-C inspired (24L x 512D)",
                "parameters": "82M",
                "vocab_size": 32,
                "max_seq_length": args.max_length,
                "target_tokens": args.target_tokens,
                "num_epochs": args.num_epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "mask_ratio": args.mask_ratio,
                "optimizer": "AdamW",
                "device": str(device),
                "mixed_precision": "bfloat16" if use_amp else "float32",
            }
        )
        print(f"\nWandB initialized: {args.wandb_project}/{args.wandb_run_name}")

    # Setup model and optimizer
    print("\nInitializing model...")
    # Use flash attention, but PyTorch's built-in version is more compatible with AMP
    model = ProteinLanguageModel(use_flash=True)
    model = model.to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    # Note: When using AMP, PyTorch's built-in flash attention (F.scaled_dot_product_attention)
    # works seamlessly. The dedicated flash-attn library requires manual dtype management.

    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=args.wandb_log_freq)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"\nLoading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
        if 'train_loss' in checkpoint:
            print(f"Previous train loss: {checkpoint['train_loss']:.4f}")
        if 'test_loss' in checkpoint:
            print(f"Previous test loss: {checkpoint['test_loss']:.4f}")

    # Load data with train/test split
    print(f"\nLoading data from: {args.data_path}")
    print(f"Target tokens per batch: {args.target_tokens}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Test split: {args.test_split * 100:.1f}%")

    train_dataloader, test_dataloader, tokenizer = create_train_test_dataloaders(
        fasta_path=args.data_path,
        target_tokens=args.target_tokens,
        max_length=args.max_length,
        num_workers=args.num_workers,
        test_split=args.test_split,
        seed=args.seed
    )

    print(f"\nTrain batches per epoch: {len(train_dataloader):,}")
    print(f"Test batches: {len(test_dataloader):,}")
    print(f"Masking ratio: {args.mask_ratio}")
    if args.eval_interval > 0:
        print(f"Periodic eval: {args.eval_batches} test batches every {args.eval_interval} training batches")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        # Wrap train dataloader with tqdm for progress tracking
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            step_start_time = time.time()

            # Unpack batch dictionary and move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bucket_size = batch['bucket_size']

            # Apply masking
            masked_input, labels, mlm_mask = apply_mask(
                input_ids,
                mask_token_id=tokenizer.mask_idx,
                mask_ratio=args.mask_ratio
            )

            # Forward pass with mixed precision
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                logits = model(masked_input, mask=attention_mask)
                loss = mlm_loss_function(logits, labels, mlm_mask)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            avg_loss = total_loss / num_batches
            num_seqs = input_ids.shape[0]
            actual_tokens = attention_mask.sum().item()
            step_time = time.time() - step_start_time
            tokens_per_sec = actual_tokens / step_time if step_time > 0 else 0

            # Update progress bar with metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'bucket': bucket_size,
                'seqs': num_seqs,
                'tokens': actual_tokens
            })

            # Log to wandb
            if args.use_wandb and (batch_idx + 1) % args.wandb_log_freq == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/bucket_size": bucket_size,
                    "train/num_sequences": num_seqs,
                    "train/actual_tokens": actual_tokens,
                    "train/step_time": step_time,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                })

            # Log to CSV
            if batch_log_path:
                with open(batch_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        batch_idx + 1,
                        loss.item(),
                        avg_loss,
                        bucket_size,
                        num_seqs,
                        actual_tokens,
                        tokens_per_sec,
                        step_time
                    ])

            # Periodic evaluation on test set sample
            if args.eval_interval > 0 and (batch_idx + 1) % args.eval_interval == 0:
                test_sample_loss = evaluate(
                    model, test_dataloader, tokenizer, device,
                    args.mask_ratio, use_amp, dtype,
                    max_batches=args.eval_batches
                )
                model.train()  # Switch back to training mode

                print(f"\n[Batch {batch_idx+1}] Test sample loss ({args.eval_batches} batches): {test_sample_loss:.4f}")

                # Log to wandb
                if args.use_wandb:
                    wandb.log({
                        "test/sample_loss": test_sample_loss,
                        "test/sample_perplexity": 2.718281828 ** test_sample_loss,
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                    })

        # Epoch summary
        train_avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time

        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        test_loss = evaluate(model, test_dataloader, tokenizer, device, args.mask_ratio, use_amp, dtype)

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.num_epochs} completed")
        print(f"Train Loss: {train_avg_loss:.4f}")
        print(f"Test Loss:  {test_loss:.4f}")
        print(f"Total Batches: {num_batches:,}")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")
        print(f"{'='*80}\n")

        # Log epoch metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch/train_loss": train_avg_loss,
                "epoch/test_loss": test_loss,
                "epoch/time_minutes": epoch_time / 60,
                "epoch/batches": num_batches,
                "epoch": epoch + 1,
            })

        # Log epoch metrics to CSV
        if epoch_log_path:
            with open(epoch_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_avg_loss,
                    test_loss,
                    epoch_time / 60,
                    num_batches
                ])

        # Save checkpoint
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.save_dir}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_avg_loss,
                'test_loss': test_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}\n")

            # Log checkpoint to wandb
            if args.use_wandb:
                wandb.save(checkpoint_path)

    print("Training complete!")

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
        print("WandB run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train protein language model with MLM")

    # Data arguments
    parser.add_argument("--data_path", type=str,
                       default="data/uniparc30_sample_10000000.fasta",
                       help="Path to FASTA file")
    parser.add_argument("--target_tokens", type=int, default=16384,
                       help="Target number of tokens per batch (adjusts batch size dynamically)")
    parser.add_argument("--max_length", type=int, default=16384,
                       help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    parser.add_argument("--test_split", type=float, default=0.1,
                       help="Fraction of data to use for test set (default: 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible train/test splits")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW optimizer (default: 0.01)")
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                       help="Ratio of tokens to mask for MLM")
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use automatic mixed precision (bfloat16) training (default: True)")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false",
                       help="Disable mixed precision training")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint file to resume training from")
    parser.add_argument("--eval_interval", type=int, default=900,
                       help="Evaluate on test set every N training batches (default: 900)")
    parser.add_argument("--eval_batches", type=int, default=100,
                       help="Number of test batches to sample during periodic evaluation (default: 100)")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log every N batches")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Directory to save CSV logs (batch_metrics.csv and epoch_metrics.csv)")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=1,
                       help="Save checkpoint every N epochs")

    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="forgefold",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (defaults to auto-generated)")
    parser.add_argument("--wandb_log_freq", type=int, default=100,
                       help="Log to WandB every N batches")

    args = parser.parse_args()

    main(args)
