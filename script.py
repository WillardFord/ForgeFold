"""
Training script for protein language model with MLM objective
Uses bucketed dataloader with constant token budget for efficient training
"""

from loss import apply_mask, mlm_loss_function
from plm import ProteinLanguageModel
from dataloader import create_dataloader
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
import wandb
import time


def main(args):
    print("=" * 80)
    print("Protein Language Model Training")
    print("=" * 80)

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
                "mask_ratio": args.mask_ratio,
                "optimizer": "Adam",
                "device": str(device),
            }
        )
        print(f"\nWandB initialized: {args.wandb_project}/{args.wandb_run_name}")

    # Setup model and optimizer
    print("\nInitializing model...")
    model = ProteinLanguageModel()
    model = model.to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=args.wandb_log_freq)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Optimizer: Adam (lr={args.lr})")

    # Load data with our bucketed dataloader
    print(f"\nLoading data from: {args.data_path}")
    print(f"Target tokens per batch: {args.target_tokens}")
    print(f"Max sequence length: {args.max_length}")

    dataloader, tokenizer = create_dataloader(
        fasta_path=args.data_path,
        target_tokens=args.target_tokens,
        shuffle=True,
        max_length=args.max_length,
        num_workers=args.num_workers
    )

    print(f"\nTotal batches per epoch: {len(dataloader):,}")
    print(f"Masking ratio: {args.mask_ratio}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        # Wrap dataloader with tqdm for progress tracking
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

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

            # Forward pass
            optimizer.zero_grad()
            logits = model(masked_input, mask=attention_mask)
            loss = mlm_loss_function(logits, labels, mlm_mask)

            # Backward pass
            loss.backward()
            optimizer.step()

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

        # Epoch summary
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.num_epochs} completed")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Total Batches: {num_batches:,}")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")
        print(f"{'='*80}\n")

        # Log epoch metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/time_minutes": epoch_time / 60,
                "epoch/batches": num_batches,
                "epoch": epoch + 1,
            })

        # Save checkpoint
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.save_dir}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
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

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                       help="Ratio of tokens to mask for MLM")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log every N batches")
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
