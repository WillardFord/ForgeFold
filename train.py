"""Training script for ForgeFold protein language model"""

import argparse
import sys
sys.path.insert(0, 'src')

from forgefold import ESMCTokenizer, ProteinLanguageModel, apply_mask, mlm_loss_function
from forgefold.data import load_bucketed_data, create_buckets_from_fasta, create_dataloader
import torch
import torch.optim as optim


def train_epoch(model, buckets, tokenizer, optimizer, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for bucket_size, bucket_seqs in sorted(buckets.items()):
        dataloader = create_dataloader(bucket_seqs, args.batch_size, tokenizer, shuffle=True)

        for input_ids, attention_mask in dataloader:
            masked_input, labels, mlm_mask = apply_mask(input_ids, tokenizer.mask_idx, mask_ratio=args.mask_ratio)

            optimizer.zero_grad()
            logits = model(masked_input, mask=attention_mask)
            loss = mlm_loss_function(logits, labels, mlm_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % args.log_interval == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else 0


def main(args):
    print("Initializing model and tokenizer...")
    tokenizer = ESMCTokenizer()
    model = ProteinLanguageModel()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Loading data...")
    if args.buckets_path:
        buckets = load_bucketed_data(args.buckets_path)
        total_seqs = sum(len(v) for v in buckets.values())
        print(f"Loaded {total_seqs} sequences from pre-bucketed data")
    else:
        buckets = create_buckets_from_fasta(args.data_path, tokenizer)
        total_seqs = sum(len(v) for v in buckets.values())
        print(f"Loaded {total_seqs} sequences from FASTA")

    print(f"Buckets: {[(k, len(v)) for k, v in sorted(buckets.items())]}")

    print(f"\nStarting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*50}")

        avg_loss = train_epoch(model, buckets, tokenizer, optimizer, args)
        print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        if args.save_checkpoints and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.checkpoint_dir}/model_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ForgeFold protein language model")
    parser.add_argument("--buckets_path", type=str, default=None, help="Path to pre-bucketed data")
    parser.add_argument("--data_path", type=str, default="data/uniparc30_sample_100.fasta", help="Path to FASTA file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.15, help="Masking ratio for MLM")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N batches")
    parser.add_argument("--save_checkpoints", action="store_true", help="Save model checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    main(args)
