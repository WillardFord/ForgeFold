from tokenizer import ESMCTokenizer
from loss import apply_mask, mlm_loss_function
from plm import ProteinLanguageModel
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import argparse



## TODO: This is a temporary dataset class, we need to modify it to be a proper dataset class
class ProteinDataset(Dataset):
    def __init__(self, fasta_path, tokenizer):
        self.sequences = tokenizer.read_fasta(fasta_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def get_bucket(length):
    buckets = [2**i for i in range(7, 15)]  # [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]

def main(args):
    # Setup
    tokenizer = ESMCTokenizer()
    model = ProteinLanguageModel()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def collate_fn(batch):
        encoded = [torch.tensor(tokenizer.encode(seq)) for seq in batch]
        input_ids = pad_sequence(encoded, batch_first=True, padding_value=tokenizer.pad_idx)
        attention_mask = (input_ids != tokenizer.pad_idx)
        return input_ids, attention_mask

    # Load sequences
    ### TODO: This is a temporary dataset class, we need to modify it to be a proper dataset class
    dataset = ProteinDataset(args.data_path, tokenizer)
    sequences = dataset.sequences

    buckets = defaultdict(list)
    for seq in sequences:
        bucket = get_bucket(len(seq))
        buckets[bucket].append(seq)

    print(f"Loaded {len(sequences)} sequences")
    print(f"Buckets: {[(k, len(v)) for k, v in sorted(buckets.items())]}")

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0
        num_batches = 0

        for bucket_size, bucket_seqs in sorted(buckets.items()):
            bucket_dataset = ProteinDataset.__new__(ProteinDataset)
            bucket_dataset.sequences = bucket_seqs
            bucket_dataset.tokenizer = tokenizer

            dataloader = DataLoader(bucket_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

            for input_ids, attention_mask in dataloader:
                # Mask tokens
                masked_input, labels, mlm_mask = apply_mask(input_ids, tokenizer.mask_idx, mask_ratio=0.15)

                # Forward
                optimizer.zero_grad()
                logits = model(masked_input, mask=attention_mask)
                loss = mlm_loss_function(logits, labels, mlm_mask)

                # Backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train protein language model")
    ### TODO: This is a temporary dataset class, we need to modify it to be a proper dataset class
    parser.add_argument("--data_path", type=str, default="/Users/liyuxin/Documents/zotero/PhD/Course/final_project/ForgeFold/uniparc30_sample_100.fasta", help="Path to FASTA file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    main(args)
