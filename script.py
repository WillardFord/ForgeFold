from tokenizer import ESMCTokenizer
from loss import apply_mask, mlm_loss_function
from plm import ProteinLanguageModel
import torch.optim as optim


tokenizer = ESMCTokenizer()
model = ProteinLanguageModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Load sequences from FASTA
fasta_path = "/Users/liyuxin/Documents/zotero/PhD/Course/final_project/ForgeFold/uniparc30_sample_100.fasta"
batch_sequences = tokenizer.read_fasta(fasta_path)

# 1. Tokenize
import torch
from torch.nn.utils.rnn import pad_sequence
encoded_seqs = [torch.tensor(tokenizer.encode(seq)) for seq in batch_sequences]
input_ids = pad_sequence(encoded_seqs, batch_first=True, padding_value=tokenizer.pad_idx)    # [B, L]

# Create attention mask (True for real tokens, False for padding)
attention_mask = (input_ids != tokenizer.pad_idx)

# 2. Mask 50% tokens
masked_input, labels, mlm_mask = apply_mask(
    input_ids,
    mask_token_id=tokenizer.mask_idx,
    mask_ratio=0.5,
)
import pdb; pdb.set_trace()
# 3. Forward through ESMC Transformer
logits = model(masked_input, mask=attention_mask)

# 4. MLM loss
loss = mlm_loss_function(logits, labels, mlm_mask)

loss.backward()
optimizer.step()
