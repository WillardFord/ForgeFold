from tokenizer import ESMCTokenizer
from loss import apply_mask, mlm_loss_function
from model import ESMCModel
import torch.optim as optim

tokenizer = ESMCTokenizer()
model = ESMCModel()
optimizer = optim.Adam(model.transformer_encoder.parameters(), lr=1e-4)

# 1. Tokenize
input_ids = tokenizer.encode(batch_sequences)    # [B, L]

# 2. Mask 50% tokens
masked_input, labels, mask = apply_mask(
    input_ids,
    mask_token_id=tokenizer.mask_idx,
    mask_ratio=0.5,
)

# 3. Forward through ESMC Transformer
outputs = model(sequence_tokens=masked_input)
logits = outputs.sequence_logits   # [B, L, vocab_size]

# 4. MLM loss
loss = mlm_loss_function(logits, labels, mask)

loss.backward()
optimizer.step()
