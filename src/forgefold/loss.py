import torch
import torch.nn as nn

def apply_mask(input_ids, mask_token_id, mask_ratio=0.5):
    """
    input_ids: [B, L]
    return:
      masked_input: masked version of input_ids
      labels:       original tokens
      mask:         boolean mask tensor
    """

    B, L = input_ids.shape
    labels = input_ids.clone()

    # Random mask positions
    rand = torch.rand(B, L, device=input_ids.device)
    mask = rand < mask_ratio

    # masked_input = replace with mask token
    masked_input = input_ids.clone()
    masked_input[mask] = mask_token_id

    return masked_input, labels, mask


def mlm_loss_function(logits, labels, mask, ignore_index=-100):
    """
    logits: [B, L, V]
    labels: [B, L]
    mask:   [B, L]  boolean mask (True → this position should contribute to loss)
    """

    # Flatten all positions
    vocab_size = logits.size(-1)
    logits = logits.view(-1, vocab_size)       # [B*L, V]
    labels = labels.view(-1)                   # [B*L]
    mask = mask.view(-1)                       # [B*L]

    # Only keep the masked positions
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    # Cross entropy across masked positions only
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(masked_logits, masked_labels)
    
    return loss
