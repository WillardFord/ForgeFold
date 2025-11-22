"""
ForgeFold Protein Language Model
82M parameter transformer for protein sequence encoding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import flash attention (dedicated package)
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# Check for PyTorch 2.0+ built-in flash attention
HAS_PYTORCH_FLASH = hasattr(F, 'scaled_dot_product_attention')

# Determine which backend we're using
if HAS_FLASH_ATTN:
    FLASH_BACKEND = "flash-attn (dedicated)"
elif HAS_PYTORCH_FLASH:
    FLASH_BACKEND = "PyTorch 2.0+ built-in"
else:
    FLASH_BACKEND = "None (using standard attention)"


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) applied to Q and K in each attention layer"""

    def __init__(self, dim, max_seq_len=16384, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[2]

        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot


class SwiGLU_FFN(nn.Module):
    """SwiGLU gated feed-forward network"""

    def __init__(self, d_model=512, expansion=3.0):
        super().__init__()
        ffn_dim = int(d_model * expansion)
        self.proj_in = nn.Linear(d_model, ffn_dim * 2, bias=False)
        self.proj_out = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        return self.proj_out(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with QK normalization and RoPE"""

    def __init__(self, d_model=512, n_heads=8, use_flash=True, max_seq_len=16384):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.k_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, seq_len)

        # Three-level fallback: dedicated flash-attn -> PyTorch flash -> standard
        if self.use_flash and HAS_FLASH_ATTN:
            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v.transpose(1, 2)

            out = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0, softmax_scale=1.0 / math.sqrt(self.head_dim), causal=False)
            out = out.transpose(1, 2)

        elif self.use_flash and HAS_PYTORCH_FLASH:
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(batch, 1, seq_len, seq_len)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(~mask_expanded, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm, attention, and FFN"""

    def __init__(self, d_model=512, n_heads=8, expansion=3.0, residual_scale=0.816, use_flash=True, max_seq_len=16384):
        super().__init__()
        self.residual_scale = residual_scale

        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, use_flash, max_seq_len)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, expansion)

    def forward(self, x, mask=None):
        x = x + self.residual_scale * self.attention(self.attn_norm(x), mask)
        x = x + self.residual_scale * self.ffn(self.ffn_norm(x))
        return x


class SequenceHead(nn.Module):
    """Output head for masked language modeling"""

    def __init__(self, d_model=512, vocab_size=32):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model, bias=True)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.fc2 = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.norm(x)
        return self.fc2(x)


class ProteinLanguageModel(nn.Module):
    """ForgeFold 82M parameter protein language model"""

    def __init__(self, vocab_size=32, d_model=512, n_layers=24, n_heads=8, expansion=3.0, max_seq_len=16384, use_flash=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        residual_scale = math.sqrt(n_layers / 36.0)

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, expansion, residual_scale, use_flash, max_seq_len)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model, bias=False)
        self.output_head = SequenceHead(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, tokens, mask=None, return_embeddings=False):
        x = self.token_embedding(tokens)

        for block in self.blocks:
            x = block(x, mask)

        embeddings = self.final_norm(x)
        logits = self.output_head(embeddings)

        if return_embeddings:
            return logits, embeddings
        return logits

    def get_embeddings(self, tokens, mask=None):
        _, embeddings = self.forward(tokens, mask, return_embeddings=True)
        return embeddings

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_forgefold_plm(vocab_size=32, max_seq_len=16384, use_flash=True):
    """Create ForgeFold PLM with default architecture"""
    model = ProteinLanguageModel(vocab_size=vocab_size, d_model=512, n_layers=24, n_heads=8, expansion=3.0, max_seq_len=max_seq_len, use_flash=use_flash)

    print(f"Created ForgeFold PLM with {model.num_parameters():,} parameters")
    print(f"Flash Attention backend: {FLASH_BACKEND}")
    return model


if __name__ == "__main__":
    model = create_forgefold_plm()

    batch_size = 2
    seq_len = 128
    tokens = torch.randint(0, 32, (batch_size, seq_len))

    print(f"\nTest input: {tokens.shape}")

    logits = model(tokens)
    print(f"Output logits: {logits.shape}")

    embeddings = model.get_embeddings(tokens)
    print(f"Output embeddings: {embeddings.shape}")

    print("\nModel created and tested successfully!")
