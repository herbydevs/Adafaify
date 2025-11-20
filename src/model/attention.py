import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """Implements scaled dot-product attention with optional causal masking.

    If causal=True, the module will mask out "future" positions so that
    the output at time t cannot attend to positions > t (useful in decoder).

    This module implements attention in a simple but efficient way and supports
    the keys/values/queries being of different source (used for cross-attention).
    """
    def __init__(self, embed_dim, num_heads=4, causal=False, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, q):
        # k: (B, T_k, C), v: (B, T_k, C), q: (B, T_q, C)
        B, T_k, C = k.shape
        Bq, T_q, Cq = q.shape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # reshape for heads: (B, heads, T, head_dim)
        q = q.view(Bq, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # causal mask if requested (applies when T_q <= T_k and causal True)
        if self.causal:
            # Create lower-triangular mask for q attending to k (T_q x T_k)
            mask = torch.tril(torch.ones(T_q, T_k, device=attn_scores.device)).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # attention output
        out = torch.matmul(attn, v)  # (B, heads, T_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(Bq, T_q, C)
        out = self.out_proj(out)
        return out