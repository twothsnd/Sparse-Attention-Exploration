import math
from typing import Optional

import torch
from torch import nn


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq_len, width = x.shape
    head_dim = width // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def safe_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    dropout: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Scaled dot-product attention that returns zero for rows with no valid keys.

    q: [B, H, Tq, D]
    k/v: [B, H, Tk, D]
    mask: broadcastable to [B, H, Tq, Tk], True means key is visible.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = mask.to(torch.bool)
    valid_rows = mask.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid_rows, probs, torch.zeros_like(probs))
    if dropout is not None:
        probs = dropout(probs)
    return torch.matmul(probs, v)


class QKVProjection(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return split_heads(q, self.num_heads), split_heads(k, self.num_heads), split_heads(v, self.num_heads)

