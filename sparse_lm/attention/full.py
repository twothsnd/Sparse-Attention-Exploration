import torch
from torch import nn

from .common import QKVProjection, causal_mask, merge_heads, safe_masked_attention


class FullCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, **_: object):
        super().__init__()
        self.proj = QKVProjection(embed_dim, num_heads, dropout)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.proj.project(x)
        seq_len = x.size(1)
        mask = causal_mask(seq_len, x.device).view(1, 1, seq_len, seq_len)
        y = safe_masked_attention(q, k, v, mask, self.proj.attn_dropout)
        return self.out(merge_heads(y))

