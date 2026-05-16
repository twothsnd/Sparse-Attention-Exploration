import torch
from torch import nn

from .common import QKVProjection, merge_heads, safe_masked_attention


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        window_size: int = 256,
        **_: object,
    ):
        super().__init__()
        self.window_size = window_size
        self.proj = QKVProjection(embed_dim, num_heads, dropout)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def make_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
        k_pos = torch.arange(seq_len, device=device).view(1, seq_len)
        return (k_pos <= q_pos) & ((q_pos - k_pos) < self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.proj.project(x)
        seq_len = x.size(1)
        mask = self.make_mask(seq_len, x.device).view(1, 1, seq_len, seq_len)
        y = safe_masked_attention(q, k, v, mask, self.proj.attn_dropout)
        return self.out(merge_heads(y))

