import torch
from torch import nn
import torch.nn.functional as F

from .common import QKVProjection, merge_heads, safe_masked_attention


class NSAAttention(nn.Module):
    """Principle-correct NSA-style attention.

    Branches:
    1. local sliding-window attention over recent tokens
    2. compressed attention over completed historical block summaries
    3. selected-block attention over original tokens from top-k historical blocks

    The compressed and selected branches only use blocks whose end position is not
    in the future relative to each query token.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        block_size: int = 64,
        local_window: int = 256,
        top_k_blocks: int = 4,
        **_: object,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.local_window = local_window
        self.top_k_blocks = top_k_blocks
        self.proj = QKVProjection(embed_dim, num_heads, dropout)
        self.out = nn.Linear(3 * embed_dim, embed_dim, bias=False)

    def local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
        k_pos = torch.arange(seq_len, device=device).view(1, seq_len)
        return (k_pos <= q_pos) & ((q_pos - k_pos) < self.local_window)

    def block_summaries(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, dim = x.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size - seq_len
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        x = x.view(batch, heads, num_blocks, self.block_size, dim)
        valid = torch.ones(seq_len, device=x.device, dtype=x.dtype)
        if pad_len:
            valid = F.pad(valid, (0, pad_len))
        valid = valid.view(num_blocks, self.block_size)
        denom = valid.sum(dim=1).clamp_min(1.0).view(1, 1, num_blocks, 1)
        return (x * valid.view(1, 1, num_blocks, self.block_size, 1)).sum(dim=3) / denom

    def completed_block_mask(self, seq_len: int, num_blocks: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
        block_end = (torch.arange(num_blocks, device=device).view(1, num_blocks) + 1) * self.block_size - 1
        return block_end <= q_pos

    def selected_token_mask(self, q: torch.Tensor, block_k: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch, heads, _, _ = q.shape
        num_blocks = block_k.size(2)
        device = q.device
        block_scores = torch.matmul(q, block_k.transpose(-2, -1))
        allowed_blocks = self.completed_block_mask(seq_len, num_blocks, device).view(1, 1, seq_len, num_blocks)
        block_scores = block_scores.masked_fill(~allowed_blocks, torch.finfo(block_scores.dtype).min)

        k = min(self.top_k_blocks, num_blocks)
        top_scores, top_idx = torch.topk(block_scores, k=k, dim=-1)
        top_valid = torch.isfinite(top_scores) & (top_scores > torch.finfo(top_scores.dtype).min / 2)
        selected_blocks = torch.zeros(batch, heads, seq_len, num_blocks, dtype=torch.bool, device=device)
        selected_blocks.scatter_(-1, top_idx, top_valid)

        key_block = torch.arange(seq_len, device=device).div(self.block_size, rounding_mode="floor")
        token_mask = selected_blocks.index_select(-1, key_block)
        causal = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1) >= torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
        return token_mask & causal

    def branch_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.proj.project(x)
        seq_len = x.size(1)
        local = safe_masked_attention(
            q,
            k,
            v,
            self.local_mask(seq_len, x.device).view(1, 1, seq_len, seq_len),
            self.proj.attn_dropout,
        )

        block_k = self.block_summaries(k)
        block_v = self.block_summaries(v)
        num_blocks = block_k.size(2)
        compressed_mask = self.completed_block_mask(seq_len, num_blocks, x.device).view(1, 1, seq_len, num_blocks)
        compressed = safe_masked_attention(q, block_k, block_v, compressed_mask, self.proj.attn_dropout)

        selected_mask = self.selected_token_mask(q, block_k, seq_len)
        selected = safe_masked_attention(q, k, v, selected_mask, self.proj.attn_dropout)
        return merge_heads(local), merge_heads(compressed), merge_heads(selected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local, compressed, selected = self.branch_outputs(x)
        return self.out(torch.cat([local, compressed, selected], dim=-1))


class GatedNSAAttention(NSAAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, **kwargs: object):
        super().__init__(embed_dim, num_heads, dropout, **kwargs)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate = nn.Linear(embed_dim, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local, compressed, selected = self.branch_outputs(x)
        weights = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        stacked = torch.stack([local, compressed, selected], dim=-2)
        merged = (stacked * weights).sum(dim=-2)
        return self.out(merged)

