import torch
from torch import nn

from .common import QKVProjection, merge_heads, safe_masked_attention


class BigBirdAttention(nn.Module):
    """Causal BigBird-style block sparse pattern.

    This implementation uses a dense boolean mask for clarity and correctness. The
    sparse pattern is local blocks + prefix global blocks + deterministic random
    historical blocks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        block_size: int = 64,
        local_blocks: int = 2,
        global_blocks: int = 1,
        random_blocks: int = 2,
        seed: int = 13,
        **_: object,
    ):
        super().__init__()
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.global_blocks = global_blocks
        self.random_blocks = random_blocks
        self.seed = seed
        self.proj = QKVProjection(embed_dim, num_heads, dropout)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def make_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        q_block = positions.div(self.block_size, rounding_mode="floor").view(seq_len, 1)
        k_block = positions.div(self.block_size, rounding_mode="floor").view(1, seq_len)
        q_pos = positions.view(seq_len, 1)
        k_pos = positions.view(1, seq_len)

        causal = k_pos <= q_pos
        local = (q_block - k_block >= 0) & (q_block - k_block <= self.local_blocks)
        global_prefix = k_block < self.global_blocks
        mask = causal & (local | global_prefix)

        num_blocks = int((seq_len + self.block_size - 1) // self.block_size)
        if self.random_blocks > 0 and num_blocks > 1:
            for qb in range(num_blocks):
                candidates = torch.arange(0, max(qb, 0), device=device)
                if candidates.numel() == 0:
                    continue
                # Deterministic pseudo-random historical blocks without storing state.
                scores = (candidates * 1103515245 + qb * 12345 + self.seed) % 2147483647
                chosen = candidates[torch.argsort(scores)[: self.random_blocks]]
                block_match = (q_block == qb) & torch.isin(k_block, chosen)
                mask = mask | (causal & block_match)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.proj.project(x)
        seq_len = x.size(1)
        mask = self.make_mask(seq_len, x.device).view(1, 1, seq_len, seq_len)
        y = safe_masked_attention(q, k, v, mask, self.proj.attn_dropout)
        return self.out(merge_heads(y))

