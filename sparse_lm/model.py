from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .attention import (
    BigBirdAttention,
    FullCausalSelfAttention,
    GatedNSAAttention,
    NSAAttention,
    SlidingWindowAttention,
)


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    attention: str = "full"
    attention_kwargs: dict[str, Any] = field(default_factory=dict)


def build_attention(config: GPTConfig) -> nn.Module:
    kwargs = dict(config.attention_kwargs)
    common = {
        "embed_dim": config.n_embd,
        "num_heads": config.n_heads,
        "dropout": config.dropout,
    }
    if config.attention == "full":
        return FullCausalSelfAttention(**common, **kwargs)
    if config.attention == "sliding":
        return SlidingWindowAttention(**common, **kwargs)
    if config.attention == "bigbird":
        return BigBirdAttention(**common, **kwargs)
    if config.attention == "nsa":
        return NSAAttention(**common, **kwargs)
    if config.attention == "nsa_gated":
        return GatedNSAAttention(**common, **kwargs)
    raise ValueError(f"unknown attention type: {config.attention}")


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = build_attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).view(1, seq_len)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return out
