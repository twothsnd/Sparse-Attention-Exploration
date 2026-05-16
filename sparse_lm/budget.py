from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .attention import BigBirdAttention, SlidingWindowAttention
from .attention.common import causal_mask
from .model import build_attention
from .utils import append_csv, load_model_config


def nsa_visible_counts(seq_len: int, block_size: int, local_window: int, top_k_blocks: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(seq_len, device=device)
    local = torch.minimum(q_pos + 1, torch.full_like(q_pos, local_window))
    completed_blocks = torch.div(q_pos + 1, block_size, rounding_mode="floor")
    compressed = completed_blocks
    selected = torch.minimum(completed_blocks, torch.full_like(completed_blocks, top_k_blocks)) * block_size
    return local + compressed + selected


def visible_counts(config_path: str, seq_len: int) -> dict[str, float]:
    cfg = load_model_config(config_path)
    device = torch.device("cpu")
    full_counts = causal_mask(seq_len, device).sum(dim=-1).float()
    attention = build_attention(cfg)
    if cfg.attention == "full":
        counts = full_counts
    elif isinstance(attention, SlidingWindowAttention):
        counts = attention.make_mask(seq_len, device).sum(dim=-1).float()
    elif isinstance(attention, BigBirdAttention):
        counts = attention.make_mask(seq_len, device).sum(dim=-1).float()
    elif cfg.attention in {"nsa", "nsa_gated"}:
        kwargs = cfg.attention_kwargs
        counts = nsa_visible_counts(
            seq_len,
            int(kwargs.get("block_size", 64)),
            int(kwargs.get("local_window", 256)),
            int(kwargs.get("top_k_blocks", 4)),
            device,
        ).float()
    else:
        raise ValueError(f"unsupported attention for budget: {cfg.attention}")
    return {
        "config": config_path,
        "attention": cfg.attention,
        "seq_len": seq_len,
        "avg_visible": counts.mean().item(),
        "max_visible": counts.max().item(),
        "full_avg_visible": full_counts.mean().item(),
        "budget_ratio": (counts.sum() / full_counts.sum()).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    for config in args.configs:
        for seq_len in args.seq_lens:
            stats = visible_counts(config, seq_len)
            print(
                f"{Path(config).stem} seq_len={seq_len} avg_visible={stats['avg_visible']:.1f} "
                f"max_visible={stats['max_visible']:.1f} ratio={stats['budget_ratio']:.4f}"
            )
            if args.csv:
                append_csv(args.csv, stats)


if __name__ == "__main__":
    main()

