from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .model import GPT
from .utils import append_csv, count_parameters, load_model_config


@torch.no_grad()
def measure(model: GPT, seq_len: int, batch_size: int, steps: int, device: torch.device) -> dict[str, float]:
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for _ in range(3):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    return {
        "latency_ms": elapsed * 1000 / steps,
        "tokens_per_s": batch_size * seq_len * steps / elapsed,
        "peak_gb": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(cfg).to(device).eval()
    print(f"parameters={count_parameters(model):,} attention={cfg.attention}")
    for seq_len in args.seq_lens:
        if seq_len > cfg.max_seq_len:
            print(f"seq_len={seq_len} skipped: exceeds max_seq_len={cfg.max_seq_len}")
            continue
        stats = measure(model, seq_len, args.batch_size, args.steps, device)
        print(
            f"seq_len={seq_len} latency_ms={stats['latency_ms']:.2f} "
            f"tok/s={stats['tokens_per_s']:.0f} peak_gb={stats['peak_gb']:.2f}"
        )
        if args.csv:
            append_csv(
                Path(args.csv),
                {
                    "config": args.config,
                    "attention": cfg.attention,
                    "parameters": count_parameters(model),
                    "seq_len": seq_len,
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                    **stats,
                },
            )


if __name__ == "__main__":
    main()
