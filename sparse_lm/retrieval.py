from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import trange

from .model import GPT
from .utils import append_csv, load_model_config, load_yaml


def retrieval_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a key-value retrieval batch.

    Sequence layout:
    noise / key value pairs / noise / query_key

    The target is the value paired with query_key. Accuracy is measured on the
    model's next-token prediction at the final position.
    """
    if seq_len < 4 * num_pairs + 8:
        raise ValueError("seq_len too small for requested num_pairs")
    reserved = 8
    noise_low = 10 + 2 * num_pairs
    x = torch.randint(noise_low, vocab_size, (batch_size, seq_len), device=device)
    keys = torch.arange(10, 10 + num_pairs, device=device)
    values = torch.arange(10 + num_pairs, 10 + 2 * num_pairs, device=device)
    query_index = torch.randint(0, num_pairs, (batch_size,), device=device)

    pair_start = reserved
    for b in range(batch_size):
        perm = torch.randperm(num_pairs, device=device)
        for j, pair_idx in enumerate(perm):
            pos = pair_start + 2 * j
            x[b, pos] = keys[pair_idx]
            x[b, pos + 1] = values[pair_idx]
        x[b, -1] = keys[query_index[b]]

    targets = values[query_index]
    return x, targets, query_index


@torch.no_grad()
def evaluate_retrieval(
    model: GPT,
    batch_size: int,
    seq_len: int,
    num_pairs: int,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    correct = 0
    total = 0
    losses = []
    for _ in range(steps):
        x, targets, _ = retrieval_batch(batch_size, seq_len, model.config.vocab_size, num_pairs, device)
        logits = model(x)["logits"][:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, targets)
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
        losses.append(loss.item())
    return {"accuracy": correct / total, "loss": sum(losses) / len(losses)}


def train_retrieval(
    model: GPT,
    batch_size: int,
    seq_len: int,
    num_pairs: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> None:
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    for step in trange(steps):
        x, targets, _ = retrieval_batch(batch_size, seq_len, model.config.vocab_size, num_pairs, device)
        logits = model(x)["logits"][:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            pred = logits.argmax(dim=-1)
            acc = (pred == targets).float().mean().item()
            trange.write(f"step={step} loss={loss.item():.4f} acc={acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_pairs", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--finetune_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_ckpt", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = load_model_config(args.config)
    seq_len = args.seq_len or min(cfg.max_seq_len, 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(cfg).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])

    if args.finetune_steps > 0:
        train_retrieval(model, args.batch_size, seq_len, args.num_pairs, args.finetune_steps, args.lr, device)
        if args.save_ckpt:
            path = Path(args.save_ckpt)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": raw}, path)

    model.eval()
    stats = evaluate_retrieval(model, args.batch_size, seq_len, args.num_pairs, args.eval_steps, device)
    print(f"retrieval_loss={stats['loss']:.4f} retrieval_acc={stats['accuracy']:.4f}")
    if args.csv:
        append_csv(
            args.csv,
            {
                "config": args.config,
                "ckpt": args.ckpt or "",
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "num_pairs": args.num_pairs,
                "eval_steps": args.eval_steps,
                "finetune_steps": args.finetune_steps,
                **stats,
            },
        )


if __name__ == "__main__":
    main()

