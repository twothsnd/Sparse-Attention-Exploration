from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .data import hf_text_batches, local_text_batches, synthetic_batches
from .model import GPT
from .utils import append_csv, load_model_config, load_yaml


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = load_model_config(args.config)
    data_cfg = raw.get("data", {})
    eval_cfg = raw.get("eval", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(cfg).to(device).eval()
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])

    batch_size = int(eval_cfg.get("batch_size", 2))
    seq_len = int(eval_cfg.get("seq_len", cfg.max_seq_len))
    if args.synthetic:
        batches = synthetic_batches(batch_size, seq_len, cfg.vocab_size, device)
    elif data_cfg.get("source") == "local_text":
        batches = local_text_batches(
            files=list(data_cfg.get("eval_files", data_cfg["files"])),
            tokenizer_name=data_cfg.get("tokenizer", "gpt2"),
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            text_key=data_cfg.get("text_key", "text"),
        )
    else:
        batches = hf_text_batches(
            data_cfg.get("name", "Salesforce/wikitext"),
            data_cfg.get("config", "wikitext-103-v1"),
            data_cfg.get("eval_split", "validation"),
            data_cfg.get("tokenizer", "gpt2"),
            batch_size,
            seq_len,
            device,
            bool(data_cfg.get("streaming", True)),
        )

    losses = []
    for _ in range(args.steps):
        input_ids, labels = next(batches)
        losses.append(model(input_ids, labels)["loss"].item())
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    print(f"loss={mean_loss:.4f} ppl={ppl:.2f}")
    if args.csv:
        append_csv(
            Path(args.csv),
            {
                "config": args.config,
                "ckpt": args.ckpt or "",
                "loss": mean_loss,
                "ppl": ppl,
                "steps": args.steps,
                "seq_len": seq_len,
                "batch_size": batch_size,
            },
        )


if __name__ == "__main__":
    main()
