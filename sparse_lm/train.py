from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from tqdm import trange

from .data import hf_text_batches, local_text_batches, synthetic_batches
from .model import GPT
from .utils import JsonlLogger, count_parameters, load_model_config, load_yaml


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--save_interval", type=int, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    raw = load_yaml(args.config)
    model_config = load_model_config(args.config)
    train_cfg = raw.get("train", {})
    data_cfg = raw.get("data", {})

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = GPT(model_config).to(device)
    raw_model = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )

    batch_size = int(train_cfg.get("batch_size", 4))
    seq_len = int(train_cfg.get("seq_len", model_config.max_seq_len))
    max_steps = args.max_steps or int(train_cfg.get("max_steps", 1000))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum = int(train_cfg.get("grad_accum", 1))

    if args.synthetic:
        batches = synthetic_batches(batch_size, seq_len, model_config.vocab_size, device)
    elif data_cfg.get("source") == "local_text":
        batches = local_text_batches(
            files=list(data_cfg["files"]),
            tokenizer_name=data_cfg.get("tokenizer", "gpt2"),
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            text_key=data_cfg.get("text_key", "text"),
            num_shards=world_size,
            shard_index=rank,
        )
    else:
        batches = hf_text_batches(
            dataset_name=data_cfg.get("name", "Salesforce/wikitext"),
            dataset_config=data_cfg.get("config", "wikitext-103-v1"),
            split=data_cfg.get("split", "train"),
            tokenizer_name=data_cfg.get("tokenizer", "gpt2"),
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            streaming=bool(data_cfg.get("streaming", True)),
            num_shards=world_size,
            shard_index=rank,
        )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    out_dir = Path(args.out_dir)
    config_stem = Path(args.config).stem
    logger = JsonlLogger(out_dir / f"{config_stem}.metrics.jsonl") if rank == 0 else None
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"parameters={count_parameters(raw_model):,} device={device} "
            f"world_size={world_size} attention={model_config.attention}"
        )
        assert logger is not None
        logger.write(
            {
                "event": "start",
                "config": args.config,
                "parameters": count_parameters(raw_model),
                "attention": model_config.attention,
                "world_size": world_size,
                "batch_size_per_rank": batch_size,
                "seq_len": seq_len,
                "max_steps": max_steps,
                "grad_accum": grad_accum,
                "time": time.time(),
            }
        )
    model.train()
    start = time.time()
    pbar = trange(max_steps, disable=rank != 0)
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum):
            input_ids, labels = next(batches)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(input_ids, labels)["loss"] / grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.detach().float()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if rank == 0 and step % int(train_cfg.get("log_interval", 10)) == 0:
            elapsed = max(time.time() - start, 1e-6)
            tokens = (step + 1) * batch_size * seq_len * world_size * grad_accum
            tok_s = tokens / elapsed
            loss_value = total_loss.item()
            pbar.set_description(f"loss={loss_value:.4f} tok/s={tok_s:.0f}")
            assert logger is not None
            logger.write(
                {
                    "event": "train",
                    "step": step,
                    "loss": loss_value,
                    "tokens": tokens,
                    "tokens_per_s": tok_s,
                    "elapsed_s": elapsed,
                    "time": time.time(),
                }
            )
        if args.save_interval and rank == 0 and step > 0 and step % args.save_interval == 0:
            ckpt = out_dir / f"{config_stem}.step{step}.pt"
            torch.save({"model": raw_model.state_dict(), "config": raw, "step": step}, ckpt)

    if rank == 0:
        ckpt = out_dir / f"{config_stem}.pt"
        torch.save({"model": raw_model.state_dict(), "config": raw, "step": max_steps}, ckpt)
        assert logger is not None
        logger.write({"event": "end", "step": max_steps, "checkpoint": str(ckpt), "time": time.time()})
        print(f"saved {ckpt}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
