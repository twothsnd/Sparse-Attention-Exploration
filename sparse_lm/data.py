from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterator

import torch


def synthetic_batches(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        x = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
        yield x[:, :-1], x[:, 1:]


def iter_local_text(files: list[str], text_key: str = "text") -> Iterator[str]:
    while True:
        for file_name in files:
            path = Path(file_name)
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(columns=[text_key], batch_size=1024):
                    column = batch.column(text_key).to_pylist()
                    for text in column:
                        if isinstance(text, str) and text.strip():
                            yield text
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = row.get(text_key)
                        if isinstance(text, str) and text.strip():
                            yield text
                else:
                    buffer: list[str] = []
                    for line in f:
                        if line.strip():
                            buffer.append(line)
                        elif buffer:
                            yield "".join(buffer)
                            buffer = []
                    if buffer:
                        yield "".join(buffer)


def local_text_batches(
    files: list[str],
    tokenizer_name: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    text_key: str = "text",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_buffer: list[int] = []
    needed = batch_size * (seq_len + 1)
    for text in iter_local_text(files, text_key=text_key):
        token_buffer.extend(tokenizer.encode(text))
        while len(token_buffer) >= needed:
            chunk = token_buffer[:needed]
            del token_buffer[:needed]
            x = torch.tensor(chunk, dtype=torch.long, device=device).view(batch_size, seq_len + 1)
            yield x[:, :-1], x[:, 1:]


def hf_text_batches(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    tokenizer_name: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    streaming: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_index)
    token_buffer: list[int] = []
    texts = (row.get("text", "") for row in ds)
    for text in texts:
        token_buffer.extend(tokenizer.encode(text))
        needed = batch_size * (seq_len + 1)
        while len(token_buffer) >= needed:
            chunk = token_buffer[:needed]
            del token_buffer[:needed]
            x = torch.tensor(chunk, dtype=torch.long, device=device).view(batch_size, seq_len + 1)
            yield x[:, :-1], x[:, 1:]


def take(iterator, n: int):
    return itertools.islice(iterator, n)
