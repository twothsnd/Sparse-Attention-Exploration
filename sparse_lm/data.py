from __future__ import annotations

import itertools
import json
import glob
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


def expand_files(files: list[str]) -> list[str]:
    expanded: list[str] = []
    for file_name in files:
        matches = sorted(glob.glob(file_name))
        expanded.extend(matches if matches else [file_name])
    return expanded


def iter_local_text(
    files: list[str],
    text_key: str = "text",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[str]:
    files = expand_files(files)
    if not files:
        raise ValueError("no local text files matched")
    file_shard = num_shards > 1 and len(files) >= num_shards
    while True:
        for file_idx, file_name in enumerate(files):
            if file_shard and file_idx % num_shards != shard_index:
                continue
            path = Path(file_name)
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(path)
                row_index = 0
                for batch in parquet_file.iter_batches(columns=[text_key], batch_size=1024):
                    column = batch.column(text_key).to_pylist()
                    for text in column:
                        if not file_shard and num_shards > 1 and row_index % num_shards != shard_index:
                            row_index += 1
                            continue
                        row_index += 1
                        if isinstance(text, str) and text.strip():
                            yield text
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if suffix == ".jsonl":
                    row_index = 0
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = row.get(text_key)
                        if not file_shard and num_shards > 1 and row_index % num_shards != shard_index:
                            row_index += 1
                            continue
                        row_index += 1
                        if isinstance(text, str) and text.strip():
                            yield text
                else:
                    buffer: list[str] = []
                    row_index = 0
                    for line in f:
                        if line.strip():
                            buffer.append(line)
                        elif buffer:
                            if not file_shard and num_shards > 1 and row_index % num_shards != shard_index:
                                row_index += 1
                            else:
                                row_index += 1
                                yield "".join(buffer)
                            buffer = []
                    if buffer:
                        if file_shard or num_shards <= 1 or row_index % num_shards == shard_index:
                            yield "".join(buffer)


def local_text_batches(
    files: list[str],
    tokenizer_name: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    text_key: str = "text",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_buffer: list[int] = []
    needed = batch_size * (seq_len + 1)
    for text in iter_local_text(files, text_key=text_key, num_shards=num_shards, shard_index=shard_index):
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
