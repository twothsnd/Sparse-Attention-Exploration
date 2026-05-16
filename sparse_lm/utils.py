from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import yaml

from .model import GPTConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path) -> GPTConfig:
    raw = load_yaml(path)
    model = raw.get("model", raw)
    valid = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in model.items() if k in valid})


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
