from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data._\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out) + "\n"


def summarize_training(results_dir: Path) -> str:
    rows = []
    for path in sorted(results_dir.glob("*.metrics.jsonl")):
        events = read_jsonl(path)
        train = [e for e in events if e.get("event") == "train"]
        start = next((e for e in events if e.get("event") == "start"), {})
        if not train:
            continue
        last = train[-1]
        rows.append(
            [
                path.name.replace(".metrics.jsonl", ""),
                start.get("attention", ""),
                start.get("parameters", ""),
                last.get("step", ""),
                f"{float(last.get('loss', 0)):.4f}",
                f"{float(last.get('tokens_per_s', 0)):.0f}",
                last.get("tokens", ""),
            ]
        )
    return markdown_table(["run", "attn", "params", "step", "loss", "tok/s", "tokens"], rows)


def summarize_benchmark(path: Path) -> str:
    rows = []
    for row in read_csv(path):
        rows.append(
            [
                Path(row["config"]).stem,
                row["attention"],
                row["seq_len"],
                f"{float(row['latency_ms']):.2f}",
                f"{float(row['tokens_per_s']):.0f}",
                f"{float(row['peak_gb']):.2f}",
            ]
        )
    return markdown_table(["run", "attn", "seq_len", "latency_ms", "tok/s", "peak_gb"], rows)


def summarize_eval(path: Path) -> str:
    rows = []
    for row in read_csv(path):
        rows.append([Path(row["config"]).stem, row["seq_len"], f"{float(row['loss']):.4f}", f"{float(row['ppl']):.2f}"])
    return markdown_table(["run", "seq_len", "loss", "ppl"], rows)


def summarize_retrieval(path: Path) -> str:
    rows = []
    for row in read_csv(path):
        rows.append(
            [
                Path(row["config"]).stem,
                row["seq_len"],
                row["num_pairs"],
                f"{float(row['loss']):.4f}",
                f"{float(row['accuracy']):.4f}",
            ]
        )
    return markdown_table(["run", "seq_len", "pairs", "loss", "accuracy"], rows)


def summarize_budget(path: Path) -> str:
    rows = []
    for row in read_csv(path):
        rows.append(
            [
                Path(row["config"]).stem,
                row["attention"],
                row["seq_len"],
                f"{float(row['avg_visible']):.1f}",
                f"{float(row['max_visible']):.1f}",
                f"{float(row['budget_ratio']):.4f}",
            ]
        )
    return markdown_table(["run", "attn", "seq_len", "avg_visible", "max_visible", "ratio"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--benchmark_csv", default="results/benchmark.csv")
    parser.add_argument("--eval_csv", default="results/eval.csv")
    parser.add_argument("--retrieval_csv", default="results/retrieval.csv")
    parser.add_argument("--budget_csv", default="results/budget.csv")
    parser.add_argument("--out", default="results/summary.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sections = [
        ("# Experiment Summary", ""),
        ("## Training", summarize_training(results_dir)),
        ("## Benchmark", summarize_benchmark(Path(args.benchmark_csv))),
        ("## Validation", summarize_eval(Path(args.eval_csv))),
        ("## Retrieval", summarize_retrieval(Path(args.retrieval_csv))),
        ("## Attention Budget", summarize_budget(Path(args.budget_csv))),
    ]
    text = "\n\n".join(title + ("\n\n" + body if body else "") for title, body in sections) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
