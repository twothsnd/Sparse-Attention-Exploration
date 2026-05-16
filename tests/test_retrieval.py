import torch

from sparse_lm.model import GPT, GPTConfig
from sparse_lm.retrieval import evaluate_retrieval, retrieval_batch


def test_retrieval_batch_shapes():
    x, targets, query = retrieval_batch(
        batch_size=3,
        seq_len=64,
        vocab_size=256,
        num_pairs=8,
        device=torch.device("cpu"),
    )
    assert x.shape == (3, 64)
    assert targets.shape == (3,)
    assert query.shape == (3,)


def test_retrieval_eval_runs():
    cfg = GPTConfig(
        vocab_size=256,
        max_seq_len=64,
        n_layers=1,
        n_heads=2,
        n_embd=32,
        attention="sliding",
        attention_kwargs={"window_size": 32},
    )
    model = GPT(cfg)
    stats = evaluate_retrieval(model, batch_size=2, seq_len=48, num_pairs=6, steps=2, device=torch.device("cpu"))
    assert 0.0 <= stats["accuracy"] <= 1.0
    assert stats["loss"] > 0

