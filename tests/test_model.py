import torch

from sparse_lm.model import GPT, GPTConfig


def test_model_forward_loss_all_attention_types():
    for attention in ["full", "sliding", "bigbird", "nsa", "nsa_gated"]:
        cfg = GPTConfig(
            vocab_size=128,
            max_seq_len=32,
            n_layers=2,
            n_heads=4,
            n_embd=32,
            attention=attention,
            attention_kwargs={"block_size": 8, "local_window": 16, "window_size": 16, "top_k_blocks": 2},
        )
        model = GPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 24))
        y = torch.randint(0, cfg.vocab_size, (2, 24))
        out = model(x, y)
        assert out["logits"].shape == (2, 24, cfg.vocab_size)
        assert out["loss"].ndim == 0


def test_no_future_leakage_all_attention_types():
    for attention in ["full", "sliding", "bigbird", "nsa", "nsa_gated"]:
        cfg = GPTConfig(
            vocab_size=128,
            max_seq_len=32,
            n_layers=1,
            n_heads=4,
            n_embd=32,
            attention=attention,
            attention_kwargs={"block_size": 8, "local_window": 16, "window_size": 16, "top_k_blocks": 2},
        )
        model = GPT(cfg).eval()
        x1 = torch.randint(0, cfg.vocab_size, (1, 24))
        x2 = x1.clone()
        x2[:, 13:] = torch.randint(0, cfg.vocab_size, (1, 11))
        with torch.no_grad():
            y1 = model(x1)["logits"][:, :13]
            y2 = model(x2)["logits"][:, :13]
        assert torch.allclose(y1, y2, atol=1e-5), attention
