import torch

from sparse_lm.attention import (
    BigBirdAttention,
    FullCausalSelfAttention,
    GatedNSAAttention,
    NSAAttention,
    SlidingWindowAttention,
)


def test_attention_shapes():
    x = torch.randn(2, 17, 32)
    modules = [
        FullCausalSelfAttention(32, 4),
        SlidingWindowAttention(32, 4, window_size=8),
        BigBirdAttention(32, 4, block_size=4),
        NSAAttention(32, 4, block_size=4, local_window=8, top_k_blocks=2),
        GatedNSAAttention(32, 4, block_size=4, local_window=8, top_k_blocks=2),
    ]
    for module in modules:
        y = module(x)
        assert y.shape == x.shape


def test_sliding_mask_is_causal():
    attn = SlidingWindowAttention(32, 4, window_size=4)
    mask = attn.make_mask(12, torch.device("cpu"))
    assert not mask[0, 1]
    assert mask[7, 7]
    assert mask[7, 4]
    assert not mask[7, 3]


def test_nsa_completed_blocks_are_causal():
    attn = NSAAttention(32, 4, block_size=4, local_window=8, top_k_blocks=2)
    mask = attn.completed_block_mask(seq_len=12, num_blocks=3, device=torch.device("cpu"))
    assert not mask[0, 0]
    assert mask[3, 0]
    assert not mask[3, 1]
    assert mask[7, 1]
    assert not mask[7, 2]

