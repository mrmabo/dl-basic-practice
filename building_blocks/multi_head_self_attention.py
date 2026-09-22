"""A from-scratch PyTorch implementation of multi-head self-attention."""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention without using ``nn.MultiheadAttention``.

    Args:
        d_model: Size of each input embedding.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, d_model=32, num_heads=4, dropout=0.0):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Each projection learns a different role for the same input:
        # Q asks what to find, K describes what can be matched, and V carries content.
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        """[B, S, D] -> [B, H, S, D/H]."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        """Compute multi-head self-attention.

        Args:
            x: Input tensor with shape ``[batch_size, seq_len, d_model]``.
            mask: Optional boolean mask broadcastable to ``[B, H, S, S]``.
                ``True`` keeps an attention connection and ``False`` blocks it.

        Returns:
            output: Contextualized features with shape ``[B, S, D]``.
            attention_weights: Attention maps with shape ``[B, H, S, S]``.
        """
        batch_size, seq_len, _ = x.shape  # [B, S, D]

        q = self._split_heads(self.q_projection(x))  # [B, H, S, D/H]
        k = self._split_heads(self.k_projection(x))  # [B, H, S, D/H]
        v = self._split_heads(self.v_projection(x))  # [B, H, S, D/H]

        # Each query position compares itself with every key position.
        scores = q @ k.transpose(-2, -1)  # [B, H, S, S]
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError("mask must be a boolean tensor")
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)  # [B, H, S, S]
        dropped_weights = self.attention_dropout(attention_weights)
        context = dropped_weights @ v  # [B, H, S, D/H]

        # Join all heads: [B, H, S, D/H] -> [B, S, D].
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, self.d_model)
        output = self.output_projection(context)  # [B, S, D]

        return output, attention_weights


if __name__ == "__main__":
    inputs = torch.randn(2, 10, 32)  # [B=2, S=10, D=32]
    attention = MultiHeadSelfAttention(d_model=32, num_heads=4, dropout=0.1)
    outputs, weights = attention(inputs)

    print("input:", inputs.shape)
    print("output:", outputs.shape)
    print("attention weights:", weights.shape)

    assert outputs.shape == (2, 10, 32)
    assert weights.shape == (2, 4, 10, 10)
