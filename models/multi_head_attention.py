"""Multi-head attention module for the future Transformer model.

This module will eventually combine multiple attention heads to learn diverse
token relationships in parallel.
"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        )
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )
        scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )
        attention_output = torch.matmul(
            attention_weights,
            V
        )
        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        output = self.W_o(attention_output)

        return output, attention_weights


import torch


batch_size = 2
seq_len = 5
d_model = 512
num_heads = 8


x = torch.randn(
    batch_size,
    seq_len,
    d_model
)


attention = MultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads
)


output, weights = attention(x)


print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)




        

        


