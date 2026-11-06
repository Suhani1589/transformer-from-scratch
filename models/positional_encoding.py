"""Positional encoding module for the future Transformer model.

This module will eventually define positional representations that inject token
order information into Transformer inputs.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

if __name__ == "__main__":
    pe = PositionalEncoding(d_model=4, max_len=10)

    x = torch.zeros(2, 5, 4)

    output = pe(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
    print(output)