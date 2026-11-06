"""Token embedding module for the Transformer model."""

import math
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Learnable token embedding layer with Transformer-style scaling."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens) * self.scale


tokens = torch.tensor([
    [1, 5, 8],
    [2, 7, 9]
])

embedding = TokenEmbedding(
    vocab_size=10000,
    d_model=512
)

output = embedding(tokens)

print(tokens.shape)
print(output.shape)
