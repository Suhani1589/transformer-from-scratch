"""Attention module for the future Transformer model.

This module will eventually define scaled dot product attention and related
masking utilities.
"""

import torch
def scaled_dot_product_attention(Q,K,V, mask=None):
    scores = Q @ K.transpose(-2, -1)
    d_k = Q.size(-1)
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V 
    return output, attention_weights