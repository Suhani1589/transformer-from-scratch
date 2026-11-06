# Transformer from Scratch

Module layout for a PyTorch Transformer following *Attention Is All You Need*.

## Modules

| File | Role |
| --- | --- |
| `models/embedding.py` | Token embeddings |
| `models/positional_encoding.py` | Sinusoidal positional encoding |
| `models/attention.py` | Scaled dot-product attention |
| `models/multi_head_attention.py` | Multi-head attention with learned Q, K, V, and output projections |
| `models/feed_forward.py` | Position-wise feed-forward block |
| `models/layer_norm.py` | Layer normalization |



## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: PyTorch, NumPy, pandas, Matplotlib, tqdm.
