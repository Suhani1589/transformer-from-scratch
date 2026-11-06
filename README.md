Layout

app.py              Flask app
main.py             Train and print metrics
data/               CSV
templates/          Login and dashboard
utils/              Data, model, recommendations
model/              Saved joblib bundle


## Transformer from Scratch

```markdown
# Transformer from Scratch

Module layout for a PyTorch Transformer following *Attention Is All You Need*. The repository is the file map for that model. The layer files are still placeholders.

## Intended modules

| File | Role |
| --- | --- |
| `models/embedding.py` | Token embeddings |
| `models/positional_encoding.py` | Sinusoidal positional encoding |
| `models/attention.py` | Scaled dot-product attention |
| `models/multi_head_attention.py` | Multi-head attention with learned Q, K, V, and output projections |
| `models/feed_forward.py` | Position-wise feed-forward block |
| `models/layer_norm.py` | Layer normalization |
| `models/encoder.py` | Encoder stack |
| `models/decoder.py` | Decoder stack |
| `models/transformer.py` | Full model |
| `train.py` | Training loop |
| `inference.py` | Generation |

Each of those files currently contains a docstring and a TODO. `notes/day1.md` is an empty study note. There is no dataset, checkpoint, or measured translation score in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
