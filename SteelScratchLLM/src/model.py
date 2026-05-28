import json
import torch
import torch.nn as nn
from config import MAX_LEN

class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<pad>": 0, "<unk>": 1}

    def fit(self, texts):
        for text in texts:
            for tok in text.split():
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    def encode(self, text, max_len=MAX_LEN):
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in text.split()]
        ids = ids[:max_len]
        ids += [self.vocab["<pad>"]] * (max_len - len(ids))
        return ids

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

class ScratchTinyLLM(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, nhead=4, num_layers=3, max_len=MAX_LEN, dropout=0.15):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        pad_mask = input_ids.eq(0)
        x = self.token_emb(input_ids) + self.pos_emb[:, :input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(pooled)
        return self.head(pooled)
