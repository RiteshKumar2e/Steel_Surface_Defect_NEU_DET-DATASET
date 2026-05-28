import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                              # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)                 # each (B, T, C)

        # reshape to (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale     # (B, heads, T, T)

        # causal mask: upper triangle → -inf
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                                # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm residual
        x = x + self.ff(self.ln2(x))
        return x


class SteelGPT(nn.Module):
    """
    Decoder-only GPT-style transformer trained on steel defect descriptions.
    Implements every layer from scratch using only PyTorch primitives.
    """

    def __init__(self, vocab_size: int, block_size: int, embed_dim: int,
                 num_heads: int, num_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.block_size = block_size

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # weight tying: share token embedding and output projection weights
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.token_embed(idx) + self.pos_embed(positions))
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)                           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 40) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
