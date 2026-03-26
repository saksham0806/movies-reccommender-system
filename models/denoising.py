"""
Module 2: Denoising Self-Attention Transformer
================================================
Implements the core GUARD recommendation model with adaptive binary
masking on self-attention heads to filter noisy implicit feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Adaptive Binary Mask
# ---------------------------------------------------------------------------

class AdaptiveBinaryMask(nn.Module):
    """
    Learnable mask applied to self-attention weights.
    During training the mask values are continuous (sigmoid).
    At inference they are binarised with a configurable threshold.
    """

    def __init__(self, num_heads: int, seq_len: int, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        # One learnable scalar per (head, query-position, key-position)
        self.logits = nn.Parameter(torch.zeros(num_heads, seq_len, seq_len))

    def forward(self, training: bool = True):
        mask = torch.sigmoid(self.logits)
        if not training:
            mask = (mask >= self.threshold).float()
        return mask


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with Denoising Mask
# ---------------------------------------------------------------------------

class DenoisingSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, seq_len: int,
                 dropout: float = 0.1, mask_threshold: float = 0.5):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.mask = AdaptiveBinaryMask(num_heads, seq_len, mask_threshold)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape
        H = self.num_heads

        Q = self.W_q(x).view(B, S, H, self.head_dim).transpose(1, 2)  # B,H,S,d
        K = self.W_k(x).view(B, S, H, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, H, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / self.scale                  # B,H,S,S

        # Apply adaptive binary mask (broadcast over batch)
        binary_mask = self.mask(training=self.training)                 # H,S,S
        attn = attn * binary_mask.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, ffn_dim, dropout, mask_threshold):
        super().__init__()
        self.attn = DenoisingSelfAttention(
            embed_dim, num_heads, seq_len, dropout, mask_threshold
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Full Denoising Transformer Model
# ---------------------------------------------------------------------------

class DenoisingTransformer(nn.Module):
    """
    GUARD Denoising Transformer for implicit-feedback recommendation.

    Input : (user_ids, item_ids)  — each of shape (B,)
    Output: prediction scores      — shape (B,)  in [0, 1]
    """

    def __init__(self, num_users: int, num_items: int,
                 embed_dim: int = 64, num_heads: int = 4,
                 ffn_dim: int = 128, dropout: float = 0.2,
                 mask_threshold: float = 0.5, num_layers: int = 2,
                 seq_len: int = 2):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, seq_len, ffn_dim,
                             dropout, mask_threshold)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        Args:
            user_ids: (B,) int tensor
            item_ids: (B,) int tensor
        Returns:
            scores:   (B,) float tensor of predicted interaction probabilities
        """
        u = self.user_emb(user_ids)   # B, D
        i = self.item_emb(item_ids)   # B, D

        # Treat (user_emb, item_emb) as a length-2 sequence
        seq = torch.stack([u, i], dim=1)  # B, 2, D
        pos = self.pos_emb(torch.arange(2, device=seq.device))
        seq = self.dropout(seq + pos)

        for block in self.blocks:
            seq = block(seq)

        # Pool: take mean of sequence positions
        pooled = seq.mean(dim=1)       # B, D
        logits = self.head(pooled).squeeze(-1)  # B
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(user_id: int, item_ids: list, model: DenoisingTransformer,
            device: str = "cpu") -> list:
    """
    Predict interaction scores for a single user across a list of item ids.

    Returns:
        List of (item_id, score) tuples sorted by descending score.
    """
    model.eval()
    user_tensor = torch.tensor([user_id] * len(item_ids), dtype=torch.long, device=device)
    item_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
    scores = model(user_tensor, item_tensor).cpu().numpy()
    results = sorted(zip(item_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    return results
