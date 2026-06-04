'''
2026.03.09:模型组件设计
'''
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, n_heads, num_layers):

        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x