import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding
from layers.Linear_extractor import Linear_extractor
from layers.kan import KAN, KANLinear
'''
4.7:这个模型已经调通，需要搬到服务器上运行
'''

# ================= MOE =================
class Expert(nn.Module):
    def __init__(self, input_dim, div):
        super().__init__()
        self.network = KAN([input_dim, input_dim // div, input_dim])

    def forward(self, x):
        return self.network(x)


class MOE(nn.Module):
    def __init__(self, expert_num, input_dim, div):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, div) for _ in range(expert_num)
        ])
        self.router = KANLinear(input_dim, expert_num)

    def forward(self, x):
        # x: [B*, P, D]
        router = torch.softmax(self.router(x), dim=-1)  # [B*, P, E]
        experts_out = torch.stack([e(x) for e in self.experts], dim=-2)  # [B*, P, E, D]
        return torch.einsum('bpe,bped->bpd', router, experts_out)


# ================= TFS =================
class TFS(nn.Module):
    def __init__(self, d_model, configs, patch_num):
        super().__init__()
        self.configs = configs

        self.extractor_his = Linear_extractor(configs)
        self.extractor_cur = Linear_extractor(configs)

        self.weight_linear = nn.Linear(d_model, patch_num)
        self.gate = nn.Linear(d_model, 1)

        self.mlp = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(configs.dropout)

        if configs.aggregated_norm:
            self.norm = nn.LayerNorm(d_model)

        if configs.expert_num > 0:
            self.moe = MOE(configs.expert_num, d_model, configs.kan_div)

    def forward(self, x):
        # x: [B*, P, D]
        origin = x

        # ---- 去非平稳 ----
        if hasattr(self, "moe"):
            x = x - self.moe(x)

        # ---- 历史建模 ----
        H = self.extractor_his(x)

        weight = torch.softmax(self.weight_linear(H), dim=-1)  # [B*, P, P]
        adj = torch.tril(weight)
        aggregated = torch.matmul(adj, x)

        H_history = self.dropout(self.mlp(aggregated))

        # ---- 当前信息 ----
        gate = self.gate(self.extractor_cur(origin))
        H_current = gate * x

        out = H_history + H_current
        if hasattr(self, "norm"):
            out = self.norm(out)

        return out, x   # 保留 stables


# ================= Attention =================
class Attention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(out))


# ================= 主模型 =================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.config = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # ===== Patch =====
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            configs.patch_len,
            configs.stride,
            configs.stride,
            configs.dropout
        )

        self.patch_num = int(
            (configs.seq_len - configs.patch_len) / configs.stride + 2
        )

        # ===== TFS =====
        self.TFSs = nn.ModuleList([
            TFS(configs.d_model, configs, self.patch_num)
            for _ in range(configs.e_layers)
        ])

        # ===== Attention =====
        self.temporal_attn = Attention(configs.d_model, configs.heads, configs.dropout)
        self.freq_attn = Attention(configs.d_model, configs.heads, configs.dropout)

        self.norm_layer = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # ===== Prediction Head（关键恢复）=====
        self.head = nn.Linear(
            2 * configs.d_model * self.patch_num,
            configs.pred_len
        )

    # ===== normalization =====
    def _norm(self, x):
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        return (x - mean) / std, mean, std

    def _denorm(self, x, mean, std):
        return x * std + mean

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x: [B, L, C]
        """

        B, L, C = x.shape

        # ===== normalization =====
        x, mean, std = self._norm(x)

        # ===== reshape → channel-independent =====
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.reshape(B * C, 1, L)  # 每个变量独立

        # ===== patch =====
        enc_out, _ = self.patch_embedding(x)  # [B*C, P, D]

        # ===== TFS =====
        for tfs in self.TFSs:
            agg, stables = tfs(enc_out)
            enc_out = self.norm_layer(enc_out + self.dropout(agg))

        # ===== temporal =====
        H_t = self.temporal_attn(enc_out)

        # ===== frequency（关键：dim=-1）=====
        freq = torch.fft.rfft(enc_out, dim=-1)

        wave = torch.zeros_like(freq.real)
        wave[:, 1:, :] = torch.exp(
            torch.abs(freq[:, 1:, :]) - torch.abs(freq[:, :-1, :])
        )

        k = self.config.k
        _, indices = torch.topk(wave, k, dim=-1)

        mask_f = torch.zeros_like(freq, dtype=torch.bool)
        mask_f.scatter_(-1, indices, True)

        filtered = torch.where(mask_f, freq, torch.zeros_like(freq))
        H_f = torch.fft.irfft(filtered, n=enc_out.shape[-1], dim=-1)

        H_f = self.freq_attn(H_f)

        # ===== fusion =====
        out = torch.cat([H_t, H_f], dim=1)  # [B*C, 2P, D]

        # ===== reshape → head（关键恢复）=====
        out = out.reshape(B * C, -1)  # [B*C, 2P*D]
        out = self.head(out)  # [B*C, pred_len]

        # ===== reshape 回多变量 =====
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # ===== denorm =====
        out = self._denorm(out, mean[:, 0:1, :], std[:, 0:1, :])

        return out