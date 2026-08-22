import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Enhanced WCFIN:
    1. Frequency Embedding（提升频谱表达）
    2. Local Frequency Interaction（原WCFIN）
    3. Global Frequency MLP（补FreTS能力）
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # ===== 频率维度 =====
        self.freq_len = self.seq_len // 2 + 1

        # ===== embedding（关键改进1）=====
        self.embed_dim = 64
        self.embedding = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))

        # ===== Local Frequency Interaction（你原来的思想，简化版）=====
        self.local_real = nn.Linear(self.embed_dim, self.embed_dim)
        self.local_imag = nn.Linear(self.embed_dim, self.embed_dim)

        # ===== Global Frequency MLP（关键改进2）=====
        self.global_real = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.GELU(),
            nn.Linear(self.freq_len, self.freq_len)
        )

        self.global_imag = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.GELU(),
            nn.Linear(self.freq_len, self.freq_len)
        )

        # ===== 融合系数（后续可改为learnable）=====
        self.alpha = 0.5

        # ===== 输出层 =====
        self.projection = nn.Linear(self.seq_len * self.embed_dim, self.pred_len)

    # ===== Embedding =====
    def token_embedding(self, x):
        # x: [B, T, N]
        x = x.permute(0, 2, 1)  # [B, N, T]
        x = x.unsqueeze(-1)     # [B, N, T, 1]
        return x * self.embedding  # [B, N, T, D]

    # ===== Local Frequency Interaction =====
    def local_freq_interaction(self, Xf):
        real = Xf.real
        imag = Xf.imag

        real = self.local_real(real)
        imag = self.local_imag(imag)

        return torch.complex(real, imag)

    # ===== Global Frequency Modeling =====
    def global_freq_mlp(self, Xf):
        real = Xf.real
        imag = Xf.imag

        # reshape: [B, N, F, D] → [B, N, D, F]
        real = real.permute(0, 1, 3, 2)
        imag = imag.permute(0, 1, 3, 2)

        real = self.global_real(real)
        imag = self.global_imag(imag)

        # reshape back
        real = real.permute(0, 1, 3, 2)
        imag = imag.permute(0, 1, 3, 2)

        return torch.complex(real, imag)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [B, T, N]
        """

        B, T, N = x_enc.shape

        # ===== Step1: embedding =====
        x = self.token_embedding(x_enc)  # [B, N, T, D]

        # ===== Step2: FFT =====
        Xf = torch.fft.rfft(x, dim=2)  # [B, N, F, D]

        # ===== Step3: Local =====
        X_local = self.local_freq_interaction(Xf)

        # ===== Step4: Global =====
        X_global = self.global_freq_mlp(Xf)

        # ===== Step5: Fusion =====
        Xf_new = self.alpha * X_local + (1 - self.alpha) * X_global

        # ===== Step6: IFFT =====
        x_time = torch.fft.irfft(Xf_new, n=self.seq_len, dim=2)

        # ===== Step7: flatten + prediction =====
        x_time = x_time.reshape(B, N, -1)

        out = self.projection(x_time)  # [B, N, pred_len]

        out = out.permute(0, 2, 1)  # [B, pred_len, N]

        return out