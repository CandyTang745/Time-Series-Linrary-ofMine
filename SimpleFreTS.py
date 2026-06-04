import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simplified FreTS
    核心：FFT → 全局频谱MLP → IFFT → 线性预测
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in  # N（实例数）

        # ===== 频率维度 =====
        self.freq_len = self.seq_len // 2 + 1

        # ===== 全局频谱MLP（核心）=====
        self.real_mlp = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.GELU(),
            nn.Linear(self.freq_len, self.freq_len)
        )

        self.imag_mlp = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.GELU(),
            nn.Linear(self.freq_len, self.freq_len)
        )

        # ===== 输出映射 =====
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [B, T, N]
        return: [B, pred_len, N]
        """

        B, T, N = x_enc.shape

        # ===== 调整为 [B, N, T] =====
        x = x_enc.permute(0, 2, 1)

        # ===== FFT =====
        Xf = torch.fft.rfft(x, dim=-1)   # [B, N, F]

        real = Xf.real
        imag = Xf.imag

        # ===== 全局频谱建模（关键）=====
        real = self.real_mlp(real)
        imag = self.imag_mlp(imag)

        Xf_new = torch.complex(real, imag)

        # ===== IFFT =====
        x_time = torch.fft.irfft(Xf_new, n=self.seq_len, dim=-1)

        # ===== 预测 =====
        out = self.projection(x_time)  # [B, N, pred_len]

        out = out.permute(0, 2, 1)  # [B, pred_len, N]

        return out