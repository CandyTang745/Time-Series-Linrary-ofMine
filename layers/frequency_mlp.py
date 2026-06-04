'''
2026.03.09:模型组件设计
'''
import torch
import torch.nn as nn


class FrequencyMLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.real_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.imag_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):

        # FFT
        x_fft = torch.fft.rfft(x, dim=1)

        real = x_fft.real
        imag = x_fft.imag

        real = self.real_mlp(real)
        imag = self.imag_mlp(imag)

        x_fft = torch.complex(real, imag)

        # IFFT
        x = torch.fft.irfft(x_fft, dim=1)

        return x