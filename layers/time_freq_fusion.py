'''
2026.03.09:模型组件设计
'''
import torch
import torch.nn as nn


class TimeFrequencyFusion(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Sigmoid()
        )

        self.proj = nn.Linear(d_model*2, d_model)

    def forward(self, time_feat, freq_feat):

        fusion = torch.cat([time_feat, freq_feat], dim=-1)

        gate = self.gate(fusion)

        fused = self.proj(fusion)

        out = gate * fused + (1-gate) * time_feat

        return out