import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencySelector(nn.Module):
    def __init__(self, d_model, sparsity_threshold=0.01):
        super().__init__()

        self.sparsity_threshold = sparsity_threshold

        # 类似 FreTS 的复数线性变换
        self.r = nn.Parameter(0.02 * torch.randn(d_model, d_model))
        self.i = nn.Parameter(0.02 * torch.randn(d_model, d_model))

        self.rb = nn.Parameter(torch.zeros(d_model))
        self.ib = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: complex [B, L, D]

        real = x.real
        imag = x.imag

        real_out = torch.einsum('bld,dd->bld', real, self.r) - \
                   torch.einsum('bld,dd->bld', imag, self.i) + self.rb

        imag_out = torch.einsum('bld,dd->bld', imag, self.r) + \
                   torch.einsum('bld,dd->bld', real, self.i) + self.ib

        out = torch.stack([real_out, imag_out], dim=-1)

        #FreTS核心：稀疏化
        out = F.softshrink(out, lambd=self.sparsity_threshold)

        out = torch.view_as_complex(out)

        return out
    
class CycleFrequencyBias(nn.Module):
    def __init__(self, seq_len):
        super().__init__()

        self.freq_len = seq_len // 2 + 1

        # 可学习频率权重（类似 Tweight）
        self.weight = nn.Parameter(torch.ones(self.freq_len))

    def forward(self, x):
        # x: [B, L_freq, D]

        weight = torch.softmax(self.weight, dim=0)

        return x * weight.view(1, -1, 1)

class EnhancedComplexInteraction(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.gate = nn.Linear(d_model, d_model)

        # 新增：frequency-aware scaling
        self.freq_scale = nn.Parameter(torch.ones(1))

    def forward(self, F_trend, F_detail):

        real_trend = F_trend.real

        gate = torch.sigmoid(self.gate(real_trend))

        real = F_detail.real * gate
        imag = F_detail.imag * gate

        F_detail_refined = torch.complex(real, imag)

        # ⭐ 增强：频率尺度调节
        F_fused = F_trend + self.freq_scale * F_detail_refined

        return F_fused
class HaarWavelet(nn.Module):

    def __init__(self):
        super().__init__()

        lp = torch.tensor([0.7071,0.7071]).view(1,1,2)
        hp = torch.tensor([-0.7071,0.7071]).view(1,1,2)

        self.register_buffer("lp", lp)
        self.register_buffer("hp", hp)


    def forward(self,x):

        """
        x: [B, L, D]
        """

        B,L,D = x.shape

        x = x.permute(0,2,1)

        trend = F.conv1d(
            x,
            self.lp.repeat(D,1,1),
            stride=2,
            groups=D
        )

        detail = F.conv1d(
            x,
            self.hp.repeat(D,1,1),
            stride=2,
            groups=D
        )

        trend = F.interpolate(trend, size=L, mode="linear", align_corners=False)
        detail = F.interpolate(detail, size=L, mode="linear", align_corners=False)

        trend = trend.permute(0,2,1)
        detail = detail.permute(0,2,1)

        return trend, detail

class TemporalMixer(nn.Module):

    def __init__(self, seq_len, d_model):

        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len)
        )


    def forward(self,x):

        """
        x: [B,L,D]
        """

        y = self.norm(x)

        y = y.permute(0,2,1)

        y = self.mlp(y)

        y = y.permute(0,2,1)

        return x + y

class PredictionHead(nn.Module):

    def __init__(self, seq_len, pred_len, d_model):

        super().__init__()

        self.proj = nn.Linear(seq_len*d_model, pred_len)


    def forward(self,x):

        B,L,D = x.shape

        x = x.reshape(B, L*D)

        out = self.proj(x)

        return out
            
class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.input_proj = nn.Linear(1, self.d_model)

        self.wavelet = HaarWavelet()

        # 新增模块
        self.freq_selector = FrequencySelector(self.d_model)
        self.cycle_bias = CycleFrequencyBias(self.seq_len)

        #  替换 interaction
        self.freq_interaction = EnhancedComplexInteraction(self.d_model)

        self.temporal_mixer = TemporalMixer(self.seq_len, self.d_model)

        self.head = PredictionHead(self.seq_len, self.pred_len, self.d_model)

    def zscore_norm(self, x):

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5

        return (x - mean) / std, mean, std

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):

        B, L, C = x.shape

        # Channel Independence
        x = x.permute(0,2,1).reshape(B*C, L, 1)

        x, mean, std = self.zscore_norm(x)

        x = self.input_proj(x)

        trend, detail = self.wavelet(x)

        # FFT
        F_trend = torch.fft.rfft(trend, dim=1)
        F_detail = torch.fft.rfft(detail, dim=1)

        # Step1: FreTS - 频率选择
        F_trend = self.freq_selector(F_trend)
        F_detail = self.freq_selector(F_detail)

        # Step2: FreqCycle - 周期bias
        F_trend = self.cycle_bias(F_trend)
        F_detail = self.cycle_bias(F_detail)

        # Step3: Interaction
        F_fused = self.freq_interaction(F_trend, F_detail)

        # IFFT
        x = torch.fft.irfft(F_fused, n=L, dim=1)

        x = self.temporal_mixer(x)

        out = self.head(x)

        out = out.reshape(B, C, self.pred_len)
        out = out[:,0,:].unsqueeze(-1)

        return out
