'''
Wavelet-Guided Complex Frequency Interaction Network (WCFIN):
Input
 │
Channel Independence
 │
Z-score Normalization
 │
Wavelet Decomposition (A1 / D1)
 │
FFT
 │
Complex Frequency Interaction
 │
IFFT
 │
Temporal Mixer
 │
Prediction Head
 │
Forecast
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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
#--------------------------------------------------------
class ComplexFreqInteraction(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.gate = nn.Linear(d_model, d_model)


    def forward(self, F_trend, F_detail):

        """
        complex tensor
        """

        real_trend = F_trend.real
        imag_trend = F_trend.imag

        gate = torch.sigmoid(
            self.gate(real_trend)
        )

        real = F_detail.real * gate
        imag = F_detail.imag * gate

        F_detail_refined = torch.complex(real, imag)

        F_fused = F_trend + F_detail_refined

        return F_fused
#------------------------------------------------------------------------
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

        # embedding
        self.input_proj = nn.Linear(1, self.d_model)

        # GPU wavelet
        self.wavelet = HaarWavelet()

        # frequency interaction
        self.freq_interaction = ComplexFreqInteraction(self.d_model)

        # temporal mixer
        self.temporal_mixer = TemporalMixer(self.seq_len, self.d_model)

        # prediction head
        self.head = PredictionHead(self.seq_len, self.pred_len, self.d_model)


    def zscore_norm(self, x):

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5

        x = (x - mean) / std

        return x, mean, std


    def forward(self, x,x_mark_enc, x_dec, x_mark_dec, mask=None):

        """
        x: [B, L, C]
        """

        B, L, C = x.shape

        # Channel Independence
        x = x.permute(0,2,1).reshape(B*C, L, 1)

        # normalization
        x, mean, std = self.zscore_norm(x)

        # embedding
        x = self.input_proj(x)

        # wavelet decomposition
        trend, detail = self.wavelet(x)

        # FFT
        F_trend = torch.fft.rfft(trend, dim=1)
        F_detail = torch.fft.rfft(detail, dim=1)

        # frequency interaction
        F_fused = self.freq_interaction(F_trend, F_detail)

        # IFFT
        x = torch.fft.irfft(F_fused, n=L, dim=1)

        # temporal mixer
        x = self.temporal_mixer(x)

        # prediction
        out = self.head(x)

        # reshape back
        out = out.reshape(B, C, self.pred_len)
        #形状从 (B, pred_len) 变为 (B, pred_len, 1)
        out = out[:,0,:].unsqueeze(-1)

        return out