import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------- 新增/修改的辅助模块 ----------
class InstanceNormalization(nn.Module):
    """
    对每个样本按通道减去最后一个时间步（按论文做法）
    forward 返回 x - last_val, 以及 last_val 以便逆归一化
    输入 x: [B, L, C]
    """
    def __init__(self):
        super(InstanceNormalization, self).__init__()

    def forward(self, x):
        # last_val shape [B,1,C]
        last_val = x[:, -1:, :].clone()
        return x - last_val, last_val

    def inverse(self, x, last_val):
        # x: [B, pred_len, C]
        return x + last_val


class ChannelIndependentStrategy(nn.Module):
    """
    将 [B, L, C] -> [B*C, L, 1] 的通道独立形式，和逆操作
    """
    def __init__(self):
        super(ChannelIndependentStrategy, self).__init__()

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        # -> [B, C, L]
        x_perm = x.permute(0, 2, 1).contiguous()
        # -> [B*C, L, 1]
        x_reshaped = x_perm.view(B * C, L, 1)
        return x_reshaped

    def inverse(self, x, original_shape):
        # x: [B*C, pred_len, 1]
        B, pred_len, C = original_shape  # original_shape expects (B, pred_len, C)
        x = x.view(B, C, pred_len).permute(0, 2, 1).contiguous()  # [B, pred_len, C]
        return x


class FrequencyEnhancedModule(nn.Module):
    """
    对每个通道的时序做短时傅里叶变换增强（作用在通道独立表示上）
    输入: [B*C, seq_len, 1]
    输出: [B*C, seq_len, 1]（和输入形状一致，便于后续流程）
    这里做了：STFT -> 放大/变换幅度谱 -> iSTFT 重建
    """
    def __init__(self, n_fft=64, hop_length=None):
        super(FrequencyEnhancedModule, self).__init__()
        #n_fft就是短时傅里叶变换中切分原始序列的长度大小，原始长度按照这个长度和hoplength切成短的片段，对每个小的片段做傅里叶变换
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        # hann window (non-trainable)
        self.register_buffer("window", torch.hann_window(self.n_fft))
        # amplitude MLP: freq_bins = n_fft//2 + 1
        freq_bins = self.n_fft // 2 + 1
        self.feed_forward = nn.Sequential(
            nn.Linear(freq_bins, freq_bins),
            nn.ReLU(),
            nn.Linear(freq_bins, freq_bins)
        )

    def forward(self, x):
        # x: [B*C, seq_len, 1]
        Bc, L, _ = x.shape
        x = x.squeeze(-1)  # [B*C, L]

        # STFT -> complex tensor: [B*C, freq_bins, time_frames]
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, return_complex=True)
        # magnitude and phase
        magnitude = torch.abs(stft)  # [B*C, freq_bins, time_frames]
        phase = torch.angle(stft)

        # we will apply FFN along freq_bins; first permute to [B*C, time_frames, freq_bins]
        mag_perm = magnitude.permute(0, 2, 1)
        # pass through feedforward (applies same MLP to each time frame)
        mag_trans = self.feed_forward(mag_perm)  # [B*C, time_frames, freq_bins]
        mag_trans = mag_trans.permute(0, 2, 1)  # back to [B*C, freq_bins, time_frames]

        # reconstruct complex spectrogram and inverse STFT
        real = mag_trans * torch.cos(phase)
        imag = mag_trans * torch.sin(phase)
        stft_trans = torch.complex(real, imag)

        x_rec = torch.istft(stft_trans, n_fft=self.n_fft, hop_length=self.hop_length,
                            window=self.window, length=L)  # [B*C, L]

        return x_rec.unsqueeze(-1)  # [B*C, L, 1]


# ---------- 原有的 decomposition 等保持不变 ----------
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ---------- 修改后的主模型 ----------
class Model(nn.Module):
    """
    Decomposition-Linear + FITS with added InstanceNorm + ChannelIndependent + FrequencyEnhance
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.channels = configs.enc_in  # number of channels

        # decomposition
        kernel_size = configs.moving_avg
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual

        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # DLinear for trend
        if self.individual:
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                lin = nn.Linear(self.seq_len, self.pred_len)
                # initialize to avg
                lin.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.append(lin)
        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # FITS freq upsampler (keep your complex approach)
        # self.alpha_param = nn.Parameter(torch.zeros(1, self.enc_in, 1))
        #调gc19_c d e时调的代码
        self.alpha_param = nn.Parameter(
            torch.full((1, self.enc_in, 1), -0.2)  # 全部填充为 -0.2
        )

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                # Linear expecting real->complex conversion done later; to keep compatibility,
                # we will keep your approach but ensure dtype handling at call time
                lin = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio))
                self.freq_upsampler.append(lin)
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio))

        # ---------------- 新增模块 ----------------
        # Instance normalization module (减去最后一个时间步)
        self.instance_norm = InstanceNormalization()
        # Channel independent helper
        self.channel_independent = ChannelIndependentStrategy()
        # Frequency enhancement module (在送入 FITS 前增强)
        # choose n_fft reasonably (keep default or set from configs if provided)
        n_fft = getattr(configs, "n_fft", 64)
        hop = getattr(configs, "hop_length", None)
        self.freq_enhancer = FrequencyEnhancedModule(n_fft=n_fft, hop_length=hop)
        # -----------------------------------------

    def forward(self, x,batch_x_mark, dec_inp, batch_y_mark):
        # x: [B, seq_len, C]
        B, L, C = x.shape

        # # === 原有的 RIN 标准化（保留） ===
        # x_mean = torch.mean(x, dim=1, keepdim=True)  # [B,1,C]
        # x_centered = x - x_mean
        # x_var = torch.var(x_centered, dim=1, keepdim=True) + 1e-5
        # x = x_centered / torch.sqrt(x_var)  # standardized [B,L,C]

        # === 新增: InstanceNormalization（减去最后时间步） ===
        # operate on standardized x to remove last-step bias
        x_in, last_val = self.instance_norm(x)  # x_in: [B,L,C], last_val: [B,1,C]

        # decomposition on normalized sequence
        seasonal_init, trend_init = self.decompsition(x_in)  # both [B,L,C]

        # ----------------- trend branch (DLinear) -----------------
        # trend_init: [B,L,C] -> permute to [B,C,L] for per-channel linear
        trend_init_p = trend_init.permute(0, 2, 1)  # [B,C,T]
        if self.individual:
            trend_output = torch.zeros([B, C, self.pred_len], dtype=trend_init_p.dtype, device=trend_init_p.device)
            for i in range(self.channels):
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init_p[:, i, :])
            # recover to [B, pred_len, C]
            trend_output = trend_output.permute(0, 2, 1)  # [B, pred_len, C]
            # undo standardization
            # trend_output = trend_output * torch.sqrt(x_var) + x_mean
            # also undo instance_norm subtraction: add last_val (after scaling), but last_val is on standardized scale
            trend_output = self.instance_norm.inverse(trend_output, last_val)
        else:
            trend_output = self.Linear_Trend(trend_init_p)  # [B, C, pred_len]
            trend_output = trend_output.permute(0, 2, 1)  # [B, pred_len, C]
            # trend_output = trend_output * torch.sqrt(x_var) + x_mean
            trend_output = self.instance_norm.inverse(trend_output, last_val)

        # ----------------- seasonal branch (FITS) -----------------
        # 在进行 rfft 对频谱上采样之前，先对 seasonal_init 做通道独立 + frequency enhancement
        # seasonal_init: [B, L, C]
        # 1) channel independent -> [B*C, L, 1]
        seasonal_ci = self.channel_independent.forward(seasonal_init)  # [B*C, L, 1]
        # 2) frequency enhancement module (STFT->FFN->iSTFT) -> [B*C, L, 1]
        seasonal_enhanced_ci = self.freq_enhancer.forward(seasonal_ci)  # [B*C, L, 1]
        # 3) inverse channel independent -> [B, L, C]
        seasonal_enhanced = self.channel_independent.inverse(seasonal_enhanced_ci, (B, L, C))

        # 接下来沿用你原来的频域 upsampling 流程，但用 seasonal_enhanced 替代 seasonal_init
        # low_specx: rfft over time dim
        low_specx = torch.fft.rfft(seasonal_enhanced, dim=1)  # [B, freq_bins, C]
        # apply LPF as before
        low_specx[:, self.dominance_freq:, :] = 0
        low_specx = low_specx[:, 0:self.dominance_freq, :]  # [B, df, C]

        # freq upsampling (你的 complex linear)
        if self.individual:
            low_specxy_ = torch.zeros([B, int(self.dominance_freq * self.length_ratio), C], dtype=low_specx.dtype, device=low_specx.device)
            for i in range(self.channels):
                # freq_upsampler[i] is a real Linear; apply per channel on real+imag concatenated representation
                # 为简单起见，这里对 real 和 imag 分别做 upsample，再合成 complex（保持和你原来思路等价）
                real = low_specx[:, :, i].real  # [B, df]
                imag = low_specx[:, :, i].imag  # [B, df]
                up_real = self.freq_upsampler[i](real)  # [B, new_df]
                up_imag = self.freq_upsampler[i](imag)  # [B, new_df]
                low_specxy_[:, :, i] = torch.complex(up_real, up_imag)
        else:
            # low_specx: [B, df, C] -> permute to [B, C, df] -> apply linear -> [B, C, df_new] -> permute
            # we apply linear on last dim by permuting
            tmp = low_specx.permute(0, 2, 1)  # [B, C, df]
            # separate real/imag and process separately
            real = tmp.real
            imag = tmp.imag
            up_real = self.freq_upsampler(real)  # [B, C, df_new]
            up_imag = self.freq_upsampler(imag)
            low_specxy_ = torch.complex(up_real, up_imag).permute(0, 2, 1)  # [B, df_new, C]

        # zero pad to expected rfft length (seq_len + pred_len)/2 + 1
        df_new = int((self.seq_len + self.pred_len) / 2 + 1)
        low_specxy = torch.zeros([B, df_new, C], dtype=low_specxy_.dtype, device=low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_

        # inverse rfft -> time domain length seq_len + pred_len
        low_xy = torch.fft.irfft(low_specxy, n=(self.seq_len + self.pred_len), dim=1)  # [B, seq_len+pred_len, C]
        low_xy = low_xy * self.length_ratio  # energy compensation

        # undo standardization for seasonal branch and instance norm inverse
        # xy = (low_xy) * torch.sqrt(x_var) + x_mean  # [B, seq_len+pred_len, C]
        seasonal_pred = low_xy[:, -self.pred_len:, :]  # [B, pred_len, C]
        seasonal_pred = self.instance_norm.inverse(seasonal_pred, last_val)

        # ----------------- 融合 -----------------
        alpha = torch.sigmoid(self.alpha_param)  # [1, C, 1]
        alpha = alpha.permute(0, 2, 1)  # [1, 1, C] to match [B, pred_len, C]
        output = alpha * trend_output + (1 - alpha) * seasonal_pred  # [B, pred_len, C]

        return output
