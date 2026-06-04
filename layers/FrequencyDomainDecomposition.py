"""
Frequency-Domain Series Decomposition
基于频域分解的时间序列分解方法

实现思路：
1. FFT 将时间序列转换到频域
2. 根据幅度和频率将分量分为：
   - 趋势分量（低频）
   - 季节分量（中频）
   - 噪声分量（高频）
3. 可学习的频率边界自适应调整
"""

import torch
import torch.nn as nn
import torch.fft


class FrequencyDomainDecomposition(nn.Module):
    """
    频域分解模块
    
    将输入时间序列分解为：
    - trend: 低频分量（长期趋势）
    - seasonal: 中高频分量（周期性和短期波动）
    """
    
    def __init__(self, seq_len, decomp_method='frequency', low_freq_ratio=0.3):
        """
        Args:
            seq_len: 输入序列长度
            decomp_method: 分解方法 ('frequency' 使用FFT，'moving_avg' 使用均值)
            low_freq_ratio: 低频分量的比例 (0.0-1.0)
        """
        super(FrequencyDomainDecomposition, self).__init__()
        
        self.seq_len = seq_len
        self.decomp_method = decomp_method
        self.low_freq_ratio = nn.Parameter(
            torch.tensor(low_freq_ratio, dtype=torch.float32),
            requires_grad=True
        )
        
        # 频率边界可学习
        self.freq_boundary = nn.Parameter(
            torch.tensor(seq_len // 3, dtype=torch.float32),
            requires_grad=True
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, N] 输入时间序列
            
        Returns:
            trend: [B, T, N] 趋势分量
            seasonal: [B, T, N] 季节分量
        """
        if self.decomp_method == 'frequency':
            return self._frequency_decomp(x)
        else:
            return self._moving_avg_decomp(x)
    
    def _frequency_decomp(self, x):
        """
        基于FFT的频域分解
        
        Args:
            x: [B, T, N]
            
        Returns:
            trend: [B, T, N]
            seasonal: [B, T, N]
        """
        B, T, N = x.shape
        
        # 对每个特征通道进行FFT
        # [B, T, N] -> [B, N, T] 便于处理
        x_transposed = x.permute(0, 2, 1)  # [B, N, T]
        
        # FFT变换
        x_fft = torch.fft.rfft(x_transposed, dim=-1)  # [B, N, T//2+1]
        
        # 计算幅度谱
        magnitude = torch.abs(x_fft)  # [B, N, T//2+1]
        phase = torch.angle(x_fft)     # [B, N, T//2+1]
        
        # 频率边界（可学习）
        freq_cut = int(torch.clamp(self.freq_boundary, min=1, max=T//2))
        
        # 分离低频和高频
        mag_low = magnitude.clone()
        mag_high = magnitude.clone()
        
        mag_low[:, :, freq_cut:] = 0      # 低频分量：只保留低频
        mag_high[:, :, :freq_cut] = 0     # 高频分量：只保留高频
        
        # 重构低频和高频分量
        low_fft = mag_low * torch.exp(1j * phase)
        high_fft = mag_high * torch.exp(1j * phase)
        
        # 逆FFT
        trend = torch.fft.irfft(low_fft, n=T, dim=-1)  # [B, N, T]
        seasonal = torch.fft.irfft(high_fft, n=T, dim=-1)  # [B, N, T]
        
        # 转回 [B, T, N]
        trend = trend.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)
        
        return trend, seasonal
    
    def _moving_avg_decomp(self, x):
        """
        基于移动平均的分解（备选方案）
        
        Args:
            x: [B, T, N]
            
        Returns:
            trend: [B, T, N]
            seasonal: [B, T, N]
        """
        # 自适应窗口大小
        window_size = int(torch.clamp(
            self.freq_boundary / 3, 
            min=3, 
            max=self.seq_len // 4
        ).item())
        
        # 计算移动平均作为趋势
        kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
        
        # [B, T, N] -> [B, N, 1, T]
        x_reshaped = x.permute(0, 2, 1).unsqueeze(1)
        
        # 使用 F.conv1d 计算移动平均
        padding = window_size // 2
        trend = torch.nn.functional.conv1d(
            x_reshaped, kernel, padding=padding
        )  # [B, 1, N, T] 实际上应该是 [B, N, T]
        
        # 调整形状回 [B, T, N]
        if trend.shape[-1] == x.shape[1]:
            trend = trend.squeeze(1).permute(0, 2, 1)
        else:
            # 如果长度不匹配，使用简单的重采样
            trend = torch.nn.functional.interpolate(
                x_reshaped,
                size=(1, x.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).permute(0, 2, 1)
        
        seasonal = x - trend
        
        return trend, seasonal


class AdaptiveFrequencyFusion(nn.Module):
    """
    自适应频域融合模块
    
    将Fourier和Wavelet两个编码器的输出进行融合
    使用可学习的权重自动找到最优融合比例
    """
    
    def __init__(self, d_model, n_heads=8):
        """
        Args:
            d_model: 模型维度
            n_heads: 多头数量
        """
        super(AdaptiveFrequencyFusion, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 融合权重学习网络
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        # 注意力融合
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 融合后的输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, fourier_out, wavelet_out):
        """
        Args:
            fourier_out: [B, T, d_model] Fourier编码器输出
            wavelet_out: [B, T, d_model] Wavelet编码器输出
            
        Returns:
            fused: [B, T, d_model] 融合后的输出
        """
        B, T, D = fourier_out.shape
        
        # 计算融合权重
        combined = torch.cat([fourier_out, wavelet_out], dim=-1)  # [B, T, 2*D]
        weight = self.fusion_weight_net(combined)  # [B, T, 1]
        
        # 加权融合
        weighted_fourier = fourier_out * weight
        weighted_wavelet = wavelet_out * (1 - weight)
        fused = weighted_fourier + weighted_wavelet
        
        # 通过注意力机制进一步融合
        attn_out, _ = self.fusion_attention(
            query=fused,
            key=torch.cat([fourier_out, wavelet_out], dim=1),
            value=torch.cat([fourier_out, wavelet_out], dim=1)
        )
        
        # 残差连接和输出投影
        fused = fused + attn_out
        fused = self.output_proj(fused)
        
        return fused


class FrequencyPatchEmbedding(nn.Module):
    """
    频域 Patch 嵌入
    
    将时间序列分割成多个patch，在频域中进行嵌入
    """
    
    def __init__(self, d_model, patch_len=16):
        """
        Args:
            d_model: 嵌入维度
            patch_len: patch长度
        """
        super(FrequencyPatchEmbedding, self).__init__()
        
        self.d_model = d_model
        self.patch_len = patch_len
        
        # Patch 线性投影
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, N] 或 [B, T]
            
        Returns:
            patches: [B, num_patches, d_model]
        """
        if x.dim() == 3:
            B, T, N = x.shape
            # 对每个特征分别处理
            x_patches = []
            for i in range(N):
                xi = x[:, :, i]  # [B, T]
                num_patches = T // self.patch_len
                
                # 将序列分割成patches
                patches = xi.unfold(1, self.patch_len, self.patch_len)  # [B, num_patches, patch_len]
                
                # 嵌入
                embedded = self.patch_embedding(patches)  # [B, num_patches, d_model]
                x_patches.append(embedded)
            
            # 合并所有特征的patch嵌入
            all_patches = torch.cat(x_patches, dim=1)  # [B, N*num_patches, d_model]
            return all_patches
        else:
            B, T = x.shape
            num_patches = T // self.patch_len
            patches = x.unfold(1, self.patch_len, self.patch_len)  # [B, num_patches, patch_len]
            embedded = self.patch_embedding(patches)  # [B, num_patches, d_model]
            return embedded
