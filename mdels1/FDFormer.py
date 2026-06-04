"""
FDFormer: Frequency-Domain Decomposition Transformer
for Cloud Load Forecasting

基于频域分解的云平台负载预测变换器

论文贡献：
1. 频域时间序列分解 (Frequency-Domain Series Decomposition)
2. 双路径傅里叶-小波融合编码器
3. 自适应频域权重学习机制
4. 云平台负载特性感知的设计

Author: CandyTang745
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer, 
    my_Layernorm, series_decomp
)
from layers.FrequencyDomainDecomposition import (
    FrequencyDomainDecomposition, AdaptiveFrequencyFusion
)


class FDFormerEncoderLayer(nn.Module):
    """
    FDFormer编码器层 - 融合Fourier和Wavelet路径
    
    包含双频域路径处理和自适应融合
    """
    
    def __init__(
        self, 
        fourier_attention,
        wavelet_attention,
        d_model, 
        d_ff=None, 
        moving_avg=25, 
        dropout=0.1, 
        activation="relu"
    ):
        super(FDFormerEncoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # 双频域路径注意力
        self.fourier_attention = fourier_attention
        self.wavelet_attention = wavelet_attention
        
        # 自适应融合
        self.fusion = AdaptiveFrequencyFusion(d_model, n_heads=8)
        
        # FFN
        self.conv1 = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_ff, 
            kernel_size=1, 
            bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, 
            out_channels=d_model, 
            kernel_size=1, 
            bias=False
        )
        
        # 分解
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [B, T, d_model]
            attn_mask: 注意力掩码
            
        Returns:
            x: [B, T, d_model] 分解后的季节分量
            attn: 注意力权重
        """
        
        # 双路径频域处理
        # Fourier路径
        fourier_out, _ = self.fourier_attention(x, x, x, attn_mask=attn_mask)
        
        # Wavelet路径
        wavelet_out, _ = self.wavelet_attention(x, x, x, attn_mask=attn_mask)
        
        # 自适应融合
        fused = self.fusion(fourier_out, wavelet_out)
        
        # 残差连接和分解
        x = x + self.dropout(fused)
        x, trend1 = self.decomp1(x)
        
        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 残差和分解
        x, trend2 = self.decomp2(x + y)
        
        return x, trend1 + trend2


class FDFormerDecoderLayer(nn.Module):
    """
    FDFormer解码器层 - 自注意力 + 交叉注意力 + 双频域融合
    """
    
    def __init__(
        self,
        self_fourier,
        self_wavelet,
        cross_fourier,
        cross_wavelet,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu"
    ):
        super(FDFormerDecoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # 自注意力（双路径）
        self.self_fourier = self_fourier
        self.self_wavelet = self_wavelet
        self.self_fusion = AdaptiveFrequencyFusion(d_model, n_heads=8)
        
        # 交叉注意力（双路径）
        self.cross_fourier = cross_fourier
        self.cross_wavelet = cross_wavelet
        self.cross_fusion = AdaptiveFrequencyFusion(d_model, n_heads=8)
        
        # FFN
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
            bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
            bias=False
        )
        
        # 分解
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        
        # 输出投影
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Args:
            x: [B, T, d_model] 解码器输入
            cross: [B, T_enc, d_model] 编码器输出
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            
        Returns:
            x: [B, T, d_model] 季节分量
            residual_trend: [B, T, c_out] 预测的趋势
        """
        
        # 自注意力（双路径）
        fourier_self, _ = self.self_fourier(x, x, x, attn_mask=x_mask)
        wavelet_self, _ = self.self_wavelet(x, x, x, attn_mask=x_mask)
        fused_self = self.self_fusion(fourier_self, wavelet_self)
        
        x = x + self.dropout(fused_self)
        x, trend1 = self.decomp1(x)
        
        # 交叉注意力（双路径）
        fourier_cross, _ = self.cross_fourier(x, cross, cross, attn_mask=cross_mask)
        wavelet_cross, _ = self.cross_wavelet(x, cross, cross, attn_mask=cross_mask)
        fused_cross = self.cross_fusion(fourier_cross, wavelet_cross)
        
        x = x + self.dropout(fused_cross)
        x, trend2 = self.decomp2(x)
        
        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        x, trend3 = self.decomp3(x + y)
        
        # 趋势聚合和投影
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(
            residual_trend.permute(0, 2, 1)
        ).transpose(1, 2)
        
        return x, residual_trend


class Model(nn.Module):
    """
    FDFormer: Frequency-Domain Decomposition Transformer
    
    核心特性：
    1. 频域时间序列分解（低/中/高频分量）
    2. 双路径编码器（Fourier + Wavelet）
    3. 自适应融合机制
    4. 编码器-解码器架构处理长期预测
    5. 云平台负载特性感知设计
    
    论文题目对应：基于频域分解的云平台负载预测方法研究
    """
    
    def __init__(
        self,
        configs,
        version='fourier_wavelet',
        mode_select='random',
        modes=32,
        decomp_method='frequency'
    ):
        """
        Args:
            configs: 配置对象
            version: 频域版本选择
            mode_select: 模式选择方法 ('random' 或 'low')
            modes: FFT模式数量
            decomp_method: 分解方法 ('frequency' 或 'moving_avg')
        """
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.decomp_method = decomp_method
        
        # ========== 频域分解模块 ==========
        # 使用方案C：频域直接分解
        self.freq_decomposer = FrequencyDomainDecomposition(
            seq_len=configs.seq_len,
            decomp_method=decomp_method,
            low_freq_ratio=0.3
        )
        
        # ========== 数据嵌入 ==========
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )
        
        # ========== 编码器注意力（双路径） ==========
        encoder_self_att_fourier = FourierBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            n_heads=configs.n_heads,
            seq_len=self.seq_len,
            modes=self.modes,
            mode_select_method=self.mode_select
        )
        
        encoder_self_att_wavelet = MultiWaveletTransform(
            ich=configs.d_model,
            L=1,
            base='legendre'
        )
        
        # ========== 编码器 ==========
        encoder_layers = [
            FDFormerEncoderLayer(
                fourier_attention=AutoCorrelationLayer(
                    encoder_self_att_fourier,
                    configs.d_model,
                    configs.n_heads
                ),
                wavelet_attention=AutoCorrelationLayer(
                    encoder_self_att_wavelet,
                    configs.d_model,
                    configs.n_heads
                ),
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation
            )
            for _ in range(configs.e_layers)
        ]
        
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=my_Layernorm(configs.d_model)
        )
        
        # ========== 解码器注意力（双路径） ==========
        decoder_self_att_fourier = FourierBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            n_heads=configs.n_heads,
            seq_len=self.seq_len // 2 + self.pred_len,
            modes=self.modes,
            mode_select_method=self.mode_select
        )
        
        decoder_self_att_wavelet = MultiWaveletTransform(
            ich=configs.d_model,
            L=1,
            base='legendre'
        )
        
        decoder_cross_att_fourier = FourierCrossAttention(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            seq_len_q=self.seq_len // 2 + self.pred_len,
            seq_len_kv=self.seq_len,
            modes=self.modes,
            mode_select_method=self.mode_select,
            num_heads=configs.n_heads
        )
        
        decoder_cross_att_wavelet = MultiWaveletCross(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            seq_len_q=self.seq_len // 2 + self.pred_len,
            seq_len_kv=self.seq_len,
            modes=self.modes,
            ich=configs.d_model,
            base='legendre',
            activation='tanh'
        )
        
        # ========== 解码器 ==========
        decoder_layers = [
            FDFormerDecoderLayer(
                self_fourier=AutoCorrelationLayer(
                    decoder_self_att_fourier,
                    configs.d_model,
                    configs.n_heads
                ),
                self_wavelet=AutoCorrelationLayer(
                    decoder_self_att_wavelet,
                    configs.d_model,
                    configs.n_heads
                ),
                cross_fourier=AutoCorrelationLayer(
                    decoder_cross_att_fourier,
                    configs.d_model,
                    configs.n_heads
                ),
                cross_wavelet=AutoCorrelationLayer(
                    decoder_cross_att_wavelet,
                    configs.d_model,
                    configs.n_heads
                ),
                d_model=configs.d_model,
                c_out=configs.c_out,
                d_ff=configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation
            )
            for _ in range(configs.d_layers)
        ]
        
        self.decoder = Decoder(
            decoder_layers,
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        # ========== 任务特定投影 ==========
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_clf = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len,
                configs.num_class
            )
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        长期/短期预测
        
        Args:
            x_enc: [B, T, N] 编码器输入
            x_mark_enc: [B, T, M] 编码器时间特征
            x_dec: [B, T+pred_len, N] 解码器输入（通常是零填充）
            x_mark_dec: [B, T+pred_len, M] 解码器时间特征
            
        Returns:
            dec_out: [B, pred_len, N] 预测结果
        """
        
        # ===== 频域分解 =====
        # 使用新的频域分解方法替代简单的均值
        trend_init, seasonal_init = self.freq_decomposer(x_enc)
        
        # 计算均值用于趋势初始化
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        
        # ===== 解码器输入初始化 =====
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean],
            dim=1
        )
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :],
            (0, 0, 0, self.pred_len)
        )
        
        # ===== 编码器 =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # ===== 解码器 =====
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=None,
            cross_mask=None,
            trend=trend_init
        )
        
        # ===== 融合 =====
        dec_out = trend_part + seasonal_part
        
        return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """缺失值填补任务"""
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out
    
    def anomaly_detection(self, x_enc):
        """异常检测任务"""
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out
    
    def classification(self, x_enc, x_mark_enc):
        """分类任务"""
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        output = self.act(enc_out)
        output = self.dropout_clf(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        
        return output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        统一的前向接口
        
        Args:
            x_enc: 编码器输入
            x_mark_enc: 编码器时间标记
            x_dec: 解码器输入
            x_mark_dec: 解码器时间标记
            mask: 掩码（可选）
            
        Returns:
            预测输出
        """
        
        if (self.task_name == 'long_term_forecast' or 
            self.task_name == 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]
        
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        
        return None
