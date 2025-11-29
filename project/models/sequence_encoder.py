#!/usr/bin/env python3
"""
序列编码流（Sequence Stream）实现
基于CNN → RNN → CTC的序列特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3作为CNN特征提取器"""
    
    def __init__(self, pretrained=True, output_channels=512):
        super(MobileNetV3Encoder, self).__init__()
        
        # 加载预训练的MobileNetV3
        try:
            self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            # 兼容旧版本PyTorch
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # 移除最后的分类层
        self.backbone.classifier = nn.Identity()
        
        # 特征维度调整
        self.feature_adjust = nn.Sequential(
            nn.Conv2d(960, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 [batch_size, 3, H, W]
        Returns:
            特征图 [batch_size, C, H', W']
        """
        # 通过MobileNetV3骨干网络
        features = self.backbone.features(x)
        
        # 调整特征维度
        features = self.feature_adjust(features)
        
        return features


class MSF_LCRNN_Encoder(nn.Module):
    """MSF-LCRNN编码器（多尺度特征 + LSTM）"""
    
    def __init__(self, input_channels=3, hidden_size=256, num_layers=2):
        super(MSF_LCRNN_Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多尺度特征提取
        self.multiscale_cnn = nn.ModuleDict({
            'scale1': self._build_cnn_block(input_channels, 64, kernel_size=7, stride=2),
            'scale2': self._build_cnn_block(64, 128, kernel_size=5, stride=2),
            'scale3': self._build_cnn_block(128, 256, kernel_size=3, stride=1),
        })
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(64 + 128 + 256, hidden_size, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 维度调整层
        self.dimension_adjust = nn.Linear(hidden_size * 4, hidden_size)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # 双向所以除以2
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
    def _build_cnn_block(self, in_channels, out_channels, kernel_size, stride):
        """构建CNN块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1)  # 使用更温和的池化
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [batch_size, 3, H, W]
        Returns:
            序列特征 [batch_size, T, D]
        """
        batch_size = x.size(0)
        
        # 多尺度特征提取
        scale_features = []
        
        # 尺度1
        feat1 = self.multiscale_cnn['scale1'](x)
        # 使用自适应池化调整高度为1，保持宽度以获得序列长度
        feat1 = F.adaptive_avg_pool2d(feat1, (1, None))
        scale_features.append(feat1)
        
        # 尺度2
        feat2 = self.multiscale_cnn['scale2'](feat1)
        feat2 = F.adaptive_avg_pool2d(feat2, (1, None))
        scale_features.append(feat2)
        
        # 尺度3
        feat3 = self.multiscale_cnn['scale3'](feat2)
        feat3 = F.adaptive_avg_pool2d(feat3, (1, None))
        scale_features.append(feat3)
        
        # 调整所有特征到相同的宽度（使用最大宽度）
        widths = [feat.shape[3] for feat in scale_features]
        max_width = max(widths)
        
        adjusted_features = []
        for feat in scale_features:
            if feat.shape[3] < max_width:
                # 使用插值调整宽度到最大值
                adjusted = F.interpolate(feat, size=(1, max_width), mode='bilinear', align_corners=False)
                adjusted_features.append(adjusted)
            else:
                adjusted_features.append(feat)
        
        # 特征融合
        fused_features = torch.cat(adjusted_features, dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 调整维度: [batch_size, C, 1, W] -> [batch_size, W, C]
        sequence_features = fused_features.squeeze(2).permute(0, 2, 1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(sequence_features)
        
        return lstm_out


class GRU_Encoder(nn.Module):
    """GRU编码器"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.1):
        super(GRU_Encoder, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, T, D]
        Returns:
            编码特征 [batch_size, T, hidden_size]
        """
        gru_out, _ = self.gru(x)
        output = self.output_proj(gru_out)
        return output


class SequenceEncoder(nn.Module):
    """序列编码流主模块"""
    
    def __init__(self, encoder_type='mobilenet', input_channels=3, hidden_size=256, 
                 vocab_size=100, use_ctc=True):
        super(SequenceEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.hidden_size = hidden_size
        self.use_ctc = use_ctc
        
        # 选择编码器类型
        if encoder_type == 'mobilenet':
            self.cnn_encoder = MobileNetV3Encoder(output_channels=hidden_size)
            cnn_output_size = hidden_size
        elif encoder_type == 'msf_lcrnn':
            self.cnn_encoder = MSF_LCRNN_Encoder(input_channels, hidden_size)
            cnn_output_size = hidden_size
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        # RNN编码器
        self.rnn_encoder = GRU_Encoder(cnn_output_size, hidden_size)
        
        # 对于MobileNetV3，需要添加投影层来处理维度调整
        if encoder_type == 'mobilenet':
            # MobileNetV3输出通道数为hidden_size，高度根据实际输入图像尺寸动态调整
            # 使用自适应投影层，在forward中根据实际维度动态创建
            self.cnn_output_size = cnn_output_size
        
        # CTC输出层（如果使用CTC）
        if use_ctc:
            self.ctc_head = nn.Linear(hidden_size, vocab_size)
        
        # 序列特征输出
        self.feature_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 [batch_size, 3, H, W]
        Returns:
            sequence_features: 序列特征 [batch_size, T, D]
            ctc_logits: CTC输出 [batch_size, T, vocab_size] (如果use_ctc=True)
        """
        # CNN特征提取
        if self.encoder_type == 'mobilenet':
            # MobileNetV3输出需要调整维度
            cnn_features = self.cnn_encoder(x)
            # [batch_size, C, H', W'] -> [batch_size, W', C]
            # 保持高度维度以获得足够的序列长度
            batch_size, channels, height, width = cnn_features.shape
            # 将高度维度展平到通道维度，以获得更长的序列
            cnn_features = cnn_features.view(batch_size, channels * height, width)
            cnn_features = cnn_features.permute(0, 2, 1)  # [batch_size, W', C*H']
            
            # 动态创建投影层以匹配实际维度
            actual_input_dim = cnn_features.size(2)
            if not hasattr(self, '_dynamic_projection') or self._dynamic_projection.in_features != actual_input_dim:
                self._dynamic_projection = nn.Linear(actual_input_dim, self.cnn_output_size).to(cnn_features.device)
            cnn_features = self._dynamic_projection(cnn_features)
        else:
            # MSF-LCRNN直接输出序列
            cnn_features = self.cnn_encoder(x)
        
        # RNN编码
        sequence_features = self.rnn_encoder(cnn_features)
        
        # 特征投影
        sequence_features = self.feature_proj(sequence_features)
        
        # CTC输出
        if self.use_ctc:
            ctc_logits = self.ctc_head(sequence_features)
            return sequence_features, ctc_logits
        else:
            return sequence_features, None
    
    def extract_features(self, x):
        """仅提取序列特征"""
        sequence_features, _ = self.forward(x)
        return sequence_features


class CTC_Loss(nn.Module):
    """CTC损失函数"""
    
    def __init__(self, blank=0, reduction='mean'):
        super(CTC_Loss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
        
    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Args:
            logits: [T, batch_size, vocab_size]
            targets: 目标序列
            input_lengths: 输入序列长度
            target_lengths: 目标序列长度
        """
        # 调整维度: [batch_size, T, vocab_size] -> [T, batch_size, vocab_size]
        logits = logits.transpose(0, 1)
        
        # 计算CTC损失
        loss = self.ctc_loss(logits.log_softmax(2), targets, input_lengths, target_lengths)
        
        return loss


def test_sequence_encoder():
    """测试序列编码器"""
    # 创建测试数据
    batch_size, channels, height, width = 2, 3, 64, 256
    test_input = torch.randn(batch_size, channels, height, width)
    
    # 测试MobileNetV3编码器
    print("测试MobileNetV3编码器...")
    encoder_mobilenet = SequenceEncoder(encoder_type='mobilenet')
    features_mobilenet, ctc_logits_mobilenet = encoder_mobilenet(test_input)
    
    print(f"MobileNetV3特征维度: {features_mobilenet.shape}")
    if ctc_logits_mobilenet is not None:
        print(f"MobileNetV3 CTC输出维度: {ctc_logits_mobilenet.shape}")
    
    # 测试MSF-LCRNN编码器
    print("\n测试MSF-LCRNN编码器...")
    encoder_msf = SequenceEncoder(encoder_type='msf_lcrnn')
    features_msf, ctc_logits_msf = encoder_msf(test_input)
    
    print(f"MSF-LCRNN特征维度: {features_msf.shape}")
    if ctc_logits_msf is not None:
        print(f"MSF-LCRNN CTC输出维度: {ctc_logits_msf.shape}")
    
    # 测试CTC损失
    print("\n测试CTC损失...")
    ctc_loss_fn = CTC_Loss()
    
    # 模拟数据
    T, batch_size, vocab_size = 50, 2, 100
    logits = torch.randn(batch_size, T, vocab_size)
    targets = torch.randint(1, vocab_size, (batch_size, 15), dtype=torch.long)
    input_lengths = torch.tensor([T, T], dtype=torch.long)
    target_lengths = torch.tensor([10, 15], dtype=torch.long)
    
    loss = ctc_loss_fn(logits, targets, input_lengths, target_lengths)
    print(f"CTC损失值: {loss.item():.4f}")
    
    print("序列编码器测试通过!")


if __name__ == "__main__":
    test_sequence_encoder()