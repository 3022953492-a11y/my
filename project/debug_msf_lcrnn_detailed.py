#!/usr/bin/env python3
"""
详细调试MSF-LCRNN编码器各阶段输出维度
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sequence_encoder import MSF_LCRNN_Encoder

def debug_msf_lcrnn_detailed():
    """详细调试MSF-LCRNN编码器各阶段输出维度"""
    
    # 创建MSF-LCRNN编码器
    encoder = MSF_LCRNN_Encoder(input_channels=3, hidden_size=256)
    
    # 创建测试输入
    batch_size, channels, height, width = 1, 3, 128, 32
    test_input = torch.randn(batch_size, channels, height, width)
    
    print(f"输入图像形状: {test_input.shape}")
    
    # 手动执行前向传播的每个步骤
    batch_size = test_input.size(0)
    
    # 多尺度特征提取
    scale_features = []
    
    print("\n=== 多尺度特征提取 ===")
    
    # 尺度1
    feat1 = encoder.multiscale_cnn['scale1'](test_input)
    print(f"尺度1原始输出形状: {feat1.shape}")
    feat1 = torch.nn.functional.adaptive_avg_pool2d(feat1, (1, None))
    print(f"尺度1池化后形状: {feat1.shape}")
    scale_features.append(feat1)
    
    # 尺度2
    feat2 = encoder.multiscale_cnn['scale2'](feat1)
    print(f"尺度2原始输出形状: {feat2.shape}")
    feat2 = torch.nn.functional.adaptive_avg_pool2d(feat2, (1, None))
    print(f"尺度2池化后形状: {feat2.shape}")
    scale_features.append(feat2)
    
    # 尺度3
    feat3 = encoder.multiscale_cnn['scale3'](feat2)
    print(f"尺度3原始输出形状: {feat3.shape}")
    feat3 = torch.nn.functional.adaptive_avg_pool2d(feat3, (1, None))
    print(f"尺度3池化后形状: {feat3.shape}")
    scale_features.append(feat3)
    
    print("\n=== 特征融合 ===")
    
    widths = [feat.shape[3] for feat in scale_features]
    print(f"各尺度特征宽度: {widths}")
    
    max_width = max(widths)
    print(f"最大宽度: {max_width}")
    
    # 调整特征到最大宽度
    adjusted_features = []
    for i, feat in enumerate(scale_features):
        if feat.shape[3] < max_width:
            adjusted = torch.nn.functional.interpolate(feat, size=(1, max_width), mode='bilinear', align_corners=False)
            adjusted_features.append(adjusted)
            print(f"尺度{i+1}调整后形状: {adjusted.shape}")
        else:
            adjusted_features.append(feat)
    
    # 特征融合
    fused_features = torch.cat(adjusted_features, dim=1)
    print(f"融合特征形状: {fused_features.shape}")
    
    # 检查特征融合层的输入通道数
    print(f"特征融合层期望输入通道数: {encoder.feature_fusion[0].in_channels}")
    print(f"实际融合特征通道数: {fused_features.shape[1]}")
    
    fused_features = encoder.feature_fusion(fused_features)
    print(f"特征融合后形状: {fused_features.shape}")
    
    # 维度调整
    sequence_features = fused_features.squeeze(2).permute(0, 2, 1)
    print(f"调整维度后形状: {sequence_features.shape}")
    
    # LSTM处理
    lstm_out, _ = encoder.lstm(sequence_features)
    print(f"LSTM输出形状: {lstm_out.shape}")
    print(f"序列长度T: {lstm_out.shape[1]}")

if __name__ == "__main__":
    debug_msf_lcrnn_detailed()