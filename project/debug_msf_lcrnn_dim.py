#!/usr/bin/env python3
"""
调试MSF-LCRNN编码器输出维度
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sequence_encoder import MSF_LCRNN_Encoder

def debug_msf_lcrnn_dimensions():
    """调试MSF-LCRNN编码器输出维度"""
    
    # 创建MSF-LCRNN编码器
    encoder = MSF_LCRNN_Encoder(input_channels=3, hidden_size=256)
    
    # 创建测试输入
    batch_size, channels, height, width = 1, 3, 128, 32
    test_input = torch.randn(batch_size, channels, height, width)
    
    print(f"输入图像形状: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        features = encoder(test_input)
    
    print(f"MSF-LCRNN输出特征形状: {features.shape}")
    
    # 检查每个阶段的输出
    print("\n检查各阶段输出:")
    
    # 尺度1
    feat1 = encoder.multiscale_cnn['scale1'](test_input)
    print(f"尺度1输出形状: {feat1.shape}")
    
    # 尺度2
    feat2 = encoder.multiscale_cnn['scale2'](feat1)
    print(f"尺度2输出形状: {feat2.shape}")
    
    # 尺度3
    feat3 = encoder.multiscale_cnn['scale3'](feat2)
    print(f"尺度3输出形状: {feat3.shape}")
    
    # 特征融合
    scale_features = []
    scale_features.append(torch.nn.functional.adaptive_avg_pool2d(feat1, (1, None)))
    scale_features.append(torch.nn.functional.adaptive_avg_pool2d(feat2, (1, None)))
    scale_features.append(torch.nn.functional.adaptive_avg_pool2d(feat3, (1, None)))
    
    print(f"尺度1池化后形状: {scale_features[0].shape}")
    print(f"尺度2池化后形状: {scale_features[1].shape}")
    print(f"尺度3池化后形状: {scale_features[2].shape}")
    
    fused_features = torch.cat(scale_features, dim=1)
    print(f"融合特征形状: {fused_features.shape}")
    
    fused_features = encoder.feature_fusion(fused_features)
    print(f"特征融合后形状: {fused_features.shape}")
    
    # 调整维度
    sequence_features = fused_features.squeeze(2).permute(0, 2, 1)
    print(f"调整维度后形状: {sequence_features.shape}")

if __name__ == "__main__":
    debug_msf_lcrnn_dimensions()