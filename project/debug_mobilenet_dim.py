#!/usr/bin/env python3
"""
调试MobileNetV3输出维度
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sequence_encoder import MobileNetV3Encoder

def debug_mobilenet_dimensions():
    """调试MobileNetV3输出维度"""
    
    # 创建MobileNetV3编码器
    encoder = MobileNetV3Encoder(output_channels=256)
    
    # 创建测试输入
    batch_size, channels, height, width = 1, 3, 128, 32
    test_input = torch.randn(batch_size, channels, height, width)
    
    print(f"输入图像形状: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        features = encoder(test_input)
    
    print(f"MobileNetV3输出特征形状: {features.shape}")
    
    # 检查维度调整
    batch_size, channels, height, width = features.shape
    print(f"特征图维度: batch_size={batch_size}, channels={channels}, height={height}, width={width}")
    
    # 测试维度调整
    cnn_features = features.view(batch_size, channels * height, width)
    print(f"展平后形状: {cnn_features.shape}")
    
    cnn_features = cnn_features.permute(0, 2, 1)  # [batch_size, W', C*H']
    print(f"转置后形状: {cnn_features.shape}")
    
    # 计算需要的投影层输入维度
    projection_input_dim = channels * height
    print(f"投影层需要的输入维度: {projection_input_dim}")

if __name__ == "__main__":
    debug_mobilenet_dimensions()