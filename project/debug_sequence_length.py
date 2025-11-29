#!/usr/bin/env python3
"""
调试序列编码器输出长度
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sequence_encoder import SequenceEncoder

def debug_sequence_length():
    """调试序列编码器输出长度"""
    print("=== 调试序列编码器输出长度 ===")
    
    # 创建测试数据
    batch_size = 2
    
    # 测试不同图像尺寸
    image_sizes = [
        (128, 128),  # 小尺寸
        (224, 224),  # 标准尺寸
        (256, 256)   # 大尺寸
    ]
    
    for img_size in image_sizes:
        print(f"\n测试图像尺寸: {img_size}")
        
        # 创建测试图像
        images = torch.randn(batch_size, 3, img_size[0], img_size[1])
        
        # 测试MobileNetV3编码器
        print("  MobileNetV3编码器:")
        try:
            encoder = SequenceEncoder(encoder_type='mobilenet', hidden_size=256)
            sequence_features, ctc_logits = encoder(images)
            print(f"    序列特征维度: {sequence_features.shape}")
            print(f"    序列长度: {sequence_features.shape[1]}")
            if ctc_logits is not None:
                print(f"    CTC输出维度: {ctc_logits.shape}")
        except Exception as e:
            print(f"    失败: {e}")
        
        # 测试MSF-LCRNN编码器
        print("  MSF-LCRNN编码器:")
        try:
            encoder = SequenceEncoder(encoder_type='msf_lcrnn', hidden_size=256)
            sequence_features, ctc_logits = encoder(images)
            print(f"    序列特征维度: {sequence_features.shape}")
            print(f"    序列长度: {sequence_features.shape[1]}")
            if ctc_logits is not None:
                print(f"    CTC输出维度: {ctc_logits.shape}")
        except Exception as e:
            print(f"    失败: {e}")

if __name__ == "__main__":
    debug_sequence_length()