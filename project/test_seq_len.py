#!/usr/bin/env python3
"""
测试序列编码器输出长度
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sequence_encoder import SequenceEncoder

def test_sequence_length():
    """测试序列编码器输出长度"""
    print("=== 测试序列编码器输出长度 ===")
    
    try:
        # 创建序列编码器
        encoder = SequenceEncoder(encoder_type='mobilenet', hidden_size=256)
        
        # 测试不同图像尺寸
        image_sizes = [(128, 128), (224, 224), (256, 256)]
        
        for img_size in image_sizes:
            print(f"\n测试图像尺寸: {img_size}")
            
            # 创建测试图像
            batch_size = 1
            images = torch.randn(batch_size, 3, img_size[0], img_size[1])
            
            # 前向传播
            with torch.no_grad():
                sequence_features, ctc_logits = encoder(images)
                
            print(f"  序列特征维度: {sequence_features.shape}")
            print(f"  序列长度: {sequence_features.shape[1]}")
            
            if ctc_logits is not None:
                print(f"  CTC输出维度: {ctc_logits.shape}")
        
        print("\n✓ 序列编码器测试成功")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_sequence_length()