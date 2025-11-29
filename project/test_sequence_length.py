#!/usr/bin/env python3
"""
测试序列编码器的输出序列长度
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sequence_encoder import SequenceEncoder
from dataset import create_data_loaders

def test_sequence_length():
    """测试序列编码器的输出序列长度"""
    
    # 创建序列编码器
    print("创建序列编码器...")
    encoder = SequenceEncoder(encoder_type='msf_lcrnn', vocab_size=243)
    
    # 创建测试数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, vocab = create_data_loaders(
        data_dir='E:\\化学公式\\dataset',
        annotation_file='E:\\化学公式\\data\\annotations_full.json',
        vocab_file='E:\\化学公式\\data\\vocab_full.json',
        batch_size=1,
        num_workers=0
    )
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 测试一个批次
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 1:  # 只测试第一个批次
            break
            
        images = batch['images']
        targets = batch['targets']
        target_lengths = batch['target_lengths']
        
        print(f"图像形状: {images.shape}")
        print(f"目标序列形状: {targets.shape}")
        print(f"目标序列长度: {target_lengths}")
        
        # 通过序列编码器获取序列特征
        with torch.no_grad():
            sequence_features, _ = encoder(images)
            
        print(f"序列特征形状: {sequence_features.shape}")
        print(f"序列长度 (T): {sequence_features.size(1)}")
        
        # 检查CTC损失的要求
        actual_seq_len = sequence_features.size(1)
        max_target_len = targets.size(1)
        
        print(f"实际序列长度: {actual_seq_len}")
        print(f"最大目标序列长度: {max_target_len}")
        
        if actual_seq_len < max_target_len:
            print(f"警告: 序列长度({actual_seq_len}) < 最大目标长度({max_target_len})")
            print("CTC损失需要序列长度 >= 目标长度")
        else:
            print("序列长度满足CTC要求")
        
        # 测试CTC损失
        try:
            # 创建输入长度张量
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), actual_seq_len, dtype=torch.long)
            
            print(f"输入长度: {input_lengths}")
            print(f"目标长度: {target_lengths}")
            
            # 测试CTC损失
            ctc_loss = torch.nn.CTCLoss()
            
            # 创建模拟的logits
            logits = torch.randn(batch_size, actual_seq_len, len(vocab))
            
            # 调整维度为CTC需要的格式 [T, N, C]
            logits_ctc = logits.transpose(0, 1)
            
            loss = ctc_loss(
                logits_ctc.log_softmax(2), 
                targets, 
                input_lengths, 
                target_lengths
            )
            
            print(f"CTC损失计算成功: {loss.item():.4f}")
            
        except Exception as e:
            print(f"CTC损失计算失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_sequence_length()