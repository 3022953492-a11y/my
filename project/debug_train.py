#!/usr/bin/env python3
"""
调试训练脚本
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import ChemicalFormulaModel, Config
from dataset import create_data_loaders

def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")
    
    try:
        config = Config()
        model = ChemicalFormulaModel(
            vocab_size=config.VOCAB_SIZE,
            num_classes=config.VOCAB_SIZE,
            d_model=256
        )
        print(f"[OK] 模型创建成功")
        print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"[ERROR] 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward():
    """测试模型前向传播"""
    print("\n测试模型前向传播...")
    
    try:
        config = Config()
        model = ChemicalFormulaModel(
            vocab_size=config.VOCAB_SIZE,
            num_classes=config.VOCAB_SIZE,
            d_model=256
        )
        
        # 创建测试输入
        batch_size = 1
        images = torch.randn(batch_size, 3, 128, 128)  # 图像输入
        
        # 创建测试图数据（使用字典格式，与数据加载器一致）
        node_features = torch.randn(3, 64)  # 3个节点
        edge_index = torch.empty(2, 0, dtype=torch.long)  # 空边索引
        edge_features = torch.empty(0, 2)  # 空边特征
        
        graph_data = [{
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features
        }]
        
        # 其他输入
        targets = torch.randint(0, config.VOCAB_SIZE, (batch_size, 10))
        target_lengths = torch.tensor([10])
        
        # 计算实际的输入长度（序列编码器的输出序列长度）
        with torch.no_grad():
            sequence_features, _ = model.sequence_encoder(images)
            actual_seq_len = sequence_features.size(1)
        input_lengths = torch.tensor([actual_seq_len])
        
        # 前向传播
        output = model(
            images,
            graph_data=graph_data,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            training=True,
            device='cpu'
        )
        
        print(f"[OK] 模型前向传播成功")
        print(f"  输出类型: {type(output)}")
        print(f"  输出形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        return True
    except Exception as e:
        print(f"[ERROR] 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    try:
        config = Config()
        
        # 创建数据加载器
        train_loader, val_loader, vocab = create_data_loaders(
            data_dir=config.DATA_ROOT,
            annotation_file=os.path.join(config.DATA_ROOT, "..", "data", "annotations_full.json"),
            vocab_file=os.path.join(config.DATA_ROOT, "..", "data", "vocab_full.json"),
            batch_size=1,
            num_workers=0,
            image_size=config.IMG_SIZE,
            max_length=50
        )
        
        print(f"[OK] 数据加载器创建成功")
        print(f"  训练批次数量: {len(train_loader)}")
        print(f"  验证批次数量: {len(val_loader)}")
        print(f"  词汇表大小: {len(vocab)}")
        
        # 测试一个批次
        batch = next(iter(train_loader))
        print(f"  批次键: {list(batch.keys())}")
        print(f"  图像形状: {batch['images'].shape}")
        print(f"  图数据数量: {len(batch['graph_data'])}")
        
        return True
    except Exception as e:
        print(f"[ERROR] 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始调试训练脚本...")
    
    # 运行所有测试
    tests = [
        test_model_creation,
        test_model_forward,
        test_data_loader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n调试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("[OK] 所有调试测试通过，训练脚本应该可以正常运行")
    else:
        print("[ERROR] 部分调试测试失败，需要进一步修复")