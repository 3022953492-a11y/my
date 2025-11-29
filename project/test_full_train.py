#!/usr/bin/env python3
"""
完整训练流程测试脚本
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset.chem_dataset import create_data_loaders
from train import ChemicalFormulaModel, Trainer

def test_full_training():
    """测试完整训练流程"""
    print("开始测试完整训练流程...")
    
    try:
        # 加载配置
        config = Config()
        print("✓ 配置加载成功")
        print(f"  数据根目录: {config.DATA_ROOT}")
        print(f"  标注文件: {config.ANNOTATION_FILE}")
        print(f"  词汇表文件: {config.VOCAB_FILE}")
        print(f"  词汇表大小: {config.VOCAB_SIZE}")
        
        # 创建数据加载器
        print("\n创建数据加载器...")
        train_loader, val_loader, vocab = create_data_loaders(
            data_dir=config.DATA_ROOT,
            annotation_file=config.ANNOTATION_FILE,
            vocab_file=config.VOCAB_FILE,
            batch_size=2,  # 使用小批量进行测试
            num_workers=0,
            image_size=config.IMG_SIZE,
            max_length=50
        )
        print("✓ 数据加载器创建成功")
        print(f"  训练批次数量: {len(train_loader)}")
        print(f"  验证批次数量: {len(val_loader)}")
        print(f"  词汇表大小: {len(vocab)}")
        
        # 测试一个批次
        batch = next(iter(train_loader))
        print(f"  批次键: {list(batch.keys())}")
        print(f"  图像形状: {batch['images'].shape}")
        print(f"  图数据数量: {len(batch['graph_data'])}")
        
        # 创建模型
        print("\n创建模型...")
        model = ChemicalFormulaModel(
            vocab_size=config.VOCAB_SIZE,
            num_classes=config.VOCAB_SIZE,
            d_model=256
        )
        print("✓ 模型创建成功")
        print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        print("\n测试前向传播...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # 使用真实数据测试
        images = batch['images'].to(device)
        graph_data = batch['graph_data']
        
        output = model(images, graph_data=graph_data, training=False)
        print("✓ 前向传播成功")
        print(f"  输出形状: {output.shape}")
        
        # 测试训练器初始化
        print("\n测试训练器初始化...")
        trainer = Trainer(config, device)
        print("✓ 训练器初始化成功")
        
        # 测试单步训练
        print("\n测试单步训练...")
        loss = trainer.train_epoch(train_loader)
        print(f"✓ 单步训练成功，损失: {loss:.4f}")
        
        # 测试验证
        print("\n测试验证...")
        val_loss = trainer.validate(val_loader)
        print(f"✓ 验证成功，验证损失: {val_loss:.4f}")
        
        print("\n🎉 完整训练流程测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_training()
    sys.exit(0 if success else 1)