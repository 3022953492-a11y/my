#!/usr/bin/env python3
"""测试数据加载器的脚本"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("测试数据加载器")
    
    # 导入必要的模块
    import torch
    from config import Config
    from dataset.chem_dataset import create_data_loaders
    
    print("✓ 导入必要模块成功")
    
    # 初始化配置
    config = Config()
    print(f"数据根目录: {config.DATA_ROOT}")
    
    # 测试数据加载器
    print("\n测试数据加载...")
    train_loader, val_loader, vocab = create_data_loaders(
        data_dir=config.DATA_ROOT,
        annotation_file=os.path.join(config.DATA_ROOT, "annotations.json"),
        vocab_file=os.path.join(config.DATA_ROOT, "vocab.json"),
        batch_size=2,  # 使用小批量
        image_size=config.IMG_SIZE,
        max_length=50
    )
    
    print(f"✓ 数据加载成功")
    print(f"训练数据加载器大小: {len(train_loader)}")
    print(f"验证数据加载器大小: {len(val_loader)}")
    
    # 测试迭代器
    print("\n测试数据迭代器...")
    for i, batch in enumerate(train_loader):
        images, labels, label_lengths, texts = batch
        print(f"批次 {i+1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签长度: {label_lengths}")
        print(f"  文本: {texts}")
        break  # 只测试一个批次
    
    print("\n数据加载器测试通过!")
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
