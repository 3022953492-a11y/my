#!/usr/bin/env python3
"""测试导入和配置的简单脚本"""

import os
import sys

try:
    print("开始测试导入...")
    
    # 导入必要的模块
    import torch
    print("✓ 导入 PyTorch 成功")
    
    # 添加项目路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # 导入配置
    from config import Config
    print("✓ 导入 Config 类成功")
    
    # 测试配置初始化
    config = Config()
    print(f"✓ 配置初始化成功")
    print(f"数据根目录: {config.DATA_ROOT}")
    print(f"图像目录1: {config.IMAGES_DIR1}")
    print(f"图像目录2: {config.IMAGES_DIR2}")
    print(f"词汇表大小: {config.VOCAB_SIZE}")
    
    # 测试数据加载器
    from dataset.chem_dataset import create_data_loaders
    print("✓ 导入 create_data_loaders 函数成功")
    
    # 尝试加载数据
    print("\n尝试加载数据...")
    train_loader, val_loader, vocab = create_data_loaders(
        data_dir=config.DATA_ROOT,
        annotation_file=os.path.join(config.DATA_ROOT, "annotations.json"),
        vocab_file=os.path.join(config.DATA_ROOT, "vocab.json"),
        batch_size=4,
        image_size=config.IMG_SIZE,
        max_length=50
    )
    print(f"✓ 数据加载成功")
    print(f"训练数据加载器大小: {len(train_loader)}")
    print(f"验证数据加载器大小: {len(val_loader)}")
    
    # 测试模型导入
    from models.graph_encoder import GraphEncoder
    from models.sequence_encoder import SequenceEncoder
    from models.fusion_encoder import FusionEncoder
    from models.ctc_crf_decoder import CTC_CRF_Decoder
    print("✓ 导入模型组件成功")
    
    # 测试完整模型
    from train import ChemicalFormulaModel
    print("✓ 导入 ChemicalFormulaModel 类成功")
    
    # 创建模型
    model = ChemicalFormulaModel(
        vocab_size=config.VOCAB_SIZE,
        num_classes=config.VOCAB_SIZE
    )
    print("✓ 创建模型成功")
    
    print("\n所有测试通过!")
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
