#!/usr/bin/env python3
"""测试模型组件的脚本"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("测试模型组件导入")
    
    # 导入PyTorch
    import torch
    import torch.nn as nn
    print(f"✓ 导入 PyTorch 成功，版本: {torch.__version__}")
    
    # 导入配置
    from config import Config
    config = Config()
    print("✓ 导入 Config 类成功")
    
    # 测试单个模型组件
    print("\n测试模型组件导入:")
    
    # 测试图编码器
    try:
        from models.graph_encoder import GraphEncoder
        graph_encoder = GraphEncoder(node_dim=3, edge_dim=2, hidden_dim=256, output_dim=256)
        print("✓ 图编码器导入和创建成功")
    except Exception as e:
        print(f"✗ 图编码器错误: {e}")
    
    # 测试序列编码器
    try:
        from models.sequence_encoder import SequenceEncoder
        sequence_encoder = SequenceEncoder(input_channels=3, hidden_dim=256, encoder_type='mobilenet')
        print("✓ 序列编码器导入和创建成功")
    except Exception as e:
        print(f"✗ 序列编码器错误: {e}")
    
    # 测试融合编码器
    try:
        from models.fusion_encoder import FusionEncoder
        fusion_encoder = FusionEncoder(d_model=256, fusion_type='cross_attention')
        print("✓ 融合编码器导入和创建成功")
    except Exception as e:
        print(f"✗ 融合编码器错误: {e}")
    
    # 测试解码器
    try:
        from models.ctc_crf_decoder import CTC_CRF_Decoder
        decoder = CTC_CRF_Decoder(input_dim=256, num_classes=config.VOCAB_SIZE)
        print("✓ 解码器导入和创建成功")
    except Exception as e:
        print(f"✗ 解码器错误: {e}")
    
    # 测试完整模型
    print("\n测试完整模型:")
    try:
        from train import ChemicalFormulaModel
        model = ChemicalFormulaModel(
            vocab_size=config.VOCAB_SIZE,
            num_classes=config.VOCAB_SIZE,
            d_model=256,
            fusion_type='cross_attention'
        )
        print("✓ 完整模型导入和创建成功")
        
        # 测试模型前向传播
        print("\n测试模型前向传播:")
        # 创建模拟输入
        images = torch.randn(2, 3, 224, 224)
        
        # 模拟图数据
        graph_data = []
        for _ in range(2):
            node_features = torch.randn(5, 3)
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
            edge_features = torch.randn(8, 2)
            graph_data.append({
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_features': edge_features
            })
        
        # 模拟标签
        targets = torch.randint(0, config.VOCAB_SIZE, (2, 10))
        target_lengths = torch.tensor([8, 10])
        
        # 前向传播
        output, loss, _ = model(images, graph_data, targets, target_lengths)
        print("✓ 模型前向传播成功")
        print(f"  输出形状: {output.shape}")
        print(f"  损失值: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ 完整模型错误: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
    
    print("\n模型测试完成!")
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
