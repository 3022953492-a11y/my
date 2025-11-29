#!/usr/bin/env python3
"""
调试模型各模块维度
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.graph_encoder import GraphEncoder
from models.sequence_encoder import SequenceEncoder
from models.fusion_encoder import FusionEncoder

def debug_dimensions():
    """调试各模块维度"""
    print("=== 调试模型维度 ===")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 50
    d_model = 256
    
    # 测试图编码器
    print("\n1. 测试图编码器:")
    try:
        graph_encoder = GraphEncoder(node_dim=64, hidden_dim=128, output_dim=d_model, num_layers=3)
        
        # 创建测试图数据
        from torch_geometric.data import Data
        node_features = torch.randn(5, 64)  # 5个节点，64维特征
        edge_index = torch.empty(2, 0, dtype=torch.long)  # 空边索引
        batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)  # 所有节点属于同一批次
        graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)
        
        graph_features = graph_encoder(graph_data)
        print(f"   ✓ 图编码器输出维度: {graph_features.shape}")
        print(f"   输入节点数: {node_features.shape[0]}, 输出批次维度: {graph_features.shape[0]}")
    except Exception as e:
        print(f"   ✗ 图编码器失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试序列编码器
    print("\n2. 测试序列编码器:")
    try:
        sequence_encoder = SequenceEncoder(input_channels=3, hidden_size=d_model)
        
        # 创建测试图像数据
        images = torch.randn(batch_size, 3, 224, 224)
        sequence_features, _ = sequence_encoder(images)
        print(f"   ✓ 序列编码器输出维度: {sequence_features.shape}")
        print(f"   输入图像: {images.shape}, 输出序列: {sequence_features.shape}")
    except Exception as e:
        print(f"   ✗ 序列编码器失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试融合编码器
    print("\n3. 测试融合编码器:")
    try:
        fusion_encoder = FusionEncoder(d_model=d_model)
        
        # 创建测试特征
        graph_features = torch.randn(batch_size, d_model)
        sequence_features = torch.randn(batch_size, seq_len, d_model)
        
        fused_features = fusion_encoder(graph_features, sequence_features)
        print(f"   ✓ 融合编码器输出维度: {fused_features.shape}")
        print(f"   图特征: {graph_features.shape}, 序列特征: {sequence_features.shape}")
    except Exception as e:
        print(f"   ✗ 融合编码器失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试完整流程
    print("\n4. 测试完整流程:")
    try:
        # 重新创建测试数据
        images = torch.randn(batch_size, 3, 224, 224)
        
        # 创建多个图数据（每个批次样本一个图）
        graph_data_list = []
        for i in range(batch_size):
            node_features = torch.randn(3, 64)  # 每个图3个节点
            edge_index = torch.empty(2, 0, dtype=torch.long)
            batch = torch.tensor([0, 0, 0], dtype=torch.long)
            graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)
            graph_data_list.append(graph_data)
        
        # 处理每个图的特征
        graph_features_list = []
        for graph_data in graph_data_list:
            graph_feat = graph_encoder(graph_data)
            graph_features_list.append(graph_feat.squeeze(0))  # 移除批次维度
        
        graph_features = torch.stack(graph_features_list)
        sequence_features, _ = sequence_encoder(images)
        
        print(f"   ✓ 完整流程维度检查:")
        print(f"   图特征堆叠后: {graph_features.shape}")
        print(f"   序列特征: {sequence_features.shape}")
        
        # 测试融合
        fused_features = fusion_encoder(graph_features, sequence_features)
        print(f"   融合特征: {fused_features.shape}")
        
    except Exception as e:
        print(f"   ✗ 完整流程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dimensions()