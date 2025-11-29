#!/usr/bin/env python3
"""
测试图编码器修复
"""

import torch
import torch.nn as nn
from models.graph_encoder import GraphEncoder
from torch_geometric.data import Data

def test_empty_edge_index():
    """测试空边索引的情况"""
    print("测试空边索引情况...")
    
    # 创建图编码器
    encoder = GraphEncoder(node_dim=64, hidden_dim=128, output_dim=256)
    
    # 创建测试数据：有节点但无边
    node_features = torch.randn(3, 64)  # 3个节点，每个节点64维特征
    edge_index = torch.empty(2, 0, dtype=torch.long)  # 空边索引
    batch = torch.tensor([0, 0, 0], dtype=torch.long)  # 所有节点属于同一个图
    
    graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)
    
    try:
        output = encoder(graph_data)
        print(f"✓ 空边索引测试通过")
        print(f"  输入节点数: {node_features.shape[0]}")
        print(f"  边索引形状: {edge_index.shape}")
        print(f"  输出特征形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ 空边索引测试失败: {e}")
        return False

def test_normal_graph():
    """测试正常图的情况"""
    print("\n测试正常图情况...")
    
    # 创建图编码器
    encoder = GraphEncoder(node_dim=64, hidden_dim=128, output_dim=256)
    
    # 创建测试数据：有节点和边
    node_features = torch.randn(3, 64)  # 3个节点，每个节点64维特征
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # 边连接
    batch = torch.tensor([0, 0, 0], dtype=torch.long)  # 所有节点属于同一个图
    
    graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)
    
    try:
        output = encoder(graph_data)
        print(f"✓ 正常图测试通过")
        print(f"  输入节点数: {node_features.shape[0]}")
        print(f"  边索引形状: {edge_index.shape}")
        print(f"  输出特征形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ 正常图测试失败: {e}")
        return False

def test_single_node():
    """测试单节点图的情况"""
    print("\n测试单节点图情况...")
    
    # 创建图编码器
    encoder = GraphEncoder(node_dim=64, hidden_dim=128, output_dim=256)
    
    # 创建测试数据：单节点
    node_features = torch.randn(1, 64)  # 1个节点，64维特征
    edge_index = torch.empty(2, 0, dtype=torch.long)  # 空边索引
    batch = torch.tensor([0], dtype=torch.long)  # 单个节点
    
    graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)
    
    try:
        output = encoder(graph_data)
        print(f"✓ 单节点图测试通过")
        print(f"  输入节点数: {node_features.shape[0]}")
        print(f"  边索引形状: {edge_index.shape}")
        print(f"  输出特征形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ 单节点图测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始图编码器测试...")
    
    # 运行所有测试
    tests = [
        test_empty_edge_index,
        test_normal_graph,
        test_single_node
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过，图编码器修复成功！")
    else:
        print("✗ 部分测试失败，需要进一步修复")