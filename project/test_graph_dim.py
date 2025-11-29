#!/usr/bin/env python3
"""
测试GraphEncoder维度问题
"""

import os
import sys
import torch

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== 测试GraphEncoder维度 ===")

# 导入模块
try:
    from models.graph_encoder import GraphEncoder
    print("✓ GraphEncoder导入成功")
except Exception as e:
    print(f"✗ GraphEncoder导入失败: {e}")
    sys.exit(1)

# 创建GraphEncoder实例
print("\n1. 创建GraphEncoder实例...")
graph_encoder = GraphEncoder(node_dim=36, hidden_dim=64, output_dim=64, num_layers=3)
print(f"GraphEncoder参数:")
print(f"  node_dim: {graph_encoder.node_dim}")
print(f"  hidden_dim: {graph_encoder.hidden_dim}")
print(f"  output_dim: {graph_encoder.output_dim}")

# 检查node_encoder的结构
print("\n2. 检查node_encoder结构...")
print(f"node_encoder: {graph_encoder.node_encoder}")

# 检查第一层线性层的输入维度
first_layer = graph_encoder.node_encoder[0]
if hasattr(first_layer, 'in_features'):
    print(f"第一层线性层输入维度: {first_layer.in_features}")
    print(f"第一层线性层输出维度: {first_layer.out_features}")

# 创建测试数据
print("\n3. 创建测试数据...")
from torch_geometric.data import Data

# 创建节点特征，维度为36（与node_dim匹配）
node_features = torch.randn(3, 36)  # 3个节点，每个节点36维特征
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
batch = torch.zeros(3, dtype=torch.long)

data_obj = Data(x=node_features, edge_index=edge_index, batch=batch)

print(f"节点特征维度: {data_obj.x.shape}")
print(f"边索引维度: {data_obj.edge_index.shape}")
print(f"批处理维度: {data_obj.batch.shape}")

# 测试node_encoder单独运行
print("\n4. 单独测试node_encoder...")
try:
    # 直接调用node_encoder
    encoded_features = graph_encoder.node_encoder(data_obj.x)
    print(f"✓ node_encoder成功，输出维度: {encoded_features.shape}")
except Exception as e:
    print(f"✗ node_encoder失败: {e}")
    import traceback
    traceback.print_exc()

# 测试完整GraphEncoder
print("\n5. 测试完整GraphEncoder...")
try:
    output = graph_encoder(data_obj)
    print(f"✓ GraphEncoder成功，输出维度: {output.shape}")
except Exception as e:
    print(f"✗ GraphEncoder失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")