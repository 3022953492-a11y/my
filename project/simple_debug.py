#!/usr/bin/env python3
"""
简单调试脚本，逐步检查维度问题
"""

import os
import sys
import torch

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== 简单维度调试 ===")

# 1. 首先检查是否能导入模块
print("\n1. 检查模块导入...")
try:
    from models.sequence_encoder import SequenceEncoder
    print("✓ SequenceEncoder导入成功")
except Exception as e:
    print(f"✗ SequenceEncoder导入失败: {e}")
    sys.exit(1)

try:
    from models.graph_encoder import GraphEncoder
    print("✓ GraphEncoder导入成功")
except Exception as e:
    print(f"✗ GraphEncoder导入失败: {e}")
    sys.exit(1)

try:
    from models.fusion_encoder import FusionEncoder
    print("✓ FusionEncoder导入成功")
except Exception as e:
    print(f"✗ FusionEncoder导入失败: {e}")
    sys.exit(1)

# 2. 创建模型实例
print("\n2. 创建模型实例...")
seq_encoder = SequenceEncoder(input_channels=3, hidden_size=64)
print(f"SequenceEncoder hidden_size: {seq_encoder.hidden_size}")

graph_encoder = GraphEncoder(node_dim=36, hidden_dim=64, output_dim=64, num_layers=3)
print(f"GraphEncoder output_dim: {graph_encoder.output_dim}")

fusion_encoder = FusionEncoder(d_model=64)
print(f"FusionEncoder d_model: {fusion_encoder.d_model}")

# 3. 测试输入数据
print("\n3. 测试输入数据...")
batch_size = 2
images = torch.randn(batch_size, 3, 128, 32)
print(f"图像维度: {images.shape}")

# 4. 测试SequenceEncoder
print("\n4. 测试SequenceEncoder...")
try:
    sequence_features, _ = seq_encoder(images)
    print(f"序列特征维度: {sequence_features.shape}")
except Exception as e:
    print(f"SequenceEncoder失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试GraphEncoder
print("\n5. 测试GraphEncoder...")
from torch_geometric.data import Data

graph_data = []
for i in range(batch_size):
    graph_data.append({
        'node_features': torch.randn(3, 36),
        'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous(),
    })

try:
    graph_features = []
    for i in range(batch_size):
        data_obj = Data(
            x=graph_data[i]['node_features'],
            edge_index=graph_data[i]['edge_index'],
            batch=torch.zeros(graph_data[i]['node_features'].size(0), dtype=torch.long)
        )
        graph_feat = graph_encoder(data_obj)
        graph_features.append(graph_feat)
    
    graph_features = torch.stack(graph_features)
    print(f"图特征维度: {graph_features.shape}")
except Exception as e:
    print(f"GraphEncoder失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 调试完成 ===")