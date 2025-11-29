#!/usr/bin/env python3
"""
测试修复后的train.py是否能正常运行
"""

import os
import sys
import torch

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# 测试导入
print("测试模块导入...")
try:
    # 尝试从当前目录导入
    from models.graph_encoder import GraphEncoder as ProjectGraphEncoder
    from models.sequence_encoder import SequenceEncoder as ProjectSequenceEncoder
    from models.fusion_encoder import FusionEncoder as ProjectFusionEncoder
    from models.ctc_crf_decoder import CTC_CRF_Decoder as ProjectCTC_CRF_Decoder
    print("✓ 模型模块从project目录导入成功")
    print(f"GraphEncoder类: {ProjectGraphEncoder}")
    print(f"GraphEncoder模块: {ProjectGraphEncoder.__module__}")
    GraphEncoder = ProjectGraphEncoder
    SequenceEncoder = ProjectSequenceEncoder
    FusionEncoder = ProjectFusionEncoder
    CTC_CRF_Decoder = ProjectCTC_CRF_Decoder
except Exception as e:
    print(f"✗ 从project目录导入失败: {e}")
    try:
        # 尝试从src目录导入
        from src.models.graph_encoder import GraphEncoder as SrcGraphEncoder
        from src.models.sequence_encoder import SequenceEncoder as SrcSequenceEncoder
        from src.models.fusion_encoder import FusionEncoder as SrcFusionEncoder
        from src.models.ctc_crf_decoder import CTC_CRF_Decoder as SrcCTC_CRF_Decoder
        print("✓ 模型模块从src目录导入成功")
        GraphEncoder = SrcGraphEncoder
        SequenceEncoder = SrcSequenceEncoder
        FusionEncoder = SrcFusionEncoder
        CTC_CRF_Decoder = SrcCTC_CRF_Decoder
    except Exception as e2:
        print(f"✗ 从src目录导入失败: {e2}")
        sys.exit(1)

try:
    from dataset.chem_dataset import ChemicalFormulaDataset
    print("✓ 数据集模块导入成功")
except Exception as e:
    print(f"✗ 数据集模块导入失败: {e}")
    sys.exit(1)

# 测试数据加载
print("\n测试数据加载...")
try:
    data_dir = "./data"
    annotation_file = "./data/annotations.json"
    vocab_file = "./data/vocab.json"
    
    dataset = ChemicalFormulaDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        vocab_file=vocab_file,
        image_size=(128, 32),
        max_length=50,
        is_training=True
    )
    print(f"✓ 数据集创建成功，大小: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"✓ 样本加载成功")
    print(f"  图像形状: {sample['image'].shape}")
    print(f"  目标序列长度: {sample['target_length'].item()}")
    print(f"  化学公式: {sample['formula']}")
    
except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    sys.exit(1)

# 测试模型创建
print("\n测试模型创建...")
try:
    class SimpleModel(torch.nn.Module):
        def __init__(self, vocab_size=100):
            super().__init__()
            self.graph_encoder = GraphEncoder(node_dim=36, hidden_dim=64, output_dim=64, num_layers=3)
            self.sequence_encoder = SequenceEncoder(input_channels=3, hidden_size=64, vocab_size=vocab_size)
            self.fusion_encoder = FusionEncoder(d_model=64)
            self.decoder = CTC_CRF_Decoder(input_dim=64, num_classes=vocab_size)
        
        def forward(self, images, graph_data):
            # 图编码
            graph_features = []
            for i in range(images.size(0)):
                # 将字典格式的图数据转换为torch_geometric.data.Data对象
                from torch_geometric.data import Data
                data_obj = Data(
                    x=graph_data[i]['node_features'],
                    edge_index=graph_data[i]['edge_index'],
                    batch=torch.zeros(graph_data[i]['node_features'].size(0), dtype=torch.long)
                )
                graph_feat = self.graph_encoder(data_obj)
                graph_features.append(graph_feat)
            graph_features = torch.stack(graph_features)
            
            # 序列编码
            sequence_features, _ = self.sequence_encoder(images)  # 只取序列特征，忽略CTC输出
            
            # 融合
            fused_features = self.fusion_encoder(graph_features, sequence_features)
            
            # 解码
            return self.decoder(fused_features, training=False)
    
    # 创建模型
    model = SimpleModel()
    print(f"✓ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 2
    images = torch.randn(batch_size, 3, 128, 32)
    
    # 创建模拟图数据
    graph_data = []
    for i in range(batch_size):
        graph_data.append({
            'node_features': torch.randn(3, 36),  # 节点特征维度改为36
            'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous(),
            'edge_features': torch.randn(2, 2)
        })
    
    output = model(images, graph_data)
    print(f"✓ 前向传播成功，输出形状: {output.shape}")
    
except Exception as e:
    print(f"✗ 模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 所有测试通过！train.py应该可以正常运行了。")
print("\n现在可以尝试运行: python train.py --epochs 1 --batch_size 1")