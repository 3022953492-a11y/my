#!/usr/bin/env python3
"""
图编码流（Graph Stream）实现
基于GNN的分子图结构编码，包含图注意力和子图模式识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data


class GraphAttentionLayer(nn.Module):
    """图注意力层（GATv2风格）"""
    
    def __init__(self, in_channels, out_channels, heads=8, concat=True):
        super(GraphAttentionLayer, self).__init__()
        # GATConv的第一个参数应该是输入维度，第二个参数是每个头的输出维度
        # 当concat=True时，总输出维度 = heads * out_channels
        # 当concat=False时，总输出维度 = out_channels
        
        # 确保GATConv的输入维度与实际输入维度一致
        # 当concat=False时，输出维度应为out_channels
        # 当concat=True时，输出维度应为heads * out_channels
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=concat)
        
    def forward(self, x, edge_index):
        # 检查输入维度是否与GATConv期望的维度匹配
        if x.size(1) != self.gat_conv.in_channels:
            raise ValueError(f"GATConv输入维度不匹配: 期望{self.gat_conv.in_channels}, 实际{x.size(1)}")
        return self.gat_conv(x, edge_index)


class SubgraphMining(nn.Module):
    """子图模式识别模块"""
    
    def __init__(self, hidden_dim, motif_dim=64):
        super(SubgraphMining, self).__init__()
        self.hidden_dim = hidden_dim
        self.motif_dim = motif_dim
        
        # 常见化学基团模式
        self.common_motifs = {
            'OH': ['O', 'H'],  # 羟基
            'COOH': ['C', 'O', 'O', 'H'],  # 羧基
            'NH2': ['N', 'H', 'H'],  # 氨基
            'SO4': ['S', 'O', 'O', 'O', 'O'],  # 硫酸根
            'PO4': ['P', 'O', 'O', 'O', 'O'],  # 磷酸根
        }
        
        # 子图嵌入网络
        self.motif_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, motif_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        识别化学基团模式
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批处理索引
        """
        num_nodes = x.size(0)
        
        # 提取局部子图特征
        motif_features = []
        
        # 对每个节点，提取其邻居信息
        for node_idx in range(num_nodes):
            # 获取当前节点的邻居
            neighbors = self._get_neighbors(node_idx, edge_index)
            
            # 即使没有邻居，也提取特征（孤立节点）
            local_features = self._extract_local_features(node_idx, neighbors, x)
            motif_features.append(local_features)
        
        if len(motif_features) > 0:
            motif_features = torch.stack(motif_features)
            # 通过MLP得到子图嵌入
            motif_embeddings = self.motif_encoder(motif_features)
            
            # 全局池化得到图级别的子图特征
            if batch is not None:
                graph_motif_features = global_mean_pool(motif_embeddings, batch)
            else:
                graph_motif_features = torch.mean(motif_embeddings, dim=0, keepdim=True)
            
            return graph_motif_features
        else:
            return torch.zeros(1, self.motif_dim, device=x.device)
    
    def _get_neighbors(self, node_idx, edge_index):
        """获取节点的邻居"""
        if edge_index.numel() == 0 or edge_index.size(0) == 0:
            # 如果边索引为空，返回空列表
            return []
        mask = edge_index[0] == node_idx
        neighbors = edge_index[1][mask].tolist()
        return neighbors
    
    def _extract_local_features(self, center_node, neighbors, x):
        """提取局部子图特征"""
        # 中心节点特征
        center_feat = x[center_node]
        
        # 邻居节点特征
        neighbor_feats = x[neighbors]
        
        # 聚合邻居特征
        if len(neighbors) > 0:
            neighbor_agg = torch.mean(neighbor_feats, dim=0)
        else:
            neighbor_agg = torch.zeros_like(center_feat)
        
        # 组合特征：中心节点 + 邻居聚合 + 交互特征
        local_features = torch.cat([
            center_feat,
            neighbor_agg,
            center_feat * neighbor_agg  # 交互特征
        ])
        
        return local_features


class GraphEncoder(nn.Module):
    """图编码流主模块"""
    
    def __init__(self, node_dim=64, hidden_dim=128, output_dim=256, num_layers=3):
        super(GraphEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 多层图注意力网络
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # 第一层：输入维度hidden_dim，输出维度hidden_dim
                # 当concat=False时，输出维度应为hidden_dim
                self.gat_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, heads=1, concat=False))
            else:
                # 后续层：输入维度hidden_dim，输出维度hidden_dim
                # 当concat=False时，输出维度应为hidden_dim
                self.gat_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, heads=1, concat=False))
        
        # 子图模式识别
        self.subgraph_mining = SubgraphMining(hidden_dim)
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + self.subgraph_mining.motif_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, graph_data):
        """
        Args:
            graph_data: 图数据对象，包含:
                - x: 节点特征 [num_nodes, node_dim]
                - edge_index: 边索引 [2, num_edges]
                - batch: 批处理索引 [num_nodes]
        """
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        
        # 节点特征编码
        x = self.node_encoder(x)
        
        # 检查边索引是否为空（没有边的图）
        if edge_index.numel() == 0 or edge_index.size(0) == 0:
            # 对于没有边的图，直接使用节点特征，跳过GAT层
            print("警告：边索引为空，跳过GAT层处理")
        else:
            # 多层图注意力传播（仅当有边时）
            for gat_layer in self.gat_layers:
                x = F.relu(gat_layer(x, edge_index))
        
        # 子图模式识别
        motif_features = self.subgraph_mining(x, edge_index, batch)
        
        # 全局图特征（平均池化）
        graph_features = global_mean_pool(x, batch)
        
        # 融合图特征和子图特征
        fused_features = torch.cat([graph_features, motif_features], dim=1)
        
        # 输出投影
        output = self.output_proj(fused_features)
        
        return output


class ChemicalGraphBuilder:
    """化学分子图构建器"""
    
    def __init__(self):
        # 化学元素到索引的映射
        self.element_to_idx = {
            'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7,
            'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14,
            'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21,
            'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28,
            'Zn': 29, 'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33, 'Br': 34, 'Kr': 35
        }
        
        # 化学键类型
        self.bond_types = {
            'single': 0,    # 单键
            'double': 1,    # 双键
            'triple': 2,    # 三键
            'ionic': 3,     # 离子键
            'coordinate': 4 # 配位键
        }
    
    def build_graph_from_formula(self, formula):
        """从化学式构建分子图"""
        # 简化的化学式解析（实际应用中需要更复杂的解析器）
        elements = self._parse_formula(formula)
        
        # 构建节点特征
        node_features = []
        for element in elements:
            # 元素类型特征
            element_idx = self.element_to_idx.get(element, 0)
            # 创建one-hot编码
            element_feat = torch.zeros(len(self.element_to_idx))
            element_feat[element_idx] = 1.0
            node_features.append(element_feat)
        
        node_features = torch.stack(node_features)
        
        # 简化的边构建（实际应用中需要分子结构信息）
        edge_index = self._build_simple_edges(len(elements))
        
        return Data(x=node_features, edge_index=edge_index)
    
    def _parse_formula(self, formula):
        """解析化学式（简化版本）"""
        # 简单的元素分割（实际需要更复杂的解析）
        elements = []
        current_element = ''
        
        for char in formula:
            if char.isupper():
                if current_element:
                    elements.append(current_element)
                current_element = char
            elif char.islower():
                current_element += char
            elif char.isdigit():
                # 处理数字（重复元素）
                if current_element:
                    count = int(char)
                    elements.extend([current_element] * (count - 1))
        
        if current_element:
            elements.append(current_element)
        
        return elements
    
    def _build_simple_edges(self, num_nodes):
        """构建简单的边连接（链式连接）"""
        edge_list = []
        
        # 构建链式连接
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])  # 无向图
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index


def test_graph_encoder():
    """测试图编码器"""
    # 创建测试数据
    graph_builder = ChemicalGraphBuilder()
    
    # 构建简单分子图
    graph_data = graph_builder.build_graph_from_formula("H2O")
    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
    
    # 创建图编码器
    encoder = GraphEncoder(node_dim=len(graph_builder.element_to_idx))
    
    # 前向传播
    output = encoder(graph_data)
    
    print(f"输入图节点数: {graph_data.x.size(0)}")
    print(f"输出特征维度: {output.shape}")
    print("图编码器测试通过!")


if __name__ == "__main__":
    test_graph_encoder()