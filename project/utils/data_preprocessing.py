#!/usr/bin/env python3
"""
数据预处理模块
包含图像预处理和分子图预处理功能
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import networkx as nx


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, image_size=(128, 32), normalize=True):
        """
        初始化图像预处理器
        
        Args:
            image_size: 目标图像尺寸 (height, width)
            normalize: 是否进行归一化
        """
        self.image_size = image_size
        self.normalize = normalize
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        if normalize:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def preprocess(self, image_path):
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            processed_image: 预处理后的图像张量
        """
        try:
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            processed_image = self.transform(image)
            
            return processed_image
            
        except Exception as e:
            print(f"图像预处理错误: {e}")
            return None
    
    def preprocess_batch(self, image_paths):
        """
        批量预处理图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            batch_images: 批量预处理后的图像张量
        """
        batch_images = []
        
        for image_path in image_paths:
            processed_image = self.preprocess(image_path)
            if processed_image is not None:
                batch_images.append(processed_image)
        
        if batch_images:
            return torch.stack(batch_images)
        else:
            return None


class GraphPreprocessor:
    """分子图预处理器"""
    
    def __init__(self, max_nodes=50, node_feature_dim=64, edge_feature_dim=32):
        """
        初始化分子图预处理器
        
        Args:
            max_nodes: 最大节点数
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
        """
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # 原子类型到索引的映射
        self.atom_types = {
            'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 
            'Cl': 7, 'Br': 8, 'I': 9, 'Na': 10, 'K': 11, 'Ca': 12, 
            'Mg': 13, 'Fe': 14, 'Cu': 15, 'Zn': 16
        }
        
        # 键类型到索引的映射
        self.bond_types = {
            'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3
        }
    
    def parse_chemical_formula(self, formula):
        """
        解析化学公式为分子图
        
        Args:
            formula: 化学公式字符串
            
        Returns:
            graph: 分子图对象
        """
        try:
            # 创建空图
            graph = nx.Graph()
            
            # 简化实现：将化学公式中的原子作为节点
            atoms = self._extract_atoms(formula)
            
            # 添加节点
            for i, atom in enumerate(atoms):
                graph.add_node(i, atom_type=atom, 
                             features=self._get_atom_features(atom))
            
            # 添加边（简化实现：连接相邻原子）
            for i in range(len(atoms) - 1):
                graph.add_edge(i, i + 1, 
                             bond_type='SINGLE',
                             features=self._get_bond_features('SINGLE'))
            
            return graph
            
        except Exception as e:
            print(f"分子图解析错误: {e}")
            return None
    
    def _extract_atoms(self, formula):
        """从化学公式中提取原子"""
        atoms = []
        i = 0
        
        while i < len(formula):
            # 检查双字符原子
            if i + 1 < len(formula) and formula[i:i+2] in self.atom_types:
                atoms.append(formula[i:i+2])
                i += 2
            # 检查单字符原子
            elif formula[i] in self.atom_types:
                atoms.append(formula[i])
                i += 1
            # 跳过数字和小写字母
            elif formula[i].isdigit() or formula[i].islower():
                i += 1
            else:
                # 未知字符，跳过
                i += 1
        
        return atoms
    
    def _get_atom_features(self, atom):
        """获取原子特征向量"""
        # 简化特征：原子类型、原子质量、电负性等
        features = np.zeros(self.node_feature_dim)
        
        # 原子类型编码
        atom_idx = self.atom_types.get(atom, -1)
        if atom_idx >= 0:
            features[atom_idx] = 1.0
        
        # 原子质量（简化）
        atomic_masses = {
            'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0, 'F': 19.0,
            'P': 31.0, 'S': 32.0, 'Cl': 35.5, 'Br': 80.0, 'I': 127.0
        }
        mass = atomic_masses.get(atom, 0.0)
        features[10] = mass / 127.0  # 归一化
        
        return features
    
    def _get_bond_features(self, bond_type):
        """获取键特征向量"""
        features = np.zeros(self.edge_feature_dim)
        
        # 键类型编码
        bond_idx = self.bond_types.get(bond_type, -1)
        if bond_idx >= 0:
            features[bond_idx] = 1.0
        
        return features
    
    def graph_to_tensor(self, graph):
        """
        将分子图转换为张量表示
        
        Args:
            graph: 分子图对象
            
        Returns:
            node_features: 节点特征张量
            edge_index: 边索引张量
            edge_features: 边特征张量
        """
        if graph is None:
            return None, None, None
        
        # 节点特征
        node_features = []
        for node_id in sorted(graph.nodes()):
            features = graph.nodes[node_id].get('features', 
                                              np.zeros(self.node_feature_dim))
            node_features.append(features)
        
        # 边索引
        edge_index = []
        edge_features = []
        for edge in graph.edges():
            edge_index.append([edge[0], edge[1]])
            features = graph.edges[edge].get('features', 
                                           np.zeros(self.edge_feature_dim))
            edge_features.append(features)
        
        # 转换为张量
        if node_features:
            node_features = torch.tensor(node_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_features = torch.tensor(edge_features, dtype=torch.float32)
        else:
            node_features = torch.zeros((1, self.node_feature_dim))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = torch.zeros((0, self.edge_feature_dim))
        
        return node_features, edge_index, edge_features


def preprocess_image(image_path, image_size=(128, 32)):
    """
    预处理图像（简化函数）
    
    Args:
        image_path: 图像文件路径
        image_size: 目标图像尺寸
        
    Returns:
        processed_image: 预处理后的图像张量
    """
    preprocessor = ImagePreprocessor(image_size)
    return preprocessor.preprocess(image_path)


def preprocess_graph(formula, max_nodes=50):
    """
    预处理分子图（简化函数）
    
    Args:
        formula: 化学公式字符串
        max_nodes: 最大节点数
        
    Returns:
        node_features: 节点特征张量
        edge_index: 边索引张量
        edge_features: 边特征张量
    """
    preprocessor = GraphPreprocessor(max_nodes)
    graph = preprocessor.parse_chemical_formula(formula)
    return preprocessor.graph_to_tensor(graph)


def test_preprocessing():
    """测试数据预处理功能"""
    # 测试图像预处理
    image_preprocessor = ImagePreprocessor()
    
    # 测试分子图预处理
    graph_preprocessor = GraphPreprocessor()
    
    # 测试化学公式解析
    test_formulas = ["H2O", "CO2", "C6H6", "H2SO4"]
    
    for formula in test_formulas:
        print(f"\n解析化学公式: {formula}")
        graph = graph_preprocessor.parse_chemical_formula(formula)
        
        if graph is not None:
            print(f"节点数量: {graph.number_of_nodes()}")
            print(f"边数量: {graph.number_of_edges()}")
            
            # 转换为张量
            node_features, edge_index, edge_features = graph_preprocessor.graph_to_tensor(graph)
            print(f"节点特征形状: {node_features.shape}")
            print(f"边索引形状: {edge_index.shape}")
            print(f"边特征形状: {edge_features.shape}")


if __name__ == "__main__":
    test_preprocessing()