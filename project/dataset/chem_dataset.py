#!/usr/bin/env python3
"""
化学公式数据集模块
支持图像和分子图数据的加载与预处理
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ChemicalFormulaDataset(Dataset):
    """化学公式识别数据集"""
    
    def __init__(self, data_dir, annotation_file, vocab_file, 
                 image_size=(128, 32), max_length=100, is_training=True):
        """
        Args:
            data_dir: 数据目录
            annotation_file: 标注文件路径
            vocab_file: 词汇表文件路径
            image_size: 图像尺寸 (height, width)
            max_length: 最大序列长度
            is_training: 是否为训练模式
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_length = max_length
        self.is_training = is_training
        
        # 加载词汇表
        self.vocab = self._load_vocab(vocab_file)
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # 加载标注数据
        self.annotations = self._load_annotations(annotation_file)
        
        # 图像预处理
        self.transform = self._get_transform()
        
        # 化学元素映射
        self.element_mapping = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
            'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
            'Fm': 100
        }
        
    def _load_vocab(self, vocab_file):
        """加载词汇表"""
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                # 检查词汇表格式，可能是列表或包含vocab字段的对象
                if isinstance(vocab_data, list):
                    vocab = vocab_data
                elif 'vocab' in vocab_data:
                    vocab = vocab_data['vocab']
                else:
                    # 如果格式不匹配，使用默认词汇表
                    vocab = self._get_default_vocab()
        else:
            vocab = self._get_default_vocab()
            
            # 保存词汇表
            os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        return vocab
    
    def _get_default_vocab(self):
        """获取默认词汇表"""
        return [
            '<blank>',  # CTC空白标签
            '<sos>',    # 序列开始
            '<eos>',    # 序列结束
            ' ',        # 空格
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 数字
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # 大写字母
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',  # 小写字母
            '+', '-', '=', '→', '↑', '↓', '(', ')', '[', ']', '{', '}',  # 符号
            '.', ',', ':', ';', '!', '?', '/', '\\', '|',  # 标点
            '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',  # 上标数字
            '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',  # 下标数字
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν',
            'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω'  # 希腊字母
        ]
    
    def _load_annotations(self, annotation_file):
        """加载标注数据"""
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        else:
            # 生成示例数据（实际使用时需要真实数据）
            annotations = [
                {
                    "image_path": "H2O.png",
                    "formula": "H₂O",
                    "graph_data": {
                        "nodes": ["H", "O", "H"],
                        "edges": [[0, 1], [1, 2]],
                        "bond_types": ["covalent", "covalent"]
                    }
                },
                {
                    "image_path": "CO2.png", 
                    "formula": "CO₂",
                    "graph_data": {
                        "nodes": ["C", "O", "O"],
                        "edges": [[0, 1], [0, 2]],
                        "bond_types": ["double", "double"]
                    }
                },
                {
                    "image_path": "H2SO4.png",
                    "formula": "H₂SO₄",
                    "graph_data": {
                        "nodes": ["H", "H", "S", "O", "O", "O", "O"],
                        "edges": [[0, 2], [1, 2], [2, 3], [2, 4], [2, 5], [2, 6]],
                        "bond_types": ["covalent", "covalent", "double", "double", "single", "single"]
                    }
                }
            ]
            
            # 保存示例标注
            os.makedirs(os.path.dirname(annotation_file), exist_ok=True)
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        return annotations
    
    def _get_transform(self):
        """获取图像预处理变换"""
        if self.is_training:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def _parse_chemical_formula(self, formula):
        """解析化学公式为字符序列"""
        # 简单的化学公式解析（实际应用需要更复杂的解析器）
        chars = []
        i = 0
        
        while i < len(formula):
            # 检查特殊字符（上标、下标等）
            if formula[i] in ['₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']:
                # 下标数字
                chars.append(formula[i])
                i += 1
            elif formula[i] in ['²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']:
                # 上标数字
                chars.append(formula[i])
                i += 1
            elif i + 1 < len(formula) and formula[i:i+2] in ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 
                                                           'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc',
                                                           'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                                                           'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                                                           'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb',
                                                           'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
                                                           'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe',
                                                           'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                                                           'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                                                           'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
                                                           'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                                                           'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
                                                           'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                                                           'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
                                                           'Bk', 'Cf', 'Es', 'Fm']:
                # 双字母元素符号
                chars.append(formula[i:i+2])
                i += 2
            else:
                # 单字符
                chars.append(formula[i])
                i += 1
        
        return chars
    
    def _build_molecular_graph(self, graph_data):
        """构建分子图数据"""
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # 处理bond_types字段，如果为空或不存在，使用默认值
        if 'bond_types' in graph_data and graph_data['bond_types']:
            bond_types = graph_data['bond_types']
        else:
            # 如果bond_types为空或不存在，为每条边使用默认的'covalent'类型
            bond_types = ['covalent'] * len(edges) if edges else []
        
        # 节点特征（元素类型）
        node_features = []
        for node in nodes:
            # 元素编码
            element_code = self.element_mapping.get(node, 0)  # 0表示未知元素
            # 创建64维节点特征：元素编码 + 零填充
            feature = [element_code] + [0.0] * 63  # 总共64维
            node_features.append(feature)
        
        # 边特征（键类型）
        edge_features = []
        bond_mapping = {'single': 1, 'double': 2, 'triple': 3, 'covalent': 1, 'ionic': 4}
        
        for bond_type in bond_types:
            bond_strength = bond_mapping.get(bond_type, 1)
            edge_features.append([bond_strength, 0.0])  # 占位符
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float),
            'edge_index': torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.tensor([], dtype=torch.long),
            'edge_features': torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.tensor([], dtype=torch.float)
        }
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 加载图像 - 处理完整的相对路径
        image_path = annotation['image_path']
        
        # 如果路径已经是绝对路径，直接使用；否则相对于data_dir构建路径
        if os.path.isabs(image_path):
            full_image_path = image_path
        else:
            # 处理包含dataset/前缀的相对路径
            if image_path.startswith('dataset/'):
                # 去掉dataset前缀，直接使用data_dir
                image_path = image_path.replace('dataset/', '')
            full_image_path = os.path.join(self.data_dir, image_path)
        
        # 如果图像文件不存在，创建示例图像
        if not os.path.exists(full_image_path):
            # 创建空白图像作为示例
            image = Image.new('RGB', (256, 64), color='white')
        else:
            image = Image.open(full_image_path).convert('RGB')
        
        # 图像预处理
        image_tensor = self.transform(image)
        
        # 解析化学公式
        formula = annotation['formula']
        chars = self._parse_chemical_formula(formula)
        
        # 转换为索引序列
        target_indices = []
        for char in chars:
            if char in self.char2idx:
                target_indices.append(self.char2idx[char])
            else:
                # 未知字符映射到空白标记
                target_indices.append(self.char2idx['<blank>'])
        
        # 填充序列
        if len(target_indices) < self.max_length:
            target_indices.extend([self.char2idx['<blank>']] * (self.max_length - len(target_indices)))
        else:
            target_indices = target_indices[:self.max_length]
        
        # 构建分子图
        graph_data = self._build_molecular_graph(annotation['graph_data'])
        
        return {
            'image': image_tensor,
            'target': torch.tensor(target_indices, dtype=torch.long),
            'target_length': torch.tensor(len(chars), dtype=torch.long),
            'formula': formula,
            'graph_data': graph_data,
            'image_path': annotation['image_path']
        }


def collate_fn(batch):
    """自定义批处理函数"""
    images = torch.stack([item['image'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    target_lengths = torch.stack([item['target_length'] for item in batch])
    
    # 图数据需要特殊处理
    graph_data = [item['graph_data'] for item in batch]
    formulas = [item['formula'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'targets': targets,
        'target_lengths': target_lengths,
        'graph_data': graph_data,
        'formulas': formulas,
        'image_paths': image_paths
    }


def create_data_loaders(data_dir, annotation_file, vocab_file, 
                       batch_size=32, num_workers=4, **kwargs):
    """创建训练和验证数据加载器"""
    
    # 根据文件名模式选择正确的训练和验证文件
    if 'test' in annotation_file:
        # 测试数据使用同一个文件
        train_annotation_file = annotation_file
        val_annotation_file = annotation_file
    elif 'train' in annotation_file:
        # 如果已经是训练文件，查找对应的验证文件
        train_annotation_file = annotation_file
        val_annotation_file = annotation_file.replace('_train.json', '_val.json')
    else:
        # 默认使用_train和_val后缀
        train_annotation_file = annotation_file.replace('.json', '_train.json')
        val_annotation_file = annotation_file.replace('.json', '_val.json')
    
    # 训练数据集
    train_dataset = ChemicalFormulaDataset(
        data_dir=data_dir,
        annotation_file=train_annotation_file,
        vocab_file=vocab_file,
        is_training=True,
        **kwargs
    )
    
    # 验证数据集
    val_dataset = ChemicalFormulaDataset(
        data_dir=data_dir,
        annotation_file=val_annotation_file,
        vocab_file=vocab_file,
        is_training=False,
        **kwargs
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.vocab


def test_dataset():
    """测试数据集"""
    # 创建测试数据集
    data_dir = "./data"
    annotation_file = "./data/annotations.json"
    vocab_file = "./data/vocab.json"
    
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    dataset = ChemicalFormulaDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        vocab_file=vocab_file,
        image_size=(128, 32),
        max_length=50
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"词汇表大小: {len(dataset.vocab)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"图像张量形状: {sample['image'].shape}")
    print(f"目标序列长度: {sample['target_length'].item()}")
    print(f"化学公式: {sample['formula']}")
    print(f"图数据节点数: {sample['graph_data']['node_features'].shape[0]}")
    print(f"图数据边数: {sample['graph_data']['edge_index'].shape[1]}")
    
    # 测试数据加载器
    train_loader, val_loader, vocab = create_data_loaders(
        data_dir, annotation_file, vocab_file, batch_size=2
    )
    
    batch = next(iter(train_loader))
    print(f"批处理图像形状: {batch['images'].shape}")
    print(f"批处理目标形状: {batch['targets'].shape}")
    print(f"批处理图数据数量: {len(batch['graph_data'])}")
    
    print("数据集测试通过!")


if __name__ == "__main__":
    test_dataset()