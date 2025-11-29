#!/usr/bin/env python3
"""
化学公式识别模型测试脚本
支持模型评估和推理演示
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.graph_encoder import GraphEncoder
from models.sequence_encoder import SequenceEncoder
from models.fusion_encoder import FusionEncoder
from models.ctc_crf_decoder import CTC_CRF_Decoder
from dataset.chem_dataset import ChemicalFormulaDataset


class ChemicalFormulaModel(nn.Module):
    """化学公式识别完整模型"""
    
    def __init__(self, vocab_size, num_classes, d_model=256, fusion_type='cross_attention'):
        super(ChemicalFormulaModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        
        # 图编码流
        self.graph_encoder = GraphEncoder(
            node_dim=3,
            edge_dim=2,
            hidden_dim=d_model,
            output_dim=d_model
        )
        
        # 序列编码流
        self.sequence_encoder = SequenceEncoder(
            input_channels=3,
            hidden_dim=d_model,
            encoder_type='mobilenet'
        )
        
        # 融合编码器
        self.fusion_encoder = FusionEncoder(
            d_model=d_model,
            fusion_type=fusion_type
        )
        
        # CTC+CRF解码器
        self.decoder = CTC_CRF_Decoder(
            input_dim=d_model,
            num_classes=num_classes
        )
        
    def forward(self, images, graph_data, training=False):
        """推理模式前向传播"""
        batch_size = images.size(0)
        
        # 图编码流
        graph_features = []
        for i in range(batch_size):
            graph_feat = self.graph_encoder(
                graph_data[i]['node_features'],
                graph_data[i]['edge_index'],
                graph_data[i]['edge_features']
            )
            graph_features.append(graph_feat)
        
        graph_features = torch.stack(graph_features)
        
        # 序列编码流
        sequence_features = self.sequence_encoder(images)
        
        # 融合编码
        fused_features = self.fusion_encoder(graph_features, sequence_features)
        
        # 解码
        output = self.decoder(fused_features, training=training)
        
        return output


class ModelTester:
    """模型测试器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载词汇表
        with open(config['vocab_file'], 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # 创建模型
        self.model = ChemicalFormulaModel(
            vocab_size=len(self.vocab),
            num_classes=len(self.vocab),
            d_model=config['d_model'],
            fusion_type=config['fusion_type']
        ).to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, checkpoint_path):
        """加载训练好的模型"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功: {checkpoint_path}")
            return True
        else:
            print(f"模型文件不存在: {checkpoint_path}")
            return False
    
    def evaluate_model(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        
        total_samples = 0
        correct_predictions = 0
        total_loss = 0
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                graph_data = batch['graph_data']
                formulas = batch['formulas']
                
                # 输入序列长度
                input_lengths = torch.tensor([images.size(3) // 4] * images.size(0)).to(self.device)
                
                # 模型推理
                output = self.model(images, graph_data, training=False)
                
                # 计算准确率
                pred_sequences = output['predictions']
                
                for i, pred_seq in enumerate(pred_sequences):
                    # 转换为字符序列
                    pred_chars = [self.idx2char.get(idx, '?') for idx in pred_seq if idx != 0]
                    pred_formula = ''.join(pred_chars)
                    
                    # 真实公式
                    true_formula = formulas[i]
                    
                    predictions.append(pred_formula)
                    ground_truths.append(true_formula)
                    
                    # 检查是否匹配
                    if pred_formula == true_formula:
                        correct_predictions += 1
                    
                    total_samples += 1
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    accuracy = correct_predictions / total_samples * 100
                    print(f'批次 {batch_idx + 1}, 准确率: {accuracy:.2f}%')
        
        # 计算最终准确率
        accuracy = correct_predictions / total_samples * 100
        
        print(f'\n评估结果:')
        print(f'总样本数: {total_samples}')
        print(f'正确预测: {correct_predictions}')
        print(f'准确率: {accuracy:.2f}%')
        
        return accuracy, predictions, ground_truths
    
    def predict_single_image(self, image_path, graph_data):
        """对单张图像进行预测"""
        self.model.eval()
        
        # 加载图像
        if not os.path.exists(image_path):
            # 创建示例图像
            image = Image.new('RGB', (256, 64), color='white')
            print(f"警告: 图像文件不存在，使用空白图像: {image_path}")
        else:
            image = Image.open(image_path).convert('RGB')
        
        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 构建图数据
        graph_batch = [graph_data]
        
        with torch.no_grad():
            # 模型推理
            output = self.model(image_tensor, graph_batch, training=False)
            
            # 解码预测结果
            pred_sequence = output['predictions'][0]
            pred_chars = [self.idx2char.get(idx, '?') for idx in pred_sequence if idx != 0]
            predicted_formula = ''.join(pred_chars)
            
            # CTC解码（对比）
            ctc_predictions = self.model.decoder.ctc_decode(
                output['logits'], 
                torch.tensor([image_tensor.size(3) // 4])
            )[0]
            ctc_chars = [self.idx2char.get(idx, '?') for idx in ctc_predictions]
            ctc_formula = ''.join(ctc_chars)
            
            return {
                'predicted_formula': predicted_formula,
                'ctc_formula': ctc_formula,
                'confidence': output['scores'][0].item(),
                'image_path': image_path
            }
    
    def visualize_predictions(self, predictions, ground_truths, save_path='./results/prediction_samples.png'):
        """可视化预测结果"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 选择前10个样本进行可视化
        num_samples = min(10, len(predictions))
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            
            # 显示预测结果
            ax.text(0.1, 0.7, f'真实: {ground_truths[i]}', fontsize=12, transform=ax.transAxes)
            ax.text(0.1, 0.4, f'预测: {predictions[i]}', fontsize=12, transform=ax.transAxes)
            
            # 颜色标记
            if predictions[i] == ground_truths[i]:
                ax.text(0.1, 0.1, '✓ 正确', fontsize=12, color='green', transform=ax.transAxes)
            else:
                ax.text(0.1, 0.1, '✗ 错误', fontsize=12, color='red', transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"预测结果可视化已保存: {save_path}")
    
    def demo_predictions(self):
        """演示预测功能"""
        print("\n=== 化学公式识别演示 ===")
        
        # 示例化学公式
        demo_examples = [
            {
                'name': '水分子',
                'formula': 'H₂O',
                'graph_data': {
                    'nodes': ['H', 'O', 'H'],
                    'edges': [[0, 1], [1, 2]],
                    'bond_types': ['covalent', 'covalent']
                }
            },
            {
                'name': '二氧化碳',
                'formula': 'CO₂',
                'graph_data': {
                    'nodes': ['C', 'O', 'O'],
                    'edges': [[0, 1], [0, 2]],
                    'bond_types': ['double', 'double']
                }
            },
            {
                'name': '硫酸',
                'formula': 'H₂SO₄',
                'graph_data': {
                    'nodes': ['H', 'H', 'S', 'O', 'O', 'O', 'O'],
                    'edges': [[0, 2], [1, 2], [2, 3], [2, 4], [2, 5], [2, 6]],
                    'bond_types': ['covalent', 'covalent', 'double', 'double', 'single', 'single']
                }
            }
        ]
        
        for example in demo_examples:
            # 构建分子图
            from dataset.chem_dataset import ChemicalFormulaDataset
            dataset = ChemicalFormulaDataset('./data', './data/annotations.json', self.config['vocab_file'])
            graph_data = dataset._build_molecular_graph(example['graph_data'])
            
            # 预测
            image_path = f"./data/{example['formula'].replace('₂', '2').replace('₄', '4')}.png"
            result = self.predict_single_image(image_path, graph_data)
            
            print(f"\n{example['name']}:")
            print(f"  真实公式: {example['formula']}")
            print(f"  CRF预测: {result['predicted_formula']}")
            print(f"  CTC预测: {result['ctc_formula']}")
            print(f"  置信度: {result['confidence']:.4f}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'vocab_file': './data/vocab.json',
        'checkpoint_path': './checkpoints/best_model.pth',
        'd_model': 256,
        'fusion_type': 'cross_attention',
        'image_size': (128, 32),
        'max_length': 100
    }
    
    # 创建测试器
    tester = ModelTester(config)
    
    # 加载模型
    if not tester.load_model(config['checkpoint_path']):
        print("无法加载模型，请先训练模型")
        return
    
    # 演示预测
    tester.demo_predictions()
    
    # 如果有测试数据，进行完整评估
    test_annotation_file = './data/annotations_test.json'
    if os.path.exists(test_annotation_file):
        print("\n=== 完整模型评估 ===")
        
        # 创建测试数据集
        test_dataset = ChemicalFormulaDataset(
            data_dir='./data',
            annotation_file=test_annotation_file,
            vocab_file=config['vocab_file'],
            image_size=config['image_size'],
            max_length=config['max_length'],
            is_training=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4
        )
        
        # 评估模型
        accuracy, predictions, ground_truths = tester.evaluate_model(test_loader)
        
        # 可视化预测结果
        tester.visualize_predictions(predictions, ground_truths)
        
        # 保存评估结果
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
        
        with open('./results/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存: ./results/evaluation_results.json")
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()