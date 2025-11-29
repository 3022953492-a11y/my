#!/usr/bin/env python3
"""
化学公式识别模型训练脚本
支持双流联合编码器和CTC+CRF混合解码器的训练
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse

# 导入自定义模块
import sys
# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.graph_encoder import GraphEncoder
from models.sequence_encoder import SequenceEncoder
from models.fusion_encoder import FusionEncoder
from models.ctc_crf_decoder import CTC_CRF_Decoder
from dataset.chem_dataset import create_data_loaders
from config import Config


class ChemicalFormulaModel(nn.Module):
    """化学公式识别完整模型"""
    
    def __init__(self, vocab_size, num_classes, d_model=256, fusion_type='cross_attention'):
        super(ChemicalFormulaModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        
        # 图编码流
        self.graph_encoder = GraphEncoder(
            node_dim=64,  # 节点特征维度（从64维节点特征）
            hidden_dim=d_model,
            output_dim=d_model,
            num_layers=3
        )
        
        # 序列编码流
        self.sequence_encoder = SequenceEncoder(
            input_channels=3,
            hidden_size=d_model,  # 修正参数名
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
        
    def forward(self, images, graph_data, targets=None, input_lengths=None, 
                target_lengths=None, training=True, device=None):
        """
        Args:
            images: 输入图像 [batch_size, 3, H, W]
            graph_data: 图数据列表
            targets: 目标序列（训练时使用）
            input_lengths: 输入序列长度
            target_lengths: 目标序列长度
            training: 是否为训练模式
            device: 设备信息
        """
        batch_size = images.size(0)
        
        # 如果没有提供设备信息，使用images的设备
        if device is None:
            device = images.device
        
        # 图编码流
        graph_features = []
        for i in range(batch_size):
            # 将字典格式的图数据转换为torch_geometric.data.Data对象，并移动到设备
            from torch_geometric.data import Data
            data_obj = Data(
                x=graph_data[i]['node_features'].to(device),
                edge_index=graph_data[i]['edge_index'].to(device),
                batch=torch.zeros(graph_data[i]['node_features'].size(0), dtype=torch.long).to(device)
            )
            graph_feat = self.graph_encoder(data_obj)
            graph_features.append(graph_feat.squeeze(0))  # 移除批次维度
        
        graph_features = torch.stack(graph_features)  # [batch_size, d_model]
        
        # 序列编码流
        sequence_features, _ = self.sequence_encoder(images)  # [batch_size, seq_len, d_model]
        
        # 融合编码
        fused_features = self.fusion_encoder(graph_features, sequence_features)
        
        # 解码
        if training:
            return self.decoder(fused_features, targets, input_lengths, target_lengths, crf_tags=None, training=True)
        else:
            return self.decoder(fused_features, training=False)


class Trainer:
    """训练器类"""
    
    def __init__(self, config, device=None):
        self.config = config
        # 强制使用GPU（如果可用）
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 如果GPU可用，设置默认张量类型为CUDA
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
        # 初始化训练损失记录
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 创建模型
        print("创建模型...")
        print(f"词汇表大小: {config.VOCAB_SIZE}")
        self.model = ChemicalFormulaModel(
            vocab_size=config.VOCAB_SIZE,
            num_classes=config.VOCAB_SIZE,  # 使用与词汇表相同的大小
            d_model=256,  # 默认值
            fusion_type='cross_attention'  # 默认值
        ).to(self.device)
        
        # 优化器
        print("配置优化器...")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01  # 默认值
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,  # 默认值
            gamma=0.1  # 默认值
        )
        
        # TensorBoard (暂时禁用)
        # log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        # os.makedirs(log_dir, exist_ok=True)
        # self.writer = SummaryWriter(log_dir)
        self.writer = None  # 暂时禁用TensorBoard
        
        # 创建检查点目录
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        print(f"开始训练epoch，数据加载器长度: {len(train_loader)}")
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            print(f"处理批次 {batch_idx}")
            
            try:
                # 解析批次数据 - 根据数据加载器的实际结构
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                graph_data = batch['graph_data']  # 这是一个列表，包含每个样本的图数据
                
                # 将图数据移动到GPU
                if torch.cuda.is_available():
                    for i in range(len(graph_data)):
                        if 'node_features' in graph_data[i]:
                            graph_data[i]['node_features'] = graph_data[i]['node_features'].to(self.device)
                        if 'edge_index' in graph_data[i]:
                            graph_data[i]['edge_index'] = graph_data[i]['edge_index'].to(self.device)
                        if 'edge_features' in graph_data[i]:
                            graph_data[i]['edge_features'] = graph_data[i]['edge_features'].to(self.device)
                
                print(f"批次数据形状 - images: {images.shape}, targets: {targets.shape}")
                print(f"图数据数量: {len(graph_data)}")
                
                # 计算输入长度（序列编码器的输出序列长度）
                batch_size = images.size(0)
                # 通过模型获取实际的序列长度
                with torch.no_grad():
                    sequence_features, _ = self.model.sequence_encoder(images)
                    actual_seq_len = sequence_features.size(1)
                input_lengths = torch.full((batch_size,), actual_seq_len, dtype=torch.long).to(self.device)
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                print("开始前向传播...")
                output = self.model(
                    images, 
                    graph_data=graph_data, 
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                    training=True,
                    device=self.device
                )
                print(f"前向传播完成，输出类型: {type(output)}, 形状: {getattr(output, 'shape', 'N/A')}")
                
                # 获取损失（CTC+CRF解码器返回的是字典）
                if isinstance(output, dict):
                    loss = output['loss']
                    print(f"损失值: {loss}")
                else:
                    loss = output
                    print(f"损失值: {loss}")
                
                # 反向传播
                print("开始反向传播...")
                loss.backward()
                print("反向传播完成")
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                print(f"批次 {batch_idx} 完成")
                
            except Exception as e:
                print(f"批次 {batch_idx} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        avg_loss = total_loss / len(train_loader)
        print(f"epoch完成，平均损失: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                # 解析批次数据
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                graph_data = batch['graph_data']
                
                # 将图数据移动到GPU
                if torch.cuda.is_available():
                    for i in range(len(graph_data)):
                        if 'node_features' in graph_data[i]:
                            graph_data[i]['node_features'] = graph_data[i]['node_features'].to(self.device)
                        if 'edge_index' in graph_data[i]:
                            graph_data[i]['edge_index'] = graph_data[i]['edge_index'].to(self.device)
                        if 'edge_features' in graph_data[i]:
                            graph_data[i]['edge_features'] = graph_data[i]['edge_features'].to(self.device)
                
                # 计算输入长度（序列编码器的输出序列长度）
                batch_size = images.size(0)
                # 通过模型获取实际的序列长度
                with torch.no_grad():
                    sequence_features, _ = self.model.sequence_encoder(images)
                    actual_seq_len = sequence_features.size(1)
                input_lengths = torch.full((batch_size,), actual_seq_len, dtype=torch.long).to(self.device)
                
                # 前向传播
                output = self.model(
                    images, 
                    graph_data=graph_data,
                    targets=targets, 
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                    training=True,
                    device=self.device
                )
                
                # 获取损失（CTC+CRF解码器返回的是字典）
                if isinstance(output, dict):
                    loss = output['loss']
                else:
                    loss = output
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """完整的训练过程"""
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model('best_model.pth')
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"早停计数器: {self.patience_counter}/{self.config.PATIENCE}")
            
            # 早停检查
            if self.patience_counter >= self.config.PATIENCE:
                print("早停触发，停止训练")
                break
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # 训练完成
        print(f"训练完成，最佳验证损失: {self.best_val_loss:.4f}")
    
    def save_model(self, filename):
        """保存模型"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch': len(self.train_losses)
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"模型保存到: {checkpoint_path}")
        
    def load_model(self, filename):
        """加载模型"""
        # 检查文件是否存在，如果是相对路径则在checkpoint目录中查找
        if not os.path.exists(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"加载模型成功，最佳验证损失: {self.best_val_loss:.4f}")


class Evaluator:
    """评估器类"""
    
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.config = config
        self.device = device
        
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        
        total_samples = 0
        correct_predictions = 0
        
        all_predictions = []
        all_targets = []
        
        print("开始评估...")
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
            for batch_idx, batch in enumerate(pbar):
                # 解析批次数据
                images = batch['images'].to(self.device)
                graph_data = batch['graph_data']
                formulas = batch['formulas']
                
                # 将图数据移动到GPU
                if torch.cuda.is_available():
                    for i in range(len(graph_data)):
                        if 'node_features' in graph_data[i]:
                            graph_data[i]['node_features'] = graph_data[i]['node_features'].to(self.device)
                        if 'edge_index' in graph_data[i]:
                            graph_data[i]['edge_index'] = graph_data[i]['edge_index'].to(self.device)
                        if 'edge_features' in graph_data[i]:
                            graph_data[i]['edge_features'] = graph_data[i]['edge_features'].to(self.device)
                
                # 计算输入长度
                batch_size = images.size(0)
                input_lengths = torch.full((batch_size,), 25, dtype=torch.long).to(self.device)
                
                # 前向传播（推理模式）
                output = self.model(images, graph_data=graph_data, training=False, device=self.device)
                
                # 获取预测结果
                predictions = self._decode_output(output)
                
                # 统计准确率
                for i, (pred, target_text) in enumerate(zip(predictions, formulas)):
                    if pred == target_text:
                        correct_predictions += 1
                    total_samples += 1
                    
                    all_predictions.append(pred)
                    all_targets.append(target_text)
                
                # 更新进度条
                accuracy = correct_predictions / total_samples * 100
                pbar.set_postfix({
                    'Accuracy': f'{accuracy:.2f}%',
                    'Correct': f'{correct_predictions}/{total_samples}'
                })
        
        accuracy = correct_predictions / total_samples * 100
        
        print(f"\n评估结果:")
        print(f"总样本数: {total_samples}")
        print(f"正确预测: {correct_predictions}")
        print(f"准确率: {accuracy:.2f}%")
        
        # 显示一些预测示例
        print("\n预测示例:")
        for i in range(min(5, len(all_predictions))):
            print(f"目标: {all_targets[i]}")
            print(f"预测: {all_predictions[i]}")
            print("---")
        
        return accuracy, all_predictions, all_targets
    
    def _decode_output(self, output):
        """解码模型输出"""
        predictions = []
        
        # 简单的贪婪解码
        for i in range(output.size(0)):
            # 获取序列中每个位置的最大概率字符
            seq_pred = []
            for j in range(output.size(1)):
                char_idx = torch.argmax(output[i, j]).item()
                # 转换为字符并跳过特殊标记
                if hasattr(self.config, 'idx2char') and char_idx < len(self.config.idx2char):
                    char = self.config.idx2char[char_idx]
                    if char not in [self.config.SOS_TOKEN, self.config.EOS_TOKEN, self.config.PAD_TOKEN]:
                        seq_pred.append(char)
            # 合并字符成字符串
            predictions.append(''.join(seq_pred))
        
        return predictions


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='化学方程式识别训练脚本')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--eval', action='store_true', help='仅进行评估')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='评估用的模型路径')
    
    args = parser.parse_args()
    
    # 配置
    config = Config()
    
    # 更新配置参数
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 打印数据路径信息
    print(f"数据根目录: {config.DATA_ROOT}")
    print(f"图像目录1: {config.IMAGES_DIR1}")
    print(f"图像目录2: {config.IMAGES_DIR2}")
    
    # 数据加载
    print("加载数据...")
    # 使用配置文件中的路径创建数据加载器（Windows下禁用多进程）
    train_loader, val_loader, vocab = create_data_loaders(
        data_dir=config.DATA_ROOT,
        annotation_file=config.ANNOTATION_FILE,
        vocab_file=config.VOCAB_FILE,
        batch_size=config.BATCH_SIZE,
        num_workers=0,  # Windows下禁用多进程
        image_size=config.IMG_SIZE,
        max_length=50
    )
    
    # 为了简化，我们使用val_loader作为test_loader
    test_loader = val_loader
    
    # 创建模型
    print("创建模型...")
    model = ChemicalFormulaModel(
        vocab_size=config.VOCAB_SIZE,
        num_classes=config.VOCAB_SIZE,
        d_model=256  # 添加缺失的d_model参数
    ).to(device)
    
    if args.eval:
        # 仅进行评估
        print("开始评估...")
        
        # 创建评估器
        evaluator = Evaluator(model, config, device)
        
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型: {args.model_path}")
        else:
            print(f"模型文件不存在: {args.model_path}")
            return
        
        # 评估
        accuracy, predictions, targets = evaluator.evaluate(test_loader)
        
    else:
        # 训练模式
        print("开始训练...")
        print(f"训练轮数: {args.epochs if args.epochs else config.NUM_EPOCHS}")
        print(f"批大小: {config.BATCH_SIZE}")
        print(f"学习率: {config.LEARNING_RATE}")
        
        # 创建训练器
        trainer = Trainer(config, device)
        
        # 恢复训练（如果指定）
        if args.resume and os.path.exists(args.resume):
            trainer.load_model(args.resume)
            print(f"恢复训练: {args.resume}")
        
        # 训练
        trainer.train(train_loader, val_loader, args.epochs)
        
        # 训练完成后进行评估
        print("\n训练完成，开始评估...")
        evaluator = Evaluator(trainer.model, config, device)
        accuracy, predictions, targets = evaluator.evaluate(test_loader)


if __name__ == "__main__":
    main()