#!/usr/bin/env python3
"""简化版化学公式识别训练脚本"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置
from config import Config
config = Config()

# 简化的模型类
class ChemicalFormulaModel(nn.Module):
    """简化版化学公式模型"""
    
    def __init__(self, vocab_size, num_classes, d_model=256, fusion_type='cross_attention'):
        super(ChemicalFormulaModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # 简化的CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),  # 假设输出大小为28x28
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, graph_data=None, targets=None, input_lengths=None, 
                target_lengths=None, training=True):
        """简化的前向传播"""
        # CNN特征提取
        features = self.cnn(images)
        
        # 展平特征
        features = features.view(features.size(0), -1)
        
        # 分类
        logits = self.classifier(features)
        
        # 模拟输出序列（假设长度为20）
        output = logits.unsqueeze(1).repeat(1, 20, 1)
        
        # 模拟损失
        loss = torch.tensor(1.0, requires_grad=True) if targets is not None else None
        
        # 模拟预测（直接返回logits）
        predictions = torch.argmax(output, dim=-1)
        
        return output, loss, predictions

# 简化的Evaluator类
class Evaluator:
    """简化版评估器"""
    
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
    
    def evaluate(self, test_loader):
        """简化的评估方法"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                images, labels, label_lengths, texts = batch[:4]  # 只取前4个元素
                
                # 移动到设备
                images = images.to(self.device)
                labels = labels.to(self.device)
                label_lengths = label_lengths.to(self.device)
                
                # 前向传播
                output, loss, _ = self.model(images)
                
                if loss is not None:
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        print(f"评估损失: {avg_loss:.4f}")
        return avg_loss

# 简化的Trainer类
class Trainer:
    """简化版训练器"""
    
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc='Training'):
            images, labels, label_lengths, texts = batch[:4]  # 只取前4个元素
            
            # 移动到设备
            images = images.to(self.device)
            labels = labels.to(self.device)
            label_lengths = label_lengths.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output, loss, _ = self.model(images, targets=labels, target_lengths=label_lengths)
            
            # 反向传播
            if loss is not None:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.scheduler.step()
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=1):
        """完整的训练过程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            evaluator = Evaluator(self.model, self.config, self.device)
            val_loss = evaluator.evaluate(val_loader)
            print(f"验证损失: {val_loss:.4f}")
    
    def save_model(self, filename):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"模型保存到 {filename}")
    
    def load_model(self, filename):
        """加载模型"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"从 {filename} 加载模型")
        else:
            print(f"模型文件不存在: {filename}")

# 主函数
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='化学方程式识别训练脚本')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--eval', action='store_true', help='仅进行评估')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='评估用的模型路径')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.lr:
        config.LEARNING_RATE = args.lr
    config.BATCH_SIZE = args.batch_size
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 打印数据路径信息
    print(f"数据根目录: {config.DATA_ROOT}")
    print(f"图像目录1: {config.IMAGES_DIR1}")
    print(f"图像目录2: {config.IMAGES_DIR2}")
    
    # 创建模型
    print("\n创建模型...")
    model = ChemicalFormulaModel(
        vocab_size=config.VOCAB_SIZE,
        num_classes=config.VOCAB_SIZE,
        d_model=256,
        fusion_type='cross_attention'
    )
    model = model.to(device)
    
    # 创建模拟数据加载器
    print("\n创建模拟数据加载器...")
    
    class MockDataLoader:
        def __init__(self, batch_size=2):
            self.batch_size = batch_size
            self.num_batches = 2  # 只训练2个批次
        
        def __iter__(self):
            for _ in range(self.num_batches):
                # 模拟图像
                images = torch.randn(self.batch_size, 3, 224, 224)
                
                # 模拟标签（前2个元素用于CTC）
                labels = torch.randint(0, config.VOCAB_SIZE, (self.batch_size, 10))
                label_lengths = torch.tensor([8, 10])  # 每个批次的标签长度
                
                # 模拟文本
                texts = ["H2O", "CO2"] * self.batch_size
                
                yield images, labels, label_lengths, texts
        
        def __len__(self):
            return self.num_batches
    
    train_loader = MockDataLoader(args.batch_size)
    val_loader = MockDataLoader(args.batch_size)
    
    if args.eval:
        # 仅评估
        print("\n开始评估...")
        evaluator = Evaluator(model, config, device)
        loss = evaluator.evaluate(val_loader)
        print(f"评估完成，损失: {loss:.4f}")
    else:
        # 训练模式
        print("\n开始训练...")
        trainer = Trainer(model, config, device)
        
        # 恢复训练
        if args.resume:
            trainer.load_model(args.resume)
        
        # 训练
        trainer.train(train_loader, val_loader, args.epochs)
        
        # 保存模型
        trainer.save_model('simple_model.pth')
        print("\n训练完成！")

if __name__ == "__main__":
    main()
