#!/usr/bin/env python3
"""测试Trainer和Evaluator类的脚本"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("测试Trainer和Evaluator类")
    
    # 导入必要的模块
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from config import Config
    
    print("✓ 导入基础模块成功")
    
    # 初始化配置
    config = Config()
    
    # 导入模型
    from train import ChemicalFormulaModel
    model = ChemicalFormulaModel(
        vocab_size=config.VOCAB_SIZE,
        num_classes=config.VOCAB_SIZE,
        d_model=256,
        fusion_type='cross_attention'
    )
    print("✓ 导入和创建模型成功")
    
    # 导入Trainer和Evaluator类
    from train import Trainer, Evaluator
    print("✓ 导入Trainer和Evaluator类成功")
    
    # 创建模拟数据加载器
    print("\n创建模拟数据加载器...")
    
    class MockDataLoader:
        def __init__(self, batch_size=2):
            self.batch_size = batch_size
            self.data = []
            for _ in range(2):  # 只有2个批次用于测试
                # 模拟图像
                images = torch.randn(batch_size, 3, 224, 224)
                
                # 模拟标签
                labels = torch.randint(0, config.VOCAB_SIZE, (batch_size, 10))
                label_lengths = torch.tensor([8, 10])
                texts = ["H2O", "CO2"] * batch_size
                
                # 模拟图数据
                graph_data = []
                for __ in range(batch_size):
                    node_features = torch.randn(5, 3)
                    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
                    edge_features = torch.randn(8, 2)
                    graph_data.append({
                        'node_features': node_features,
                        'edge_index': edge_index,
                        'edge_features': edge_features
                    })
                
                self.data.append((images, labels, label_lengths, texts, graph_data))
        
        def __iter__(self):
            return iter(self.data)
        
        def __len__(self):
            return len(self.data)
    
    train_loader = MockDataLoader(batch_size=2)
    val_loader = MockDataLoader(batch_size=2)
    
    print("✓ 创建模拟数据加载器成功")
    
    # 测试Evaluator类
    print("\n测试Evaluator类...")
    evaluator = Evaluator(model, val_loader, device='cpu')
    metrics = evaluator.evaluate()
    print(f"✓ Evaluator测试成功，准确率: {metrics['accuracy']:.4f}")
    
    # 测试Trainer类初始化
    print("\n测试Trainer类初始化...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu'
    )
    print("✓ Trainer初始化成功")
    
    # 测试单步训练
    print("\n测试单步训练...")
    loss = trainer.train_epoch()
    print(f"✓ 单步训练成功，损失: {loss:.4f}")
    
    # 测试保存模型
    print("\n测试保存模型...")
    model_path = "test_model.pth"
    trainer.save_model(model_path)
    print(f"✓ 模型保存成功到 {model_path}")
    
    # 测试加载模型
    print("\n测试加载模型...")
    trainer.load_model(model_path)
    print("✓ 模型加载成功")
    
    # 清理测试文件
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"✓ 清理测试文件 {model_path}")
    
    print("\nTrainer和Evaluator类测试通过!")
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
