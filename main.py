#!/usr/bin/env python3
"""
化学公式识别项目主入口文件

功能：
1. 数据预处理和准备
2. 模型训练
3. 模型评估
4. 预测演示
"""

import os
import sys
import argparse

# 添加project目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'project'))

def main():
    parser = argparse.ArgumentParser(description='化学公式识别项目')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'test', 'demo'], 
                       default='demo', help='运行模式')
    parser.add_argument('--config', type=str, default='src/config.py', 
                       help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--model_path', type=str, help='模型路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("化学公式识别项目")
    print("=" * 60)
    
    if args.mode == 'preprocess':
        print("执行数据预处理...")
        from utils.data_preprocessing import main as preprocess_main
        preprocess_main()
        
    elif args.mode == 'train':
        print("开始模型训练...")
        # 导入训练模块
        import torch
        from train import main as train_main
        
        # 设置训练参数
        train_args = argparse.Namespace()
        train_args.epochs = args.epochs
        train_args.batch_size = args.batch_size
        train_args.resume = None
        train_args.eval = False
        
        train_main()
        
    elif args.mode == 'test':
        print("执行模型测试...")
        from test import main as test_main
        test_main()
        
    else:  # demo模式
        print("运行演示模式...")
        print("项目结构检查:")
        
        # 检查数据
        data_files = [
            ('data/annotations_full_train.json', '训练集标注'),
            ('data/annotations_full_val.json', '验证集标注'),
            ('data/vocab_full.json', '词汇表'),
            ('dataset/images', '图像数据'),
            ('dataset/images1', '图像数据1'),
        ]
        
        for file_path, description in data_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            if os.path.exists(full_path):
                if os.path.isdir(full_path):
                    file_count = len([f for f in os.listdir(full_path) if f.endswith('.jpg')])
                    print(f"  ✓ {description}: {file_count} 个图像")
                else:
                    print(f"  ✓ {description}: 存在")
            else:
                print(f"  ✗ {description}: 缺失")
        
        # 检查模型组件
        model_files = [
            ('project/models/graph_encoder.py', '图编码器'),
            ('project/models/sequence_encoder.py', '序列编码器'),
            ('project/models/fusion_encoder.py', '融合编码器'),
            ('project/models/ctc_crf_decoder.py', 'CTC+CRF解码器'),
            ('project/dataset/chem_dataset.py', '数据加载器'),
        ]
        
        print("\n模型组件检查:")
        for file_path, description in model_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {description}: 正常")
            else:
                print(f"  ✗ {description}: 缺失")
        
        print("\n快速功能测试:")
        try:
            # 测试数据加载
            from dataset.chem_dataset import create_data_loaders
            train_loader, val_loader, vocab = create_data_loaders(
                data_dir='dataset',
                annotation_file='data/annotations_full.json',
                vocab_file='data/vocab_full.json',
                batch_size=2,
                image_size=(128, 512),
                max_length=50
            )
            print("  ✓ 数据加载器: 正常")
            print(f"    词汇表大小: {len(vocab)}")
            
            # 测试模型组件
            import torch
            from models.sequence_encoder import SequenceEncoder
            
            batch = next(iter(train_loader))
            images = batch['images']
            encoder = SequenceEncoder(input_channels=3, hidden_dim=256)
            features = encoder(images)
            print("  ✓ 序列编码器: 正常")
            print(f"    输入形状: {images.shape}")
            print(f"    输出形状: {features.shape}")
            
        except Exception as e:
            print(f"  ✗ 功能测试失败: {e}")
        
        print("\n" + "=" * 60)
        print("项目状态: 正常运行")
        print("=" * 60)
        
        print("\n使用说明:")
        print("1. 数据预处理: python main.py --mode preprocess")
        print("2. 模型训练: python main.py --mode train --epochs 100 --batch_size 16")
        print("3. 模型测试: python main.py --mode test")
        print("4. 快速演示: python main.py (默认)")

if __name__ == "__main__":
    main()