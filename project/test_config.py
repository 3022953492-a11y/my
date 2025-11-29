#!/usr/bin/env python3
"""测试Config类的脚本"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("测试Config类导入和使用")
    
    # 导入Config类
    from config import Config
    print("✓ 导入 Config 类成功")
    
    # 测试配置初始化
    config = Config()
    print("✓ 配置初始化成功")
    
    # 打印配置信息
    print("\n配置信息:")
    print(f"数据根目录: {config.DATA_ROOT}")
    print(f"图像目录1: {config.IMAGES_DIR1}")
    print(f"图像目录2: {config.IMAGES_DIR2}")
    print(f"标签文件: {config.LABELS_FILE}")
    print(f"词汇表大小: {config.VOCAB_SIZE}")
    print(f"批大小: {config.BATCH_SIZE}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"训练轮数: {config.NUM_EPOCHS}")
    
    # 测试配置属性访问
    print("\n测试配置属性访问:")
    required_attrs = ['DATA_ROOT', 'IMAGES_DIR1', 'IMAGES_DIR2', 'LABELS_FILE', 
                     'VOCAB_SIZE', 'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS']
    
    for attr in required_attrs:
        if hasattr(config, attr):
            print(f"✓ 配置包含 {attr}")
        else:
            print(f"✗ 配置缺少 {attr}")
    
    print("\nConfig类测试通过!")
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
