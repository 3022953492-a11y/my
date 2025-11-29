#!/usr/bin/env python3
"""最基本的测试脚本"""

print("Python基本功能测试")
print(f"Python版本: {__import__('sys').version}")

# 测试基本导入
try:
    import os
    print("✓ 导入 os 模块成功")
    
    import json
    print("✓ 导入 json 模块成功")
    
    import numpy as np
    print("✓ 导入 numpy 模块成功")
    
    import torch
    print(f"✓ 导入 PyTorch 成功，版本: {torch.__version__}")
    
    # 测试路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前文件目录: {current_dir}")
    print(f"项目根目录: {os.path.abspath(os.path.join(current_dir, '..'))}")
    
    # 测试数据根目录
    data_root = os.path.abspath(os.path.join(current_dir, '..', 'dataset'))
    print(f"数据集根目录: {data_root}")
    print(f"数据集目录存在: {os.path.exists(data_root)}")
    
    # 测试配置文件存在
    config_path = os.path.join(current_dir, 'config.py')
    print(f"配置文件路径: {config_path}")
    print(f"配置文件存在: {os.path.exists(config_path)}")
    
    print("\n基本测试通过!")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    print("详细错误信息:")
    traceback.print_exc()
