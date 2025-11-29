#!/usr/bin/env python3
"""简单的环境测试脚本"""

import os
print("Python环境测试")
print(f"当前工作目录: {os.getcwd()}")
print(f"项目根目录: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
print(f"数据根目录: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))}")

# 检查文件是否存在
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
images_path = os.path.join(dataset_path, 'images')
images1_path = os.path.join(dataset_path, 'images1')

print(f"\n检查数据目录是否存在:")
print(f"数据集根目录存在: {os.path.exists(dataset_path)}")
print(f"images目录存在: {os.path.exists(images_path)}")
print(f"images1目录存在: {os.path.exists(images1_path)}")

# 列出一些图片文件
if os.path.exists(images_path):
    print(f"\nimages目录中的部分文件:")
    images_files = os.listdir(images_path)
    for i, file in enumerate(images_files[:5]):
        print(f"  {file}")

if os.path.exists(images1_path):
    print(f"\nimages1目录中的部分文件:")
    images1_files = os.listdir(images1_path)
    for i, file in enumerate(images1_files[:5]):
        print(f"  {file}")

# 尝试导入必要的模块
try:
    import torch
    print(f"\nPyTorch版本: {torch.__version__}")
except ImportError:
    print("\n错误: 无法导入PyTorch")

try:
    from config import Config
    config = Config()
    print(f"\n配置文件测试:")
    print(f"数据根目录配置: {config.DATA_ROOT}")
    print(f"图像目录1配置: {config.IMAGES_DIR1}")
    print(f"图像目录2配置: {config.IMAGES_DIR2}")
except ImportError as e:
    print(f"\n错误: 无法导入Config类 - {e}")
except Exception as e:
    print(f"\n错误: 配置文件测试失败 - {e}")
