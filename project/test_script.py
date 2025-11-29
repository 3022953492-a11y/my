import torch
import sys
import os

print("=== 环境测试 ===")
print(f"Python 版本: {sys.version}")
print(f"Torch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\n=== 数据目录检查 ===")
data_dir = "E:\\化学公式\\dataset"
print(f"数据目录存在: {os.path.exists(data_dir)}")
if os.path.exists(data_dir):
    images_dir = os.path.join(data_dir, "images")
    images1_dir = os.path.join(data_dir, "images1")
    print(f"images目录存在: {os.path.exists(images_dir)}")
    print(f"images1目录存在: {os.path.exists(images1_dir)}")
    
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        print(f"images目录图片数量: {len(image_files)}")
    
    if os.path.exists(images1_dir):
        image1_files = [f for f in os.listdir(images1_dir) if f.endswith(('.jpg', '.png'))]
        print(f"images1目录图片数量: {len(image1_files)}")

print("=== 测试结束 ===")