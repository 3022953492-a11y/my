import sys
import os
import torch

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置编码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=== 简化数据加载器测试 ===")

try:
    # 导入数据加载器
    from dataset.chem_dataset import create_data_loaders
    print("[OK] 成功导入数据加载器模块")
except ImportError as e:
    print(f"[ERROR] 导入数据加载器失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建数据加载器（禁用多进程以避免Windows问题）
try:
    print("创建训练数据加载器...")
    train_loader, val_loader, vocab = create_data_loaders(
        data_dir="E:\\化学公式\\dataset",
        annotation_file="E:\\化学公式\\data\\annotations_full.json",
        vocab_file="E:\\化学公式\\data\\vocab_full.json",
        batch_size=2,  # 小批次
        num_workers=0,  # 禁用多进程
        image_size=(224, 224),
        max_length=50
    )
    print(f"[OK] 数据加载器创建成功")
    print(f"训练数据批次数量: {len(train_loader)}")
    print(f"验证数据批次数量: {len(val_loader)}")
    print(f"词汇表大小: {len(vocab)}")
    
    # 测试第一个批次
    print("\n测试第一个训练批次...")
    batch = next(iter(train_loader))
    
    print(f"批次数据结构:")
    print(f"  - images 形状: {batch['images'].shape}")
    print(f"  - targets 形状: {batch['targets'].shape}")
    print(f"  - target_lengths: {batch['target_lengths']}")
    print(f"  - formulas: {batch['formulas'][:2]}...")  # 显示前两个公式
    print(f"  - graph_data 类型: {type(batch['graph_data'])}")
    print(f"  - graph_data 长度: {len(batch['graph_data'])}")
    
    # 检查图数据的第一个样本
    if len(batch['graph_data']) > 0:
        first_graph = batch['graph_data'][0]
        print(f"\n第一个图数据结构:")
        print(f"  - 类型: {type(first_graph)}")
        if isinstance(first_graph, dict):
            for key, value in first_graph.items():
                if hasattr(value, 'shape'):
                    print(f"  - {key}: {value.shape}")
                else:
                    print(f"  - {key}: {type(value)}")
    
    print("\n[OK] 数据加载器测试通过!")
    
except Exception as e:
    print(f"[ERROR] 数据加载器测试失败: {e}")
    import traceback
    traceback.print_exc()

print("=== 测试完成 ===")