# 项目配置说明

## 配置文件结构

### 主配置文件 (src/config.py)

```python
class Config:
    # ========== 数据配置 ==========
    DATA_DIR = "dataset"
    IMAGE_SIZE = (128, 512)  # 图像尺寸 (高度, 宽度)
    MAX_LENGTH = 50          # 最大序列长度
    VOCAB_SIZE = 44          # 词汇表大小
    
    # ========== 模型配置 ==========
    EMBED_DIM = 256          # 嵌入维度
    HIDDEN_DIM = 512         # 隐藏层维度
    NUM_HEADS = 8            # 注意力头数
    NUM_LAYERS = 3           # 编码器层数
    DROPOUT = 0.1            # Dropout率
    
    # ========== 训练配置 ==========
    BATCH_SIZE = 16          # 批次大小
    LEARNING_RATE = 0.001    # 学习率
    WEIGHT_DECAY = 0.0001    # 权重衰减
    NUM_EPOCHS = 100         # 训练轮数
    WARMUP_STEPS = 1000      # 预热步数
    
    # ========== 路径配置 ==========
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    MODEL_SAVE_PATH = "best_model.pth"
```

## 环境变量配置

### 创建环境配置文件 (.env)

```bash
# 项目根目录创建 .env 文件
DATA_PATH=./dataset
MODEL_PATH=./checkpoints
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0
```

### 使用环境变量

```python
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "dataset")
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

## 命令行参数配置

### 主程序参数 (main.py)

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="化学公式识别系统")
    
    # 运行模式
    parser.add_argument("--mode", type=str, default="demo",
                       choices=["preprocess", "train", "test", "demo"],
                       help="运行模式: preprocess/train/test/demo")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="dataset",
                       help="数据目录路径")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="学习率")
    
    # 模型参数
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="模型检查点路径")
    
    return parser.parse_args()
```

## 数据集配置

### 数据目录结构

```
dataset/
├── images/           # 图像文件
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── labels.txt        # 原始标签文件
└── metadata.json     # 元数据文件
```

### 标注文件格式

```json
{
  "annotations": [
    {
      "image_path": "dataset/images/1.jpg",
      "formula": "H₂O",
      "graph_data": {
        "nodes": ["H", "O", "H"],
        "edges": [[0, 1], [1, 2]],
        "bond_types": ["covalent", "covalent"]
      }
    }
  ],
  "vocab": {
    "chars": ["H", "O", "C", "N", "2", "=", "+", "-", ...],
    "size": 44
  }
}
```

## 模型配置

### 图编码器配置

```python
graph_encoder_config = {
    "node_dim": 64,           # 节点特征维度
    "edge_dim": 16,           # 边特征维度
    "hidden_dim": 256,        # 隐藏层维度
    "num_layers": 3,          # GNN层数
    "num_heads": 8,           # 注意力头数
    "dropout": 0.1            # Dropout率
}
```

### 序列编码器配置

```python
sequence_encoder_config = {
    "backbone": "mobilenet",  # 骨干网络
    "hidden_dim": 256,        # 隐藏层维度
    "num_layers": 2,          # RNN层数
    "bidirectional": True,    # 双向RNN
    "dropout": 0.1            # Dropout率
}
```

### 融合编码器配置

```python
fusion_encoder_config = {
    "fusion_type": "cross_attention",  # 融合类型
    "hidden_dim": 512,                 # 隐藏层维度
    "num_heads": 8,                    # 注意力头数
    "num_layers": 3,                   # 编码器层数
    "dropout": 0.1                     # Dropout率
}
```

## 训练配置

### 优化器配置

```python
train_config = {
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "scheduler": "CosineAnnealingLR",
    "warmup_steps": 1000,
    "max_epochs": 100,
    "batch_size": 16,
    "grad_clip": 1.0
}
```

### 损失函数配置

```python
loss_config = {
    "ctc_weight": 0.7,        # CTC损失权重
    "crf_weight": 0.3,        # CRF损失权重
    "label_smoothing": 0.1,   # 标签平滑
    "ignore_index": -100      # 忽略索引
}
```

## 日志配置

### 日志格式设置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
```

### 训练日志示例

```
2024-01-01 10:00:00 - train - INFO - 开始训练
2024-01-01 10:00:01 - train - INFO - 加载数据完成，样本数: 1880
2024-01-01 10:00:02 - train - INFO - 模型初始化完成
2024-01-01 10:00:03 - train - INFO - Epoch 1/100, Loss: 2.3456
```

## 部署配置

### 推理配置

```python
inference_config = {
    "model_path": "checkpoints/best_model.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 1,
    "max_length": 50,
    "beam_size": 5
}
```

### API配置

```python
api_config = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 4
}
```

## 性能优化配置

### 内存优化

```python
memory_config = {
    "gradient_checkpointing": True,    # 梯度检查点
    "mixed_precision": True,           # 混合精度训练
    "pin_memory": True,                # 固定内存
    "num_workers": 4                   # 数据加载进程数
}
```

### GPU配置

```python
gpu_config = {
    "cuda_visible_devices": "0,1",     # 可见GPU
    "allow_growth": True,              # 内存增长模式
    "per_process_gpu_memory_fraction": 0.8  # GPU内存限制
}
```

## 测试配置

### 单元测试配置

```python
test_config = {
    "test_data_ratio": 0.2,            # 测试数据比例
    "random_seed": 42,                 # 随机种子
    "test_batch_size": 32,             # 测试批次大小
    "metrics": ["accuracy", "precision", "recall", "f1"]
}
```

## 配置验证

### 配置验证函数

```python
def validate_config(config):
    """验证配置参数的有效性"""
    
    # 检查数据路径
    if not os.path.exists(config.DATA_DIR):
        raise ValueError(f"数据目录不存在: {config.DATA_DIR}")
    
    # 检查模型参数
    if config.HIDDEN_DIM <= 0:
        raise ValueError("隐藏层维度必须大于0")
    
    # 检查训练参数
    if config.BATCH_SIZE <= 0:
        raise ValueError("批次大小必须大于0")
    
    return True
```

## 配置示例

### 开发环境配置

```python
# config_dev.py
class DevConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
```

### 生产环境配置

```python
# config_prod.py
class ProdConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
```

## 配置管理最佳实践

1. **版本控制**: 将配置文件纳入版本控制
2. **环境分离**: 为不同环境创建不同配置
3. **参数验证**: 添加配置参数验证逻辑
4. **文档化**: 为每个配置参数添加注释说明
5. **默认值**: 为所有参数设置合理的默认值

通过合理的配置管理，可以确保项目在不同环境下的稳定运行和性能优化。