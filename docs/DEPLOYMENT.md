# 部署指南

## 概述

本文档提供化学公式识别系统的完整部署指南，包括本地部署、Docker部署、云服务部署和边缘设备部署。

## 本地部署

### 1. 环境要求

#### 硬件要求
- **CPU**: 4核以上
- **内存**: 8GB以上
- **GPU** (可选): NVIDIA GPU，支持CUDA 11.0+
- **存储**: 至少1GB可用空间

#### 软件要求
- **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8+
- **包管理器**: pip 20.0+

### 2. 安装步骤

#### 克隆项目
```bash
git clone <repository-url>
cd 化学公式
```

#### 创建虚拟环境
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 安装依赖
```bash
pip install -r requirements.txt

# 如果有GPU，安装GPU版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 数据准备
```bash
# 创建数据目录结构
mkdir -p dataset/images data checkpoints logs

# 准备数据文件
# 将图像文件放入 dataset/images/
# 将标签文件放入 dataset/labels.txt
```

#### 预训练模型下载
```bash
# 下载预训练模型（如果有）
wget -O checkpoints/best_model.pth <model-download-url>
```

### 3. 验证安装

```bash
# 运行测试
python test_restructured.py

# 运行演示
python main.py
```

## Docker部署

### 1. Dockerfile

```dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY src/ ./src/
COPY main.py .
COPY data/ ./data/
COPY dataset/ ./dataset/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p checkpoints logs

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/checkpoints/best_model.pth

# 启动命令
CMD ["python", "main.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose配置

```yaml
version: '3.8'

services:
  chemical-formula-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/checkpoints/best_model.pth
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  # 可选：添加Redis缓存
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### 3. 构建和运行

```bash
# 构建镜像
docker build -t chemical-formula-recognition .

# 运行容器
docker run -d -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  chemical-formula-recognition

# 使用Docker Compose
docker-compose up -d
```

## 云服务部署

### 1. AWS部署

#### EC2实例配置

```bash
# 用户数据脚本（实例启动时自动执行）
#!/bin/bash
apt-get update
apt-get install -y python3-pip docker.io

# 安装Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 克隆项目
git clone <repository-url>
cd 化学公式

# 启动服务
docker-compose up -d
```

#### ECS部署配置

```yaml
# task-definition.json
{
  "family": "chemical-formula-api",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "chemical-formula-recognition:latest",
      "cpu": 1024,
      "memory": 2048,
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/checkpoints/best_model.pth"
        }
      ]
    }
  ]
}
```

### 2. Azure部署

#### Azure Container Instances

```bash
# 创建容器实例
az container create \
  --resource-group myResourceGroup \
  --name chemical-formula-api \
  --image chemical-formula-recognition:latest \
  --ports 8000 \
  --environment-variables MODEL_PATH=/app/checkpoints/best_model.pth
```

#### Azure App Service

```bash
# 创建Web应用
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name chemical-formula-api \
  --runtime "PYTHON:3.9"

# 部署代码
az webapp up --name chemical-formula-api
```

### 3. Google Cloud部署

#### Google Cloud Run

```bash
# 构建并推送镜像
gcloud builds submit --tag gcr.io/my-project/chemical-formula-recognition

# 部署到Cloud Run
gcloud run deploy chemical-formula-api \
  --image gcr.io/my-project/chemical-formula-recognition \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 边缘设备部署

### 1. Raspberry Pi部署

#### 环境准备

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# 安装OpenCV依赖
sudo apt-get install -y libopencv-dev python3-opencv

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装轻量级依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-deps
```

#### 优化配置

```python
# config_raspberry.py
class RaspberryConfig(Config):
    BATCH_SIZE = 1  # 单张图像处理
    IMAGE_SIZE = (64, 256)  # 减小图像尺寸
    HIDDEN_DIM = 128  # 减小模型维度
    USE_GPU = False  # 禁用GPU
```

### 2. NVIDIA Jetson部署

#### JetPack SDK安装

```bash
# 安装JetPack SDK
sudo apt-get update
sudo apt-get install nvidia-jetpack

# 安装PyTorch for Jetson
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

#### TensorRT优化

```python
import tensorrt as trt

def optimize_model(model_path):
    """使用TensorRT优化模型"""
    # TensorRT优化代码
    pass
```

## 性能优化

### 1. 模型优化

#### 模型量化

```python
import torch
from torch.quantization import quantize_dynamic

# 动态量化
model = torch.load("checkpoints/best_model.pth")
model_quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(model_quantized, "checkpoints/model_quantized.pth")
```

#### 模型剪枝

```python
import torch.nn.utils.prune as prune

# 权重剪枝
parameters_to_prune = [
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
```

### 2. 推理优化

#### 批处理优化

```python
class BatchProcessor:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.batch_buffer = []
    
    def process_batch(self, images):
        """批量处理图像"""
        # 批量推理代码
        pass
```

#### 缓存策略

```python
import redis

class CachedPredictor:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
    
    def predict_with_cache(self, image_hash, image_data):
        """带缓存的预测"""
        # 检查缓存
        cached_result = self.redis.get(image_hash)
        if cached_result:
            return json.loads(cached_result)
        
        # 执行预测
        result = self.model.predict(image_data)
        
        # 保存缓存
        self.redis.setex(image_hash, 3600, json.dumps(result))
        return result
```

## 监控和日志

### 1. 应用监控

#### Prometheus配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'chemical-formula-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

#### 自定义指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
requests_total = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'API request duration')
predictions_total = Counter('predictions_total', 'Total predictions made')

@request_duration.time()
def predict_endpoint(image_data):
    requests_total.inc()
    predictions_total.inc()
    # 预测逻辑
    pass
```

### 2. 日志配置

#### 结构化日志

```python
import structlog

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

## 安全配置

### 1. 网络安全

#### SSL/TLS配置

```python
# 使用HTTPS
import ssl

ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('cert.pem', 'key.pem')
```

#### 防火墙规则

```bash
# 只允许必要端口
ufw allow 8000/tcp
ufw allow 22/tcp  # SSH
ufw enable
```

### 2. 应用安全

#### 输入验证

```python
def validate_image_file(file_path):
    """验证图像文件安全性"""
    # 检查文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    # 检查文件大小
    max_size = 10 * 1024 * 1024  # 10MB
    # 检查文件内容
    # ...
```

#### API密钥认证

```python
from flask_httpauth import HTTPTokenAuth

auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    "secret-token-1": "user1",
    "secret-token-2": "user2"
}

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]
```

## 备份和恢复

### 1. 数据备份

#### 自动备份脚本

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/chemical-formula"

# 备份模型
cp checkpoints/best_model.pth $BACKUP_DIR/model_$DATE.pth

# 备份数据
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/ dataset/

# 备份日志
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# 清理旧备份（保留最近7天）
find $BACKUP_DIR -name "*.pth" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 2. 灾难恢复

#### 恢复脚本

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

# 停止服务
docker-compose down

# 恢复数据
tar -xzf $BACKUP_FILE

# 启动服务
docker-compose up -d
```

通过以上部署指南，您可以在各种环境中成功部署化学公式识别系统。