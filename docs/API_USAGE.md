# API使用说明

## 概述

化学公式识别系统提供完整的RESTful API接口，支持图像上传、批量识别、模型管理和状态监控等功能。

## 基础配置

### 启动API服务

```bash
# 方式1：使用主程序
python main.py --mode api --host 0.0.0.0 --port 8000

# 方式2：直接运行API脚本
python src/api.py --host 0.0.0.0 --port 8000
```

### 环境变量配置

```bash
# .env 文件
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=checkpoints/best_model.pth
LOG_LEVEL=INFO
DEBUG=False
```

## API接口文档

### 1. 健康检查

**端点**: `GET /health`

**描述**: 检查API服务状态

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T10:00:00Z",
  "version": "1.0.0",
  "model_loaded": true
}
```

### 2. 单张图像识别

**端点**: `POST /predict`

**描述**: 识别单张化学公式图像

**请求格式**:
- **Content-Type**: `multipart/form-data`
- **参数**:
  - `image`: 图像文件 (JPEG/PNG)
  - `graph_data` (可选): 分子图数据 (JSON)

**响应示例**:
```json
{
  "success": true,
  "prediction": "H₂O",
  "confidence": 0.95,
  "processing_time": 0.12,
  "graph_analysis": {
    "nodes": ["H", "O", "H"],
    "edges": [[0, 1], [1, 2]],
    "bond_types": ["covalent", "covalent"]
  }
}
```

### 3. 批量识别

**端点**: `POST /batch_predict`

**描述**: 批量识别多张化学公式图像

**请求格式**:
- **Content-Type**: `multipart/form-data`
- **参数**:
  - `images[]`: 多个图像文件
  - `batch_size` (可选): 批次大小，默认32

**响应示例**:
```json
{
  "success": true,
  "results": [
    {
      "image_id": "1.jpg",
      "prediction": "H₂O",
      "confidence": 0.95,
      "processing_time": 0.12
    },
    {
      "image_id": "2.jpg", 
      "prediction": "CO₂",
      "confidence": 0.92,
      "processing_time": 0.11
    }
  ],
  "total_time": 0.23,
  "average_time": 0.115
}
```

### 4. 模型管理

**端点**: `POST /model/reload`

**描述**: 重新加载模型

**请求参数**:
- `model_path` (可选): 新模型路径

**响应示例**:
```json
{
  "success": true,
  "message": "模型重新加载成功",
  "model_path": "checkpoints/best_model.pth"
}
```

### 5. 性能统计

**端点**: `GET /stats`

**描述**: 获取API性能统计信息

**响应示例**:
```json
{
  "total_requests": 1000,
  "successful_requests": 980,
  "failed_requests": 20,
  "average_response_time": 0.15,
  "uptime": "2 days, 3 hours, 15 minutes",
  "memory_usage": "256MB",
  "gpu_available": true
}
```

## 客户端使用示例

### Python客户端

```python
import requests
import json

class ChemicalFormulaClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict(self, image_path, graph_data=None):
        """单张图像识别"""
        files = {"image": open(image_path, "rb")}
        data = {}
        if graph_data:
            data["graph_data"] = json.dumps(graph_data)
        
        response = requests.post(f"{self.base_url}/predict", files=files, data=data)
        return response.json()
    
    def batch_predict(self, image_paths, batch_size=32):
        """批量识别"""
        files = [("images[]", open(path, "rb")) for path in image_paths]
        data = {"batch_size": batch_size}
        
        response = requests.post(f"{self.base_url}/batch_predict", files=files, data=data)
        return response.json()

# 使用示例
client = ChemicalFormulaClient()

# 健康检查
print(client.health_check())

# 单张图像识别
result = client.predict("test_image.jpg")
print(result)

# 批量识别
results = client.batch_predict(["img1.jpg", "img2.jpg", "img3.jpg"])
print(results)
```

### JavaScript客户端

```javascript
class ChemicalFormulaClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
    
    async predict(imageFile, graphData = null) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        if (graphData) {
            formData.append('graph_data', JSON.stringify(graphData));
        }
        
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    async batchPredict(imageFiles, batchSize = 32) {
        const formData = new FormData();
        
        imageFiles.forEach((file, index) => {
            formData.append('images[]', file);
        });
        
        formData.append('batch_size', batchSize);
        
        const response = await fetch(`${this.baseUrl}/batch_predict`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
}

// 使用示例
const client = new ChemicalFormulaClient();

// 健康检查
client.healthCheck().then(console.log);

// 单张图像识别
const imageInput = document.getElementById('image-input');
client.predict(imageInput.files[0]).then(console.log);
```

### cURL示例

```bash
# 健康检查
curl -X GET http://localhost:8000/health

# 单张图像识别
curl -X POST -F "image=@test_image.jpg" http://localhost:8000/predict

# 批量识别
curl -X POST -F "images[]=@img1.jpg" -F "images[]=@img2.jpg" http://localhost:8000/batch_predict

# 带图数据的识别
curl -X POST -F "image=@test_image.jpg" -F 'graph_data={"nodes":["H","O","H"],"edges":[[0,1],[1,2]]}' http://localhost:8000/predict
```

## 错误处理

### 错误码说明

| 状态码 | 错误类型 | 描述 |
|--------|----------|------|
| 200 | SUCCESS | 请求成功 |
| 400 | BAD_REQUEST | 请求参数错误 |
| 404 | NOT_FOUND | 资源不存在 |
| 415 | UNSUPPORTED_MEDIA_TYPE | 不支持的媒体类型 |
| 500 | INTERNAL_ERROR | 服务器内部错误 |

### 错误响应格式

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "不支持的文件格式，请使用JPEG或PNG格式",
    "details": "文件扩展名: .bmp"
  }
}
```

## 性能优化建议

### 1. 图像预处理

```python
# 客户端预处理示例
def preprocess_image(image_path):
    """图像预处理"""
    import cv2
    
    # 读取图像
    image = cv2.imread(image_path)
    
    # 调整尺寸
    image = cv2.resize(image, (512, 128))
    
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    return image
```

### 2. 批量请求优化

```python
# 批量处理优化
import asyncio
import aiohttp

async def batch_predict_async(image_paths, batch_size=32):
    """异步批量识别"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # 分批处理
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            task = process_batch(session, batch)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
```

### 3. 缓存策略

```python
# 客户端缓存示例
import hashlib
import pickle

class CachedClient(ChemicalFormulaClient):
    def __init__(self, base_url, cache_dir=".cache"):
        super().__init__(base_url)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, image_path, graph_data):
        """生成缓存键"""
        content = image_path + (json.dumps(graph_data) if graph_data else "")
        return hashlib.md5(content.encode()).hexdigest()
    
    def predict(self, image_path, graph_data=None):
        """带缓存的预测"""
        cache_key = self._get_cache_key(image_path, graph_data)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 调用API
        result = super().predict(image_path, graph_data)
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
```

## 安全考虑

### 1. 认证授权

```python
# 添加API密钥认证
class SecureClient(ChemicalFormulaClient):
    def __init__(self, base_url, api_key):
        super().__init__(base_url)
        self.api_key = api_key
    
    def _add_auth_headers(self, headers=None):
        """添加认证头"""
        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
```

### 2. 输入验证

```python
# 客户端输入验证
def validate_image_file(file_path):
    """验证图像文件"""
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    max_size = 10 * 1024 * 1024  # 10MB
    
    if not os.path.exists(file_path):
        raise ValueError("文件不存在")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    if os.path.getsize(file_path) > max_size:
        raise ValueError("文件大小超过限制")
    
    return True
```

## 监控和日志

### 1. 客户端日志

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class LoggingClient(ChemicalFormulaClient):
    def predict(self, image_path, graph_data=None):
        """带日志的预测"""
        logger.info(f"开始识别图像: {image_path}")
        
        try:
            result = super().predict(image_path, graph_data)
            logger.info(f"识别成功: {result['prediction']}")
            return result
        except Exception as e:
            logger.error(f"识别失败: {str(e)}")
            raise
```

通过以上API使用说明，您可以轻松集成化学公式识别功能到您的应用程序中。