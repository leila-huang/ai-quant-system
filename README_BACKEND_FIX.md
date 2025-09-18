# 后端启动问题修复说明

## 问题描述

后端启动时遇到 `ModuleNotFoundError: No module named 'backend'` 错误。

## 解决方案

### 1. 问题根源

- Python 模块导入路径设置不正确
- 缺少必要的依赖包

### 2. 修复步骤

#### 步骤 1: 修复导入路径

所有 `backend/app/` 目录下的文件中，将相对导入改为绝对导入：

```python
# 修改前
from app.core.config import settings

# 修改后
from backend.app.core.config import settings
```

#### 步骤 2: 安装依赖包

```bash
pip3 install pydantic-settings "fastapi[standard]" uvicorn
```

#### 步骤 3: 正确启动后端

```bash
cd /path/to/ai-quant-system
PYTHONPATH=/path/to/ai-quant-system python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 验证结果

### ✅ 成功启动的标志

1. **根路径测试**

   ```bash
   curl http://localhost:8000/
   ```

   返回: `{"message":"Welcome to AI量化系统",...}`

2. **健康检查**

   ```bash
   curl http://localhost:8000/api/v1/health/ping
   ```

   返回: `{"message":"pong","timestamp":"..."}`

3. **API 文档**
   访问: http://localhost:8000/docs

### 🔧 快速启动脚本

已创建 `start_backend_local.py` 脚本，可直接运行：

```bash
python3 start_backend_local.py
```

## 服务访问地址

- **FastAPI 应用**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/api/v1/health
- **OpenAPI 规范**: http://localhost:8000/api/v1/openapi.json

## 可用 API 接口

- 健康检查: `/api/v1/health/*`
- 数据管理: `/api/v1/data/*`
- 股票数据查询: `/api/v1/data/stocks/{symbol}`
- AKShare 数据: `/api/v1/data/akshare/stocks/{symbol}`

修复完成时间: $(date)
状态: ✅ 完全正常运行
