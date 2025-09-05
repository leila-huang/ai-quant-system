# API 接口文档

AI 量化系统 RESTful API 接口完整说明

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API Version**: v1
- **Content-Type**: `application/json`
- **编码**: UTF-8

## 认证

当前版本未启用认证，P1 阶段将添加 JWT Token 认证。

## 响应格式

### 成功响应

```json
{
  "success": true,
  "message": "操作成功",
  "data": {
    // 响应数据
  },
  "timestamp": "2024-12-09T10:30:00Z"
}
```

### 错误响应

```json
{
  "success": false,
  "message": "错误描述",
  "error_code": "ERROR_CODE",
  "details": {
    // 错误详情
  },
  "timestamp": "2024-12-09T10:30:00Z"
}
```

## API 接口

### 1. 健康检查

#### 1.1 基础健康检查

**接口**: `GET /api/v1/health/ping`

**描述**: 检查 API 服务状态

**响应示例**:

```json
{
  "success": true,
  "message": "pong",
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-12-09T10:30:00Z"
  }
}
```

#### 1.2 数据库健康检查

**接口**: `GET /api/v1/health/database`

**描述**: 检查数据库连接状态

**响应示例**:

```json
{
  "success": true,
  "message": "数据库连接正常",
  "data": {
    "database": "healthy",
    "connection_count": 5,
    "max_connections": 100
  }
}
```

#### 1.3 系统健康检查

**接口**: `GET /api/v1/health/system`

**描述**: 检查整个系统状态

**响应示例**:

```json
{
  "success": true,
  "message": "系统运行正常",
  "data": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "storage": "healthy",
    "akshare": "healthy"
  }
}
```

### 2. 数据管理

#### 2.1 获取股票列表

**接口**: `GET /api/v1/data/stocks/list`

**描述**: 获取支持的股票列表

**查询参数**:

- `page` (int, optional): 页码，默认 1
- `size` (int, optional): 每页大小，默认 100
- `market` (str, optional): 市场类型 (sz/sh)

**响应示例**:

```json
{
  "success": true,
  "data": {
    "stocks": [
      {
        "symbol": "000001",
        "name": "平安银行",
        "market": "sz"
      }
    ],
    "pagination": {
      "page": 1,
      "size": 100,
      "total": 5000,
      "pages": 50
    }
  }
}
```

#### 2.2 查询单只股票数据

**接口**: `GET /api/v1/data/stocks/{symbol}`

**描述**: 获取指定股票的历史数据

**路径参数**:

- `symbol` (str): 股票代码，如 "000001"

**查询参数**:

- `start_date` (str, optional): 开始日期，格式 YYYY-MM-DD
- `end_date` (str, optional): 结束日期，格式 YYYY-MM-DD
- `fields` (str, optional): 返回字段，逗号分隔

**响应示例**:

```json
{
  "success": true,
  "data": {
    "symbol": "000001",
    "name": "平安银行",
    "records": [
      {
        "date": "2024-01-02",
        "open_price": 10.5,
        "close_price": 10.8,
        "high_price": 10.9,
        "low_price": 10.4,
        "volume": 12500000,
        "amount": 134375000.0,
        "pct_change": 2.86
      }
    ],
    "total_records": 1
  }
}
```

#### 2.3 批量查询股票数据

**接口**: `POST /api/v1/data/stocks/batch`

**描述**: 批量获取多只股票的历史数据

**请求体**:

```json
{
  "symbols": ["000001", "000002", "600000"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "fields": ["date", "open_price", "close_price", "volume"]
}
```

**响应示例**:

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "symbol": "000001",
        "name": "平安银行",
        "records": [...],
        "count": 21
      }
    ],
    "summary": {
      "requested_symbols": 3,
      "successful_symbols": 3,
      "failed_symbols": 0,
      "total_records": 63
    }
  }
}
```

### 3. 数据同步

#### 3.1 创建同步任务

**接口**: `POST /api/v1/data-sync/sync`

**描述**: 创建新的数据同步任务

**请求体**:

```json
{
  "task_type": "stock_data",
  "symbols": ["000001", "000002"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "priority": "normal",
  "async": true
}
```

**响应示例**:

```json
{
  "success": true,
  "message": "同步任务创建成功",
  "data": {
    "task_id": "123e4567-e89b-12d3-a456-426614174000",
    "task_type": "stock_data",
    "status": "pending",
    "symbols": ["000001", "000002"],
    "created_at": "2024-12-09T10:30:00Z"
  }
}
```

#### 3.2 查询任务状态

**接口**: `GET /api/v1/data-sync/tasks/{task_id}/status`

**描述**: 查询同步任务执行状态

**路径参数**:

- `task_id` (str): 任务 ID

**响应示例**:

```json
{
  "success": true,
  "data": {
    "task_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "running",
    "progress": 65.5,
    "total_count": 100,
    "completed_count": 65,
    "failed_count": 1,
    "created_at": "2024-12-09T10:30:00Z",
    "started_at": "2024-12-09T10:31:00Z",
    "estimated_completion": "2024-12-09T10:35:00Z"
  }
}
```

#### 3.3 任务列表查询

**接口**: `GET /api/v1/data-sync/tasks`

**描述**: 获取同步任务列表

**查询参数**:

- `status` (str, optional): 任务状态过滤
- `task_type` (str, optional): 任务类型过滤
- `page` (int, optional): 页码
- `size` (int, optional): 每页大小

**响应示例**:

```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "task_id": "123e4567-e89b-12d3-a456-426614174000",
        "task_type": "stock_data",
        "status": "completed",
        "progress": 100.0,
        "created_at": "2024-12-09T10:30:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "size": 10,
      "total": 25,
      "pages": 3
    }
  }
}
```

#### 3.4 取消同步任务

**接口**: `DELETE /api/v1/data-sync/tasks/{task_id}`

**描述**: 取消指定的同步任务

**路径参数**:

- `task_id` (str): 任务 ID

**响应示例**:

```json
{
  "success": true,
  "message": "任务取消成功",
  "data": {
    "task_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "cancelled",
    "cancelled_at": "2024-12-09T10:35:00Z"
  }
}
```

### 4. 系统统计

#### 4.1 获取系统统计信息

**接口**: `GET /api/v1/data-sync/stats`

**描述**: 获取数据同步统计信息

**响应示例**:

```json
{
  "success": true,
  "data": {
    "tasks": {
      "total": 150,
      "completed": 145,
      "running": 2,
      "pending": 1,
      "failed": 2
    },
    "data_volume": {
      "total_stocks": 4500,
      "total_records": 15000000,
      "storage_size_mb": 2048
    },
    "performance": {
      "avg_sync_time": 120.5,
      "success_rate": 96.7,
      "last_sync": "2024-12-09T10:00:00Z"
    }
  }
}
```

#### 4.2 清理历史数据

**接口**: `POST /api/v1/data-sync/cleanup`

**描述**: 清理过期的历史数据

**请求体**:

```json
{
  "cleanup_type": "expired_tasks",
  "retention_days": 30
}
```

**响应示例**:

```json
{
  "success": true,
  "message": "数据清理完成",
  "data": {
    "cleaned_tasks": 25,
    "freed_space_mb": 512,
    "cleanup_time": "2024-12-09T10:30:00Z"
  }
}
```

## 错误码说明

| 错误码            | HTTP 状态码 | 说明             |
| ----------------- | ----------- | ---------------- |
| VALIDATION_ERROR  | 400         | 请求参数验证失败 |
| NOT_FOUND         | 404         | 资源不存在       |
| INTERNAL_ERROR    | 500         | 内部服务器错误   |
| DATA_SOURCE_ERROR | 502         | 数据源服务异常   |
| STORAGE_ERROR     | 503         | 存储服务异常     |
| TASK_CONFLICT     | 409         | 任务冲突         |
| RATE_LIMIT        | 429         | 请求频率限制     |

## 请求限制

- **单次批量查询**: 最多 100 只股票
- **日期范围**: 最长 1 年
- **请求频率**: 100 请求/分钟/IP
- **并发连接**: 50 个/IP

## SDK 和客户端

### Python SDK 示例

```python
import requests
import json

class AIQuantClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def get_stock_data(self, symbol, start_date=None, end_date=None):
        url = f"{self.base_url}/api/v1/data/stocks/{symbol}"
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        response = requests.get(url, params=params)
        return response.json()

    def create_sync_task(self, symbols, start_date, end_date):
        url = f"{self.base_url}/api/v1/data-sync/sync"
        data = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "task_type": "stock_data"
        }

        response = requests.post(url, json=data)
        return response.json()

# 使用示例
client = AIQuantClient()
result = client.get_stock_data("000001", "2024-01-01", "2024-01-31")
print(result)
```

### cURL 示例

```bash
# 获取股票数据
curl -X GET "http://localhost:8000/api/v1/data/stocks/000001?start_date=2024-01-01&end_date=2024-01-31"

# 创建同步任务
curl -X POST "http://localhost:8000/api/v1/data-sync/sync" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["000001", "000002"],
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "task_type": "stock_data"
  }'

# 查询任务状态
curl -X GET "http://localhost:8000/api/v1/data-sync/tasks/task-id/status"
```

## 变更日志

### v1.0.0 (2024-12-09)

- 初始版本发布
- 基础数据查询接口
- 数据同步任务管理
- 健康检查接口

### 即将发布

- 用户认证和授权 (v1.1.0)
- 实时数据推送 (v1.2.0)
- 策略回测接口 (v1.3.0)

---

**文档更新时间**: 2024 年 12 月 09 日  
**API 版本**: v1.0.0
