# 架构设计文档

AI 量化系统技术架构设计和实现说明

## 总体架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    客户端层 (Client Layer)                    │
│                   Web UI / API Clients                      │
├─────────────────────────────────────────────────────────────┤
│                    API网关层 (API Gateway)                   │
│         FastAPI + 中间件 (CORS, Auth, Rate Limit)           │
├─────────────────────────────────────────────────────────────┤
│                   应用服务层 (Application Layer)              │
│    ┌──────────────┬──────────────┬──────────────────────┐    │
│    │   数据服务    │   业务服务    │     任务队列服务      │    │
│    │ Data Service │Business Logic│   Task Queue         │    │
│    └──────────────┴──────────────┴──────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                   数据访问层 (Data Access Layer)             │
│    ┌──────────────┬──────────────┬──────────────────────┐    │
│    │ ORM/SQLAlchemy│   Parquet    │      Redis           │    │
│    │   (Business)  │  (Time Series)│     (Cache)         │    │
│    └──────────────┴──────────────┴──────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                   数据存储层 (Storage Layer)                 │
│    ┌──────────────┬──────────────┬──────────────────────┐    │
│    │ PostgreSQL   │ Parquet Files│      Redis           │    │
│    │ (ACID/关系)   │  (列式存储)   │    (内存缓存)         │    │
│    └──────────────┴──────────────┴──────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                   外部数据层 (External Data)                 │
│         AKShare API / Eastmoney API / Other Sources         │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计原则

1. **分层架构**: 清晰的层次划分，高内聚低耦合
2. **微服务思想**: 模块化设计，便于扩展和维护
3. **异步处理**: 高并发支持，非阻塞 IO
4. **数据分离**: 业务数据和时序数据分离存储
5. **容器化**: Docker 部署，环境一致性
6. **可观测性**: 完整的监控、日志、指标体系

## 详细设计

### 1. API 网关层

#### 技术栈

- **Web 框架**: FastAPI (异步, 高性能)
- **ASGI 服务器**: Uvicorn
- **API 文档**: OpenAPI/Swagger

#### 中间件组件

```python
# 中间件栈
app.add_middleware(SecurityHeadersMiddleware)    # 安全头
app.add_middleware(CORSMiddleware)              # 跨域
app.add_middleware(PerformanceMiddleware)       # 性能监控
app.add_middleware(RequestTrackingMiddleware)   # 请求追踪
```

#### 路由组织

```
/api/v1/
├── health/              # 健康检查
│   ├── ping            # 基础检查
│   ├── database        # 数据库检查
│   └── system          # 系统检查
├── data/               # 数据管理
│   ├── stocks/         # 股票数据
│   └── markets/        # 市场数据
└── data-sync/          # 数据同步
    ├── sync           # 创建任务
    ├── tasks/         # 任务管理
    └── stats          # 统计信息
```

### 2. 应用服务层

#### 数据服务 (Data Service)

负责数据获取、处理、存储的核心业务逻辑。

```python
class DataService:
    def __init__(self):
        self.akshare_adapter = AKShareAdapter()
        self.parquet_storage = ParquetStorage()
        self.task_queue = TaskQueue()

    async def sync_stock_data(self, symbols, date_range):
        """异步股票数据同步"""
        task = DataSyncTask(symbols, date_range)
        return await self.task_queue.submit(task)

    async def get_stock_data(self, symbol, date_range):
        """获取股票数据（缓存优先）"""
        # 1. 检查缓存
        # 2. 查询Parquet存储
        # 3. 实时获取（如需要）
```

#### 任务队列服务 (Task Queue)

处理异步任务的调度和执行。

```python
class TaskQueue:
    def __init__(self):
        self.pending_tasks = asyncio.Queue()
        self.running_tasks = {}
        self.completed_tasks = {}

    async def submit(self, task):
        """提交任务"""
        await self.pending_tasks.put(task)
        return task.task_id

    async def worker(self):
        """任务工作者"""
        while True:
            task = await self.pending_tasks.get()
            await self.execute_task(task)
```

### 3. 数据访问层

#### PostgreSQL (业务数据)

存储用户、策略、订单、任务等业务数据。

**表设计**:

```sql
-- 用户表
users (id, username, email, created_at, ...)

-- 策略表
strategies (id, name, user_id, config, status, ...)

-- 同步任务表
sync_tasks (id, task_id, symbols, status, progress, ...)

-- 系统日志表
system_logs (id, level, message, created_at, ...)
```

**连接池配置**:

```python
DATABASE_CONFIG = {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "pool_recycle": 3600
}
```

#### Parquet (时序数据)

高效存储和查询股票历史数据。

**目录结构**:

```
data/parquet/
├── stocks/
│   ├── 000001/
│   │   ├── 2024/
│   │   │   ├── 2024-01.parquet
│   │   │   └── 2024-02.parquet
│   │   └── metadata.json
│   └── 000002/
└── indexes/
    └── symbol_index.parquet
```

**分片策略**:

- 按股票代码分目录
- 按年月分文件
- 支持增量更新
- 自动压缩优化

```python
class ParquetStorage:
    def save_stock_data(self, stock_data: StockData):
        """保存股票数据"""
        file_path = self._get_file_path(
            stock_data.symbol,
            stock_data.bars[0].date
        )
        # 使用Snappy压缩
        df.to_parquet(file_path, compression='snappy')
```

#### Redis (缓存)

提供高速缓存和会话存储。

**缓存策略**:

```python
CACHE_CONFIG = {
    "stock_list": {"ttl": 3600},      # 股票列表缓存1小时
    "stock_data": {"ttl": 1800},      # 股票数据缓存30分钟
    "task_status": {"ttl": 300},      # 任务状态缓存5分钟
    "api_rate_limit": {"ttl": 60}     # API限流1分钟
}
```

### 4. 数据模型设计

#### 核心数据模型

```python
@dataclass
class StockDailyBar:
    """股票日线数据模型"""
    date: date
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: float
    amount: Optional[float] = None

@dataclass
class StockData:
    """股票完整数据模型"""
    symbol: str
    name: Optional[str] = None
    bars: List[StockDailyBar] = None
```

#### API 数据模型

```python
class DataSyncRequest(BaseModel):
    """数据同步请求模型"""
    symbols: List[str]
    start_date: str
    end_date: str
    task_type: str = "stock_data"
    priority: str = "normal"
    async_mode: bool = True

class DataSyncResponse(BaseModel):
    """数据同步响应模型"""
    task_id: str
    status: str
    created_at: datetime
```

### 5. 外部数据适配器

#### AKShare 适配器

```python
class AKShareAdapter:
    def __init__(self):
        self.timeout = 30
        self.retry_config = RetryConfig(max_retries=3, delay=1.0)

    def get_stock_daily_data(self, symbol: str,
                           start_date: str, end_date: str):
        """获取股票日线数据"""
        @retry(self.retry_config)
        def _fetch_data():
            return ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

        raw_data = _fetch_data()
        return self._standardize_data(raw_data, symbol)
```

### 6. 性能优化设计

#### 数据库优化

1. **索引策略**:

   ```sql
   -- 核心索引
   CREATE INDEX idx_sync_tasks_status ON sync_tasks(status);
   CREATE INDEX idx_sync_tasks_created_at ON sync_tasks(created_at);
   ```

2. **连接池优化**:
   - 合理设置池大小
   - 连接复用和回收
   - 预连接预热

#### 缓存优化

1. **多级缓存**:

   - L1: 应用内存缓存
   - L2: Redis 缓存
   - L3: Parquet 文件缓存

2. **缓存策略**:
   - LRU 淘汰策略
   - 分层 TTL 设置
   - 缓存预热机制

#### 存储优化

1. **Parquet 优化**:

   - Snappy 压缩，压缩比高
   - 列式存储，查询高效
   - 分片存储，并行处理

2. **查询优化**:
   - 谓词下推
   - 列剪裁
   - 分区扫描

### 7. 可扩展性设计

#### 水平扩展

```python
# 多实例部署配置
services:
  app-1:
    build: .
    ports: ["8001:8000"]
  app-2:
    build: .
    ports: ["8002:8000"]

  nginx:  # 负载均衡
    image: nginx
    depends_on: [app-1, app-2]
```

#### 垂直扩展

- CPU 密集型：增加 worker 进程
- IO 密集型：增加异步并发数
- 内存密集型：优化数据结构

### 8. 容错和高可用

#### 故障处理

1. **重试机制**:

   ```python
   @retry(max_attempts=3, backoff=ExponentialBackoff())
   async def fetch_data():
       # 数据获取逻辑
   ```

2. **熔断器**:

   ```python
   @circuit_breaker(failure_threshold=5, timeout=30)
   async def external_api_call():
       # 外部API调用
   ```

3. **降级策略**:
   - 数据源降级
   - 功能降级
   - 服务降级

#### 数据一致性

1. **事务管理**:

   - ACID 事务保证
   - 分布式事务(Saga 模式)
   - 最终一致性

2. **数据校验**:
   - 输入验证
   - 业务规则校验
   - 数据完整性检查

### 9. 安全设计

#### API 安全

1. **认证授权** (P1 阶段):

   - JWT Token 认证
   - RBAC 权限控制
   - API Key 管理

2. **访问控制**:
   - 请求频率限制
   - IP 白名单
   - SQL 注入防护

#### 数据安全

1. **敏感数据保护**:

   - 数据脱敏
   - 加密存储
   - 审计日志

2. **网络安全**:
   - HTTPS 加密
   - 安全头设置
   - CORS 配置

### 10. 监控和运维

#### 监控指标

1. **业务指标**:

   - API 响应时间
   - 数据同步成功率
   - 错误率统计

2. **系统指标**:
   - CPU/内存使用率
   - 数据库连接数
   - 缓存命中率

#### 日志管理

```python
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    }
}
```

## 部署架构

### 开发环境

```yaml
# docker-compose.dev.yml
version: "3.8"
services:
  app:
    build: .
    volumes:
      - .:/app # 代码热更新
  postgres:
    image: postgres:16
  redis:
    image: redis:7
  adminer: # 数据库管理
    image: adminer
```

### 生产环境

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  app:
    image: ai-quant:latest
    replicas: 3 # 多实例
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
  nginx:
    image: nginx
    # 负载均衡配置
  postgres:
    image: postgres:16
    volumes:
      - postgres_data:/var/lib/postgresql/data
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
```

## 未来架构演进

### P1 阶段扩展

- 实时数据推送 (WebSocket)
- 分布式任务调度 (Celery)
- 多数据源聚合

### P2 阶段扩展

- 微服务拆分
- 事件驱动架构
- 机器学习服务集成

### P3 阶段扩展

- 云原生架构 (Kubernetes)
- 服务网格 (Istio)
- 多区域部署

---

**文档版本**: v1.0.0  
**更新时间**: 2024 年 12 月 09 日  
**负责人**: 架构设计团队
