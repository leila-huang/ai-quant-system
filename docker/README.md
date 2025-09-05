# AI 量化系统 Docker 开发环境

本目录包含 AI 量化系统的 Docker 开发环境配置文件。

## 目录结构

```
docker/
├── postgres/
│   ├── init.sql              # PostgreSQL初始化脚本
│   ├── dev-init.sql          # 开发环境数据脚本
│   └── postgresql.conf       # PostgreSQL配置
├── redis/
│   ├── redis.conf           # Redis生产配置
│   └── redis-dev.conf       # Redis开发配置
├── prometheus/
│   └── prometheus.yml       # Prometheus监控配置
└── grafana/
    ├── dashboards/
    │   └── dashboard.yml    # 仪表板配置
    └── datasources/
        └── prometheus.yml   # 数据源配置
```

## 服务说明

### PostgreSQL

- **镜像**: postgres:16-alpine
- **端口**: 5432
- **数据库**: ai_quant_db
- **用户**: ai_quant_user
- **密码**: ai_quant_password

### Redis

- **镜像**: redis:7-alpine
- **端口**: 6379
- **持久化**: 启用 RDB + AOF
- **配置**: 开发和生产环境分离

### 监控服务

#### Prometheus

- **端口**: 9090
- **功能**: 指标收集和存储
- **配置**: 监控应用、数据库、缓存

#### Grafana

- **端口**: 3000
- **默认账户**: admin/admin
- **功能**: 监控仪表板

## 使用方法

### 基本命令

```bash
# 启动开发环境
./scripts/start-dev.sh start

# 启动开发环境和工具
./scripts/start-dev.sh start --with-tools

# 启动监控服务
./scripts/start-dev.sh start --with-monitoring

# 查看服务状态
./scripts/start-dev.sh status

# 查看应用日志
./scripts/start-dev.sh logs app

# 进入容器
./scripts/start-dev.sh exec app

# 停止服务
./scripts/start-dev.sh stop
```

### 数据持久化

所有重要数据都通过 Docker 卷进行持久化：

- `ai-quant-postgres-data`: PostgreSQL 数据
- `ai-quant-redis-data`: Redis 数据
- `ai-quant-app-data`: 应用数据文件
- `ai-quant-app-logs`: 应用日志

### 开发模式特性

1. **代码热更新**: 本地代码变更自动同步到容器
2. **调试支持**: 详细的日志输出和错误信息
3. **管理工具**: Adminer 和 Redis Commander
4. **监控工具**: Prometheus 和 Grafana

## 配置自定义

### 环境变量

复制 `env.template` 为 `.env` 并根据需要修改配置：

```bash
cp env.template .env
```

### 数据库初始化

- `init.sql`: 生产环境基础表结构
- `dev-init.sql`: 开发环境测试数据

### 服务配置

- PostgreSQL: `docker/postgres/postgresql.conf`
- Redis 生产: `docker/redis/redis.conf`
- Redis 开发: `docker/redis/redis-dev.conf`

## 故障排除

### 常见问题

1. **端口冲突**: 确保 8000、5432、6379 端口未被占用
2. **权限问题**: 确保有 Docker 执行权限
3. **磁盘空间**: 确保有足够的磁盘空间

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
./scripts/start-dev.sh logs postgres
./scripts/start-dev.sh logs redis
./scripts/start-dev.sh logs app
```

### 重置环境

```bash
# 清理所有容器和数据
./scripts/start-dev.sh clean

# 重新构建镜像
./scripts/start-dev.sh build
```

## 性能优化

### PostgreSQL 优化

- 调整 `shared_buffers` 和 `effective_cache_size`
- 根据硬件配置修改 `max_connections`

### Redis 优化

- 配置合适的 `maxmemory` 策略
- 根据使用模式调整持久化策略

### 应用优化

- 配置合适的工作进程数
- 调整数据库连接池大小
