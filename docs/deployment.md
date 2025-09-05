# 部署说明文档

AI 量化系统生产环境部署完整指南

## 环境要求

### 硬件要求

#### 最低配置

- **CPU**: 4 核心
- **内存**: 8GB RAM
- **存储**: 100GB SSD
- **网络**: 10Mbps 带宽

#### 推荐配置

- **CPU**: 8 核心 (Intel i7/AMD Ryzen 7)
- **内存**: 16GB RAM
- **存储**: 500GB NVMe SSD
- **网络**: 100Mbps 带宽

#### 生产配置

- **CPU**: 16 核心 (Intel Xeon/AMD EPYC)
- **内存**: 32GB+ RAM
- **存储**: 1TB+ NVMe SSD + 数据盘
- **网络**: 1Gbps 带宽

### 软件要求

- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.11+ (如果不使用容器)
- **Git**: 2.0+

## 部署方式

### 1. Docker 部署（推荐）

#### 1.1 准备工作

```bash
# 1. 安装Docker和Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 2. 验证安装
docker --version
docker-compose --version
```

#### 1.2 下载项目

```bash
# 克隆项目
git clone <repository-url>
cd ai-quant-system

# 检查项目完整性
ls -la
```

#### 1.3 配置环境

```bash
# 1. 复制环境变量模板
cp env.template .env

# 2. 编辑环境变量
vim .env
```

**关键配置项**:

```bash
# 基础配置
PROJECT_NAME=AI量化系统-生产环境
DEBUG=false
LOG_LEVEL=INFO

# 数据库配置
DB_HOST=postgres
DB_PORT=5432
DB_USERNAME=ai_quant_user
DB_PASSWORD=your_secure_password_here
DB_NAME=ai_quant_db

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password_here

# 安全配置
SECRET_KEY=your_very_secure_secret_key_here
```

#### 1.4 启动服务

```bash
# 生产环境启动
docker-compose -f docker-compose.yml up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

#### 1.5 验证部署

```bash
# 健康检查
curl http://localhost:8000/api/v1/health/ping

# API文档访问
curl http://localhost:8000/docs
```

### 2. 生产环境优化部署

#### 2.1 生产配置文件

创建 `docker-compose.prod.yml`:

```yaml
version: "3.8"

services:
  # 应用服务（多实例）
  app:
    image: ai-quant:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - WORKERS=4
      - DEBUG=false
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # 负载均衡器
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

  # 数据库（主从配置）
  postgres-master:
    image: postgres:16-alpine
    environment:
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: replication_password
    volumes:
      - postgres_master_data:/var/lib/postgresql/data
      - ./postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    restart: unless-stopped

  postgres-slave:
    image: postgres:16-alpine
    environment:
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_MASTER_SERVICE: postgres-master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: replication_password
    depends_on:
      - postgres-master
    restart: unless-stopped

  # Redis集群
  redis-master:
    image: redis:7-alpine
    command: redis-server /etc/redis/redis.conf --appendonly yes
    volumes:
      - ./redis/redis-master.conf:/etc/redis/redis.conf:ro
      - redis_master_data:/data
    restart: unless-stopped

  redis-slave:
    image: redis:7-alpine
    command: redis-server /etc/redis/redis.conf --slaveof redis-master 6379
    volumes:
      - ./redis/redis-slave.conf:/etc/redis/redis.conf:ro
      - redis_slave_data:/data
    depends_on:
      - redis-master
    restart: unless-stopped

volumes:
  postgres_master_data:
  redis_master_data:
  redis_slave_data:
```

#### 2.2 Nginx 配置

创建 `nginx/nginx.conf`:

```nginx
upstream app_backend {
    least_conn;
    server app_1:8000;
    server app_2:8000;
    server app_3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL配置
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # 安全头
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    # 代理配置
    location / {
        proxy_pass http://app_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时配置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # 静态文件缓存
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, no-transform";
    }
}
```

#### 2.3 生产环境启动

```bash
# 构建生产镜像
docker build -t ai-quant:latest .

# 启动生产环境
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 验证服务
curl -k https://your-domain.com/api/v1/health/ping
```

### 3. 手动部署

#### 3.1 系统准备

```bash
# 安装Python和依赖
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip postgresql-client redis-tools

# 创建应用用户
sudo useradd -m -s /bin/bash ai-quant
sudo su - ai-quant
```

#### 3.2 应用部署

```bash
# 1. 下载代码
git clone <repository-url>
cd ai-quant-system

# 2. 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp env.template .env
vim .env

# 5. 初始化数据库
alembic upgrade head

# 6. 启动应用
gunicorn backend.app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

#### 3.3 系统服务配置

创建 systemd 服务文件 `/etc/systemd/system/ai-quant.service`:

```ini
[Unit]
Description=AI量化系统
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=ai-quant
Group=ai-quant
WorkingDirectory=/home/ai-quant/ai-quant-system
Environment=PATH=/home/ai-quant/ai-quant-system/venv/bin
ExecStart=/home/ai-quant/ai-quant-system/venv/bin/gunicorn backend.app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-quant
sudo systemctl start ai-quant
sudo systemctl status ai-quant
```

## 数据库配置

### PostgreSQL 优化配置

#### 1. 连接配置

```sql
-- postgresql.conf
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

#### 2. 性能配置

```sql
-- WAL配置
wal_level = replica
wal_buffers = 16MB
checkpoint_completion_target = 0.9

-- 查询优化
random_page_cost = 1.1
seq_page_cost = 1.0
effective_io_concurrency = 200
```

#### 3. 监控配置

```sql
-- 启用统计收集
shared_preload_libraries = 'pg_stat_statements'
track_activities = on
track_counts = on
track_io_timing = on
```

### Redis 配置优化

```conf
# redis.conf 生产环境配置
# 内存配置
maxmemory 1gb
maxmemory-policy allkeys-lru

# 持久化配置
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# 网络配置
timeout 300
tcp-keepalive 300
tcp-backlog 511

# 安全配置
bind 127.0.0.1
requirepass your_redis_password
```

## 监控和日志

### 1. 应用监控

#### Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "ai-quant-app"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: "/metrics"

  - job_name: "postgres"
    static_configs:
      - targets: ["postgres:5432"]

  - job_name: "redis"
    static_configs:
      - targets: ["redis:6379"]
```

#### Grafana 仪表板

导入预配置的监控面板:

```bash
# 启动Grafana
docker run -d -p 3000:3000 --name grafana grafana/grafana

# 导入仪表板配置
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/dashboard.json
```

### 2. 日志管理

#### 日志配置

```python
# logging.conf
[loggers]
keys=root,app,data,api

[handlers]
keys=console,file,error_file

[formatters]
keys=standard,error

[logger_root]
level=INFO
handlers=console,file

[logger_app]
level=INFO
handlers=file
qualname=app
propagate=0

[handler_file]
class=handlers.RotatingFileHandler
args=('logs/app.log', 'a', 10485760, 5)
formatter=standard

[formatter_standard]
format=%(asctime)s [%(levelname)s] %(name)s: %(message)s
```

#### 日志轮转

配置 logrotate `/etc/logrotate.d/ai-quant`:

```
/home/ai-quant/ai-quant-system/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    create 644 ai-quant ai-quant
}
```

## 备份和恢复

### 数据库备份

#### 自动备份脚本

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="ai_quant_db"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 数据库备份
pg_dump -h localhost -U ai_quant_user $DB_NAME > $BACKUP_DIR/backup_$DATE.sql

# 压缩备份文件
gzip $BACKUP_DIR/backup_$DATE.sql

# 清理30天前的备份
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "备份完成: backup_$DATE.sql.gz"
```

#### 设置 cron 任务

```bash
# 编辑crontab
crontab -e

# 每天2点执行备份
0 2 * * * /home/ai-quant/scripts/backup.sh
```

### 数据恢复

```bash
# 1. 停止应用
sudo systemctl stop ai-quant

# 2. 恢复数据库
gunzip backup_20241209_020000.sql.gz
psql -h localhost -U ai_quant_user ai_quant_db < backup_20241209_020000.sql

# 3. 重启应用
sudo systemctl start ai-quant
```

## 安全配置

### 1. 防火墙配置

```bash
# UFW防火墙配置
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432  # 内网PostgreSQL
sudo ufw allow from 10.0.0.0/8 to any port 6379  # 内网Redis
```

### 2. SSL 证书配置

```bash
# 使用Let's Encrypt获取证书
sudo apt install certbot
sudo certbot certonly --webroot -w /var/www/html -d your-domain.com

# 设置证书自动续期
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

### 3. 访问控制

#### API 访问限制

```python
# 中间件配置
RATE_LIMIT_CONFIG = {
    "default": "100/minute",
    "data_sync": "10/minute",
    "heavy_query": "5/minute"
}
```

#### 数据库访问控制

```sql
-- 创建只读用户
CREATE USER readonly_user WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE ai_quant_db TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
```

## 性能优化

### 1. 应用优化

```python
# gunicorn配置
workers = 4  # CPU核心数
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 5
```

### 2. 数据库优化

```sql
-- 创建必要索引
CREATE INDEX CONCURRENTLY idx_sync_tasks_status_created ON sync_tasks(status, created_at);
CREATE INDEX CONCURRENTLY idx_system_logs_level_created ON system_logs(level, created_at);

-- 定期清理
DELETE FROM system_logs WHERE created_at < NOW() - INTERVAL '90 days';
```

### 3. 缓存优化

```python
# Redis缓存配置
CACHE_CONFIG = {
    "default_timeout": 300,
    "key_prefix": "ai_quant:",
    "version": 1
}
```

## 故障处理

### 常见问题解决

#### 1. 应用启动失败

```bash
# 检查日志
docker-compose logs app
# 或
journalctl -u ai-quant -f

# 检查端口占用
netstat -tlnp | grep :8000

# 检查配置文件
python -c "from backend.app.core.config import get_settings; print(get_settings())"
```

#### 2. 数据库连接失败

```bash
# 检查数据库状态
docker-compose exec postgres pg_isready

# 测试连接
psql -h localhost -U ai_quant_user ai_quant_db -c "SELECT 1;"

# 检查连接数
SELECT count(*) FROM pg_stat_activity;
```

#### 3. 性能问题

```bash
# 检查系统资源
htop
iotop
df -h

# 检查数据库性能
SELECT * FROM pg_stat_activity WHERE state = 'active';

# 检查Redis内存使用
redis-cli info memory
```

## 更新和维护

### 应用更新流程

```bash
# 1. 备份数据
./scripts/backup.sh

# 2. 拉取最新代码
git pull origin main

# 3. 更新依赖
pip install -r requirements.txt

# 4. 数据库迁移
alembic upgrade head

# 5. 重启服务
docker-compose restart app
# 或
sudo systemctl restart ai-quant

# 6. 验证更新
curl http://localhost:8000/api/v1/health/ping
```

### 定期维护任务

```bash
# 每周执行的维护脚本 - weekly_maintenance.sh
#!/bin/bash

# 1. 数据库维护
psql -c "VACUUM ANALYZE;"

# 2. 日志清理
find /var/log -name "*.log" -mtime +7 -exec truncate -s 0 {} \;

# 3. 缓存清理
redis-cli FLUSHDB

# 4. Docker清理
docker system prune -f

echo "定期维护完成: $(date)"
```

---

**部署文档版本**: v1.0.0  
**更新时间**: 2024 年 12 月 09 日  
**技术支持**: 运维团队
