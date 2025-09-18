# AI 量化系统 - 部署指南

> 🚀 完整的开发环境和生产环境部署指南

## 📋 目录

- [开发环境部署](#-开发环境部署)
- [生产环境部署](#-生产环境部署)
- [部署验证](#-部署验证)
- [故障排除](#-故障排除)
- [性能优化](#-性能优化)

## 🛠️ 开发环境部署

### 准备工作

1. **安装必要软件**

```bash
# Docker Desktop
# 从官网下载安装: https://www.docker.com/products/docker-desktop

# Node.js (推荐使用nvm管理)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20

# Python 3.11
# 从官网下载安装: https://www.python.org/downloads/
python --version  # 确保版本 >= 3.9
```

2. **克隆项目**

```bash
git clone <your-repository-url>
cd ai-quant-system
```

### 快速启动 (推荐)

```bash
# 1. 配置环境变量
cp env.template .env

# 2. 一键启动所有服务
docker-compose --profile dev up -d

# 3. 等待服务启动完成 (约2-3分钟)
docker-compose logs -f app frontend

# 4. 验证服务状态
curl http://localhost:8000/api/v1/health/ping
curl http://localhost:3000
```

### 分别启动前后端

#### 启动后端

```bash
# 1. 启动数据库服务
docker-compose up -d postgres redis

# 2. 等待数据库启动
sleep 10

# 3. 创建Python虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 4. 安装依赖
pip install -r requirements.txt

# 5. 数据库迁移
alembic upgrade head

# 6. 启动后端
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 启动前端

```bash
# 在新的终端窗口中
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### 开发环境验证

| 服务       | 地址                                     | 状态检查               |
| ---------- | ---------------------------------------- | ---------------------- |
| 前端应用   | http://localhost:3000                    | 显示 React 应用界面    |
| 后端 API   | http://localhost:8000/docs               | 显示 FastAPI 文档      |
| 健康检查   | http://localhost:8000/api/v1/health/ping | 返回{"message":"pong"} |
| 数据库管理 | http://localhost:8080                    | Adminer 登录界面       |
| Redis 管理 | http://localhost:8081                    | Redis Commander 界面   |

## 🚀 生产环境部署

### 服务器准备

1. **系统要求**

   - Ubuntu 20.04+ / CentOS 8+ / Debian 11+
   - CPU: 4 核心+ (推荐 8 核心)
   - 内存: 16GB+ (推荐 32GB)
   - 存储: 100GB+ SSD
   - 网络: 稳定的互联网连接

2. **安装 Docker**

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 重新登录或执行
newgrp docker

# 验证安装
docker --version
docker-compose --version
```

### 生产部署步骤

#### 1. 项目部署

```bash
# 克隆到服务器
git clone <your-repository-url> /opt/ai-quant-system
cd /opt/ai-quant-system

# 设置权限
sudo chown -R $USER:$USER /opt/ai-quant-system
chmod +x scripts/*.sh
```

#### 2. 环境配置

```bash
# 复制环境配置
cp env.template .env.production

# 编辑生产环境配置
nano .env.production
```

**关键生产环境配置**:

```bash
# 安全设置
DEBUG=false
SECRET_KEY=your-super-secret-production-key-at-least-32-characters-long
LOG_LEVEL=WARNING

# 数据库密码 (使用强密码)
DB_PASSWORD=your-very-strong-database-password-2024

# Redis密码
REDIS_PASSWORD=your-redis-password-2024

# 性能设置
WORKERS=4
DB_POOL_SIZE=20
```

#### 3. 创建生产配置

```bash
# 创建生产环境覆盖配置
cat > docker-compose.prod.yml << 'EOF'
version: "3.8"

services:
  app:
    environment:
      - DEBUG=false
      - LOG_LEVEL=WARNING
      - WORKERS=4
    volumes:
      # 生产环境不挂载源码
      - app_data:/app/data
      - app_logs:/app/logs
    restart: always

  frontend:
    environment:
      - NODE_ENV=production
    restart: always

  postgres:
    restart: always
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    restart: always
    volumes:
      - redis_data:/data

  # 禁用开发工具
  adminer:
    profiles: [disabled]

  redis-commander:
    profiles: [disabled]

  # 可选: Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: ai-quant-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro
    depends_on:
      - app
      - frontend
    networks:
      - ai-quant-network
    restart: always
EOF
```

#### 4. 启动生产环境

```bash
# 构建并启动服务
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 查看启动状态
docker-compose ps

# 监控启动日志
docker-compose logs -f app frontend
```

#### 5. 数据库初始化

```bash
# 执行数据库迁移
docker-compose exec app alembic upgrade head

# 验证数据库连接
docker-compose exec app python -c "
from backend.app.database.engine import get_session
with get_session() as session:
    print('数据库连接成功')
"
```

### SSL/HTTPS 配置

```bash
# 安装Certbot
sudo apt update
sudo apt install certbot

# 获取SSL证书 (替换your-domain.com)
sudo certbot certonly --standalone -d your-domain.com

# 创建SSL目录并复制证书
mkdir -p ssl
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/
sudo chown $USER:$USER ssl/*

# 创建Nginx SSL配置
cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    upstream frontend {
        server frontend:80;
    }

    # HTTP重定向到HTTPS
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS配置
    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/ssl/certs/fullchain.pem;
        ssl_certificate_key /etc/ssl/certs/privkey.pem;

        # 前端
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # 后端API
        location /api/ {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # WebSocket
        location /ws {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
    }
}
EOF
```

## ✅ 部署验证

### 自动化验证脚本

```bash
# 创建验证脚本
cat > scripts/verify_deployment.sh << 'EOF'
#!/bin/bash

echo "🔍 开始验证AI量化系统部署..."

# 检查服务状态
echo "1. 检查Docker服务状态"
docker-compose ps

# 检查健康状态
echo "2. 检查API健康状态"
curl -f http://localhost:8000/api/v1/health/ping || echo "❌ 后端健康检查失败"

# 检查前端
echo "3. 检查前端状态"
curl -f http://localhost:3000 > /dev/null || echo "❌ 前端检查失败"

# 检查数据库连接
echo "4. 检查数据库连接"
docker-compose exec -T app python -c "
from backend.app.database.engine import get_session
try:
    with get_session() as session:
        print('✅ 数据库连接成功')
except Exception as e:
    print(f'❌ 数据库连接失败: {e}')
"

# 检查Redis连接
echo "5. 检查Redis连接"
docker-compose exec -T redis redis-cli ping || echo "❌ Redis连接失败"

echo "🎉 验证完成！"
EOF

chmod +x scripts/verify_deployment.sh
./scripts/verify_deployment.sh
```

### 手动验证清单

- [ ] **前端应用**: http://localhost:3000 正常访问
- [ ] **API 文档**: http://localhost:8000/docs 正常显示
- [ ] **健康检查**: `curl http://localhost:8000/api/v1/health/ping` 返回 pong
- [ ] **数据库**: 能正常连接和查询
- [ ] **Redis**: 缓存功能正常
- [ ] **WebSocket**: 实时功能正常
- [ ] **日志**: 无严重错误日志
- [ ] **性能**: 响应时间符合预期

## 🔧 故障排除

### 常见问题及解决方案

#### 1. 服务启动失败

```bash
# 检查端口占用
sudo netstat -tlnp | grep :8000
sudo netstat -tlnp | grep :3000

# 检查Docker状态
docker system info
docker-compose ps

# 查看详细错误日志
docker-compose logs app frontend postgres redis
```

#### 2. 数据库连接失败

```bash
# 检查PostgreSQL状态
docker-compose exec postgres pg_isready -U ai_quant_user -d ai_quant_db

# 检查数据库日志
docker-compose logs postgres

# 重置数据库
docker-compose down
docker volume rm ai-quant-postgres-data
docker-compose up -d postgres
```

#### 3. 前端构建失败

```bash
# 检查Node.js版本
node --version
npm --version

# 清理缓存重新构建
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

#### 4. 内存不足

```bash
# 检查系统资源
free -h
df -h
docker stats

# 优化Docker配置
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## ⚡ 性能优化

### 生产环境优化

```bash
# 1. 启用Redis持久化
echo 'save 900 1' >> docker/redis/redis.conf

# 2. PostgreSQL性能调优
cat >> docker/postgres/postgresql.conf << 'EOF'
# 内存设置
shared_buffers = 512MB
effective_cache_size = 2GB
work_mem = 64MB

# 连接设置
max_connections = 200

# 写入优化
wal_buffers = 16MB
checkpoint_completion_target = 0.9
EOF

# 3. 应用性能监控
docker-compose exec app pip install psutil
```

### 监控设置

```bash
# 创建监控脚本
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "$(date): 系统资源使用情况"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    echo "---"
    sleep 60
done
EOF

chmod +x scripts/monitor.sh
# 后台运行监控
nohup ./scripts/monitor.sh > logs/monitor.log 2>&1 &
```

## 🔄 维护指南

### 日常维护

```bash
# 定期备份数据库
docker-compose exec postgres pg_dump -U ai_quant_user ai_quant_db > backups/backup_$(date +%Y%m%d_%H%M%S).sql

# 清理Docker空间
docker system prune -f
docker volume prune -f

# 更新应用
git pull origin main
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 查看日志
tail -f logs/app.log
docker-compose logs -f --tail=100 app
```

### 安全维护

```bash
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 更新Docker镜像
docker-compose pull
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 检查安全漏洞
docker scan postgres:16-alpine
docker scan redis:7-alpine
```

---

**🎯 部署成功标志**:

- 前端正常访问 ✅
- API 响应正常 ✅
- 数据库连接成功 ✅
- 无严重错误日志 ✅

如有问题，请查看 [故障排除](#-故障排除) 或提交 Issue。
