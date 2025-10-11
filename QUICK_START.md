# AI 量化系统 - 快速启动指南

> 🚀 5 分钟快速启动 AI 量化交易系统，立即开始量化投资之旅

## 📋 前提条件

在开始之前，请确保你的系统已安装：

- **Docker Desktop** 4.0+ ([下载地址](https://www.docker.com/products/docker-desktop))
- **Docker Compose** V1 或 V2 (支持 `docker-compose` 或 `docker compose` 命令)
- **Git** 2.0+
- **至少 8GB RAM** 和 10GB 硬盘空间

## 🚀 一键启动（推荐）

### 方式一：使用启动脚本（最简单）

```bash
# 1. 克隆项目
git clone <your-repository-url>
cd ai-quant-system

# 2. 给脚本执行权限
chmod +x scripts/start-dev.sh

# 3. 一键启动（包含管理工具）
./scripts/start-dev.sh start --with-tools

# 4. 等待启动完成（约2-3分钟）
```

### 方式二：Docker Compose 快速启动

```bash
# 1. 克隆项目
git clone <your-repository-url>
cd ai-quant-system

# 2. 配置环境变量
cp env.template .env

# 3. 启动所有服务
docker-compose --profile dev up -d

# 4. 查看启动状态
docker-compose ps
```

## ✅ 验证安装

服务启动后，访问以下地址验证安装成功：

| 服务              | 地址                                     | 说明               |
| ----------------- | ---------------------------------------- | ------------------ |
| **🌐 前端应用**   | http://localhost:3000                    | React 应用主界面   |
| **📡 后端 API**   | http://localhost:8000/docs               | FastAPI 交互文档   |
| **💓 健康检查**   | http://localhost:8000/api/v1/health/ping | 系统状态检查       |
| **🗃️ 数据库管理** | http://localhost:8080                    | Adminer 数据库工具 |
| **🔑 缓存管理**   | http://localhost:8081                    | Redis Commander    |

## 🛠️ 前后端分别启动

### 后端启动

```bash
# 1. 启动基础服务
docker-compose up -d postgres redis

# 2. 进入后端开发
cd backend
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证依赖安装
python ../scripts/verify_p1_dependencies.py

# 5. 数据库迁移
alembic upgrade head

# 6. 启动后端服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端启动

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装Node.js依赖
npm install

# 3. 启动开发服务器
npm run dev

# 前端将在 http://localhost:3000 启动
```

## 🚀 开始使用

### 1. 数据概览页面

访问 http://localhost:3000 查看系统概览，包括：

- 系统状态监控
- 数据同步状态
- 快速操作面板

### 2. API 测试

访问 http://localhost:8000/docs 进行 API 测试：

- 数据获取接口
- 回测功能接口
- 系统监控接口

### 3. 基础功能验证

```bash
# 测试API连接
curl http://localhost:8000/api/v1/health/ping

# 查看系统状态
curl http://localhost:8000/api/v1/health/status

# 测试数据接口
curl http://localhost:8000/api/v1/data/stocks/000001?start_date=2024-01-01&end_date=2024-01-31
```

## 🔧 常用操作命令

### 服务管理

```bash
# 使用脚本管理（推荐）
./scripts/start-dev.sh status      # 查看服务状态
./scripts/start-dev.sh logs        # 查看日志
./scripts/start-dev.sh stop        # 停止服务
./scripts/start-dev.sh restart     # 重启服务

# 或使用Docker Compose
docker-compose ps                  # 查看服务状态
docker-compose logs -f app         # 查看应用日志
docker-compose down                # 停止所有服务
docker-compose restart app         # 重启应用服务
```

### 数据库操作

```bash
# 连接数据库
docker-compose exec postgres psql -U ai_quant_user -d ai_quant_db

# 数据库迁移
docker-compose exec app alembic upgrade head

# 查看迁移历史
docker-compose exec app alembic history
```

### 进入容器调试

```bash
# 进入后端容器
./scripts/start-dev.sh exec app

# 进入数据库容器
./scripts/start-dev.sh exec postgres

# 进入Redis容器
./scripts/start-dev.sh exec redis
```

## 🔍 故障排除

### 1. Docker 相关问题

```bash
# 检查Docker状态
docker --version
docker-compose --version
docker info

# 清理Docker资源
docker system prune -f
docker volume prune -f
```

### 2. 端口占用问题

```bash
# 检查端口占用
lsof -ti:8000  # 后端端口
lsof -ti:3000  # 前端端口
lsof -ti:5432  # 数据库端口

# 杀死占用进程
kill -9 $(lsof -ti:8000)
```

### 3. 服务启动失败

```bash
# 查看详细错误日志
docker-compose logs app
docker-compose logs postgres
docker-compose logs redis

# 重新构建镜像
docker-compose build --no-cache app
```

### 4. 依赖问题

```bash
# Python依赖验证
python scripts/verify_p1_dependencies.py

# Node.js依赖重新安装
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## 📚 下一步

成功启动后，你可以：

1. **查看详细文档**：

   - [完整 README](./README.md) - 项目详细介绍
   - [前端开发指南](./frontend/README.md) - 前端开发文档
   - [部署指南](./DEPLOYMENT_GUIDE.md) - 生产环境部署

2. **开始开发**：

   - 后端 API 开发：`backend/` 目录
   - 前端界面开发：`frontend/` 目录
   - 添加新功能或修改现有功能

3. **学习使用**：
   - 数据获取和处理
   - 策略开发和回测
   - 系统监控和维护

## 🆘 获取帮助

- **查看帮助**：`./scripts/start-dev.sh help`
- **API 文档**：http://localhost:8000/docs
- **系统监控**：http://localhost:8000/api/v1/health
- **日志查看**：`docker-compose logs -f app`

## ⚡ 性能提示

- 首次启动需要下载镜像，可能需要 5-10 分钟
- 确保有足够的内存（推荐 16GB）
- SSD 硬盘可以显著提升性能
- 关闭不必要的程序以释放资源

---

**🎉 恭喜！** 你已经成功启动了 AI 量化系统，可以开始你的量化投资之旅了！

**⚠️ 风险提示**：本系统仅用于量化研究和学习，不构成投资建议。投资有风险，入市需谨慎。
