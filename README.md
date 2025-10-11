# AI 量化系统 (AI-Quant-System)

> 基于 AKShare 的现代化 A 股量化交易平台 - P1 级核心业务逻辑

## 🚀 快速开始

**新用户推荐**: 查看 [📋 快速启动指南](./QUICK_START.md) - 5 分钟完成系统部署

**一键启动命令**:

```bash
git clone <repository-url>
cd ai-quant-system
chmod +x scripts/start-dev.sh
./scripts/start-dev.sh start --with-tools
```

访问地址: [前端应用](http://localhost:3000) | [API 文档](http://localhost:8000/docs)

## 📚 文档导航

| 文档                                      | 说明              | 适用人群             |
| ----------------------------------------- | ----------------- | -------------------- |
| [📋 快速启动指南](./QUICK_START.md)       | 5 分钟快速上手    | 新用户、快速体验     |
| [🚀 部署指南](./DEPLOYMENT_GUIDE.md)      | 完整生产环境部署  | 运维人员、生产部署   |
| [💻 前端开发指南](./frontend/README.md)   | 前端开发详细说明  | 前端开发者           |
| [📖 API 文档](http://localhost:8000/docs) | 后端 API 接口文档 | 后端开发者、接口调用 |

## 项目概述

AI 量化系统是一个完整的 A 股量化交易平台，采用现代化技术栈构建高性能、高可用的量化投资基础设施。

### 核心特性

- 🚀 **高性能数据获取**: 基于 AKShare 免费数据源，数据获取成功率>95%
- 📊 **列式存储**: Parquet 格式存储历史数据，查询性能优异
- ⚡ **现代化 API**: FastAPI 异步框架，响应时间<200ms
- 🐘 **可靠数据库**: PostgreSQL 存储业务数据，支持 ACID 事务
- 🐳 **容器化部署**: Docker 一键启动开发环境
- 📈 **量化回测**: vectorbt 高性能向量化回测引擎，支持多策略并行
- 🤖 **机器学习**: 集成 XGBoost 预测模型，支持多种预测目标
- 📊 **技术指标**: 完整的技术指标计算库(MA、RSI、MACD、布林带等)
- ⚡ **特征工程**: 自动化特征生成和选择流水线
- 📋 **A 股约束**: T+1、涨跌停、停牌等 A 股市场约束建模
- 💰 **成本建模**: 精确的交易成本和滑点建模
- 📑 **专业报告**: 完整的回测报告生成，包含风险收益分析
- 🎯 **智能风控**: 多维度风险管理系统(P3 阶段)

## 技术栈

### 后端技术

- **Web 框架**: FastAPI + Uvicorn
- **数据获取**: AKShare (A 股数据)
- **数据处理**: Pandas + NumPy
- **数据存储**: PostgreSQL + Redis + Parquet
- **ORM**: SQLAlchemy 2.0
- **异步编程**: asyncio/aioredis
- **回测引擎**: vectorbt (高性能向量化回测)
- **机器学习**: XGBoost + scikit-learn
- **技术指标**: TA-Lib + 自定义指标库
- **数值计算**: NumPy + Numba (JIT 加速)
- **数据可视化**: Matplotlib + Seaborn + Plotly
- **并行计算**: joblib + 多进程

### 前端技术 (P2 阶段)

- **框架**: React + TypeScript
- **构建工具**: Vite
- **UI 组件**: Ant Design
- **图表库**: ECharts
- **状态管理**: Zustand

### 开发工具

- **代码质量**: Black + isort + flake8 + mypy
- **测试框架**: pytest + pytest-asyncio
- **容器化**: Docker + Docker Compose
- **数据库迁移**: Alembic
- **数据分析**: Jupyter Lab + IPython
- **交互式图表**: Plotly + Dash
- **快速原型**: Streamlit
- **性能分析**: memory_profiler + line_profiler

## 项目结构

```
ai-quant-system/
├── backend/                    # 后端应用
│   ├── app/                   # FastAPI应用
│   │   ├── api/              # API路由
│   │   ├── core/             # 核心配置
│   │   └── main.py           # 应用入口
│   └── src/                   # 核心业务逻辑
│       ├── data/             # 数据适配器
│       ├── storage/          # 存储引擎
│       ├── database/         # 数据库模型
│       ├── models/           # 数据模型
│       └── engine/           # P1级核心引擎
│           ├── features/     # 特征工程模块
│           ├── modeling/     # 机器学习建模
│           ├── backtest/     # 回测引擎
│           └── utils/        # 工具函数
├── data/                      # 数据存储
│   └── parquet/              # Parquet文件存储
├── tests/                     # 测试文件
│   ├── integration/          # 集成测试
│   └── performance/          # 性能测试
├── docker/                    # Docker配置
├── scripts/                   # 脚本文件
├── requirements.txt           # Python依赖
└── docker-compose.yml         # 容器编排
```

## 💻 开发环境设置

> 详细指南请参考 [📋 快速启动指南](./QUICK_START.md)

### 环境要求

- **Docker Desktop** 4.0+
- **Docker Compose** V1 或 V2 (支持 `docker-compose` 或 `docker compose` 命令)
- **Python** 3.9+ (本地开发)
- **Node.js** 18+ (前端开发)
- **内存**: 8GB+ (推荐 16GB)

### 三种启动方式

1. **一键启动** (推荐新手)

   ```bash
   ./scripts/start-dev.sh start --with-tools
   ```

2. **Docker Compose**

   ```bash
   cp env.template .env
   docker-compose --profile dev up -d
   ```

3. **本地开发** (高级用户)
   - 后端: `uvicorn backend.app.main:app --reload`
   - 前端: `cd frontend && npm run dev`

### 验证安装

服务启动后访问以下地址：

| 服务       | 地址                                | 说明             |
| ---------- | ----------------------------------- | ---------------- |
| 前端应用   | http://localhost:3000               | React 主界面     |
| API 文档   | http://localhost:8000/docs          | FastAPI 交互文档 |
| 健康检查   | http://localhost:8000/api/v1/health | 系统状态         |
| 数据库管理 | http://localhost:8080               | Adminer 工具     |

**数据库信息**: `localhost:5432 / ai_quant_db / ai_quant_user / ai_quant_password`

## 🚀 生产环境部署

> 完整部署指南请参考 [📋 部署指南](./DEPLOYMENT_GUIDE.md)

### 服务器要求

- **CPU**: 4 核+ (推荐 8 核)
- **内存**: 16GB+ (推荐 32GB)
- **存储**: 100GB+ SSD
- **系统**: Ubuntu 20.04+ / CentOS 8+

### 快速部署

```bash
# 1. 服务器准备
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh

# 2. 项目部署
git clone <repository-url> /opt/ai-quant-system
cd /opt/ai-quant-system

# 3. 配置生产环境
cp env.template .env.production
# 编辑 .env.production 设置生产密码和密钥

# 4. 启动生产服务
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 5. 验证部署
curl http://localhost:8000/api/v1/health/ping
```

### 安全建议

- 修改默认密码和密钥
- 配置 HTTPS 和防火墙
- 定期备份数据库
- 监控系统日志

## 开发指南

### 代码规范

项目使用严格的代码质量标准：

```bash
# 代码格式化
black backend/ tests/
isort backend/ tests/

# 类型检查
mypy backend/

# 代码检查
flake8 backend/ tests/

# 运行测试
pytest tests/ -v --cov=backend
```

### 数据库操作

```bash
# 创建迁移
alembic revision --autogenerate -m "描述信息"

# 执行迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

## API 文档

### 核心接口

#### 健康检查

```http
GET /health
```

#### 数据同步

```http
POST /api/v1/data/sync
Content-Type: application/json

{
    "symbols": ["000001", "000002"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
}
```

#### 查询股票数据

```http
GET /api/v1/data/stocks/000001?start_date=2024-01-01&end_date=2024-12-31
```

详细 API 文档请访问: http://localhost:8000/docs

## 性能指标

### P0 级目标

- ✅ 数据获取成功率: >95%
- ✅ API 响应时间: <200ms
- ✅ 数据质量校验通过率: >99%
- ✅ Docker 环境启动成功率: 100%

### 系统容量

- 支持同时获取 100+股票数据
- 单日数据处理能力: 10 万+条记录
- 并发 API 请求: 100+/秒
- 数据存储: TB 级历史数据

## 📝 常用命令

### 开发测试

```bash
# 运行后端测试
pytest tests/ -v --cov=backend

# 运行前端测试
cd frontend && npm run test

# 代码格式化
black backend/ tests/
isort backend/ tests/

# 类型检查
mypy backend/

# 前端代码检查
cd frontend && npm run lint
```

### 数据库操作

```bash
# 创建迁移文件
alembic revision --autogenerate -m "描述信息"

# 执行迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1

# 查看迁移历史
alembic history
```

### Docker 管理

```bash
# 查看服务状态
docker-compose ps

# 重启服务
docker-compose restart app

# 查看日志
docker-compose logs -f app frontend

# 清理未使用的镜像
docker system prune -f
```

## 监控和日志

- **应用日志**: logs/app.log
- **API 访问日志**: logs/access.log
- **数据质量日志**: logs/data_quality.log
- **系统监控**: Prometheus + Grafana (P1 阶段)

## 故障排除

### 常见问题

1. **数据获取失败**

   - 检查网络连接
   - 验证 AKShare 服务状态
   - 查看重试机制日志

2. **数据库连接失败**

   - 确认 PostgreSQL 服务运行
   - 检查连接参数配置
   - 验证用户权限

3. **API 响应慢**
   - 检查 Redis 缓存状态
   - 优化 SQL 查询
   - 调整连接池大小

## 🗓️ 版本计划

- ✅ **P0 (Week 1-4)**: 数据基础设施
- ✅ **P1 (Week 5-8)**: 特征工程 + XGBoost 建模 + vectorbt 回测引擎 + A 股约束建模
- ✅ **P2 (Week 9-12)**: React 前端 + ECharts 可视化 + WebSocket 实时通信
- ⏳ **P3 (Week 13-16)**: 纸上交易 + 智能风控
- ⏳ **P4 (Week 17-20)**: AI 增强功能 + 高级策略

### P1 级功能详情 (已完成 ✅)

- ✅ **依赖环境**: vectorbt + XGBoost + scikit-learn + TA-Lib + numba
- ✅ **技术指标**: MA、EMA、RSI、MACD、布林带、KDJ、Williams %R、成交量均线
- ✅ **特征工程**: 自动化特征生成、缩放、选择和存储流水线
- ✅ **机器学习**: XGBoost 预测模型，支持回归和分类任务
- ✅ **回测引擎**: 高性能向量化回测，支持多策略并行执行
- ✅ **市场约束**: T+1、涨跌停、停牌等 A 股特有规则完整建模
- ✅ **成本建模**: 精确的手续费、印花税、过户费、滑点计算
- ✅ **专业报告**: HTML/JSON 多格式风险收益分析报告生成

### P2 级功能详情 (已完成 ✅)

- ✅ **前端技术栈**: React 18 + TypeScript + Vite + Ant Design
- ✅ **数据可视化**: ECharts 专业图表库，支持 K 线图、折线图、柱状图
- ✅ **实时通信**: WebSocket 连接管理，支持断线重连和消息订阅
- ✅ **状态管理**: Zustand 轻量级状态管理，支持持久化和开发工具
- ✅ **页面功能**:
  - 数据概览 Dashboard - 系统状态监控和快速操作
  - 回测工作台 - 策略配置、结果展示、历史管理
  - 策略管理 - 策略 CRUD、性能监控
  - AI 助手 - 智能对话界面
  - 模拟交易 - 纸上交易操作面板
- ✅ **性能优化**: 代码分离、懒加载、React.memo 优化
- ✅ **测试覆盖**: 36 个单元测试通过，性能测试框架完备
- ✅ **容器化部署**: Docker 多阶段构建，Nginx 生产级配置

## 贡献指南

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交变更: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系我们

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 开发文档: [Wiki]

---

**⚠️ 风险提示**: 本系统仅用于量化研究和学习目的，不构成投资建议。投资有风险，入市需谨慎。
