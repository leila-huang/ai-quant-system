# AI 量化系统 - 前端应用

> 基于 React + TypeScript + Vite 的现代化量化交易前端界面

## 📋 项目概述

AI量化系统前端是一个现代化的React应用，提供直观的量化交易界面，包含数据可视化、回测工作台、策略管理等核心功能。

### 🚀 核心特性

- **现代化技术栈**: React 18 + TypeScript + Vite
- **UI组件库**: Ant Design 5.x + 自定义组件
- **数据可视化**: ECharts 专业图表库
- **状态管理**: Zustand 轻量级状态管理
- **实时通信**: WebSocket + 自动重连机制
- **性能优化**: 代码分割 + 懒加载 + React.memo
- **测试覆盖**: Vitest + React Testing Library
- **容器化**: Docker 多阶段构建

### 🛠️ 技术栈

```
前端技术栈:
├── React 18            # 用户界面框架
├── TypeScript         # 静态类型检查
├── Vite              # 构建工具和开发服务器
├── Ant Design        # UI组件库
├── ECharts           # 图表可视化
├── Zustand           # 状态管理
├── React Router      # 路由管理
├── Axios             # HTTP客户端
├── Day.js            # 日期处理
└── Vitest            # 测试框架
```

## 🏗️ 项目结构

```
frontend/
├── public/                 # 静态资源
├── src/
│   ├── components/         # 可复用组件
│   │   ├── Charts/        # 图表组件
│   │   ├── ErrorBoundary.tsx
│   │   ├── MainLayout.tsx
│   │   └── ...
│   ├── pages/             # 页面组件
│   │   ├── Dashboard.tsx  # 数据概览
│   │   ├── Backtest.tsx   # 回测工作台
│   │   ├── Strategy.tsx   # 策略管理
│   │   ├── AI.tsx         # AI助手
│   │   └── Trading.tsx    # 模拟交易
│   ├── services/          # API服务层
│   │   ├── api.ts         # 基础API客户端
│   │   ├── backtestApi.ts # 回测相关API
│   │   ├── dataApi.ts     # 数据相关API
│   │   └── websocket.ts   # WebSocket服务
│   ├── stores/            # Zustand状态管理
│   │   ├── appStore.ts    # 应用状态
│   │   ├── backtestStore.ts # 回测状态
│   │   ├── dataStore.ts   # 数据状态
│   │   └── websocketStore.ts # WebSocket状态
│   ├── types/             # TypeScript类型定义
│   ├── utils/             # 工具函数
│   └── tests/             # 测试文件
├── Dockerfile             # Docker构建配置
├── nginx.conf             # Nginx配置
├── package.json           # 项目依赖
├── vite.config.ts         # Vite配置
├── vitest.config.ts       # Vitest测试配置
└── tsconfig.json          # TypeScript配置
```

## 🚀 开发环境启动

### 环境要求

- **Node.js**: 18.x 或更高版本 (推荐 20.x)
- **npm**: 9.x 或更高版本
- **内存**: 至少 4GB 可用内存

### 快速启动

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev
```

开发服务器将在 http://localhost:3000 启动。

### 详细步骤

#### 1. 检查Node.js版本

```bash
# 检查Node.js版本
node --version  # 应该 >= 18.0.0
npm --version   # 应该 >= 9.0.0

# 如果版本过低，推荐使用nvm安装最新版本
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20
```

#### 2. 安装项目依赖

```bash
# 清理缓存（如果之前安装过）
rm -rf node_modules package-lock.json

# 安装所有依赖
npm install

# 或者使用yarn（如果偏好）
yarn install
```

#### 3. 环境配置

前端应用通过环境变量配置API连接：

```bash
# 开发环境默认配置（自动生效）
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/ws

# 如需自定义配置，创建 .env.local 文件
echo "VITE_API_URL=http://your-backend-host:8000/api" > .env.local
echo "VITE_WS_URL=ws://your-backend-host:8000/ws" >> .env.local
```

#### 4. 启动开发服务器

```bash
# 启动前端开发服务器
npm run dev

# 或指定端口
npm run dev -- --port 3001

# 或在后台运行
nohup npm run dev &
```

#### 5. 验证启动

访问以下链接验证前端启动成功：

- **前端应用**: http://localhost:3000
- **开发者工具**: 浏览器F12查看无错误信息

## 🧪 开发工具和命令

### 开发命令

```bash
# 开发服务器
npm run dev              # 启动开发服务器（热更新）
npm run preview          # 预览生产构建结果

# 代码质量
npm run lint             # ESLint代码检查
npm run lint:fix         # 自动修复ESLint问题

# 构建部署
npm run build            # 生产环境构建
npm run build:analyze    # 构建并分析包大小
```

### 测试命令

```bash
# 单元测试
npm run test             # 交互式测试模式
npm run test:run         # 运行所有测试
npm run test:coverage    # 生成测试覆盖率报告
npm run test:ui          # 启动测试UI界面

# 专项测试
npm run test:unit        # 单元测试
npm run test:integration # 集成测试
npm run test:performance # 性能测试
```

### 类型检查

```bash
# TypeScript类型检查
npx tsc --noEmit        # 检查类型错误
npx tsc --watch         # 监听模式类型检查
```

## 🔧 开发配置详解

### Vite配置 (vite.config.ts)

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000', // 后端API代理
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'antd-vendor': ['antd', '@ant-design/icons'],
          'chart-vendor': ['echarts', 'echarts-for-react'],
        },
      },
    },
  },
});
```

### API代理配置

开发环境下，前端会自动将 `/api/*` 请求代理到后端服务：

```
前端请求: http://localhost:3000/api/v1/health
实际请求: http://localhost:8000/api/v1/health
```

### 环境变量

支持的环境变量：

```bash
# API配置
VITE_API_URL=http://localhost:8000/api    # 后端API地址
VITE_WS_URL=ws://localhost:8000/ws        # WebSocket地址

# 功能开关
VITE_ENABLE_MOCK=false                    # 是否启用模拟数据
VITE_ENABLE_DEVTOOLS=true                 # 是否启用开发工具
```

## 📱 页面功能说明

### 1. 数据概览 (Dashboard)

- 系统状态监控
- 数据源连接状态
- 快速操作面板
- 实时数据更新

### 2. 回测工作台 (Backtest)

- 策略参数配置
- 回测结果展示
- 历史回测记录
- 性能分析图表

### 3. 策略管理 (Strategy)

- 策略创建和编辑
- 策略性能监控
- 策略比较分析

### 4. AI助手 (AI)

- 智能对话界面
- 策略建议
- 市场分析

### 5. 模拟交易 (Trading)

- 模拟交易界面
- 持仓管理
- 交易历史

## 🐳 Docker开发

### 使用Docker开发

```bash
# 构建前端Docker镜像
docker build -t ai-quant-frontend .

# 运行前端容器
docker run -p 3000:80 ai-quant-frontend

# 使用docker-compose（推荐）
docker-compose up frontend
```

### Docker配置说明

- **多阶段构建**: 减少最终镜像大小
- **Nginx优化**: Gzip压缩、静态资源缓存
- **健康检查**: 自动监控容器状态

## 🔍 故障排除

### 常见问题

#### 1. 端口占用

```bash
# 检查端口占用
lsof -ti:3000
# 或
netstat -tulpn | grep :3000

# 杀死占用进程
kill -9 $(lsof -ti:3000)

# 使用其他端口启动
npm run dev -- --port 3001
```

#### 2. 依赖安装失败

```bash
# 清理npm缓存
npm cache clean --force

# 删除node_modules重新安装
rm -rf node_modules package-lock.json
npm install

# 使用国内镜像源
npm config set registry https://registry.npmmirror.com/
npm install
```

#### 3. TypeScript类型错误

```bash
# 检查TypeScript版本
npx tsc --version

# 重新生成类型声明
rm -rf node_modules/@types
npm install

# 检查配置文件
npx tsc --showConfig
```

#### 4. API连接问题

```bash
# 检查后端服务状态
curl http://localhost:8000/api/v1/health/ping

# 检查代理配置
# 确保vite.config.ts中的proxy配置正确
```

#### 5. 热更新不工作

```bash
# 检查文件监听
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# 或使用polling模式
npm run dev -- --force
```

## 📊 性能优化

### 1. 构建优化

- **代码分割**: 自动分包减少初始加载
- **Tree Shaking**: 移除未使用代码
- **压缩优化**: 生产环境代码压缩

### 2. 运行时优化

- **React.memo**: 防止无效重渲染
- **懒加载**: 路由和组件按需加载
- **虚拟滚动**: 大数据列表优化

### 3. 网络优化

- **HTTP缓存**: 静态资源缓存策略
- **Gzip压缩**: 减少传输大小
- **CDN**: 静态资源CDN分发

## 📚 开发指南

### 代码规范

项目使用严格的代码规范：

```bash
# 代码格式化
npm run lint              # 检查代码规范
npm run lint:fix          # 自动修复问题

# 类型检查
npx tsc --noEmit         # TypeScript类型检查
```

### 组件开发

创建新组件的步骤：

```bash
# 1. 创建组件文件
mkdir src/components/NewComponent
touch src/components/NewComponent/index.tsx
touch src/components/NewComponent/index.test.tsx

# 2. 添加到组件导出
echo "export { default as NewComponent } from './NewComponent';" >> src/components/index.ts
```

### API集成

添加新的API接口：

```typescript
// src/services/newApi.ts
import { apiClient } from './api';

export const newApi = {
  getData: () => apiClient.get('/new-endpoint'),
  postData: (data: any) => apiClient.post('/new-endpoint', data),
};
```

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
npm run test:run

# 监听模式测试
npm run test

# 生成覆盖率报告
npm run test:coverage

# 测试UI界面
npm run test:ui
```

### 编写测试

```typescript
// 组件测试示例
import { render, screen } from '@testing-library/react';
import { NewComponent } from './NewComponent';

test('renders component correctly', () => {
  render(<NewComponent />);
  expect(screen.getByText('Expected Text')).toBeInTheDocument();
});
```

## 🚀 生产部署

### 构建生产版本

```bash
# 生产构建
npm run build

# 预览构建结果
npm run preview

# 分析包大小
npx vite-bundle-analyzer
```

### 部署验证

```bash
# 检查构建产物
ls -la dist/

# 验证资源完整性
find dist/ -name "*.js" -o -name "*.css" | wc -l
```

## 📖 更多资源

- **API文档**: http://localhost:8000/docs (后端运行时)
- **技术文档**: [项目Wiki](./docs/)
- **问题反馈**: [GitHub Issues](./issues/)

---

**开发状态**: ✅ Ready for Development
**最后更新**: 2024年12月
