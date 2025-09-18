# 多阶段构建 Dockerfile for React Frontend - 优化版本

# Build stage - 使用更新的Node.js版本以获得更好的性能
FROM node:20-alpine AS builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖（包括Python）
RUN apk add --no-cache python3 make g++

# 复制package文件（利用Docker缓存）
COPY package*.json ./

# 设置npm配置以提高安装速度
RUN npm config set registry https://registry.npmmirror.com/

# 安装所有依赖（包括devDependencies，用于构建）
RUN npm ci --silent && npm cache clean --force

# 复制源代码
COPY . .

# 设置生产环境变量
ENV NODE_ENV=production
ENV VITE_API_URL=/api
ENV VITE_WS_URL=/ws

# 构建应用
RUN npm run build

# 验证构建文件
RUN ls -la /app/dist

# Production stage - 使用优化的nginx镜像
FROM nginx:1.25-alpine AS production

# 安装必要工具
RUN apk add --no-cache tzdata

# 设置时区
ENV TZ=Asia/Shanghai

# 复制构建文件到nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制nginx配置文件
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 添加nginx性能优化配置
RUN echo 'user nginx;' > /etc/nginx/nginx.conf && \
    echo 'worker_processes auto;' >> /etc/nginx/nginx.conf && \
    echo 'error_log /var/log/nginx/error.log warn;' >> /etc/nginx/nginx.conf && \
    echo 'pid /var/run/nginx.pid;' >> /etc/nginx/nginx.conf && \
    echo '' >> /etc/nginx/nginx.conf && \
    echo 'events {' >> /etc/nginx/nginx.conf && \
    echo '    worker_connections 1024;' >> /etc/nginx/nginx.conf && \
    echo '    use epoll;' >> /etc/nginx/nginx.conf && \
    echo '    multi_accept on;' >> /etc/nginx/nginx.conf && \
    echo '}' >> /etc/nginx/nginx.conf && \
    echo '' >> /etc/nginx/nginx.conf && \
    echo 'http {' >> /etc/nginx/nginx.conf && \
    echo '    include /etc/nginx/mime.types;' >> /etc/nginx/nginx.conf && \
    echo '    default_type application/octet-stream;' >> /etc/nginx/nginx.conf && \
    echo '' >> /etc/nginx/nginx.conf && \
    echo '    # Performance optimizations' >> /etc/nginx/nginx.conf && \
    echo '    sendfile on;' >> /etc/nginx/nginx.conf && \
    echo '    tcp_nopush on;' >> /etc/nginx/nginx.conf && \
    echo '    tcp_nodelay on;' >> /etc/nginx/nginx.conf && \
    echo '    keepalive_timeout 65;' >> /etc/nginx/nginx.conf && \
    echo '    types_hash_max_size 2048;' >> /etc/nginx/nginx.conf && \
    echo '' >> /etc/nginx/nginx.conf && \
    echo '    # Gzip compression' >> /etc/nginx/nginx.conf && \
    echo '    gzip on;' >> /etc/nginx/nginx.conf && \
    echo '    gzip_vary on;' >> /etc/nginx/nginx.conf && \
    echo '    gzip_min_length 1000;' >> /etc/nginx/nginx.conf && \
    echo '    gzip_comp_level 6;' >> /etc/nginx/nginx.conf && \
    echo '    gzip_types' >> /etc/nginx/nginx.conf && \
    echo '        text/plain' >> /etc/nginx/nginx.conf && \
    echo '        text/css' >> /etc/nginx/nginx.conf && \
    echo '        text/xml' >> /etc/nginx/nginx.conf && \
    echo '        text/javascript' >> /etc/nginx/nginx.conf && \
    echo '        application/javascript' >> /etc/nginx/nginx.conf && \
    echo '        application/xml+rss' >> /etc/nginx/nginx.conf && \
    echo '        application/json;' >> /etc/nginx/nginx.conf && \
    echo '' >> /etc/nginx/nginx.conf && \
    echo '    include /etc/nginx/conf.d/*.conf;' >> /etc/nginx/nginx.conf && \
    echo '}' >> /etc/nginx/nginx.conf

# 创建非特权用户
RUN addgroup -g 1001 -S frontend && \
    adduser -S frontend -u 1001

# 设置文件权限
RUN chown -R frontend:frontend /usr/share/nginx/html && \
    chown -R frontend:frontend /var/cache/nginx && \
    chown -R frontend:frontend /var/log/nginx && \
    chown -R frontend:frontend /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && \
    chown -R frontend:frontend /var/run/nginx.pid

# 切换到非特权用户
USER frontend

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

# 暴露端口
EXPOSE 80

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
