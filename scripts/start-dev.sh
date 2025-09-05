#!/bin/bash
# AI量化系统开发环境启动脚本

set -e  # 遇到错误时停止执行

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查Docker是否运行
    if ! docker info &> /dev/null; then
        log_error "Docker服务未启动，请启动Docker"
        exit 1
    fi
    
    log_success "Docker环境检查通过"
}

# 清理旧容器和网络
cleanup() {
    log_info "清理旧容器和网络..."
    
    # 停止并删除容器
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml down --remove-orphans
    
    # 删除悬空的镜像
    if [ "$(docker images -f dangling=true -q)" ]; then
        docker rmi $(docker images -f dangling=true -q) 2>/dev/null || true
    fi
    
    log_success "清理完成"
}

# 构建镜像
build_images() {
    log_info "构建Docker镜像..."
    
    # 构建应用镜像
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache app
    
    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动开发环境服务..."
    
    # 启动基础服务（数据库、缓存）
    log_info "启动基础服务（PostgreSQL、Redis）..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres redis
    
    # 等待数据库启动
    log_info "等待数据库服务启动..."
    sleep 10
    
    # 检查数据库连接
    if ! docker-compose exec -T postgres pg_isready -U ai_quant_user -d ai_quant_db; then
        log_warning "数据库未完全启动，再等待10秒..."
        sleep 10
    fi
    
    # 启动应用服务
    log_info "启动FastAPI应用服务..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d app
    
    # 启动开发工具（可选）
    if [ "$1" == "--with-tools" ]; then
        log_info "启动开发工具（Adminer、Redis Commander）..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev up -d
    fi
    
    # 启动监控工具（可选）
    if [ "$1" == "--with-monitoring" ]; then
        log_info "启动监控工具（Prometheus、Grafana）..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile monitoring up -d
    fi
    
    log_success "所有服务启动完成"
}

# 显示服务状态
show_status() {
    log_info "服务状态："
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps
    
    echo ""
    log_info "服务访问地址："
    echo "  FastAPI应用:     http://localhost:8000"
    echo "  API文档:         http://localhost:8000/docs"
    echo "  健康检查:        http://localhost:8000/api/v1/health"
    echo "  Adminer:         http://localhost:8080 (如果启用)"
    echo "  Redis Commander: http://localhost:8081 (如果启用)"
    echo "  Grafana:         http://localhost:3000 (如果启用)"
    echo "  Prometheus:      http://localhost:9090 (如果启用)"
    
    echo ""
    log_info "数据库连接信息："
    echo "  主机: localhost:5432"
    echo "  数据库: ai_quant_db"
    echo "  用户名: ai_quant_user"
    echo "  密码: ai_quant_password"
}

# 查看日志
show_logs() {
    local service=${1:-app}
    log_info "显示 $service 服务日志..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f $service
}

# 进入容器
exec_container() {
    local service=${1:-app}
    log_info "进入 $service 容器..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec $service bash
}

# 停止服务
stop_services() {
    log_info "停止开发环境服务..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
    log_success "服务已停止"
}

# 重启服务
restart_services() {
    log_info "重启开发环境服务..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart
    log_success "服务已重启"
}

# 显示帮助信息
show_help() {
    echo "AI量化系统开发环境管理脚本"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  start              启动开发环境"
    echo "  start --with-tools 启动开发环境和开发工具"
    echo "  start --with-monitoring 启动开发环境和监控工具"
    echo "  stop               停止开发环境"
    echo "  restart            重启开发环境"
    echo "  status             显示服务状态"
    echo "  logs [service]     查看服务日志（默认：app）"
    echo "  exec [service]     进入容器（默认：app）"
    echo "  build              重新构建镜像"
    echo "  clean              清理容器和镜像"
    echo "  help               显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start                    # 启动基础开发环境"
    echo "  $0 start --with-tools       # 启动开发环境和管理工具"
    echo "  $0 logs app                 # 查看应用日志"
    echo "  $0 exec postgres            # 进入PostgreSQL容器"
    echo ""
}

# 主函数
main() {
    # 切换到项目根目录
    cd "$(dirname "$0")/.."
    
    case "${1:-start}" in
        "start")
            check_docker
            start_services $2
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs $2
            ;;
        "exec")
            exec_container $2
            ;;
        "build")
            check_docker
            build_images
            ;;
        "clean")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
