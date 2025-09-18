"""
FastAPI主应用

AI量化系统的主要应用入口，包含路由配置、中间件设置、异常处理等。
"""

import os
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

from backend.app.core.config import settings
from backend.app.core.exceptions import setup_exception_handlers
from backend.app.middleware import (
    setup_cors,
    RequestTrackingMiddleware,
    PerformanceMonitoringMiddleware,
    SecurityHeadersMiddleware
)
from backend.app.api import api_router
from backend.app.api.websocket import router as websocket_router
from backend.src.database.connection import init_database, close_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    
    # 创建日志目录
    os.makedirs(settings.LOG_STORAGE_PATH, exist_ok=True)
    
    # 配置日志
    logging.config.dictConfig(settings.get_logging_config())
    logger = logging.getLogger(__name__)
    logger.info(f"Application starting: {settings.PROJECT_NAME}")
    
    # 初始化数据库连接 (开发环境可选)
    try:
        if not settings.DEBUG:  # 生产环境才初始化数据库
            from backend.src.database.connection import DatabaseConfig
            db_config = DatabaseConfig.from_env()
            init_database(db_config)
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization skipped or failed: {e}")
    
    logger.info(f"Server starting on {settings.HOST}:{settings.PORT}")
    
    yield
    
    # 关闭时执行
    logger.info("Application shutting down...")
    try:
        close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
    
    print("👋 Application shutdown complete")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="AI量化交易系统 - 提供股票数据获取、策略回测、实时交易等功能",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=None,  # 我们将自定义文档路由
    redoc_url=None,
    lifespan=lifespan
)

# 设置CORS中间件
setup_cors(app)

# 添加自定义中间件
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(PerformanceMonitoringMiddleware, warning_threshold=0.2)  # 200ms警告阈值
app.add_middleware(RequestTrackingMiddleware)

# 设置异常处理器
setup_exception_handlers(app)

# 包含API路由
app.include_router(api_router, prefix=settings.API_V1_STR)

# 包含WebSocket路由
app.include_router(websocket_router)

# 根路径
@app.get("/", include_in_schema=False)
async def root():
    """根路径重定向到API文档"""
    return JSONResponse({
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.PROJECT_VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_url": settings.API_V1_STR,
        "health_check": f"{settings.API_V1_STR}/health"
    })


# 自定义API文档路由
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI"""
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - API文档",
        swagger_favicon_url="/static/favicon.ico" if os.path.exists("static/favicon.ico") else None,
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc API文档"""
    return get_redoc_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - API文档",
        redoc_favicon_url="/static/favicon.ico" if os.path.exists("static/favicon.ico") else None,
    )


# 自定义OpenAPI schema
def custom_openapi():
    """自定义OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        description="""
## AI量化交易系统 API

这是一个现代化的量化交易系统，提供以下核心功能：

### 🏥 健康检查
- 系统状态监控
- 数据库连接检查  
- 性能指标监控

### 📊 数据管理
- 股票历史数据查询
- 多数据源支持 (AKShare等)
- 高性能Parquet存储

### 🔒 安全特性
- 请求追踪和日志
- 异常处理和错误码
- CORS和安全头配置

### 📈 后续功能 (开发中)
- 策略回测引擎
- 实时交易接口
- 用户权限管理
- 风控管理

### 技术栈
- **后端**: FastAPI + SQLAlchemy + PostgreSQL
- **存储**: Parquet列式存储 + Redis缓存
- **数据源**: AKShare + 东财接口
- **部署**: Docker + Docker Compose
        """,
        routes=app.routes,
    )
    
    # 添加额外的API信息
    openapi_schema["info"]["contact"] = {
        "name": "AI量化系统开发团队",
        "email": "dev@ai-quant.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT License"
    }
    
    # 添加服务器信息
    openapi_schema["servers"] = [
        {
            "url": f"http://{settings.HOST}:{settings.PORT}",
            "description": "开发服务器"
        }
    ]
    
    # 添加标签信息
    openapi_schema["tags"] = [
        {
            "name": "健康检查",
            "description": "系统健康状态监控接口"
        },
        {
            "name": "数据管理", 
            "description": "股票数据查询和管理接口"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# 中间件性能统计接口 (开发模式下可用)
@app.get("/internal/performance", include_in_schema=False)
async def get_performance_stats():
    """获取性能统计数据 (内部接口)"""
    if not settings.DEBUG:
        return JSONResponse(
            {"error": "Performance stats only available in debug mode"},
            status_code=403
        )
    
    # 尝试从中间件获取性能数据
    try:
        # 这里需要从应用的中间件栈中获取PerformanceMonitoringMiddleware实例
        # 这是一个简化实现，实际中需要更复杂的中间件管理
        return {
            "message": "Performance monitoring middleware data",
            "note": "Full implementation in Task 6"
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    # 开发模式运行
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        loop="asyncio"
    )
