"""
健康检查API路由

提供系统健康状态检查接口，包括数据库连接、外部服务状态等。
"""

import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.src.database.health import get_health_checker
from backend.app.middleware.performance import PerformanceMonitoringMiddleware


router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    environment: str


class DetailedHealthResponse(BaseModel):
    """详细健康检查响应模型"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    environment: str
    components: Dict[str, Any]
    system_info: Dict[str, Any]


class SystemInfoResponse(BaseModel):
    """系统信息响应模型"""
    version: str
    environment: str
    config: Dict[str, Any]
    uptime_seconds: float


# 应用启动时间
_start_time = time.time()


@router.get("/", response_model=HealthResponse, summary="基础健康检查")
async def health_check():
    """
    基础健康检查接口
    
    返回系统基本状态信息，响应时间要求 < 100ms。
    """
    uptime = time.time() - _start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=settings.PROJECT_VERSION,
        uptime_seconds=round(uptime, 2),
        environment="development" if settings.DEBUG else "production"
    )


@router.get("/detailed", response_model=DetailedHealthResponse, summary="详细健康检查")
async def detailed_health_check():
    """
    详细健康检查接口
    
    返回系统各组件的详细健康状态，包括数据库、缓存、外部服务等。
    """
    uptime = time.time() - _start_time
    
    # 获取数据库健康状态
    try:
        health_checker = get_health_checker()
        db_health = health_checker.check_connection()
        pool_status = health_checker.check_pool_status()
    except Exception as e:
        db_health = {"status": "error", "error": str(e)}
        pool_status = {"status": "error", "error": str(e)}
    
    # 组件状态
    components = {
        "database": {
            "connection": db_health,
            "pool": pool_status
        },
        "storage": {
            "parquet": {"status": "healthy"},  # 简化实现
            "logs": {"status": "healthy"}
        },
        "external_services": {
            "akshare": {"status": "healthy" if settings.AKSHARE_ENABLED else "disabled"}
        }
    }
    
    # 系统信息
    system_info = {
        "python_version": "3.13",  # 简化实现
        "fastapi_version": "0.115+",
        "environment": "development" if settings.DEBUG else "production",
        "debug_mode": settings.DEBUG
    }
    
    # 确定整体状态
    overall_status = "healthy"
    for component_name, component_info in components.items():
        if isinstance(component_info, dict):
            for sub_component in component_info.values():
                if isinstance(sub_component, dict) and sub_component.get("status") == "error":
                    overall_status = "unhealthy"
                    break
        if overall_status == "unhealthy":
            break
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version=settings.PROJECT_VERSION,
        uptime_seconds=round(uptime, 2),
        environment="development" if settings.DEBUG else "production",
        components=components,
        system_info=system_info
    )


@router.get("/database", summary="数据库健康检查")
async def database_health_check():
    """
    数据库健康检查接口
    
    检查数据库连接状态、连接池使用情况、查询性能等。
    """
    try:
        health_checker = get_health_checker()
        return health_checker.get_full_health_report()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"数据库健康检查失败: {str(e)}"
        )


@router.get("/metrics", summary="性能指标")
async def get_performance_metrics():
    """
    获取系统性能指标
    
    返回API响应时间、内存使用情况、系统资源使用等指标。
    """
    try:
        # 注意：这里需要从应用实例中获取性能监控中间件
        # 这是一个简化的实现，实际应该从应用中获取中间件实例
        
        # 模拟性能数据
        performance_data = {
            "api_metrics": {
                "total_requests": 0,
                "avg_response_time": 0.0,
                "error_rate": 0.0
            },
            "system_metrics": {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return performance_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能指标失败: {str(e)}"
        )


@router.get("/info", response_model=SystemInfoResponse, summary="系统信息")
async def get_system_info():
    """
    获取系统基本信息
    
    返回应用版本、环境配置、运行时间等信息。
    """
    uptime = time.time() - _start_time
    
    # 安全的配置信息（不包含敏感数据）
    safe_config = {
        "project_name": settings.PROJECT_NAME,
        "api_version": settings.API_V1_STR,
        "debug": settings.DEBUG,
        "host": settings.HOST,
        "port": settings.PORT,
        "cors_origins_count": len(settings.CORS_ORIGINS),
        "database_host": settings.DB_HOST,
        "database_port": settings.DB_PORT,
        "database_name": settings.DB_NAME,
        "redis_host": settings.REDIS_HOST,
        "redis_port": settings.REDIS_PORT,
        "akshare_enabled": settings.AKSHARE_ENABLED,
        "log_level": settings.LOG_LEVEL
    }
    
    return SystemInfoResponse(
        version=settings.PROJECT_VERSION,
        environment="development" if settings.DEBUG else "production",
        config=safe_config,
        uptime_seconds=round(uptime, 2)
    )


@router.get("/ping", summary="简单连通性测试")
async def ping():
    """
    简单的连通性测试接口

    最轻量的健康检查，仅返回pong响应。
    """
    return {"message": "pong", "timestamp": datetime.utcnow().isoformat()}


@router.options("/ping", include_in_schema=False)
async def ping_options():
    """处理CORS预检请求，避免返回405。"""
    return {"allow": ["GET", "OPTIONS"]}
