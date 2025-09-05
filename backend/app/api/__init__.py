"""
API路由模块

包含所有API路由定义。
"""

from fastapi import APIRouter
from .health import router as health_router
from .data import router as data_router
# TODO: 暂时注释数据同步路由，等待Pydantic配置问题解决
# from .data_sync import router as data_sync_router

# 创建主路由器
api_router = APIRouter()

# 包含各模块路由
api_router.include_router(health_router, prefix="/health", tags=["健康检查"])
api_router.include_router(data_router, prefix="/data", tags=["数据管理"])
# api_router.include_router(data_sync_router, prefix="/data-sync", tags=["数据同步"])

__all__ = ["api_router"]
