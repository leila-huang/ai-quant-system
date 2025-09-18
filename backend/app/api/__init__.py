"""
API路由模块

包含所有API路由定义。
"""

from fastapi import APIRouter
from .health import router as health_router
from .data import router as data_router
from .websocket import router as websocket_router
# 导入所有API路由模块
from .features import router as features_router
from .models import router as models_router
from .backtest import router as backtest_router
from .data_sync import router as data_sync_router
from .strategies import router as strategies_router
from .ai import router as ai_router

# 创建主路由器
api_router = APIRouter()

# 包含各模块路由
api_router.include_router(health_router, prefix="/health", tags=["健康检查"])
api_router.include_router(data_router, prefix="/data", tags=["数据管理"])
api_router.include_router(websocket_router, tags=["实时通信"])
# 注册所有API路由
api_router.include_router(features_router, prefix="/features", tags=["特征工程"])
api_router.include_router(models_router, prefix="/models", tags=["机器学习"])
api_router.include_router(backtest_router, prefix="/backtest", tags=["策略回测"])
api_router.include_router(data_sync_router, prefix="/data-sync", tags=["数据同步"])
api_router.include_router(strategies_router, prefix="/strategies", tags=["策略管理"])
api_router.include_router(ai_router, prefix="/ai", tags=["AI助手"])

__all__ = ["api_router"]
