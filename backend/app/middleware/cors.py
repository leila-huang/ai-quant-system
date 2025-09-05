"""
CORS中间件配置

配置跨域资源共享，支持前端应用调用API。
"""

from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings


def setup_cors(app: FastAPI) -> None:
    """设置CORS中间件"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
        allow_headers=[
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "X-API-Key",
            "Cache-Control",
            "Pragma",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time", 
            "X-Process-Time",
            "X-Memory-Usage",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset",
        ],
        max_age=86400,  # 24小时缓存预检请求
    )


def get_cors_config() -> dict:
    """获取CORS配置信息"""
    return {
        "allowed_origins": settings.CORS_ORIGINS,
        "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
        "allowed_headers": [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "X-API-Key",
            "Cache-Control",
            "Pragma",
        ],
        "exposed_headers": [
            "X-Request-ID",
            "X-Response-Time", 
            "X-Process-Time",
            "X-Memory-Usage",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset",
        ],
        "allow_credentials": True,
        "max_age": 86400
    }
