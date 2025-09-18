"""
FastAPI应用配置管理

统一管理应用配置，支持环境变量配置和开发/生产环境切换。
"""

import os
from typing import List, Optional, Any, Dict
from pydantic import validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置类"""
    # Pydantic v2 settings 配置：读取 .env，区分大小写，忽略未知环境变量
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )
    
    # 基础应用配置
    PROJECT_NAME: str = "AI量化系统"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # 安全配置
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS配置
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://frontend:80",
        "http://ai-quant-frontend:80"
    ]
    
    # 数据库配置
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USERNAME: str = "postgres"
    DB_PASSWORD: str = "password"
    DB_NAME: str = "ai_quant"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # 数据存储配置
    DATA_STORAGE_PATH: str = "data"
    PARQUET_STORAGE_PATH: str = "data/parquet"
    LOG_STORAGE_PATH: str = "logs"
    
    # 外部服务配置
    AKSHARE_ENABLED: bool = True
    AKSHARE_TIMEOUT: int = 30
    EASTMONEY_ENABLED: bool = False
    EASTMONEY_TIMEOUT: int = 30
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_SIZE: str = "10MB"
    LOG_BACKUP_COUNT: int = 5
    
    # 性能配置
    REQUEST_TIMEOUT: int = 30
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    API_RATE_LIMIT: int = 100  # 每分钟请求次数
    
    # 缓存配置
    CACHE_TTL: int = 300  # 5分钟
    CACHE_PREFIX: str = "ai_quant"
    
    # 监控配置
    ENABLE_METRICS: bool = True
    METRICS_PATH: str = "/metrics"
    HEALTH_CHECK_PATH: str = "/health"
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        """处理CORS origins配置"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError("CORS_ORIGINS must be a list or comma-separated string")
    
    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        return (
            f"postgresql+psycopg2://{self.DB_USERNAME}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def async_database_url(self) -> str:
        """构建异步数据库连接URL"""
        return (
            f"postgresql+asyncpg://{self.DB_USERNAME}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def redis_url(self) -> str:
        """构建Redis连接URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.LOG_FORMAT,
                },
                "access": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "access": {
                    "formatter": "access", 
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "formatter": "default",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": f"{self.LOG_STORAGE_PATH}/app.log",
                    "maxBytes": self._parse_size(self.LOG_MAX_SIZE),
                    "backupCount": self.LOG_BACKUP_COUNT,
                },
            },
            "loggers": {
                "": {
                    "level": self.LOG_LEVEL,
                    "handlers": ["default", "file"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["access"],
                    "level": "INFO",
                    "propagate": False,
                },
                "sqlalchemy": {
                    "level": "WARNING",
                    "handlers": ["default"],
                    "propagate": False,
                },
            },
        }
    
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串为字节数"""
        size_str = size_str.upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    # 旧版 Pydantic v1 Config 已由 model_config 取代


@lru_cache()
def get_settings() -> Settings:
    """获取应用配置单例"""
    return Settings()


# 全局配置实例
settings = get_settings()


# 环境相关的配置函数
def is_development() -> bool:
    """是否为开发环境"""
    return settings.DEBUG


def is_production() -> bool:
    """是否为生产环境"""
    return not settings.DEBUG


def get_api_prefix() -> str:
    """获取API前缀"""
    return settings.API_V1_STR


def get_cors_origins() -> List[str]:
    """获取CORS允许的源"""
    return settings.CORS_ORIGINS


def get_database_config() -> Dict[str, Any]:
    """获取数据库配置字典"""
    return {
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "username": settings.DB_USERNAME,
        "password": settings.DB_PASSWORD,
        "database": settings.DB_NAME,
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "echo": settings.DB_ECHO,
    }
