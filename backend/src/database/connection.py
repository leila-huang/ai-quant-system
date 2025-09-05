"""
数据库连接管理

提供数据库连接池、session管理、健康检查等功能。
支持PostgreSQL连接池配置和连接参数管理。
"""

import os
import asyncio
from typing import Optional, AsyncGenerator, Dict, Any
from contextlib import contextmanager, asynccontextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text, MetaData, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from loguru import logger

from backend.src.database.models import Base


class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        username: str = "postgres", 
        password: str = "password",
        database: str = "ai_quant",
        # 连接池配置
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        # 连接配置
        connect_timeout: int = 10,
        statement_timeout: int = 30000,  # 30秒
        # SSL配置
        sslmode: str = "prefer",
        **kwargs
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        
        self.connect_timeout = connect_timeout
        self.statement_timeout = statement_timeout
        self.sslmode = sslmode
        self.extra_params = kwargs
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """从环境变量创建配置"""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            username=os.getenv("DB_USERNAME", "postgres"),
            password=os.getenv("DB_PASSWORD", "password"),
            database=os.getenv("DB_NAME", "ai_quant"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
            statement_timeout=int(os.getenv("DB_STATEMENT_TIMEOUT", "30000")),
            sslmode=os.getenv("DB_SSLMODE", "prefer"),
        )
    
    def get_database_url(self, driver: str = "postgresql", async_driver: bool = False) -> str:
        """构建数据库连接URL"""
        if async_driver:
            driver = f"{driver}+asyncpg"
        else:
            driver = f"{driver}+psycopg2"
            
        # URL编码密码中的特殊字符
        encoded_password = quote_plus(self.password)
        
        url = (
            f"{driver}://{self.username}:{encoded_password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
        
        # 添加连接参数
        params = [
            f"sslmode={self.sslmode}",
            f"connect_timeout={self.connect_timeout}",
            f"options=-c statement_timeout={self.statement_timeout}ms"
        ]
        
        if self.extra_params:
            for key, value in self.extra_params.items():
                params.append(f"{key}={value}")
        
        if params:
            url += "?" + "&".join(params)
            
        return url


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self.async_engine = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.AsyncSessionLocal = None
        self._initialized = False
    
    def initialize(self, create_tables: bool = False):
        """初始化数据库连接"""
        if self._initialized:
            return
            
        try:
            # 创建同步引擎
            database_url = self.config.get_database_url(async_driver=False)
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                future=True
            )
            
            # 创建session工厂
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            # 创建异步引擎
            async_database_url = self.config.get_database_url(async_driver=True)
            self.async_engine = create_async_engine(
                async_database_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
            )
            
            # 创建异步session工厂
            self.AsyncSessionLocal = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            # 添加连接事件监听
            self._setup_connection_events()
            
            # 创建表
            if create_tables:
                self.create_tables()
            
            self._initialized = True
            logger.info(f"Database initialized successfully: {self.config.host}:{self.config.port}/{self.config.database}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _setup_connection_events(self):
        """设置连接事件监听"""
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """连接时设置数据库参数"""
            pass
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接检出时的处理"""
            logger.debug("Database connection checked out")
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """连接归还时的处理"""
            logger.debug("Database connection checked in")
    
    def create_tables(self):
        """创建所有表"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """删除所有表"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """获取数据库session（同步）"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库session（异步）"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    def check_connection(self) -> bool:
        """检查数据库连接"""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    async def check_async_connection(self) -> bool:
        """检查数据库连接（异步）"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database async connection check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        if not self.engine:
            return {}
            
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
            "url": str(self.engine.url).replace(self.config.password, "***"),
            "echo": self.engine.echo,
        }
    
    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed")
        
        if self.async_engine:
            asyncio.create_task(self.async_engine.dispose())
            logger.info("Async database engine disposed")
        
        self._initialized = False


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _db_manager
    
    if _db_manager is None:
        config = DatabaseConfig.from_env()
        _db_manager = DatabaseManager(config)
        _db_manager.initialize()
    
    return _db_manager


def get_session() -> Session:
    """获取数据库session的便捷函数"""
    manager = get_database_manager()
    return manager.get_session()


def get_async_session() -> AsyncSession:
    """获取异步数据库session的便捷函数"""
    manager = get_database_manager()
    return manager.get_async_session()


def init_database(config: Optional[DatabaseConfig] = None, create_tables: bool = False):
    """初始化数据库"""
    global _db_manager
    
    if config is None:
        config = DatabaseConfig.from_env()
    
    _db_manager = DatabaseManager(config)
    _db_manager.initialize(create_tables=create_tables)


def close_database():
    """关闭数据库连接"""
    global _db_manager
    
    if _db_manager:
        _db_manager.close()
        _db_manager = None
