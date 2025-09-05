"""
数据库模块

包含数据库模型、连接管理、CRUD操作等核心功能。
"""

from .models import (
    Base,
    User, Strategy, Order, Position, 
    SyncTask, SystemLog, AppConfig,
    UserRole, UserStatus, StrategyStatus,
    OrderStatus, OrderSide, OrderType,
    SyncTaskStatus, LogLevel
)
from .connection import (
    DatabaseConfig, DatabaseManager,
    get_database_manager, get_session, get_async_session,
    init_database, close_database
)
from .crud import (
    BaseCRUD, CRUDError, PaginatedResponse,
    paginate, execute_raw_query, 
    bulk_insert_mappings, bulk_update_mappings
)

__all__ = [
    # Models
    "Base",
    "User", "Strategy", "Order", "Position",
    "SyncTask", "SystemLog", "AppConfig",
    # Enums
    "UserRole", "UserStatus", "StrategyStatus",
    "OrderStatus", "OrderSide", "OrderType", 
    "SyncTaskStatus", "LogLevel",
    # Connection
    "DatabaseConfig", "DatabaseManager",
    "get_database_manager", "get_session", "get_async_session",
    "init_database", "close_database",
    # CRUD
    "BaseCRUD", "CRUDError", "PaginatedResponse",
    "paginate", "execute_raw_query",
    "bulk_insert_mappings", "bulk_update_mappings"
]