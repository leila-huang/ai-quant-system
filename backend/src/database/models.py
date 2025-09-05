"""
SQLAlchemy数据库模型定义

定义AI量化系统的核心业务数据表结构，包括：
- 用户管理
- 策略管理 
- 订单和持仓
- 系统日志
- 数据同步任务
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Date, Boolean, 
    Numeric, JSON, ForeignKey, Index, UniqueConstraint,
    Enum, TIMESTAMP, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid


Base = declarative_base()


class UserRole(PyEnum):
    """用户角色枚举"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class UserStatus(PyEnum):
    """用户状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class StrategyStatus(PyEnum):
    """策略状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    BACKTEST = "backtest"


class OrderStatus(PyEnum):
    """订单状态枚举"""
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(PyEnum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"


class OrderType(PyEnum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class SyncTaskStatus(PyEnum):
    """数据同步任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LogLevel(PyEnum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(128), nullable=False)
    full_name = Column(String(100))
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 用户配置
    config = Column(JSON, default={})
    
    # 关联关系
    strategies = relationship("Strategy", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class Strategy(Base):
    """策略表"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    status = Column(Enum(StrategyStatus), default=StrategyStatus.STOPPED, nullable=False)
    
    # 策略参数配置
    config = Column(JSON, default={})
    
    # 策略代码和版本
    code = Column(Text)
    version = Column(String(20), default="1.0.0")
    
    # 风控参数
    max_position_size = Column(Numeric(15, 4), default=10000.0)
    max_daily_loss = Column(Numeric(15, 4), default=1000.0)
    max_drawdown = Column(Numeric(10, 4), default=0.20)  # 最大回撤比例
    
    # 时间字段
    last_run_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 关联关系
    user = relationship("User", back_populates="strategies")
    orders = relationship("Order", back_populates="strategy", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="strategy", cascade="all, delete-orphan")
    
    # 索引
    __table_args__ = (
        Index("idx_strategy_user_status", user_id, status),
        Index("idx_strategy_name", name),
    )
    
    def __repr__(self):
        return f"<Strategy(id={self.id}, name='{self.name}', status='{self.status.value}')>"


class Order(Base):
    """订单表"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    
    # 基本订单信息
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING, nullable=False)
    
    # 价格和数量
    quantity = Column(Numeric(15, 4), nullable=False)
    price = Column(Numeric(15, 4))  # 限价单价格
    stop_price = Column(Numeric(15, 4))  # 止损价格
    filled_quantity = Column(Numeric(15, 4), default=0.0, nullable=False)
    avg_fill_price = Column(Numeric(15, 4))
    
    # 手续费和金额
    commission = Column(Numeric(15, 4), default=0.0)
    total_amount = Column(Numeric(15, 4))
    
    # 外部系统ID
    external_id = Column(String(100))  # 券商系统订单ID
    client_order_id = Column(String(100), unique=True)  # 客户端订单ID
    
    # 时间字段
    placed_at = Column(DateTime(timezone=True), server_default=func.now())
    filled_at = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))
    
    # 订单备注和元数据
    notes = Column(Text)
    order_metadata = Column(JSON, default={})
    
    # 关联关系
    user = relationship("User", back_populates="orders")
    strategy = relationship("Strategy", back_populates="orders")
    
    # 索引
    __table_args__ = (
        Index("idx_order_symbol_status", symbol, status),
        Index("idx_order_user_placed", user_id, placed_at.desc()),
        Index("idx_order_strategy", strategy_id, placed_at.desc()),
        UniqueConstraint("client_order_id", name="uq_client_order_id"),
    )
    
    def __repr__(self):
        return f"<Order(id={self.id}, symbol='{self.symbol}', side='{self.side.value}', status='{self.status.value}')>"


class Position(Base):
    """持仓表"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # 持仓信息
    quantity = Column(Numeric(15, 4), default=0.0, nullable=False)  # 持仓数量，正数为多头，负数为空头
    available_quantity = Column(Numeric(15, 4), default=0.0, nullable=False)  # 可用数量
    frozen_quantity = Column(Numeric(15, 4), default=0.0, nullable=False)  # 冻结数量
    
    # 成本和价值
    avg_cost = Column(Numeric(15, 4), default=0.0, nullable=False)  # 平均成本价
    total_cost = Column(Numeric(15, 4), default=0.0, nullable=False)  # 总成本
    market_value = Column(Numeric(15, 4), default=0.0, nullable=False)  # 市值
    unrealized_pnl = Column(Numeric(15, 4), default=0.0, nullable=False)  # 浮动盈亏
    realized_pnl = Column(Numeric(15, 4), default=0.0, nullable=False)  # 已实现盈亏
    
    # 时间字段
    first_buy_date = Column(Date)  # 首次买入日期
    last_update_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # 关联关系
    user = relationship("User", back_populates="positions")
    strategy = relationship("Strategy", back_populates="positions")
    
    # 索引和约束
    __table_args__ = (
        UniqueConstraint("user_id", "strategy_id", "symbol", name="uq_position_user_strategy_symbol"),
        Index("idx_position_user_symbol", user_id, symbol),
        Index("idx_position_strategy", strategy_id),
    )
    
    def __repr__(self):
        return f"<Position(id={self.id}, symbol='{self.symbol}', quantity={self.quantity}, avg_cost={self.avg_cost})>"


class SyncTask(Base):
    """数据同步任务表"""
    __tablename__ = "sync_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    
    # 任务基本信息
    task_name = Column(String(100), nullable=False, index=True)
    task_type = Column(String(50), nullable=False)  # 任务类型：daily_sync, historical_sync等
    status = Column(Enum(SyncTaskStatus), default=SyncTaskStatus.PENDING, nullable=False)
    
    # 任务参数
    symbols = Column(JSON)  # 股票代码列表
    start_date = Column(Date)
    end_date = Column(Date)
    data_source = Column(String(50), default="akshare")
    
    # 执行结果
    total_symbols = Column(Integer, default=0)
    completed_symbols = Column(Integer, default=0)
    failed_symbols = Column(Integer, default=0)
    total_records = Column(Integer, default=0)
    
    # 错误信息
    error_message = Column(Text)
    error_details = Column(JSON)
    
    # 时间字段
    scheduled_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index("idx_sync_task_status_created", status, created_at.desc()),
        Index("idx_sync_task_type", task_type),
    )
    
    def __repr__(self):
        return f"<SyncTask(id={self.id}, name='{self.task_name}', status='{self.status.value}')>"


class SystemLog(Base):
    """系统日志表"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 日志基本信息
    level = Column(Enum(LogLevel), nullable=False, index=True)
    module = Column(String(100), nullable=False, index=True)
    function = Column(String(100))
    message = Column(Text, nullable=False)
    
    # 上下文信息
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    
    # 请求信息
    request_id = Column(String(100), index=True)
    session_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # 异常信息
    exception_type = Column(String(100))
    stack_trace = Column(Text)
    
    # 扩展数据
    extra_data = Column(JSON)
    
    # 时间字段
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # 索引
    __table_args__ = (
        Index("idx_log_level_created", level, created_at.desc()),
        Index("idx_log_module_created", module, created_at.desc()),
        Index("idx_log_user", user_id, created_at.desc()),
    )
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level.value}', module='{self.module}')>"


class AppConfig(Base):
    """应用配置表"""
    __tablename__ = "app_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text)
    category = Column(String(50), default="general", index=True)
    is_encrypted = Column(Boolean, default=False, nullable=False)
    is_readonly = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<AppConfig(key='{self.key}', category='{self.category}')>"
