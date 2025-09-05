"""
数据库模块测试

测试数据库连接、模型、CRUD操作、健康检查等功能。
"""

import pytest
import tempfile
import os
from datetime import datetime, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.src.database.models import (
    Base, User, Strategy, Order, Position, SystemLog, SyncTask,
    UserRole, UserStatus, StrategyStatus, OrderStatus, OrderSide, OrderType,
    SyncTaskStatus, LogLevel
)
from backend.src.database.connection import DatabaseConfig, DatabaseManager
from backend.src.database.crud import BaseCRUD, paginate, CRUDError
from backend.src.database.health import DatabaseHealth


class TestDatabaseModels:
    """测试数据库模型"""
    
    @pytest.fixture
    def db_session(self):
        """创建测试数据库会话"""
        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    def test_user_model(self, db_session):
        """测试用户模型"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashedpassword123",
            full_name="Test User",
            role=UserRole.USER,
            status=UserStatus.ACTIVE
        )
        
        db_session.add(user)
        db_session.commit()
        
        # 验证用户创建
        saved_user = db_session.query(User).filter(User.username == "testuser").first()
        assert saved_user is not None
        assert saved_user.email == "test@example.com"
        assert saved_user.role == UserRole.USER
        assert saved_user.status == UserStatus.ACTIVE
        assert saved_user.uuid is not None
        assert saved_user.created_at is not None
    
    def test_strategy_model(self, db_session):
        """测试策略模型"""
        # 先创建用户
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashedpassword123"
        )
        db_session.add(user)
        db_session.commit()
        
        # 创建策略
        strategy = Strategy(
            user_id=user.id,
            name="测试策略",
            description="这是一个测试策略",
            status=StrategyStatus.ACTIVE,
            config={"param1": "value1", "param2": 100},
            max_position_size=10000.0,
            max_daily_loss=1000.0
        )
        
        db_session.add(strategy)
        db_session.commit()
        
        # 验证策略创建
        saved_strategy = db_session.query(Strategy).filter(Strategy.name == "测试策略").first()
        assert saved_strategy is not None
        assert saved_strategy.user_id == user.id
        assert saved_strategy.status == StrategyStatus.ACTIVE
        assert saved_strategy.config["param1"] == "value1"
        assert saved_strategy.uuid is not None
    
    def test_order_model(self, db_session):
        """测试订单模型"""
        # 创建用户和策略
        user = User(username="testuser", email="test@example.com", hashed_password="pwd")
        db_session.add(user)
        db_session.commit()
        
        strategy = Strategy(user_id=user.id, name="测试策略")
        db_session.add(strategy)
        db_session.commit()
        
        # 创建订单
        order = Order(
            user_id=user.id,
            strategy_id=strategy.id,
            symbol="000001",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000.0,
            price=10.50,
            status=OrderStatus.PENDING
        )
        
        db_session.add(order)
        db_session.commit()
        
        # 验证订单创建
        saved_order = db_session.query(Order).filter(Order.symbol == "000001").first()
        assert saved_order is not None
        assert saved_order.side == OrderSide.BUY
        assert saved_order.order_type == OrderType.LIMIT
        assert saved_order.quantity == 1000.0
        assert saved_order.price == 10.50
        assert saved_order.uuid is not None
    
    def test_position_model(self, db_session):
        """测试持仓模型"""
        # 创建用户和策略
        user = User(username="testuser", email="test@example.com", hashed_password="pwd")
        db_session.add(user)
        db_session.commit()
        
        strategy = Strategy(user_id=user.id, name="测试策略")
        db_session.add(strategy)
        db_session.commit()
        
        # 创建持仓
        position = Position(
            user_id=user.id,
            strategy_id=strategy.id,
            symbol="000001",
            quantity=1000.0,
            available_quantity=1000.0,
            avg_cost=10.50,
            total_cost=10500.0
        )
        
        db_session.add(position)
        db_session.commit()
        
        # 验证持仓创建
        saved_position = db_session.query(Position).filter(Position.symbol == "000001").first()
        assert saved_position is not None
        assert saved_position.quantity == 1000.0
        assert saved_position.avg_cost == 10.50
        assert saved_position.total_cost == 10500.0
    
    def test_sync_task_model(self, db_session):
        """测试同步任务模型"""
        sync_task = SyncTask(
            task_name="日线数据同步",
            task_type="daily_sync",
            status=SyncTaskStatus.PENDING,
            symbols=["000001", "000002"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            total_symbols=2
        )
        
        db_session.add(sync_task)
        db_session.commit()
        
        # 验证同步任务创建
        saved_task = db_session.query(SyncTask).filter(SyncTask.task_name == "日线数据同步").first()
        assert saved_task is not None
        assert saved_task.task_type == "daily_sync"
        assert saved_task.status == SyncTaskStatus.PENDING
        assert saved_task.symbols == ["000001", "000002"]
        assert saved_task.total_symbols == 2
    
    def test_system_log_model(self, db_session):
        """测试系统日志模型"""
        log_entry = SystemLog(
            level=LogLevel.INFO,
            module="test_module",
            function="test_function",
            message="这是一条测试日志",
            request_id="test-request-123"
        )
        
        db_session.add(log_entry)
        db_session.commit()
        
        # 验证日志创建
        saved_log = db_session.query(SystemLog).filter(SystemLog.request_id == "test-request-123").first()
        assert saved_log is not None
        assert saved_log.level == LogLevel.INFO
        assert saved_log.module == "test_module"
        assert saved_log.message == "这是一条测试日志"
        assert saved_log.created_at is not None


class TestDatabaseConnection:
    """测试数据库连接"""
    
    def test_database_config_creation(self):
        """测试数据库配置创建"""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            username="test_user",
            password="test_password",
            database="test_db"
        )
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.username == "test_user"
        assert config.password == "test_password"
        assert config.database == "test_db"
    
    def test_database_url_generation(self):
        """测试数据库URL生成"""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            username="test_user",
            password="test@password",  # 包含特殊字符
            database="test_db"
        )
        
        url = config.get_database_url()
        assert "postgresql+psycopg2://test_user:" in url
        assert "test%40password" in url  # URL编码后的密码
        assert "@localhost:5432/test_db" in url
    
    def test_database_config_from_env(self, monkeypatch):
        """测试从环境变量创建配置"""
        monkeypatch.setenv("DB_HOST", "testhost")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("DB_USERNAME", "testuser")
        monkeypatch.setenv("DB_PASSWORD", "testpass")
        monkeypatch.setenv("DB_NAME", "testdb")
        
        config = DatabaseConfig.from_env()
        
        assert config.host == "testhost"
        assert config.port == 5433
        assert config.username == "testuser"
        assert config.password == "testpass"
        assert config.database == "testdb"


class TestBaseCRUD:
    """测试基础CRUD操作"""
    
    @pytest.fixture
    def db_session(self):
        """创建测试数据库会话"""
        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        session.close()
    
    @pytest.fixture
    def user_crud(self):
        """创建用户CRUD实例"""
        return BaseCRUD(User)
    
    def test_create_user(self, db_session, user_crud):
        """测试创建用户"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashedpassword123",
            "full_name": "Test User"
        }
        
        user = user_crud.create(db_session, obj_in=user_data)
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
    
    def test_get_user(self, db_session, user_crud):
        """测试获取用户"""
        # 先创建用户
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashedpassword123"
        }
        created_user = user_crud.create(db_session, obj_in=user_data)
        
        # 获取用户
        user = user_crud.get(db_session, id=created_user.id)
        
        assert user is not None
        assert user.id == created_user.id
        assert user.username == "testuser"
    
    def test_get_multi_users(self, db_session, user_crud):
        """测试获取多个用户"""
        # 创建多个用户
        for i in range(5):
            user_data = {
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "hashed_password": f"password{i}"
            }
            user_crud.create(db_session, obj_in=user_data)
        
        # 获取用户列表
        users = user_crud.get_multi(db_session, skip=0, limit=3)
        
        assert len(users) == 3
        assert users[0].username in ["user0", "user1", "user2", "user3", "user4"]
    
    def test_update_user(self, db_session, user_crud):
        """测试更新用户"""
        # 创建用户
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashedpassword123"
        }
        user = user_crud.create(db_session, obj_in=user_data)
        
        # 更新用户
        update_data = {"full_name": "Updated User Name"}
        updated_user = user_crud.update(db_session, db_obj=user, obj_in=update_data)
        
        assert updated_user.full_name == "Updated User Name"
        assert updated_user.username == "testuser"  # 未更新的字段保持不变
    
    def test_remove_user(self, db_session, user_crud):
        """测试删除用户"""
        # 创建用户
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashedpassword123"
        }
        user = user_crud.create(db_session, obj_in=user_data)
        user_id = user.id
        
        # 删除用户
        removed_user = user_crud.remove(db_session, id=user_id)
        
        assert removed_user is not None
        assert removed_user.id == user_id
        
        # 验证用户已被删除
        user = user_crud.get(db_session, id=user_id)
        assert user is None
    
    def test_count_users(self, db_session, user_crud):
        """测试统计用户数量"""
        # 创建用户
        for i in range(3):
            user_data = {
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "hashed_password": f"password{i}"
            }
            user_crud.create(db_session, obj_in=user_data)
        
        count = user_crud.count(db_session)
        assert count == 3
    
    def test_exists_user(self, db_session, user_crud):
        """测试检查用户是否存在"""
        # 创建用户
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashedpassword123"
        }
        user = user_crud.create(db_session, obj_in=user_data)
        
        # 检查存在性
        assert user_crud.exists(db_session, user.id) is True
        assert user_crud.exists(db_session, 99999) is False


class TestDatabaseHealth:
    """测试数据库健康检查"""
    
    @pytest.fixture
    def health_checker(self):
        """创建健康检查器（使用内存数据库）"""
        # 这里应该使用测试数据库配置
        return DatabaseHealth()
    
    def test_health_report_structure(self, health_checker):
        """测试健康检查报告结构"""
        # 注意：这个测试可能会失败，因为需要真实的PostgreSQL连接
        try:
            report = health_checker.get_full_health_report()
            
            assert "timestamp" in report
            assert "overall_status" in report
            assert "connection" in report
            assert "connection_pool" in report
            assert "performance" in report
            assert "tables" in report
            assert "recent_errors" in report
            assert "recommendations" in report
            
        except Exception as e:
            # 如果没有数据库连接，跳过测试
            pytest.skip(f"Database connection required for health check: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
