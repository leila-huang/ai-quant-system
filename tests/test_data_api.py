"""
数据API集成测试

测试数据同步、查询等API功能。
"""

import pytest
import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.data_schemas import DataSyncRequest, SyncStatus
from backend.src.models.basic_models import StockData, StockDailyBar


class TestDataSyncAPI:
    """数据同步API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_data_service(self):
        """模拟数据服务"""
        with patch('backend.app.api.data_sync.get_data_service_dep') as mock:
            service = MagicMock()
            mock.return_value = service
            yield service
    
    @pytest.fixture
    def mock_task_queue(self):
        """模拟任务队列"""
        with patch('backend.app.api.data_sync.get_task_queue_dep') as mock:
            queue = MagicMock()
            mock.return_value = queue
            yield queue
    
    def test_create_sync_task_success(self, client, mock_data_service):
        """测试创建同步任务成功"""
        # 模拟返回
        mock_data_service.create_sync_task.return_value = {
            "task_id": "test-task-123",
            "status": SyncStatus.PENDING,
            "message": "任务已创建",
            "symbols_count": 10,
            "estimated_time": 20
        }
        
        # 发送请求
        response = client.post("/api/v1/data-sync/sync", json={
            "symbols": ["000001", "000002"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "async_mode": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == SyncStatus.PENDING.value
        assert "symbols_count" in data
    
    def test_create_sync_task_validation_error(self, client, mock_data_service):
        """测试同步任务参数验证错误"""
        # 测试股票数量超限
        symbols = [f"{i:06d}" for i in range(101)]  # 101个股票代码
        
        response = client.post("/api/v1/data-sync/sync", json={
            "symbols": symbols,
            "async_mode": True
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "单次同步股票数量不能超过100个" in data["error"]["message"]
    
    def test_create_sync_task_date_range_error(self, client, mock_data_service):
        """测试日期范围验证错误"""
        start_date = date.today() - timedelta(days=400)
        end_date = date.today()
        
        response = client.post("/api/v1/data-sync/sync", json={
            "symbols": ["000001"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "async_mode": True
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "同步日期范围不能超过365天" in data["error"]["message"]
    
    def test_get_sync_task_status(self, client, mock_data_service):
        """测试查询同步任务状态"""
        # 模拟任务状态
        mock_data_service.get_sync_task_status.return_value = {
            "task_id": "test-task-123",
            "status": SyncStatus.RUNNING,
            "progress": 50.0,
            "symbols_total": 10,
            "symbols_completed": 5,
            "symbols_failed": 0,
            "current_symbol": "000005",
            "start_time": datetime.utcnow(),
            "end_time": None,
            "error_message": None,
            "result": None
        }
        
        response = client.get("/api/v1/data-sync/sync/test-task-123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-123"
        assert data["status"] == SyncStatus.RUNNING.value
        assert data["progress"] == 50.0
        assert data["symbols_completed"] == 5
    
    def test_get_sync_task_status_not_found(self, client, mock_data_service):
        """测试查询不存在的任务状态"""
        mock_data_service.get_sync_task_status.return_value = None
        
        response = client.get("/api/v1/data-sync/sync/nonexistent-task/status")
        
        assert response.status_code == 404
        data = response.json()
        assert "未找到任务ID" in data["error"]["message"]
    
    def test_cancel_sync_task(self, client, mock_data_service):
        """测试取消同步任务"""
        mock_data_service.cancel_sync_task.return_value = True
        
        response = client.delete("/api/v1/data-sync/sync/test-task-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "任务已取消"
        assert data["task_id"] == "test-task-123"
    
    def test_cancel_sync_task_not_found(self, client, mock_data_service):
        """测试取消不存在的任务"""
        mock_data_service.cancel_sync_task.return_value = False
        
        response = client.delete("/api/v1/data-sync/sync/nonexistent-task")
        
        assert response.status_code == 404
        data = response.json()
        assert "不存在或无法取消" in data["error"]["message"]


class TestStockQueryAPI:
    """股票查询API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_data_service(self):
        """模拟数据服务"""
        with patch('backend.app.api.data_sync.get_data_service_dep') as mock:
            service = MagicMock()
            mock.return_value = service
            yield service
    
    def test_get_stock_data_success(self, client, mock_data_service):
        """测试查询单个股票数据成功"""
        # 模拟返回数据
        mock_stock_response = {
            "symbol": "000001",
            "name": "平安银行",
            "data": [
                {
                    "date": "2024-01-01",
                    "open_price": 10.0,
                    "close_price": 10.5,
                    "high_price": 10.8,
                    "low_price": 9.8,
                    "volume": 1000000,
                    "amount": 10500000
                }
            ],
            "total_count": 1,
            "has_more": False,
            "next_offset": None
        }
        
        mock_data_service.query_stock_data.return_value = mock_stock_response
        
        response = client.get("/api/v1/data-sync/stocks/000001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "000001"
        assert data["name"] == "平安银行"
        assert len(data["data"]) == 1
        assert data["data"][0]["close_price"] == 10.5
    
    def test_get_stock_data_not_found(self, client, mock_data_service):
        """测试查询不存在的股票"""
        mock_data_service.query_stock_data.return_value = None
        
        response = client.get("/api/v1/data-sync/stocks/NONEXISTENT")
        
        assert response.status_code == 404
        data = response.json()
        assert "未找到股票" in data["error"]["message"]
    
    def test_get_stock_data_with_params(self, client, mock_data_service):
        """测试带参数查询股票数据"""
        mock_stock_response = {
            "symbol": "000001",
            "name": "平安银行",
            "data": [],
            "total_count": 100,
            "has_more": True,
            "next_offset": 50
        }
        
        mock_data_service.query_stock_data.return_value = mock_stock_response
        
        response = client.get(
            "/api/v1/data-sync/stocks/000001",
            params={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "limit": 50,
                "offset": 0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["has_more"] is True
        assert data["next_offset"] == 50
    
    def test_batch_query_stocks_success(self, client, mock_data_service):
        """测试批量查询股票数据成功"""
        mock_batch_response = {
            "stocks": [
                {
                    "symbol": "000001",
                    "name": "平安银行",
                    "data": [{"date": "2024-01-01", "close_price": 10.0}],
                    "total_count": 1,
                    "has_more": False
                }
            ],
            "total_symbols": 1,
            "success_count": 1,
            "failed_symbols": [],
            "query_time": datetime.utcnow().isoformat()
        }
        
        mock_data_service.batch_query_stocks.return_value = mock_batch_response
        
        response = client.post(
            "/api/v1/data-sync/stocks/batch",
            json=["000001", "000002"]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success_count"] == 1
        assert len(data["stocks"]) == 1
    
    def test_batch_query_stocks_limit_exceeded(self, client, mock_data_service):
        """测试批量查询股票数量超限"""
        symbols = [f"{i:06d}" for i in range(51)]  # 51个股票代码
        
        response = client.post(
            "/api/v1/data-sync/stocks/batch",
            json=symbols
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "批量查询股票数量不能超过50个" in data["error"]["message"]


class TestDataManagementAPI:
    """数据管理API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_data_service(self):
        """模拟数据服务"""
        with patch('backend.app.api.data_sync.get_data_service_dep') as mock:
            service = MagicMock()
            mock.return_value = service
            yield service
    
    def test_get_available_symbols(self, client, mock_data_service):
        """测试获取可用股票代码"""
        mock_symbols = ["000001", "000002", "600000", "600036"]
        mock_data_service.get_available_symbols.return_value = mock_symbols
        
        response = client.get("/api/v1/data-sync/symbols")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 4
        assert "000001" in data["symbols"]
    
    def test_get_available_symbols_with_pattern(self, client, mock_data_service):
        """测试带过滤模式获取股票代码"""
        mock_symbols = ["000001", "000002", "600000", "600036"]
        mock_data_service.get_available_symbols.return_value = mock_symbols
        
        response = client.get("/api/v1/data-sync/symbols?pattern=000")
        
        assert response.status_code == 200
        data = response.json()
        # 应该只返回包含"000"的代码
        filtered_symbols = [s for s in data["symbols"] if "000" in s]
        assert len(filtered_symbols) == len(data["symbols"])
    
    def test_get_data_statistics(self, client, mock_data_service):
        """测试获取数据统计信息"""
        mock_stats = {
            "total_symbols": 100,
            "total_records": 50000,
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "last_update": datetime.utcnow().isoformat(),
            "data_sources": ["akshare"],
            "storage_size_mb": 256.5
        }
        
        mock_data_service.get_data_statistics.return_value = mock_stats
        
        response = client.get("/api/v1/data-sync/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_symbols"] == 100
        assert data["storage_size_mb"] == 256.5
        assert "akshare" in data["data_sources"]
    
    def test_data_health_check(self, client, mock_data_service):
        """测试数据健康检查"""
        mock_health = {
            "status": "healthy",
            "checks": {
                "akshare_adapter": True,
                "parquet_storage": True,
                "sync_tasks": True
            },
            "statistics": {
                "total_symbols": 100,
                "total_records": 50000
            },
            "last_sync_tasks": [],
            "issues": [],
            "recommendations": []
        }
        
        mock_data_service.health_check.return_value = mock_health
        
        response = client.get("/api/v1/data-sync/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert all(data["checks"].values())
        assert len(data["issues"]) == 0


class TestTaskQueueAPI:
    """任务队列API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_task_queue(self):
        """模拟任务队列"""
        with patch('backend.app.api.data_sync.get_task_queue_dep') as mock:
            queue = MagicMock()
            mock.return_value = queue
            yield queue
    
    def test_get_task_queue_status(self, client, mock_task_queue):
        """测试获取任务队列状态"""
        mock_stats = {
            "total_tasks": 10,
            "pending_tasks": 2,
            "running_tasks": 1,
            "completed_tasks": 6,
            "failed_tasks": 1,
            "cancelled_tasks": 0,
            "max_concurrent": 3
        }
        
        mock_task_queue.get_queue_stats.return_value = mock_stats
        
        response = client.get("/api/v1/data-sync/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_tasks"] == 10
        assert data["running_tasks"] == 1
        assert data["max_concurrent"] == 3
    
    def test_schedule_background_task(self, client, mock_task_queue):
        """测试调度后台任务"""
        mock_task_queue.add_task.return_value = "bg-task-456"
        
        response = client.post(
            "/api/v1/data-sync/tasks/schedule",
            params={
                "task_type": "data_sync",
                "priority": 7,
                "symbols": ["000001", "000002"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "bg-task-456"
        assert data["task_type"] == "data_sync"
        assert data["priority"] == 7
    
    def test_schedule_background_task_invalid_type(self, client, mock_task_queue):
        """测试调度无效类型的后台任务"""
        response = client.post(
            "/api/v1/data-sync/tasks/schedule",
            params={"task_type": "invalid_type"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "不支持的任务类型" in data["error"]["message"]
    
    def test_cancel_background_task(self, client, mock_task_queue):
        """测试取消后台任务"""
        mock_task_queue.cancel_task.return_value = True
        
        response = client.delete("/api/v1/data-sync/tasks/bg-task-456")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "任务已取消"
        assert data["task_id"] == "bg-task-456"


class TestMaintenanceAPI:
    """维护API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture  
    def mock_services(self):
        """模拟所有服务"""
        with patch('backend.app.api.data_sync.get_data_service_dep') as data_mock, \
             patch('backend.app.api.data_sync.get_task_queue_dep') as queue_mock:
            
            data_service = MagicMock()
            task_queue = MagicMock()
            
            data_mock.return_value = data_service
            queue_mock.return_value = task_queue
            
            yield data_service, task_queue
    
    def test_maintenance_cleanup(self, client, mock_services):
        """测试清理维护"""
        data_service, task_queue = mock_services
        
        response = client.post(
            "/api/v1/data-sync/maintenance/cleanup",
            params={
                "max_age_hours": 48,
                "cleanup_tasks": True,
                "cleanup_logs": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "清理维护完成"
        assert data["results"]["tasks_cleaned"] is True
        
        # 验证清理方法被调用
        data_service.cleanup_old_tasks.assert_called_once_with(48)
        task_queue.cleanup_completed_tasks.assert_called_once_with(48)


# 开发模式测试
class TestDevAPI:
    """开发模式API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_data_service(self):
        """模拟数据服务"""
        with patch('backend.app.api.data_sync.get_data_service_dep') as mock:
            service = MagicMock()
            service.parquet_storage.save_stock_data.return_value = True
            mock.return_value = service
            yield service
    
    @patch('backend.app.core.config.settings.DEBUG', True)
    def test_create_test_data(self, client, mock_data_service):
        """测试创建测试数据"""
        response = client.post(
            "/api/v1/data-sync/dev/test-data",
            params={"symbol": "TEST001", "days": 30}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "TEST001"
        assert data["bars_count"] == 30
        assert "date_range" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
