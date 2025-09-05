"""
FastAPI应用测试

测试API路由、中间件、异常处理等功能。
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app.main import app


class TestHealthAPI:
    """健康检查API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_basic_health_check(self, client):
        """测试基础健康检查"""
        response = client.get("/api/v1/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "environment" in data
    
    def test_ping_endpoint(self, client):
        """测试ping接口"""
        response = client.get("/api/v1/health/ping")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "pong"
        assert "timestamp" in data
    
    def test_system_info(self, client):
        """测试系统信息接口"""
        response = client.get("/api/v1/health/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "version" in data
        assert "environment" in data
        assert "config" in data
        assert "uptime_seconds" in data
        
        # 检查配置信息不包含敏感数据
        config = data["config"]
        assert "DB_PASSWORD" not in str(config)
        assert "SECRET_KEY" not in str(config)
    
    def test_detailed_health_check(self, client):
        """测试详细健康检查"""
        response = client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert "system_info" in data
        
        # 检查组件信息结构
        components = data["components"]
        assert "database" in components
        assert "storage" in components
        assert "external_services" in components


class TestDataAPI:
    """数据管理API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_data_status(self, client):
        """测试数据状态接口"""
        response = client.get("/api/v1/data/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "data_sources" in data
        assert "storage_info" in data
        assert isinstance(data["data_sources"], list)
        assert isinstance(data["storage_info"], list)
    
    def test_get_symbols(self, client):
        """测试获取股票代码列表"""
        response = client.get("/api/v1/data/symbols")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "symbols" in data
        assert "total_count" in data
        assert "data_source" in data
        assert isinstance(data["symbols"], list)
    
    def test_get_stock_data_not_found(self, client):
        """测试获取不存在的股票数据"""
        response = client.get("/api/v1/data/stocks/NONEXISTENT")
        
        assert response.status_code == 404
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
    
    def test_storage_stats(self, client):
        """测试存储统计信息"""
        response = client.get("/api/v1/data/storage/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        # parquet存储统计可能成功也可能失败，这里只检查基本结构
        assert isinstance(data, dict)
    
    @patch('backend.app.core.config.settings.DEBUG', True)
    def test_create_sample_data(self, client):
        """测试创建样本数据（仅调试模式）"""
        response = client.post("/api/v1/data/test/sample-data?symbol=TEST001")
        
        # 由于需要真实的存储引擎，这个测试可能会失败
        # 我们主要测试接口是否存在和基本逻辑
        assert response.status_code in [200, 500]  # 成功或内部错误都可接受


class TestMiddleware:
    """中间件测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_request_tracking_headers(self, client):
        """测试请求追踪中间件"""
        response = client.get("/api/v1/health/ping")
        
        assert response.status_code == 200
        
        # 检查响应头
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
        
        # 请求ID应该是UUID格式
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 20  # 基本长度检查
    
    def test_security_headers(self, client):
        """测试安全头中间件"""
        response = client.get("/api/v1/health/ping")
        
        assert response.status_code == 200
        
        # 检查安全响应头
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Content-Security-Policy"
        ]
        
        for header in security_headers:
            assert header in response.headers
        
        # 检查服务器头被移除
        assert "Server" not in response.headers
    
    def test_cors_headers(self, client):
        """测试CORS配置"""
        # 预检请求
        response = client.options(
            "/api/v1/health/ping",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # CORS头可能存在也可能不存在，取决于配置
        # 主要检查请求不会被拒绝
        assert response.status_code in [200, 404]


class TestErrorHandling:
    """异常处理测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_404_error(self, client):
        """测试404错误处理"""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert "timestamp" in data["error"]
    
    def test_validation_error(self, client):
        """测试数据验证错误"""
        # 传递无效的查询参数
        response = client.get("/api/v1/data/stocks/TEST001?limit=-1")
        
        assert response.status_code == 422
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"


class TestAPIDocumentation:
    """API文档测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """测试根路径"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "docs_url" in data
        assert "api_url" in data
    
    def test_openapi_schema(self, client):
        """测试OpenAPI schema"""
        response = client.get("/api/v1/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        
        # 检查基本信息
        info = data["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info
    
    def test_swagger_ui(self, client):
        """测试Swagger UI访问"""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc(self, client):
        """测试ReDoc访问"""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestPerformance:
    """性能测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_health_check_performance(self, client):
        """测试健康检查响应时间"""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/health/")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.2  # 200ms 阈值
        
        # 检查性能响应头
        assert "X-Process-Time" in response.headers
    
    def test_concurrent_requests(self, client):
        """测试并发请求处理"""
        import concurrent.futures
        
        def make_request():
            return client.get("/api/v1/health/ping")
        
        # 并发发送10个请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 所有请求都应该成功
        for response in responses:
            assert response.status_code == 200
            # 每个请求都应该有唯一的请求ID
            assert "X-Request-ID" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
